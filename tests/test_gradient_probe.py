from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from embed_optim.geometry import _sha256
from embed_optim.gradient_probe import (
    _hidden_parameter_mapping,
    _sample_ids_sha256,
    _selection_records,
    balanced_probe_indices,
    export_gradient_probe,
)
from embed_optim.update_geometry import analyze_common_state_updates


def _rows() -> list[dict]:
    rows = []
    for sample_id, source in ((11, "fiqa"), (19, "nq"), (23, "fiqa"), (29, "nq")):
        row = {
            "sample_id": sample_id,
            "source": source,
            "query": f"query {sample_id}",
            "positive": f"positive {sample_id}",
        }
        for index in range(7):
            row[f"negative_{index}"] = f"negative {sample_id} {index}"
        rows.append(row)
    return rows


def _probe_fixture(root: Path) -> Path:
    probe = root / "probe"
    dataset = Dataset.from_list(_rows())
    dataset.save_to_disk(str(probe / "dataset"))
    serialized = Dataset.load_from_disk(str(probe / "dataset"))
    selected_ids = hashlib.sha256()
    for sample_id in serialized["sample_id"]:
        selected_ids.update(f"{int(sample_id)}\n".encode())
    (probe / "selection.jsonl").write_text("fixture selection\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "count": len(serialized),
        "negative_candidates": 7,
        "positive_candidate_index": 0,
        "selected_sample_ids_sha256": selected_ids.hexdigest(),
        "selection_sha256": _sha256(probe / "selection.jsonl"),
        "serialized_probe_dataset_fingerprint": serialized._fingerprint,
    }
    (probe / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return probe


class FakeGradientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        with torch.no_grad():
            self.encoder.layers[0].weight.copy_(torch.tensor([[1.0, 0.5], [-0.5, 2.0]]))


class FakeCollator:
    def __call__(self, rows):
        batch = {"return_loss": True}
        columns = ("query", "positive", *(f"negative_{index}" for index in range(7)))
        for column_index, column in enumerate(columns):
            batch[f"{column}_input_ids"] = torch.tensor(
                [[float(row["sample_id"]), float(column_index + 1)] for row in rows]
            )
        return batch


class FakeLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        weight = self.model.encoder.layers[0].weight
        values = torch.stack([feature["input_ids"].float() @ weight.T for feature in features])
        return values.square().mean()


def _checkpoint_fixture(root: Path) -> Path:
    checkpoint = root / "outputs" / "dense" / "fixture" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    save_file(
        {
            "encoder.layers.0.weight": torch.tensor([[1.0, 0.5], [-0.5, 2.0]]),
        },
        checkpoint / "model.safetensors",
    )
    (checkpoint / "config.json").write_text('{"fixture": true}\n', encoding="utf-8")
    (checkpoint.parent / "run_config.json").write_text(
        json.dumps({"model_family": "dense", "temperature": 0.02}) + "\n",
        encoding="utf-8",
    )
    return checkpoint


def _common_state_spec(root: Path, probe: Path) -> Path:
    dataset = Dataset.load_from_disk(str(probe / "dataset"))
    records, selection_sha256 = _selection_records(
        dataset, balanced_probe_indices(dataset, count=4, seed=2718)
    )
    source_counts = {
        source: sum(record["source"] == source for record in records)
        for source in sorted({record["source"] for record in records})
    }
    path = root / "common-state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "probe_manifest_sha256": _sha256(probe / "manifest.json"),
                "selection": {
                    "seed": 2718,
                    "gradient_steps": 2,
                    "examples_per_gradient": 2,
                    "count": 4,
                    "expected_selection_sha256": selection_sha256,
                    "expected_sample_ids_sha256": _sample_ids_sha256(records),
                    "expected_source_counts": source_counts,
                },
                "gradient_protocol": {
                    "micro_batch_size": 1,
                    "max_grad_norm": 1.0,
                    "model_dtype": "float32",
                    "forward_dtype": "float32",
                    "storage_dtype": "float32",
                    "model_mode": "eval",
                    "gradient_checkpointing": False,
                    "weights_advanced": False,
                },
                "operator_protocol": {
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.999,
                    "adam_eps": 1e-8,
                    "muon_momentum": 0.95,
                    "normuon_beta2": 0.95,
                    "ns_steps": 5,
                    "adjust_lr_fn": "original",
                },
                "operator_runtime": {"device": "cpu"},
                "matched_update_normalization": ("per-tensor-frobenius-equals-weight-frobenius"),
                "analysis_protocol": {
                    "sketch_rank": 2,
                    "oversample": 0,
                    "power_iterations": 0,
                    "seed": 42,
                    "matched_update_storage_dtype": "float32",
                    "weight_decay_included": False,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_balanced_probe_selection_is_seeded_and_round_robin():
    dataset = Dataset.from_list(_rows())
    first = balanced_probe_indices(dataset, count=4, seed=17)
    second = balanced_probe_indices(dataset, count=4, seed=17)

    assert first == second
    assert len(set(first)) == 4
    assert [dataset[index]["source"] for index in first] == ["fiqa", "nq", "fiqa", "nq"]
    with pytest.raises(ValueError, match="count must be"):
        balanced_probe_indices(dataset, count=5, seed=17)


def test_hidden_parameter_mapping_canonicalizes_transformer_model_namespace(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file({"0.layers.0.weight": torch.ones(3, 2)}, checkpoint / "model.safetensors")
    parameter = nn.Parameter(torch.ones(3, 2))

    mapping = _hidden_parameter_mapping([("0.model.layers.0.weight", parameter)], checkpoint)

    assert mapping == [("0.model.layers.0.weight", "0.layers.0.weight", parameter)]


def test_gradient_export_is_resumable_and_feeds_common_state_analysis(tmp_path: Path, monkeypatch):
    checkpoint = _checkpoint_fixture(tmp_path)
    probe = _probe_fixture(tmp_path)
    common_state_spec = _common_state_spec(tmp_path, probe)
    model = FakeGradientModel()
    monkeypatch.setattr("embed_optim.gradient_probe._load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        "embed_optim.gradient_probe._probe_components",
        lambda model, family, temperature: (FakeLoss(model), FakeCollator()),
    )
    output = tmp_path / "gradients"

    manifest = export_gradient_probe(
        checkpoint,
        probe,
        output,
        family="dense",
        common_state_spec=common_state_spec,
        gradient_steps=2,
        examples_per_gradient=2,
        micro_batch_size=1,
        model_dtype="float32",
        forward_dtype="float32",
        storage_dtype="float32",
        device="cpu",
        flash_attention=False,
        gradient_checkpointing=False,
    )

    assert manifest["status"] == "complete"
    assert manifest["common_state_spec"]["sha256"] == _sha256(common_state_spec)
    assert len(manifest["gradient_shards"]) == 2
    assert manifest["partition_summary"]["hidden"] == {"tensors": 1, "parameters": 4}
    assert (
        len({sample_id for item in manifest["gradient_shards"] for sample_id in item["sample_ids"]})
        == 4
    )
    for item in manifest["gradient_shards"]:
        path = output / item["path"]
        assert item["sha256"] == _sha256(path)
        assert item["pre_clip_grad_norm"] > 1.0
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert handle.keys() == ["encoder.layers.0.weight"]
            assert handle.metadata()["weights_advanced"] == "false"

    before = (output / "gradient-0001.safetensors").stat().st_mtime_ns
    resumed = export_gradient_probe(
        checkpoint,
        probe,
        output,
        family="dense",
        common_state_spec=common_state_spec,
        gradient_steps=2,
        examples_per_gradient=2,
        micro_batch_size=1,
        model_dtype="float32",
        forward_dtype="float32",
        storage_dtype="float32",
        device="cpu",
        flash_attention=False,
        gradient_checkpointing=False,
    )
    assert resumed == manifest
    assert (output / "gradient-0001.safetensors").stat().st_mtime_ns == before

    update_manifest = analyze_common_state_updates(
        checkpoint,
        output / "manifest.json",
        tmp_path / "updates",
        common_state_spec=common_state_spec,
        sketch_rank=2,
        oversample=0,
        power_iterations=0,
    )
    assert update_manifest["gradient_steps"] == 2
    assert update_manifest["tensors"] == 1


def test_gradient_export_rejects_temperature_drift(tmp_path: Path):
    checkpoint = _checkpoint_fixture(tmp_path)
    probe = _probe_fixture(tmp_path)
    with pytest.raises(ValueError, match="differs from checkpoint"):
        export_gradient_probe(
            checkpoint,
            probe,
            tmp_path / "gradients",
            family="dense",
            gradient_steps=1,
            examples_per_gradient=1,
            temperature=0.03,
            model_dtype="float32",
            forward_dtype="float32",
            device="cpu",
            flash_attention=False,
        )
