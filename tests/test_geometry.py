from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from embed_optim.geometry import TensorStore, analyze_run, matrix_metrics
from embed_optim.optimizers import parameter_partition_name


def test_parameter_partition_name_matches_training_routing():
    assert parameter_partition_name("encoder.layers.0.weight", 2) == "hidden"
    assert parameter_partition_name("layers.1.weight", 2) == "hidden"
    assert parameter_partition_name("encoder.layers.0.bias", 1) == "aux_no_decay"
    assert parameter_partition_name("encoder.embeddings.weight", 2) == "aux_decay"
    assert parameter_partition_name("encoder.layers.0.head.weight", 2) == "aux_decay"
    assert parameter_partition_name("encoder.layer_norm.weight", 1) == "aux_no_decay"


def test_matrix_metrics_exact_spectrum():
    metrics = matrix_metrics(torch.diag(torch.tensor([2.0, 1.0])), sketch_rank=2)
    assert metrics["algorithm"] == "exact"
    assert metrics["frobenius_norm"] == pytest.approx(math.sqrt(5))
    assert metrics["spectral_norm"] == pytest.approx(2.0)
    assert metrics["approx_stable_rank"] == pytest.approx(1.25)
    assert metrics["sketched_nuclear_norm"] == pytest.approx(3.0)
    assert metrics["captured_frobenius_energy"] == pytest.approx(1.0)
    assert metrics["row_norms"]["cv"] == pytest.approx(1 / 3)


def test_matrix_metrics_randomized_sketch_is_seeded():
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(12, 7, generator=generator)
    first = matrix_metrics(matrix, sketch_rank=3, oversample=2, power_iterations=1, seed=99)
    second = matrix_metrics(matrix, sketch_rank=3, oversample=2, power_iterations=1, seed=99)
    assert first == second
    assert first["algorithm"] == "randomized"
    assert first["rank"] == 3
    assert 0 < first["captured_frobenius_energy"] <= 1


def _write_checkpoint(path: Path, hidden: torch.Tensor) -> None:
    path.mkdir(parents=True)
    save_file(
        {
            "encoder.layers.0.weight": hidden,
            "encoder.embeddings.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "encoder.layers.0.bias": torch.ones(2),
        },
        path / "model.safetensors",
    )


def test_tensor_store_namespaces_sentence_transformer_modules(tmp_path: Path):
    save_file(
        {"encoder.layers.0.weight": torch.eye(2), "encoder.embeddings.weight": torch.eye(2)},
        tmp_path / "model.safetensors",
    )
    module = tmp_path / "1_Dense"
    module.mkdir()
    save_file({"linear.weight": torch.ones(3, 2)}, module / "model.safetensors")
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "name": "0", "path": ""},
                {"idx": 1, "name": "1", "path": "1_Dense"},
            ]
        ),
        encoding="utf-8",
    )

    with TensorStore(tmp_path) as store:
        assert store.keys() == [
            "0.encoder.embeddings.weight",
            "0.encoder.layers.0.weight",
            "1.linear.weight",
        ]
        assert store.shape("1.linear.weight") == (3, 2)
        assert torch.equal(store.tensor("1.linear.weight"), torch.ones(3, 2))
    assert parameter_partition_name("0.encoder.layers.0.weight", 2) == "hidden"
    assert parameter_partition_name("1.linear.weight", 2) == "aux_decay"


def _write_tiny_run(root: Path) -> tuple[Path, Path]:
    run = root / "outputs" / "dense" / "muon-lr1e-4"
    _write_checkpoint(run / "checkpoint-1", torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    _write_checkpoint(run / "checkpoint-2", torch.tensor([[2.0, 0.0], [0.0, 1.0]]))
    (run / "completed.json").write_text(
        json.dumps(
            {
                "run_id": "muon-lr1e-4",
                "model_family": "dense",
                "checkpoints": [1, 2],
                "dataset_fingerprint": "fixed-data",
                "optimizer_partition": {
                    "hidden": {"tensors": 1, "parameters": 4},
                    "aux_decay": {"tensors": 1, "parameters": 8},
                    "aux_no_decay": {"tensors": 1, "parameters": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "run_config.json").write_text(
        json.dumps({"optimizer": {"name": "muon", "lr": 1e-4}}), encoding="utf-8"
    )
    reference = root / "reference"
    _write_checkpoint(reference, torch.tensor([[0.0, 0.0], [0.0, 1.0]]))
    return run, reference


def test_analyze_run_streams_partitioned_records_and_resumes(tmp_path: Path):
    run, reference = _write_tiny_run(tmp_path)
    output = tmp_path / "analysis"
    manifest = analyze_run(
        run,
        output,
        reference=reference,
        sketch_rank=2,
        oversample=0,
        power_iterations=0,
    )

    assert set(manifest["records"]) == {"1", "2"}
    assert manifest["partition_summary"] == {
        "hidden": {"tensors": 1, "parameters": 4},
        "aux_decay": {"tensors": 1, "parameters": 8},
        "aux_no_decay": {"tensors": 1, "parameters": 2},
    }
    first = json.loads((output / "records" / "checkpoint-1.jsonl").read_text())
    second = json.loads((output / "records" / "checkpoint-2.jsonl").read_text())
    assert first["tensor"] == "encoder.layers.0.weight"
    assert first["delta_from_reference"]["frobenius_norm"] == pytest.approx(1.0)
    assert "delta_from_previous" not in first
    assert second["delta_from_previous"]["frobenius_norm"] == pytest.approx(1.0)
    assert second["delta_from_reference"]["frobenius_norm"] == pytest.approx(2.0)

    before = (output / "records" / "checkpoint-2.jsonl").stat().st_mtime_ns
    resumed = analyze_run(
        run,
        output,
        reference=reference,
        sketch_rank=2,
        oversample=0,
        power_iterations=0,
    )
    assert resumed == manifest
    assert (output / "records" / "checkpoint-2.jsonl").stat().st_mtime_ns == before

    checkpoint_two = output / "records" / "checkpoint-2.jsonl"
    checkpoint_two.write_text(checkpoint_two.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its manifest"):
        analyze_run(
            run,
            output,
            reference=reference,
            sketch_rank=2,
            oversample=0,
            power_iterations=0,
        )


def test_analyze_run_rejects_partition_manifest_mismatch(tmp_path: Path):
    run, _ = _write_tiny_run(tmp_path)
    completed_path = run / "completed.json"
    completed = json.loads(completed_path.read_text())
    completed["optimizer_partition"]["hidden"]["tensors"] = 2
    completed_path.write_text(json.dumps(completed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match completed.json"):
        analyze_run(run, tmp_path / "analysis", sketch_rank=0, max_checkpoints=1)
