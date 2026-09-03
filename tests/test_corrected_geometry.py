from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from embed_optim.config import load_matrix
from embed_optim.corrected_geometry_matrix import _load_protocol, _validate_matrix
from embed_optim.corrected_geometry_summary import summarize_corrected_geometry
from embed_optim.geometry import _sha256, analyze_run

REPOSITORY = Path(__file__).resolve().parents[1]


def _save_model(path: Path, hidden: torch.Tensor) -> None:
    path.mkdir(parents=True)
    save_file(
        {
            "encoder.layers.0.weight": hidden,
            "encoder.embeddings.weight": torch.ones(2, 2),
            "encoder.layers.0.bias": torch.ones(2),
        },
        path / "model.safetensors",
    )


def _fixture(root: Path) -> tuple[list, Path, Path]:
    output_root = root / "outputs"
    optimizers = []
    learning_rates = {
        "adamw": (1e-6, 3e-6, 1e-5, 3e-5),
        "muon": (1e-4, 3e-4, 1e-3, 3e-3),
        "normuon": (1e-4, 3e-4, 1e-3, 3e-3),
    }
    for optimizer, rates in learning_rates.items():
        for index, learning_rate in enumerate(rates):
            optimizers.append(
                f"  - id: padded-{optimizer}-{index}\n"
                f"    name: {optimizer}\n"
                f"    lr: {learning_rate:.1e}\n"
            )
    matrix = root / "matrix.yaml"
    matrix.write_text(
        (
            "common:\n"
            f"  output_root: {output_root}\n"
            f"  dataset_path: {root / 'data'}\n"
            "  dense_can_flatten_inputs: false\n"
            "  checkpoint_fractions: [0.2, 0.4, 0.6, 0.8, 1.0]\n"
            "models:\n"
            "  dense:\n"
            "    model_name: example/base\n"
            "    model_revision: revision\n"
            "optimizers:\n" + "".join(optimizers)
        ),
        encoding="utf-8",
    )
    configs = load_matrix(matrix)
    reference = root / "reference"
    _save_model(reference, torch.zeros(2, 2))
    directions = {
        "adamw": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "muon": torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
        "normuon": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
    }
    geometry_root = root / "geometry"
    for config in configs:
        for step in range(1, 6):
            _save_model(
                config.output_dir / f"checkpoint-{step}",
                directions[config.optimizer.name] * step,
            )
        (config.output_dir / "completed.json").write_text(
            json.dumps(
                {
                    "run_id": config.run_id,
                    "model_family": "dense",
                    "checkpoints": [1, 2, 3, 4, 5],
                    "dataset_fingerprint": "fixed-data",
                    "optimizer_partition": {
                        "hidden": {"tensors": 1, "parameters": 4},
                        "aux_decay": {"tensors": 1, "parameters": 4},
                        "aux_no_decay": {"tensors": 1, "parameters": 2},
                    },
                }
            ),
            encoding="utf-8",
        )
        (config.output_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "optimizer": {
                        "name": config.optimizer.name,
                        "lr": config.optimizer.lr,
                    }
                }
            ),
            encoding="utf-8",
        )
        analyze_run(
            config.output_dir,
            geometry_root / "dense" / f"{config.run_id}-rank64",
            reference=reference,
            sketch_rank=2,
            oversample=0,
            power_iterations=0,
            seed=19,
        )
    protocol = root / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    return configs, reference, protocol


def test_corrected_geometry_emits_complete_grid_and_all_rate_pair_subspaces(tmp_path: Path):
    configs, reference, protocol = _fixture(tmp_path)
    output = tmp_path / "report"

    manifest = summarize_corrected_geometry(
        tmp_path / "geometry",
        output,
        configs,
        reference,
        protocol_path=protocol,
        sketch_rank=2,
        subspace_rank=1,
        oversample=0,
        power_iterations=0,
        seed=19,
    )

    assert manifest["status"] == "complete"
    assert manifest["checkpoint_rows"] == 60
    assert manifest["run_pair_subspace_rows"] == 660
    assert manifest["optimizer_pair_rows"] == 60
    assert "not a per-step optimizer update" in manifest["claim_boundary"]
    with (output / "checkpoint_geometry.csv").open(newline="") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    with (output / "run_pair_subspace_overlap.csv").open(newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    assert len(checkpoint_rows) == 60
    assert len(pair_rows) == 660
    first = checkpoint_rows[0]
    assert float(first["saved_segment_to_weight_ratio"]) == pytest.approx(1.0)
    assert float(first["saved_segment_stable_rank_parameter_weighted"]) == pytest.approx(1.0)
    adamw_muon = next(
        row
        for row in pair_rows
        if row["stage"] == "1"
        and row["displacement_kind"] == "saved_segment"
        and row["first_optimizer"] == "adamw"
        and row["second_optimizer"] == "muon"
    )
    adamw_normuon = next(
        row
        for row in pair_rows
        if row["stage"] == "1"
        and row["displacement_kind"] == "saved_segment"
        and row["first_optimizer"] == "adamw"
        and row["second_optimizer"] == "normuon"
    )
    assert float(adamw_muon["mean_subspace_overlap"]) == pytest.approx(1.0)
    assert float(adamw_normuon["left_subspace_overlap"]) == pytest.approx(1.0)
    assert float(adamw_normuon["right_subspace_overlap"]) == pytest.approx(0.0)
    assert float(adamw_normuon["mean_subspace_overlap"]) == pytest.approx(0.5)


def test_corrected_geometry_matrix_validation_rejects_packed_config(tmp_path: Path):
    configs, _, _ = _fixture(tmp_path)
    _validate_matrix(configs)
    configs[0] = replace(configs[0], dense_can_flatten_inputs=True)
    with pytest.raises(ValueError, match="12-run padded Dense"):
        _validate_matrix(configs)


def test_corrected_geometry_protocol_is_source_and_parent_bound(tmp_path: Path):
    source = tmp_path / "source.py"
    parent = tmp_path / "parent.json"
    source.write_text("value = 1\n", encoding="utf-8")
    parent.write_text("{}\n", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "status": "prospective_corrected_analysis_lock",
                "source_bindings": {
                    "source": {
                        "path": "source.py",
                        "bytes": source.stat().st_size,
                        "sha256": _sha256(source),
                    }
                },
                "parent_bindings": {"parent": {"path": "parent.json", "sha256": _sha256(parent)}},
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_protocol(protocol, tmp_path)
    assert loaded["status"] == "prospective_corrected_analysis_lock"
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source binding mismatch"):
        _load_protocol(protocol, tmp_path)


def test_checked_in_corrected_geometry_protocol_matches_current_sources():
    protocol = _load_protocol(
        REPOSITORY / "configs/dense_no_packing_analysis_protocol.json",
        REPOSITORY,
    )
    assert protocol["visibility_at_freeze"]["corrected_checkpoint_weights_visible"] is False
    assert protocol["weight_space"]["expected_outputs"] == {
        "checkpoint_geometry_rows": 60,
        "run_pair_subspace_rows": 660,
        "optimizer_pair_subspace_rows": 60,
    }
