from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from embed_optim.geometry import analyze_run
from embed_optim.geometry_summary import summarize_geometry


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


def _geometry_fixture(root: Path) -> tuple[Path, Path]:
    run = root / "outputs" / "dense" / "muon-lr1e-4"
    _save_model(run / "checkpoint-1", torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    _save_model(run / "checkpoint-2", torch.tensor([[2.0, 0.0], [0.0, 1.0]]))
    (run / "completed.json").write_text(
        json.dumps(
            {
                "run_id": "muon-lr1e-4",
                "model_family": "dense",
                "checkpoints": [1, 2],
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
    (run / "run_config.json").write_text(
        json.dumps({"optimizer": {"name": "muon", "lr": 1e-4}}), encoding="utf-8"
    )
    reference = root / "reference"
    _save_model(reference, torch.tensor([[0.0, 0.0], [0.0, 1.0]]))
    geometry_root = root / "geometry"
    analyze_run(
        run,
        geometry_root / "dense" / "muon-lr1e-4-exact",
        reference=reference,
        sketch_rank=0,
    )
    matrix = root / "matrix.yaml"
    matrix.write_text(
        """
common:
  dataset_path: data/shared
  checkpoint_fractions: [0.5, 1.0]
models:
  dense:
    model_name: example/base
    model_revision: revision
optimizers:
  - id: muon-lr1e-4
    name: muon
    lr: 0.0001
""".lstrip(),
        encoding="utf-8",
    )
    return geometry_root, matrix


def _paired_geometry_fixture(root: Path) -> tuple[Path, Path]:
    geometry_root, matrix = _geometry_fixture(root)
    run = root / "outputs" / "dense" / "normuon-lr1e-4"
    _save_model(run / "checkpoint-1", torch.tensor([[0.5, 0.0], [0.5, 1.0]]))
    _save_model(run / "checkpoint-2", torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
    (run / "completed.json").write_text(
        json.dumps(
            {
                "run_id": "normuon-lr1e-4",
                "model_family": "dense",
                "checkpoints": [1, 2],
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
    (run / "run_config.json").write_text(
        json.dumps({"optimizer": {"name": "normuon", "lr": 1e-4}}), encoding="utf-8"
    )
    reference = root / "reference"
    analyze_run(
        run,
        geometry_root / "dense" / "normuon-lr1e-4-exact",
        reference=reference,
        sketch_rank=0,
    )
    with matrix.open("a", encoding="utf-8") as handle:
        handle.write(
            """
  - id: normuon-lr1e-4
    name: normuon
    lr: 0.0001
"""
        )
    return geometry_root, matrix


def test_summarize_geometry_validates_and_aggregates(tmp_path: Path):
    geometry_root, matrix = _geometry_fixture(tmp_path)
    output = tmp_path / "reports"
    summary = summarize_geometry(
        geometry_root,
        output,
        matrix_path=matrix,
        verify_inputs=True,
    )

    assert summary["complete"] is True
    assert summary["observed_runs"] == 1
    assert summary["checkpoint_rows"] == 2
    assert "not individual optimizer updates" in summary["contrast_interpretation"]
    with (output / "checkpoint_trajectory.csv").open(newline="") as handle:
        checkpoints = list(csv.DictReader(handle))
    with (output / "run_trajectory_summary.csv").open(newline="") as handle:
        runs = list(csv.DictReader(handle))
    with (output / "optimizer_pair_contrasts.csv").open(newline="") as handle:
        contrasts = list(csv.DictReader(handle))
    with (output / "optimizer_pair_contrast_trajectory.csv").open(newline="") as handle:
        contrast_trajectory = list(csv.DictReader(handle))
    assert b"\r\n" not in (output / "checkpoint_trajectory.csv").read_bytes()
    assert float(checkpoints[0]["reference_displacement_frobenius_norm"]) == pytest.approx(1.0)
    assert checkpoints[0]["previous_checkpoint_displacement_frobenius_norm"] == ""
    assert float(
        checkpoints[1]["previous_checkpoint_displacement_frobenius_norm"]
    ) == pytest.approx(1.0)
    assert float(runs[0]["final_reference_displacement_frobenius_norm"]) == pytest.approx(2.0)
    assert float(runs[0]["coarse_checkpoint_path_length"]) == pytest.approx(2.0)
    assert float(runs[0]["coarse_checkpoint_path_efficiency"]) == pytest.approx(1.0)
    assert contrasts == []
    assert contrast_trajectory == []


def test_summarize_geometry_emits_matched_muon_normuon_contrasts(tmp_path: Path):
    geometry_root, matrix = _paired_geometry_fixture(tmp_path)
    output = tmp_path / "reports"
    summary = summarize_geometry(geometry_root, output, matrix_path=matrix)

    with (output / "optimizer_pair_contrasts.csv").open(newline="") as handle:
        contrasts = list(csv.DictReader(handle))
    with (output / "optimizer_pair_contrast_trajectory.csv").open(newline="") as handle:
        contrast_trajectory = list(csv.DictReader(handle))
    assert summary["outputs"]["optimizer_pair_contrasts.csv"]["rows"] == 1
    assert summary["outputs"]["optimizer_pair_contrast_trajectory.csv"]["rows"] == 2
    assert len(contrasts) == 1
    assert [row["stage"] for row in contrast_trajectory] == ["1", "2"]
    contrast = contrasts[0]
    assert contrast["model_family"] == "dense"
    assert contrast["stage"] == "2"
    assert contrast["step"] == "2"
    assert float(contrast["normuon_to_muon_displacement_ratio"]) == pytest.approx(2**-0.5)
    assert float(contrast["normuon_to_muon_row_cv_ratio"]) == pytest.approx(0.0)
    assert float(contrast["normuon_to_muon_top_1pct_row_energy_ratio"]) == pytest.approx(0.5)


def test_summarize_geometry_rejects_record_corruption(tmp_path: Path):
    geometry_root, matrix = _geometry_fixture(tmp_path)
    record = geometry_root / "dense" / "muon-lr1e-4-exact" / "records" / "checkpoint-2.jsonl"
    record.write_text(record.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its manifest"):
        summarize_geometry(geometry_root, tmp_path / "reports", matrix_path=matrix)
