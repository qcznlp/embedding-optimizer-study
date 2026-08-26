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
    with (output / "checkpoint_trajectory.csv").open(newline="") as handle:
        checkpoints = list(csv.DictReader(handle))
    with (output / "run_trajectory_summary.csv").open(newline="") as handle:
        runs = list(csv.DictReader(handle))
    assert float(checkpoints[0]["reference_displacement_frobenius_norm"]) == pytest.approx(1.0)
    assert checkpoints[0]["previous_checkpoint_displacement_frobenius_norm"] == ""
    assert float(
        checkpoints[1]["previous_checkpoint_displacement_frobenius_norm"]
    ) == pytest.approx(1.0)
    assert float(runs[0]["final_reference_displacement_frobenius_norm"]) == pytest.approx(2.0)
    assert float(runs[0]["coarse_checkpoint_path_length"]) == pytest.approx(2.0)
    assert float(runs[0]["coarse_checkpoint_path_efficiency"]) == pytest.approx(1.0)


def test_summarize_geometry_rejects_record_corruption(tmp_path: Path):
    geometry_root, matrix = _geometry_fixture(tmp_path)
    record = geometry_root / "dense" / "muon-lr1e-4-exact" / "records" / "checkpoint-2.jsonl"
    record.write_text(record.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its manifest"):
        summarize_geometry(geometry_root, tmp_path / "reports", matrix_path=matrix)
