from __future__ import annotations

import csv
from pathlib import Path

import pytest

from embed_optim.geometry_plot import plot_pair_contrasts
from embed_optim.geometry_summary import PAIR_CONTRAST_FIELDS


def _write_contrasts(path: Path, *, duplicate: bool = False) -> None:
    rows = []
    for family in ("dense", "late"):
        for learning_rate in (1e-4, 3e-4):
            for stage in (1, 2):
                rows.append(
                    {
                        "model_family": family,
                        "learning_rate": learning_rate,
                        "stage": stage,
                        "step": stage * 10,
                        "muon_run_id": f"muon-{learning_rate}",
                        "normuon_run_id": f"normuon-{learning_rate}",
                        "muon_reference_displacement_frobenius_norm": 1.0,
                        "normuon_reference_displacement_frobenius_norm": 1.0,
                        "normuon_to_muon_displacement_ratio": 1.0,
                        "muon_reference_delta_row_cv_parameter_weighted": 0.2,
                        "normuon_reference_delta_row_cv_parameter_weighted": 0.1,
                        "normuon_to_muon_row_cv_ratio": 0.5,
                        "muon_reference_delta_top_1pct_row_energy_parameter_weighted": 0.03,
                        "normuon_reference_delta_top_1pct_row_energy_parameter_weighted": 0.02,
                        "normuon_to_muon_top_1pct_row_energy_ratio": 2 / 3,
                    }
                )
    if duplicate:
        rows.append(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_CONTRAST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_plot_pair_contrasts_is_deterministic(tmp_path: Path):
    input_path = tmp_path / "contrasts.csv"
    output_path = tmp_path / "figure.svg"
    _write_contrasts(input_path)

    first = plot_pair_contrasts(input_path, output_path)
    first_bytes = output_path.read_bytes()
    second = plot_pair_contrasts(input_path, output_path)

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["rows"] == 8
    assert first["families"] == ["dense", "late"]
    assert first_bytes.startswith(b"<?xml")


def test_plot_pair_contrasts_rejects_duplicate_stage(tmp_path: Path):
    input_path = tmp_path / "duplicate.csv"
    _write_contrasts(input_path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate contrast stage"):
        plot_pair_contrasts(input_path, tmp_path / "figure.svg")
