import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from embed_optim.config import load_matrix
from embed_optim.corrected_execution_sensitivity import (
    _load_protocol,
    assemble_sensitivity_tables,
    build_report,
)


def _synthetic_rows():
    configs = load_matrix("configs/dense_no_packing_retrain.yaml")
    historical_effect = {"adamw": 0.00, "muon": 0.02, "normuon": -0.01}
    corrected_effect = {"adamw": 0.00, "muon": -0.03, "normuon": 0.01}
    historical_rows = []
    corrected_rows = []
    for config in configs:
        optimizer = config.optimizer.name
        rates = sorted(
            candidate.optimizer.lr for candidate in configs if candidate.optimizer.name == optimizer
        )
        dose_index = rates.index(config.optimizer.lr) + 1
        historical_run_id = config.run_id.removeprefix("padded-").replace("adamw", "adamw")
        for stage in range(1, 6):
            base = 0.50 + 0.01 * stage + 0.001 * dose_index
            historical_rows.append(
                {
                    "model_family": "dense",
                    "optimizer": optimizer,
                    "learning_rate": config.optimizer.lr,
                    "run_id": historical_run_id,
                    "stage": stage,
                    "fraction": stage / 5,
                    "checkpoint_step": (782, 1563, 2345, 3126, 3907)[stage - 1],
                    "mean_ndcg_at_10": base + historical_effect[optimizer],
                    "tasks_completed": 14,
                }
            )
            corrected_rows.append(
                {
                    "run_id": config.run_id,
                    "optimizer": optimizer,
                    "learning_rate": config.optimizer.lr,
                    "stage": stage,
                    "progress_fraction": stage / 5,
                    "tasks": 14,
                    "mean_ndcg_at_10": base + corrected_effect[optimizer],
                }
            )
    return configs, historical_rows, corrected_rows


def test_sensitivity_matches_rates_stages_and_reports_rank_reversal() -> None:
    configs, historical_rows, corrected_rows = _synthetic_rows()
    tables = assemble_sensitivity_tables(historical_rows, corrected_rows, configs)
    assert {name: len(rows) for name, rows in tables.items()} == {
        "matched_run_stage_sensitivity": 60,
        "optimizer_stage_sensitivity": 15,
        "optimizer_minus_adamw_sensitivity": 10,
        "stage_optimizer_rankings": 5,
    }
    assert all(row["ranking_changed"] for row in tables["stage_optimizer_rankings"])
    ranking = tables["stage_optimizer_rankings"][0]
    assert ranking["historical_optimizer_order"] == "muon>adamw>normuon"
    assert ranking["corrected_optimizer_order"] == "normuon>adamw>muon"

    muon = next(
        row
        for row in tables["optimizer_minus_adamw_sensitivity"]
        if row["optimizer"] == "muon" and row["stage"] == 5
    )
    assert muon["historical_optimizer_minus_adamw"] == pytest.approx(0.02)
    assert muon["corrected_optimizer_minus_adamw"] == pytest.approx(-0.03)
    assert muon["corrected_minus_historical_contrast_shift"] == pytest.approx(-0.05)
    assert muon["direction_changed"] is True


def test_sensitivity_rejects_incomplete_historical_grid() -> None:
    configs, historical_rows, corrected_rows = _synthetic_rows()
    with pytest.raises(ValueError, match="Historical Dense coverage"):
        assemble_sensitivity_tables(historical_rows[:-1], corrected_rows, configs)


def test_real_historical_table_is_complete_and_matches_corrected_grid() -> None:
    repository = Path(__file__).resolve().parents[1]
    configs = load_matrix(repository / "configs/dense_no_packing_retrain.yaml")
    with (repository / "reports/dense-discovery/checkpoint_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        historical_rows = list(csv.DictReader(handle))
    _, _, corrected_rows = _synthetic_rows()
    tables = assemble_sensitivity_tables(historical_rows, corrected_rows, configs)
    assert len(tables["matched_run_stage_sensitivity"]) == 60


def test_checked_in_sensitivity_protocol_binds_sources_and_history() -> None:
    repository = Path(__file__).resolve().parents[1]
    payload = _load_protocol(
        repository / "configs/dense_no_packing_sensitivity_implementation_protocol.json",
        repository,
    )
    assert payload["expected_outputs"] == {
        "matched_run_stage_sensitivity_rows": 60,
        "optimizer_stage_sensitivity_rows": 15,
        "optimizer_minus_adamw_sensitivity_rows": 10,
        "stage_optimizer_rankings_rows": 5,
    }


def test_build_report_audits_sources_and_writes_all_tables(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    _, _, corrected_rows = _synthetic_rows()
    outcomes_dir = tmp_path / "outcomes"
    outcomes_dir.mkdir()
    table_path = outcomes_dir / "run_stage_scores.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(corrected_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(corrected_rows)
    outcome_protocol = repository / "configs/dense_no_packing_outcome_protocol.json"
    identity = {
        "path": str(table_path),
        "rows": len(corrected_rows),
        "bytes": table_path.stat().st_size,
        "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
    }
    (outcomes_dir / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "coverage": {"task_units": 840},
                "protocol": {"sha256": hashlib.sha256(outcome_protocol.read_bytes()).hexdigest()},
                "outputs": {"run_stage_scores": identity},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "sensitivity"
    report = build_report(
        argparse.Namespace(
            protocol=repository
            / "configs/dense_no_packing_sensitivity_implementation_protocol.json",
            matrix=repository / "configs/dense_no_packing_retrain.yaml",
            historical_coverage=repository / "reports/dense-discovery/coverage.json",
            outcomes_dir=outcomes_dir,
            output_dir=output_dir,
        )
    )
    assert report["status"] == "complete"
    assert report["no_pooling"] is True
    assert report["coverage"]["matched_rows"] == 60
    assert report["outputs"]["optimizer_minus_adamw_sensitivity.csv"]["rows"] == 10
    assert (output_dir / "summary_manifest.json").is_file()
