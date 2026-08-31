from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.geometry import SCHEMA_VERSION, _sha256
from embed_optim.tail_stability import SHORT_BRANCH_FIELDS
from embed_optim.temporal_short_branch import (
    analyze_rows,
    audit_report,
    build_report,
    support_decision,
)


def _spec() -> dict:
    return {
        "schema_version": 1,
        "status": "prospective_before_short_branch_outputs",
        "frozen_at_utc": "x",
        "family": "dense",
        "seeds": [314159, 271828, 161803],
        "operators": ["adamw", "muon", "normuon"],
        "early_stages": [1, 2],
        "final_stage": 5,
        "predictors": ["spectral"],
        "negative_controls": ["norm"],
        "outcomes": ["validation_loss_p95", "unseen_margin_p05"],
        "beneficial_direction": {
            "validation_loss_p95": "negative",
            "unseen_margin_p05": "positive",
        },
        "analysis": {"claim_rule": "falsifiable", "primary_predictor": "spectral"},
        "claim_boundary": "not formal mediation",
    }


def _rows():
    predictors, outcomes = [], []
    effects = {"adamw": 0.0, "muon": 1.0, "normuon": 1.5}
    for seed_index, seed in enumerate((314159, 271828, 161803)):
        for operator in ("adamw", "muon", "normuon"):
            for stage in range(1, 6):
                effect = effects[operator] * (1 + seed_index * 0.1) + seed_index * 0.1
                predictors.append(
                    {
                        "family": "dense",
                        "seed": str(seed),
                        "operator": operator,
                        "stage": str(stage),
                        "spectral": str(effect),
                        "norm": str(seed_index + stage * 0.01),
                    }
                )
                row = {field: "0" for field in SHORT_BRANCH_FIELDS}
                row.update(
                    {
                        "family": "dense",
                        "seed": str(seed),
                        "operator": operator,
                        "run_id": operator,
                        "stage": str(stage),
                        "fraction": str(stage / 5),
                        "step": str(stage),
                    }
                )
                row["validation_loss_p95"] = str(1.0 - effect)
                row["unseen_margin_p05"] = str(effect)
                outcomes.append(row)
    return predictors, outcomes


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _audit(
    output: Path,
    *,
    protocol: Path,
    predictor_csv: Path,
    predictor_manifest: Path,
    outcome_csv: Path,
    outcome_manifest: Path,
    scope_amendment: Path | None = None,
) -> dict:
    return audit_report(
        output,
        protocol=protocol,
        scope_amendment=scope_amendment or protocol,
        predictor_csv=predictor_csv,
        predictor_manifest=predictor_manifest,
        outcome_csv=outcome_csv,
        outcome_manifest=outcome_manifest,
    )


def test_analysis_covers_pairing_loso_baseline_controls_and_shrinkage():
    paired, predictions, estimates = analyze_rows(_spec(), *_rows())
    assert len(paired) == 6
    assert len(predictions) == 24
    assert len(estimates) == 4
    assert {row["predictor_kind"] for row in estimates} == {"mechanism", "negative_control"}
    spectral = [row for row in estimates if row["predictor"] == "spectral"]
    assert all(row["mediator_rmse"] < row["label_only_rmse"] for row in spectral)
    assert all("muon_absolute_coefficient_shrinkage" in row for row in estimates)


def _passing_estimates() -> list[dict]:
    rows = []
    for outcome in _spec()["outcomes"]:
        rows.extend(
            [
                {
                    "outcome": outcome,
                    "predictor": "spectral",
                    "relative_rmse_improvement": 0.2,
                    "muon_coefficient_label_only": 1.0,
                    "muon_coefficient_with_predictor": 0.5,
                    "normuon_coefficient_label_only": 1.0,
                    "normuon_coefficient_with_predictor": 0.5,
                },
                {
                    "outcome": outcome,
                    "predictor": "norm",
                    "relative_rmse_improvement": 0.1,
                    "muon_coefficient_label_only": 1.0,
                    "muon_coefficient_with_predictor": 1.0,
                    "normuon_coefficient_label_only": 1.0,
                    "normuon_coefficient_with_predictor": 1.0,
                },
            ]
        )
    return rows


def _passing_paired() -> list[dict]:
    return [
        {
            "seed": seed,
            "challenger": challenger,
            "delta_spectral": 1.0,
            "delta_validation_loss_p95": -1.0,
            "delta_unseen_margin_p05": 1.0,
        }
        for seed in _spec()["seeds"]
        for challenger in ("muon", "normuon")
    ]


def test_support_requires_both_outcomes_to_favor_challenger_in_same_seed():
    paired = _passing_paired()
    for challenger in ("muon", "normuon"):
        members = [row for row in paired if row["challenger"] == challenger]
        members[0]["delta_unseen_margin_p05"] = -1.0
        members[2]["delta_validation_loss_p95"] = 1.0

    decision = support_decision(_spec(), paired, _passing_estimates())

    assert decision["criteria"]["outcome_shift"] is False
    assert decision["spectral_temporal_bridge_supported"] is False


def test_support_rejects_coefficient_growth_from_exact_zero_baseline():
    estimates = _passing_estimates()
    primary_loss = next(
        row
        for row in estimates
        if row["predictor"] == "spectral" and row["outcome"] == "validation_loss_p95"
    )
    primary_loss["muon_coefficient_label_only"] = 0.0
    primary_loss["muon_coefficient_with_predictor"] = 0.1

    decision = support_decision(_spec(), _passing_paired(), estimates)

    assert decision["criteria"]["coefficient_behavior"] is False
    assert decision["spectral_temporal_bridge_supported"] is False


def test_missing_upstream_writes_pending_receipt_that_audit_rejects(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(_spec()))
    monkeypatch.setattr("embed_optim.temporal_short_branch._load_spec", lambda path: _spec())
    output = tmp_path / "report"
    result = build_report(
        protocol=protocol,
        predictor_csv=tmp_path / "p.csv",
        predictor_manifest=tmp_path / "p.json",
        outcome_csv=tmp_path / "o.csv",
        outcome_manifest=tmp_path / "o.json",
        output_dir=output,
    )
    assert result["status"] == "pending-not-claimable"
    assert result["complete"] is result["claimable"] is False
    with pytest.raises(RuntimeError, match="pending"):
        _audit(
            output,
            protocol=protocol,
            predictor_csv=tmp_path / "p.csv",
            predictor_manifest=tmp_path / "p.json",
            outcome_csv=tmp_path / "o.csv",
            outcome_manifest=tmp_path / "o.json",
        )


def test_complete_report_is_hash_bound_and_auditable(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(_spec()))
    monkeypatch.setattr("embed_optim.temporal_short_branch._load_spec", lambda path: _spec())
    predictors, outcomes = _rows()
    predictor_csv = tmp_path / "predictors.csv"
    _csv(predictor_csv, predictors)
    predictor_manifest = tmp_path / "predictors.json"
    predictor_manifest.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "analysis_protocol": {
                    "path": str(protocol.resolve()),
                    "bytes": protocol.stat().st_size,
                    "sha256": _sha256(protocol),
                },
                "scope_amendment": {
                    "path": str(protocol.resolve()),
                    "bytes": protocol.stat().st_size,
                    "sha256": _sha256(protocol),
                },
                "output": {
                    "path": str(predictor_csv.resolve()),
                    "bytes": predictor_csv.stat().st_size,
                    "sha256": _sha256(predictor_csv),
                    "rows": 45,
                },
            }
        )
    )
    outcome_csv = tmp_path / "outcomes.csv"
    _csv(outcome_csv, outcomes)
    outcome_manifest = tmp_path / "outcomes.json"
    outcome_manifest.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "outputs": {
                    "short_branch_checkpoint_tail": {
                        "path": str(outcome_csv.resolve()),
                        "bytes": outcome_csv.stat().st_size,
                        "sha256": _sha256(outcome_csv),
                        "rows": 45,
                    }
                },
            }
        )
    )
    output = tmp_path / "report"
    result = build_report(
        protocol=protocol,
        predictor_csv=predictor_csv,
        predictor_manifest=predictor_manifest,
        outcome_csv=outcome_csv,
        outcome_manifest=outcome_manifest,
        output_dir=output,
    )
    assert result["complete"] is result["claimable"] is True
    persisted_paths = [
        result["protocol"]["path"],
        *(record["path"] for record in result["sources"]),
        *(record["path"] for record in result["outputs"].values()),
    ]
    assert all(not Path(path).is_absolute() for path in persisted_paths)
    assert all(Path(path).as_posix() == path for path in persisted_paths)
    assert (
        _audit(
            output,
            protocol=protocol,
            predictor_csv=predictor_csv,
            predictor_manifest=predictor_manifest,
            outcome_csv=outcome_csv,
            outcome_manifest=outcome_manifest,
        )["coverage"]["checkpoints"]
        == 45
    )
    manifest_path = output / "summary_manifest.json"
    original = json.loads(manifest_path.read_text())
    alternate_scope = tmp_path / "alternate-scope.json"
    alternate_scope.write_bytes(protocol.read_bytes())
    with pytest.raises(RuntimeError, match="different protocol/scope"):
        _audit(
            output,
            protocol=protocol,
            scope_amendment=alternate_scope,
            predictor_csv=predictor_csv,
            predictor_manifest=predictor_manifest,
            outcome_csv=outcome_csv,
            outcome_manifest=outcome_manifest,
        )
    alternate_predictors = tmp_path / "alternate-predictors.csv"
    alternate_predictors.write_bytes(predictor_csv.read_bytes())
    with pytest.raises(RuntimeError, match="CLI protocol/input bindings"):
        _audit(
            output,
            protocol=protocol,
            predictor_csv=alternate_predictors,
            predictor_manifest=predictor_manifest,
            outcome_csv=outcome_csv,
            outcome_manifest=outcome_manifest,
        )
    altered = json.loads(manifest_path.read_text())
    altered["decision"]["spectral_temporal_bridge_supported"] = not altered["decision"][
        "spectral_temporal_bridge_supported"
    ]
    manifest_path.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="manifest fields"):
        _audit(
            output,
            protocol=protocol,
            predictor_csv=predictor_csv,
            predictor_manifest=predictor_manifest,
            outcome_csv=outcome_csv,
            outcome_manifest=outcome_manifest,
        )
    manifest_path.write_text(json.dumps(original))
    (output / "estimates.csv").write_text("tampered")
    with pytest.raises(RuntimeError, match="differs"):
        _audit(
            output,
            protocol=protocol,
            predictor_csv=predictor_csv,
            predictor_manifest=predictor_manifest,
            outcome_csv=outcome_csv,
            outcome_manifest=outcome_manifest,
        )
