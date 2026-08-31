from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim import dose_band_analysis
from embed_optim.dose_band_analysis import (
    _anchor_tests,
    _forward_bridge_supported,
    _load_protocol,
    _predictions,
    _validate_evaluation_manifest,
    analyze,
)
from embed_optim.geometry import _sha256


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_missing_upstream_writes_auditable_pending_receipt(tmp_path):
    output = tmp_path / "out"
    manifest = analyze(tmp_path / "missing", tmp_path / "eval.csv", output)
    assert manifest["status"] == "pending"
    assert manifest["complete"] is False
    assert manifest["claimability"] == "pending"
    assert "formal mediation" in manifest["claim_boundary"]
    assert (output / "anchor_tests.csv").read_text().startswith("family,anchor")
    assert json.loads((output / "summary_manifest.json").read_text()) == manifest
    with pytest.raises(ValueError, match="not complete"):
        analyze(tmp_path / "missing", tmp_path / "eval.csv", output, audit=True)


def test_checked_in_protocol_exposes_the_frozen_dose_claim_boundary():
    dose, provenance, anchors = _load_protocol(Path("configs/causal_chain_analysis.json").resolve())
    assert "formal mediation" in dose["claim_boundary"]
    assert provenance["protocol"]["sha256"] == _sha256(Path("configs/causal_chain_analysis.json"))
    assert len(anchors) == 10


def test_anchor_tail_summary_contract_synthesizes_omitted_adam_lambda_zero(tmp_path):
    summary = tmp_path / "summary"
    anchor = "dense/muon-lr1e-3/checkpoint-10"
    conditions = [
        ("muon-native", -0.6, 0.6),
        ("adam-basis__spectrum-lambda-0.25", -0.1, 0.1),
        ("adam-basis__spectrum-lambda-0.50", -0.2, 0.2),
        ("adam-basis__spectrum-lambda-0.75", -0.3, 0.3),
        ("adam-basis__muon-spectrum", -0.5, 0.5),
        ("muon-basis__adam-spectrum", -0.1, 0.1),
        ("adam-basis__muon-head-spectrum", -0.1, 0.1),
        ("adam-basis__muon-middle-spectrum", -0.2, 0.2),
        ("adam-basis__muon-tail-spectrum", -0.4, 0.4),
    ]
    _csv(
        summary / "anchor_query_tail_effects.csv",
        [
            {
                "family": "dense",
                "anchor": anchor,
                "condition": condition,
                "p95_pairwise_loss_contrast": loss,
                "p05_pairwise_margin_contrast": margin,
                "mean_pairwise_loss_contrast": loss,
                "mean_pairwise_margin_contrast": margin,
            }
            for condition, loss, margin in conditions
        ],
    )
    rows, features = _anchor_tests(summary)
    assert rows[0]["loss_dose_monotone"] is True
    assert rows[0]["margin_dose_monotone"] is True
    assert rows[0]["loss_lambda_0.00"] == 0.0
    assert rows[0]["margin_lambda_0.00"] == 0.0
    assert rows[0]["basis_swap_negative_control"] is True
    assert rows[0]["tail_band_best_both_metrics"] is True
    assert features[anchor]["spectrum_margin"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="anchor identities"):
        _anchor_tests(summary, {"dense/pretrained"})

    with (summary / "anchor_query_tail_effects.csv").open("a") as handle:
        handle.write(f"dense,{anchor},unexpected,0,0,0,0\n")
    with pytest.raises(ValueError, match="exact frozen set"):
        _anchor_tests(summary)


def test_anchor_identity_and_family_must_match_exact_dense_grid(tmp_path):
    summary = tmp_path / "summary"
    _csv(
        summary / "anchor_query_tail_effects.csv",
        [
            {
                "family": "late",
                "anchor": "late/pretrained",
                "condition": condition,
                "p95_pairwise_loss_contrast": -0.1,
                "p05_pairwise_margin_contrast": 0.1,
                "mean_pairwise_loss_contrast": -0.1,
                "mean_pairwise_margin_contrast": 0.1,
            }
            for condition in dose_band_analysis.TAIL_CONDITIONS
        ],
    )
    with pytest.raises(ValueError, match="Dense family"):
        _anchor_tests(summary)


def test_tail_band_localization_rejects_co_best_ties(tmp_path):
    summary = tmp_path / "summary"
    anchor = "dense/pretrained"
    rows = []
    for condition in dose_band_analysis.TAIL_CONDITIONS:
        loss, margin = -0.1, 0.1
        if condition == "adam-basis__muon-spectrum":
            loss, margin = -0.5, 0.5
        elif condition == "muon-basis__adam-spectrum":
            loss, margin = -0.1, 0.1
        rows.append(
            {
                "family": "dense",
                "anchor": anchor,
                "condition": condition,
                "p95_pairwise_loss_contrast": loss,
                "p05_pairwise_margin_contrast": margin,
                "mean_pairwise_loss_contrast": loss,
                "mean_pairwise_margin_contrast": margin,
            }
        )
    _csv(summary / "anchor_query_tail_effects.csv", rows)

    tests, _ = _anchor_tests(summary)

    assert tests[0]["tail_band_best_both_metrics"] is False


def test_forward_bridge_uses_only_the_matched_basis_control():
    improvements = {
        "spectrum_loss": 0.20,
        "basis_loss": 0.10,
        "spectrum_margin": 0.05,
        "basis_margin": 0.30,
    }

    assert _forward_bridge_supported(improvements) is True
    improvements["basis_loss"] = 0.20
    assert _forward_bridge_supported(improvements) is False


def test_audit_recomputes_and_rejects_manifest_only_tampering(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    for name in dose_band_analysis.OUTPUTS:
        (output / name).write_text(f"canonical-{name}\n")
    canonical = {
        "schema_version": 1,
        "status": "complete",
        "complete": True,
        "claimability": "claimable",
        "supported": True,
        "outputs": {
            name: {
                "path": str(output / name),
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in dose_band_analysis.OUTPUTS
        },
    }
    tampered = {**canonical, "supported": False}
    (output / "summary_manifest.json").write_text(json.dumps(tampered))

    def fake_analyze(_summary, _evaluation, expected_dir, **_kwargs):
        for name in dose_band_analysis.OUTPUTS:
            (expected_dir / name).write_text(f"canonical-{name}\n")
        return {
            **canonical,
            "outputs": {
                name: {
                    "path": str(expected_dir / name),
                    "bytes": (expected_dir / name).stat().st_size,
                    "sha256": _sha256(expected_dir / name),
                }
                for name in dose_band_analysis.OUTPUTS
            },
        }

    monkeypatch.setattr(dose_band_analysis, "analyze", fake_analyze)
    with pytest.raises(ValueError, match="manifest differs"):
        dose_band_analysis.audit_receipt(
            output,
            summary_dir=tmp_path / "summary",
            evaluation=tmp_path / "evaluation.csv",
            protocol=tmp_path / "protocol.json",
            evaluation_manifest=tmp_path / "coverage.json",
        )


def test_discovery_evaluation_is_bound_to_coverage_manifest(tmp_path):
    repository = tmp_path
    report = repository / "reports/dense-discovery"
    rows = [{"model_family": "dense", "run_id": f"run-{index}"} for index in range(840)]
    evaluation = report / "evaluation_long.csv"
    _csv(evaluation, rows)
    coverage = report / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "complete": True,
                "outputs": {
                    "evaluation_long": {
                        "path": "reports/dense-discovery/evaluation_long.csv",
                        "bytes": evaluation.stat().st_size,
                        "sha256": _sha256(evaluation),
                        "rows": 840,
                    }
                },
            }
        )
    )
    assert _validate_evaluation_manifest(coverage, evaluation)["sha256"] == _sha256(coverage)
    payload = json.loads(coverage.read_text())
    payload["outputs"]["evaluation_long"]["rows"] = 839
    coverage.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen contract"):
        _validate_evaluation_manifest(coverage, evaluation)


def test_forward_bridge_is_84_task_transition_rows_with_run_lr_holdout(tmp_path):
    runs = ("adamw-lr1e-5", "muon-lr1e-3", "normuon-lr1e-3")
    tasks = [f"task-{index}" for index in range(14)]
    features, evaluations = {}, []
    for run_index, run in enumerate(runs):
        for stage, step in ((1, 10), (3, 30)):
            anchor = f"dense/{run}/checkpoint-{step}"
            features[anchor] = {
                task: {
                    "spectrum_loss": run_index + task_index / 10,
                    "spectrum_margin": stage + task_index / 20,
                    "basis_loss": run_index - task_index / 30,
                    "basis_margin": stage - task_index / 40,
                }
                for task_index, task in enumerate(tasks)
            }
            for task_index, task in enumerate(tasks):
                base = run_index + stage + task_index / 100
                for observed_stage, observed_step, score in (
                    (stage, step, base),
                    (stage + 1, step + 10, base + 0.01 * (run_index + task_index + stage)),
                ):
                    evaluations.append(
                        {
                            "model_family": "dense",
                            "run_id": run,
                            "learning_rate": str(10 ** (-5 + run_index)),
                            "stage": observed_stage,
                            "task": task,
                            "checkpoint_step": observed_step,
                            "ndcg_at_10": score,
                        }
                    )
    evaluation = tmp_path / "evaluation.csv"
    _csv(evaluation, evaluations)
    rows, summary = _predictions(
        evaluation,
        features,
        expected_runs=runs,
        expected_tasks=tuple(tasks),
    )
    assert len(rows) == 84
    assert {row["held_out_run"] for row in rows} == set(runs)
    assert set(summary["rmse"]) == {
        "baseline",
        "spectrum_loss",
        "spectrum_margin",
        "basis_loss",
        "basis_margin",
    }

    duplicated = [*evaluations, dict(evaluations[0])]
    _csv(evaluation, duplicated)
    with pytest.raises(ValueError, match="duplicate run/stage/task"):
        _predictions(
            evaluation,
            features,
            expected_runs=runs,
            expected_tasks=tuple(tasks),
        )

    _csv(evaluation, evaluations)
    incomplete = {anchor: dict(task_rows) for anchor, task_rows in features.items()}
    incomplete[next(iter(incomplete))].pop(tasks[0])
    with pytest.raises(ValueError, match="84-row grid"):
        _predictions(
            evaluation,
            incomplete,
            expected_runs=runs,
            expected_tasks=tuple(tasks),
        )
