from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.geometry import _sha256
from embed_optim.outcome_report import render_outcome_report


def _csv(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _hybrid(root: Path) -> Path:
    rows = []
    for family in ("dense", "late"):
        for learning_rate in (1e-6, 3e-6, 1e-5, 3e-5):
            rows.append(
                {
                    "model_family": family,
                    "learning_rate": learning_rate,
                    "tasks": 14,
                    "adamw_mean_ndcg_at_10": 0.4,
                    "hybrid_adamw_mean_ndcg_at_10": 0.41,
                    "hybrid_minus_adamw_mean": 0.01,
                    "hybrid_task_wins": 8,
                    "task_ties": 0,
                    "hybrid_task_losses": 6,
                }
            )
    declared = _csv(root / "final_summary.csv", rows)
    _manifest(
        root / "summary_manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "evaluations": {
                "native_five_stage_units": 560,
                "native_final_units": 112,
                "hybrid_final_units": 112,
                "tasks": 14,
            },
            "outputs": {"final_summary": declared},
        },
    )
    return root


def _functional(root: Path) -> Path:
    rows = []
    for family in ("dense", "late"):
        for algorithm in ("adamw", "muon", "normuon"):
            for direction, scales in (
                ("descent", (0.0001, 0.0003, 0.001)),
                ("sign_reversal", (0.001,)),
            ):
                for scale in scales:
                    rows.append(
                        {
                            "family": family,
                            "algorithm": algorithm,
                            "direction": direction,
                            "relative_scale": scale,
                            "anchors": 10,
                            "mean_anchor_delta_contrastive_loss": -0.01,
                            "mean_anchor_delta_positive_margin": 0.02,
                            "mean_anchor_delta_reciprocal_rank": 0.03,
                            "mean_anchor_delta_top1_accuracy": 0.04,
                            "anchors_with_lower_loss_fraction": 0.8,
                        }
                    )
    declared = _csv(root / "family_summary.csv", rows)
    _manifest(
        root / "manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "anchors": 20,
            "conditions_per_anchor": 13,
            "anchor_effect_records": 240,
            "optimizer_contrast_records": 160,
            "family_summary_records": 24,
            "outputs": {"family_summary": declared},
        },
    )
    return root


def _short(root: Path) -> Path:
    rows = []
    for family in ("dense", "late"):
        for stage in range(1, 6):
            for treatment, baseline in (
                ("muon", "adamw"),
                ("normuon", "adamw"),
                ("normuon", "muon"),
            ):
                for metric in (
                    "contrastive_loss",
                    "positive_margin",
                    "reciprocal_rank",
                    "top1_accuracy",
                ):
                    rows.append(
                        {
                            "family": family,
                            "stage": stage,
                            "fraction": stage / 5,
                            "treatment": treatment,
                            "baseline": baseline,
                            "metric": metric,
                            "seeds": 3,
                            "mean_delta": -0.01 if metric == "contrastive_loss" else 0.01,
                            "seed_delta_standard_deviation": 0.001,
                            "treatment_seed_wins": 3,
                            "seed_ties": 0,
                            "treatment_seed_losses": 0,
                            "beneficial_direction": (
                                "negative" if metric == "contrastive_loss" else "positive"
                            ),
                        }
                    )
    declared = _csv(root / "paired_dynamics_summary.csv", rows)
    _manifest(
        root / "summary_manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "coverage": {
                "runs": 18,
                "checkpoints": 90,
                "paired_checkpoint_contrasts": 90,
                "paired_dynamics_summaries": 120,
            },
            "outputs": {"paired_summary": declared},
        },
    )
    return root


def _confirmatory(root: Path) -> Path:
    rows = []
    for family in ("dense", "late"):
        for treatment, baseline in (
            ("muon", "adamw"),
            ("normuon", "adamw"),
            ("normuon", "muon"),
        ):
            rows.append(
                {
                    "model_family": family,
                    "treatment": treatment,
                    "baseline": baseline,
                    "seeds": 3,
                    "tasks": 14,
                    "mean_delta_ndcg_at_10": 0.01,
                    "bootstrap_ci_95_lower": 0.001,
                    "bootstrap_ci_95_upper": 0.019,
                    "seed_wins": 3,
                    "seed_ties": 0,
                    "seed_losses": 0,
                    "task_wins_after_seed_average": 9,
                    "task_ties_after_seed_average": 0,
                    "task_losses_after_seed_average": 5,
                }
            )
    declared = _csv(root / "paired_summary.csv", rows)
    _manifest(
        root / "summary_manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "coverage": {
                "seeds": 3,
                "runs": 18,
                "tasks": 14,
                "evaluation_units": 252,
                "paired_contrast_units": 252,
            },
            "outputs": {"paired_summary": declared},
        },
    )
    return root


def _inputs(tmp_path: Path):
    functional = _functional(tmp_path / "functional")
    hybrid = _hybrid(tmp_path / "hybrid")
    short = _short(tmp_path / "short")
    confirmatory = _confirmatory(tmp_path / "confirmatory")
    mechanism = tmp_path / "reports" / "mechanism-summary.md"
    mechanism.parent.mkdir(parents=True, exist_ok=True)
    mechanism.write_text("mechanism evidence\n", encoding="utf-8")
    _manifest(
        mechanism.with_suffix(".manifest.json"),
        {
            "schema_version": 1,
            "complete": True,
            "output": {
                "path": str(mechanism.resolve()),
                "bytes": mechanism.stat().st_size,
                "sha256": _sha256(mechanism),
            },
        },
    )
    blog = tmp_path / "blog.md"
    blog.write_text(
        "before\n<!-- MECHANISM:BEGIN -->\nmechanism evidence\n<!-- MECHANISM:END -->\n"
        "<!-- OUTCOMES:BEGIN -->\nold\n<!-- OUTCOMES:END -->\nafter\n",
        encoding="utf-8",
    )
    return functional, hybrid, short, confirmatory, mechanism, blog


def test_outcome_report_renders_all_causal_and_confirmation_tiers(tmp_path: Path):
    functional, hybrid, short, confirmatory, mechanism, blog = _inputs(tmp_path)
    output = tmp_path / "reports" / "outcome-summary.md"

    manifest = render_outcome_report(
        functional, hybrid, short, confirmatory, mechanism, blog, output
    )
    first = (output.read_bytes(), blog.read_bytes())
    repeated = render_outcome_report(
        functional, hybrid, short, confirmatory, mechanism, blog, output
    )

    assert manifest == repeated
    assert manifest["complete"] is True
    assert (output.read_bytes(), blog.read_bytes()) == first
    text = output.read_text(encoding="utf-8")
    assert "AdamW parameter routing" in text
    assert "matched optimizer directions" in text
    assert "shared checkpoint" in text
    assert "validation-frozen recipe" in text
    assert "old" not in blog.read_text(encoding="utf-8")


def test_outcome_report_rejects_hashed_table_drift(tmp_path: Path):
    functional, hybrid, short, confirmatory, mechanism, blog = _inputs(tmp_path)
    (confirmatory / "paired_summary.csv").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Declared table differs"):
        render_outcome_report(
            functional,
            hybrid,
            short,
            confirmatory,
            mechanism,
            blog,
            tmp_path / "outcome.md",
        )


def test_outcome_report_rejects_stale_mechanism_marker(tmp_path: Path):
    functional, hybrid, short, confirmatory, mechanism, blog = _inputs(tmp_path)
    blog.write_text(blog.read_text().replace("mechanism evidence", "stale"), encoding="utf-8")

    with pytest.raises(ValueError, match="mechanism marker differs"):
        render_outcome_report(
            functional,
            hybrid,
            short,
            confirmatory,
            mechanism,
            blog,
            tmp_path / "outcome.md",
        )
