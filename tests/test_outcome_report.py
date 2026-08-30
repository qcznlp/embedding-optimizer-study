from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.geometry import _sha256
from embed_optim.mechanism_report import MECHANISM_MARKERS, _marked_block_record
from embed_optim.outcome_report import render_outcome_report
from embed_optim.scope import resolve_scope


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


def _hybrid(
    root: Path,
    families: tuple[str, ...] = ("dense", "late"),
    scope: dict[str, object] | None = None,
) -> Path:
    rows = []
    for family in families:
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
    payload = {
        "schema_version": 1,
        "complete": True,
        "evaluations": {
            "native_five_stage_units": 280 * len(families),
            "native_final_units": 56 * len(families),
            "hybrid_final_units": 56 * len(families),
            "tasks": 14,
        },
        "outputs": {"final_summary": declared},
    }
    if scope is not None:
        payload.update({"families": list(families), "scope_amendment": scope})
    _manifest(root / "summary_manifest.json", payload)
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


def _short(
    root: Path,
    families: tuple[str, ...] = ("dense", "late"),
    scope: dict[str, object] | None = None,
) -> Path:
    rows = []
    for family in families:
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
                            "mean_delta": (-0.01 if metric == "contrastive_loss" else 0.01),
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
    payload = {
        "schema_version": 1,
        "complete": True,
        "coverage": {
            "runs": 9 * len(families),
            "checkpoints": 45 * len(families),
            "paired_checkpoint_contrasts": 45 * len(families),
            "paired_dynamics_summaries": 60 * len(families),
        },
        "outputs": {"paired_summary": declared},
    }
    if scope is not None:
        payload.update({"families": list(families), "scope_amendment": scope})
    _manifest(root / "summary_manifest.json", payload)
    return root


def _confirmatory(
    root: Path,
    families: tuple[str, ...] = ("dense", "late"),
    scope: dict[str, object] | None = None,
) -> Path:
    rows = []
    for family in families:
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
                    "familywise_method": "bonferroni",
                    "familywise_contrasts": 6,
                    "familywise_ci_95_lower": -0.001,
                    "familywise_ci_95_upper": 0.021,
                    "seed_wins": 3,
                    "seed_ties": 0,
                    "seed_losses": 0,
                    "task_wins_after_seed_average": 9,
                    "task_ties_after_seed_average": 0,
                    "task_losses_after_seed_average": 5,
                }
            )
    declared = _csv(root / "paired_summary.csv", rows)
    payload = {
        "schema_version": 1,
        "complete": True,
        "coverage": {
            "seeds": 3,
            "runs": 9 * len(families),
            "tasks": 14,
            "evaluation_units": 126 * len(families),
            "paired_contrast_units": 126 * len(families),
        },
        "outputs": {"paired_summary": declared},
    }
    if scope is not None:
        payload.update({"families": list(families), "scope_amendment": scope})
    _manifest(root / "summary_manifest.json", payload)
    return root


def _tail(
    root: Path,
    families: tuple[str, ...],
    scope: dict[str, object] | None,
) -> Path:
    family_count = len(families)
    discovery = [
        {
            "family": family,
            "challenger": challenger,
            "reference": "adamw",
            "anchors": 10,
            "tail_fraction": 0.05,
            "tail_size": 12,
            "adam_tail_contrast_mean": -0.10,
            "challenger_tail_contrast_mean": 0.01,
            "tail_jaccard_mean": 0.30,
            "tail_identity_regime": "tail redistribution",
        }
        for family in families
        for challenger in ("muon", "normuon")
    ]
    final = [
        {
            "family": family,
            "challenger": challenger,
            "reference": "adamw",
            "stage": 5,
            "seeds": 3,
            "median_delta_validation_loss_p95": -0.02,
            "validation_loss_p95_seed_wins": 3,
            "median_delta_unseen_margin_p05": 0.01,
            "unseen_margin_p05_seed_wins": 2,
            "tail_stability_decision": "supported",
        }
        for family in families
        for challenger in ("muon", "normuon")
    ]
    counts = {
        "discovery_anchor_tail": 30 * family_count,
        "discovery_family_contrasts": 2 * family_count,
        "discovery_cross_tail": 20 * family_count,
        "discovery_cross_tail_summary": 2 * family_count,
        "short_branch_checkpoint_tail": 45 * family_count,
        "short_branch_checkpoint_contrasts": 30 * family_count,
        "short_branch_final_summary": 2 * family_count,
    }
    outputs = {}
    for name, count in counts.items():
        rows = (
            discovery
            if name == "discovery_cross_tail_summary"
            else (
                final
                if name == "short_branch_final_summary"
                else [{"row": index} for index in range(count)]
            )
        )
        outputs[name] = _csv(root / f"{name}.csv", rows)
    readme = root / "README.md"
    readme.write_text("tail stability\n", encoding="utf-8")
    outputs["readme"] = {
        "path": str(readme.resolve()),
        "bytes": readme.stat().st_size,
        "sha256": _sha256(readme),
    }
    _manifest(
        root / "summary_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "complete": True,
            "discovery_complete": True,
            "short_branch_confirmation_complete": True,
            "analysis_status": ("post_hoc_discovery_with_prospective_short_branch_confirmation"),
            "families": list(families),
            "scope_amendment": scope,
            "pending_reason": None,
            "discovery_anchors": 10 * family_count,
            "discovery_anchor_operator_rows": 30 * family_count,
            "discovery_contrasts": 2 * family_count,
            "discovery_cross_tail_rows": 20 * family_count,
            "discovery_cross_tail_summaries": 2 * family_count,
            "short_branch_checkpoint_rows": 45 * family_count,
            "short_branch_contrast_rows": 30 * family_count,
            "short_branch_final_rows": 2 * family_count,
            "outputs": outputs,
            "claim_boundary": "Post-hoc discovery; persistence does not prove mediation.",
        },
    )
    return root


def _spectral(
    root: Path,
    families: tuple[str, ...],
    scope: dict[str, object] | None,
) -> Path:
    family_count = len(families)
    metrics = (
        "contrastive_loss",
        "positive_score",
        "hardest_negative_score",
        "positive_margin",
        "reciprocal_rank",
        "top1_accuracy",
    )
    conditions = (
        "muon-native",
        "adam-basis__spectrum-lambda-0.25",
        "adam-basis__spectrum-lambda-0.50",
        "adam-basis__spectrum-lambda-0.75",
        "adam-basis__muon-spectrum",
        "muon-basis__adam-spectrum",
        "adam-basis__muon-head-spectrum",
        "adam-basis__muon-middle-spectrum",
        "adam-basis__muon-tail-spectrum",
    )
    factorial = [
        {
            "family": family,
            "metric": metric,
            "anchors": 10,
            "median_spectrum_main_effect": -0.02,
            "median_basis_main_effect": 0.01,
            "median_spectrum_basis_interaction": 0.001,
        }
        for family in families
        for metric in metrics
    ]
    tail = [
        {
            "family": family,
            "condition": condition,
            "anchors": 10,
            "median_p95_pairwise_loss_contrast": -0.02,
            "median_p05_pairwise_margin_contrast": 0.01,
            "median_mean_loss_contrast_on_adam_tail": -0.10,
            "median_mean_loss_contrast_on_condition_tail": -0.05,
            "median_worst_loss_tail_jaccard": 0.40,
        }
        for family in families
        for condition in conditions
    ]
    counts = {
        "anchor_condition_effects": 100 * family_count,
        "family_condition_summary": 60 * family_count,
        "anchor_factorial_effects": 60 * family_count,
        "family_factorial_summary": 6 * family_count,
        "anchor_spectral_path": 300 * family_count,
        "family_spectral_path": 30 * family_count,
        "anchor_band_effects": 180 * family_count,
        "family_band_summary": 18 * family_count,
        "anchor_query_tail_effects": 90 * family_count,
        "family_query_tail_summary": 9 * family_count,
    }
    outputs = {}
    for name, count in counts.items():
        rows = (
            factorial
            if name == "family_factorial_summary"
            else (
                tail
                if name == "family_query_tail_summary"
                else [{"row": index} for index in range(count)]
            )
        )
        outputs[name] = _csv(root / f"{name}.csv", rows)
    _manifest(
        root / "summary_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "complete": True,
            "analysis_status": "post_hoc_explanatory_intervention",
            "families": list(families),
            "scope_amendment": scope,
            "anchors": 10 * family_count,
            "anchor_effect_records": 100 * family_count,
            "anchor_tail_effect_records": 90 * family_count,
            "outputs": outputs,
            "claim_boundary": "Fixed-state causal attribution cannot establish BEIR mediation.",
        },
    )
    return root


def _inputs(
    tmp_path: Path,
    families: tuple[str, ...] = ("dense", "late"),
    scope: dict[str, object] | None = None,
):
    functional = _functional(tmp_path / "functional")
    hybrid = _hybrid(tmp_path / "hybrid", families, scope)
    short = _short(tmp_path / "short", families, scope)
    tail = _tail(tmp_path / "tail", families, scope)
    spectral = _spectral(tmp_path / "spectral", families, scope)
    confirmatory = _confirmatory(tmp_path / "confirmatory", families, scope)
    mechanism = tmp_path / "reports" / "mechanism-summary.md"
    mechanism.parent.mkdir(parents=True, exist_ok=True)
    mechanism.write_text("mechanism evidence\n", encoding="utf-8")
    blog = tmp_path / "blog.md"
    blog.write_text(
        "before\n<!-- MECHANISM:BEGIN -->\nmechanism evidence\n<!-- MECHANISM:END -->\n"
        "<!-- OUTCOMES:BEGIN -->\nold\n<!-- OUTCOMES:END -->\nafter\n",
        encoding="utf-8",
    )
    mechanism_payload = {
        "schema_version": 1,
        "complete": True,
        "output": {
            "path": str(mechanism.resolve()),
            "bytes": mechanism.stat().st_size,
            "sha256": _sha256(mechanism),
        },
        "blog": _marked_block_record(blog, MECHANISM_MARKERS),
    }
    if scope is not None:
        mechanism_payload.update({"families": list(families), "scope_amendment": scope})
    _manifest(mechanism.with_suffix(".manifest.json"), mechanism_payload)
    return functional, hybrid, short, tail, spectral, confirmatory, mechanism, blog


def test_outcome_report_renders_all_causal_and_confirmation_tiers(tmp_path: Path):
    functional, hybrid, short, tail, spectral, confirmatory, mechanism, blog = _inputs(tmp_path)
    output = tmp_path / "reports" / "outcome-summary.md"

    manifest = render_outcome_report(
        functional,
        hybrid,
        short,
        confirmatory,
        mechanism,
        blog,
        output,
        tail_stability_dir=tail,
        spectral_transplant_dir=spectral,
    )
    first = (output.read_bytes(), blog.read_bytes())
    repeated = render_outcome_report(
        functional,
        hybrid,
        short,
        confirmatory,
        mechanism,
        blog,
        output,
        tail_stability_dir=tail,
        spectral_transplant_dir=spectral,
    )

    assert manifest == repeated
    assert manifest["complete"] is True
    assert (output.read_bytes(), blog.read_bytes()) == first
    text = output.read_text(encoding="utf-8")
    assert "AdamW parameter routing" in text
    assert "matched optimizer directions" in text
    assert "shared checkpoint" in text
    assert "validation-frozen recipe" in text
    assert "tail signature survive accumulation" in text
    assert "Post-hoc spectrum-versus-basis causal decomposition" in text
    assert set(manifest["sources"]) == {
        "mechanism_report",
        "functional_intervention",
        "hybrid_adamw",
        "short_branch",
        "tail_stability",
        "spectral_transplant",
        "confirmation",
    }
    assert len(manifest["source_tables"]) == 8
    assert "sha256" not in manifest["blog"]
    assert "bytes" not in manifest["blog"]
    assert manifest["blog"]["block_bytes"] > 0
    assert "familywise 95% CI" in text
    assert "Only the familywise interval" in text
    assert "\nold\n" not in blog.read_text(encoding="utf-8")


def test_outcome_report_renders_strict_dense_scope_without_late_rows(tmp_path: Path):
    families, scope = resolve_scope(("dense",), "configs/dense_scope_amendment.json")
    functional, hybrid, short, tail, spectral, confirmatory, mechanism, blog = _inputs(
        tmp_path, families, scope
    )
    output = tmp_path / "reports" / "outcome-summary.md"

    manifest = render_outcome_report(
        functional,
        hybrid,
        short,
        confirmatory,
        mechanism,
        blog,
        output,
        tail_stability_dir=tail,
        spectral_transplant_dir=spectral,
        families=families,
        scope_amendment=Path("configs/dense_scope_amendment.json"),
    )

    text = output.read_text(encoding="utf-8")
    assert manifest["families"] == ["dense"]
    assert manifest["scope_amendment"] == scope
    assert manifest["sources"]["hybrid_adamw"]["hybrid_units"] == 56
    assert manifest["sources"]["short_branch"]["runs"] == 9
    assert manifest["sources"]["tail_stability"]["anchors"] == 10
    assert manifest["sources"]["spectral_transplant"]["anchors"] == 10
    assert manifest["sources"]["confirmation"]["units"] == 126
    assert "DenseOn" in text
    assert "LateOn" not in text
    assert "all six comparisons prespecified" in text


def test_outcome_report_rejects_incomplete_dense_summary(tmp_path: Path):
    families, scope = resolve_scope(("dense",), "configs/dense_scope_amendment.json")
    functional, hybrid, short, tail, spectral, confirmatory, mechanism, blog = _inputs(
        tmp_path, families, scope
    )
    manifest_path = short / "summary_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["coverage"]["runs"] = 8
    _manifest(manifest_path, payload)

    with pytest.raises(ValueError, match="9-run-per-family"):
        render_outcome_report(
            functional,
            hybrid,
            short,
            confirmatory,
            mechanism,
            blog,
            tmp_path / "outcome.md",
            tail_stability_dir=tail,
            spectral_transplant_dir=spectral,
            families=families,
            scope_amendment=Path("configs/dense_scope_amendment.json"),
        )


def test_outcome_report_rejects_hashed_table_drift(tmp_path: Path):
    functional, hybrid, short, tail, spectral, confirmatory, mechanism, blog = _inputs(tmp_path)
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
            tail_stability_dir=tail,
            spectral_transplant_dir=spectral,
        )


def test_outcome_report_recomputes_tail_decision_from_effect_signs(tmp_path: Path):
    functional, hybrid, short, tail, spectral, confirmatory, mechanism, blog = _inputs(tmp_path)
    table = tail / "short_branch_final_summary.csv"
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["tail_stability_decision"] = "mixed"
    record = _csv(table, rows)
    manifest_path = tail / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["short_branch_final_summary"] = record
    _manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="decision is invalid"):
        render_outcome_report(
            functional,
            hybrid,
            short,
            confirmatory,
            mechanism,
            blog,
            tmp_path / "outcome.md",
            tail_stability_dir=tail,
            spectral_transplant_dir=spectral,
        )


def test_outcome_report_rejects_stale_mechanism_marker(tmp_path: Path):
    functional, hybrid, short, tail, spectral, confirmatory, mechanism, blog = _inputs(tmp_path)
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
            tail_stability_dir=tail,
            spectral_transplant_dir=spectral,
        )


@pytest.mark.parametrize(
    ("directory", "message"),
    (("tail", "Tail-stability"), ("spectral", "Spectral-transplant")),
)
def test_outcome_report_requires_complete_posthoc_decomposition(
    tmp_path: Path, directory: str, message: str
):
    functional, hybrid, short, tail, spectral, confirmatory, mechanism, blog = _inputs(tmp_path)
    target = tail if directory == "tail" else spectral
    manifest_path = target / "summary_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["complete"] = False
    _manifest(manifest_path, payload)

    with pytest.raises(ValueError, match=message):
        render_outcome_report(
            functional,
            hybrid,
            short,
            confirmatory,
            mechanism,
            blog,
            tmp_path / "outcome.md",
            tail_stability_dir=tail,
            spectral_transplant_dir=spectral,
        )


def test_outcome_blog_hash_is_owned_marker_block_not_whole_document(tmp_path: Path):
    functional, hybrid, short, tail, spectral, confirmatory, mechanism, blog = _inputs(tmp_path)
    output = tmp_path / "reports/outcome-summary.md"
    first = render_outcome_report(
        functional,
        hybrid,
        short,
        confirmatory,
        mechanism,
        blog,
        output,
        tail_stability_dir=tail,
        spectral_transplant_dir=spectral,
    )
    blog.write_text("external renderer edit\n" + blog.read_text(encoding="utf-8"), encoding="utf-8")

    repeated = render_outcome_report(
        functional,
        hybrid,
        short,
        confirmatory,
        mechanism,
        blog,
        output,
        tail_stability_dir=tail,
        spectral_transplant_dir=spectral,
    )

    assert repeated["blog"] == first["blog"]
    assert blog.read_text(encoding="utf-8").startswith("external renderer edit\n")
