import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from embed_optim.corrected_publication import (
    FEATURE_LABELS,
    OPTIMIZERS,
    SYSTEM_FIELDS,
    _bridge_table,
    _load_publication_protocol,
    _validate_contrasts,
    build_conclusion,
    load_publication_evidence,
    render_latex,
    render_markdown,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _contrast(treatment: str, baseline: str, mean: float, *, secondary: bool = False):
    row = {
        "treatment": treatment,
        "baseline": baseline,
        "mean": mean,
        "lower": mean - 0.01,
        "upper": mean + 0.01,
        "support": "positive"
        if mean - 0.01 > 0
        else "negative"
        if mean + 0.01 < 0
        else "inconclusive",
    }
    if secondary:
        row.update(
            {
                "treatment_run_id": f"selected-{treatment}",
                "baseline_run_id": f"selected-{baseline}",
            }
        )
    return row


def _evidence():
    primary = [
        _contrast("muon", "adamw", 0.02),
        _contrast("normuon", "adamw", -0.03),
        _contrast("normuon", "muon", -0.05),
    ]
    secondary = [
        _contrast("muon", "adamw", 0.005, secondary=True),
        _contrast("normuon", "adamw", -0.025, secondary=True),
        _contrast("normuon", "muon", -0.03, secondary=True),
    ]
    optimizer_stage = []
    for stage in range(1, 6):
        for optimizer_index, optimizer in enumerate(OPTIMIZERS):
            optimizer_stage.append(
                {
                    "stage": stage,
                    "progress_fraction": stage / 5,
                    "optimizer": optimizer,
                    "mean": 0.5 + 0.01 * stage + 0.001 * optimizer_index,
                    "median": 0.5 + 0.01 * stage + 0.001 * optimizer_index,
                }
            )
    validation_selected = [
        {
            "optimizer": optimizer,
            "run_id": f"selected-{optimizer}",
            "learning_rate_value": 10 ** (-6 + index),
            "loss": 0.5 - 0.01 * index,
            "margin": 0.1 + 0.01 * index,
        }
        for index, optimizer in enumerate(OPTIMIZERS)
    ]
    systems = []
    for index, optimizer in enumerate(OPTIMIZERS):
        row = {"optimizer": optimizer, "runs": 4}
        for field_index, field in enumerate(SYSTEM_FIELDS):
            row[f"mean_{field}"] = 1 + index + field_index / 10
        systems.append(row)
    final_geometry = []
    for index, optimizer in enumerate(OPTIMIZERS):
        final_geometry.append(
            {
                "optimizer": optimizer,
                "saved_segment_to_weight_ratio": 0.001 + index * 0.0001,
                "saved_segment_stable_rank_fraction_parameter_weighted": 0.1 + index * 0.01,
                "saved_segment_sketch_effective_rank_fraction_parameter_weighted": (
                    0.2 + index * 0.01
                ),
                "saved_segment_row_cv_parameter_weighted": 0.3 + index * 0.01,
                "saved_segment_top_1pct_row_energy_parameter_weighted": 0.4 + index * 0.01,
                "cumulative_displacement_to_weight_ratio": 0.01 + index * 0.001,
            }
        )
    bridge = []
    for index, (feature, label) in enumerate(FEATURE_LABELS.items()):
        bridge.append(
            {
                "feature": feature,
                "label": label,
                "baseline_rmse": 0.03,
                "feature_rmse": 0.029 if index == 0 else 0.031,
                "reduction": 0.001 if index == 0 else -0.001,
                "folds_improved": 4 if index == 0 else 2,
                "supported": index == 0,
                "pearson": 0.5 - index * 0.05,
                "spearman": 0.4 - index * 0.04,
            }
        )
    sensitivity = []
    for stage in range(1, 6):
        for index, optimizer in enumerate(("muon", "normuon")):
            historical = 0.01 - index * 0.005
            corrected = -0.02 + index * 0.01
            sensitivity.append(
                {
                    "stage": stage,
                    "optimizer": optimizer,
                    "historical": historical,
                    "corrected": corrected,
                    "shift": corrected - historical,
                }
            )
    rankings = [
        {
            "stage": stage,
            "historical_order": "muon>normuon>adamw",
            "corrected_order": "adamw>normuon>muon",
            "changed": True,
        }
        for stage in range(1, 6)
    ]
    return {
        "primary": primary,
        "secondary": secondary,
        "optimizer_stage": optimizer_stage,
        "validation_selected": validation_selected,
        "systems": systems,
        "final_geometry": final_geometry,
        "bridge": bridge,
        "sensitivity": sensitivity,
        "rankings": rankings,
        "sources": {},
    }


def test_rendered_publication_contains_every_frozen_feature_and_claim_boundary() -> None:
    evidence = _evidence()
    markdown = render_markdown(evidence)
    latex = render_latex(evidence)
    for label in FEATURE_LABELS.values():
        assert label in markdown
    assert "Muon versus AdamW is positive" in markdown
    assert "NorMuon versus AdamW is negative" in markdown
    assert "not causal mediation" in markdown
    assert "Corrected Independently Padded Replication" in latex
    assert "Final-stage execution-path sensitivity" in latex
    assert r"\newcommand{\CorrectedGeometryBridgeTable}" in latex
    assert r"\newcommand{\CorrectedExecutionSensitivityTable}" in latex
    assert latex.index(r"\newcommand{\CorrectedGeometryBridgeTable}") < latex.index(
        r"\section{Corrected Independently Padded Replication}"
    )
    assert "Contrast & Mean & 95\\% CI & Decision" in latex


def test_corrected_detail_tables_are_called_only_after_the_appendix_boundary() -> None:
    manuscript = (REPOSITORY / "paper/main.tex").read_text(encoding="utf-8")
    appendix = manuscript.index(r"\appendix")
    corrected_input = manuscript.index(r"\input{generated/corrected-no-packing}")
    bridge_call = manuscript.index(r"\CorrectedGeometryBridgeTable")
    sensitivity_call = manuscript.index(r"\CorrectedExecutionSensitivityTable")

    assert corrected_input < appendix < bridge_call < sensitivity_call
    makefile = (REPOSITORY / "paper/Makefile").read_text(encoding="utf-8")
    assert "generated/corrected-no-packing.tex" in makefile


def test_conclusion_reports_supported_features_without_cherry_picking() -> None:
    conclusion = build_conclusion(_evidence())
    assert FEATURE_LABELS[next(iter(FEATURE_LABELS))] in conclusion
    assert "one-seed, pinned-grid" in conclusion

    evidence = _evidence()
    for row in evidence["bridge"]:
        row["supported"] = False
    conclusion = build_conclusion(evidence)
    assert "none of the nine frozen features" in conclusion


def test_contrast_validation_recomputes_support_and_common_max_t() -> None:
    rows = []
    for treatment, baseline, mean in (
        ("muon", "adamw", 0.02),
        ("normuon", "adamw", -0.03),
        ("normuon", "muon", -0.05),
    ):
        rows.append(
            {
                "treatment": treatment,
                "baseline": baseline,
                "mean_delta_ndcg_at_10": mean,
                "simultaneous_ci_95_lower": mean - 0.01,
                "simultaneous_ci_95_upper": mean + 0.01,
                "simultaneous_critical_value": 2.5,
                "support": "positive" if mean > 0 else "negative",
                "bootstrap_samples": 50_000,
                "bootstrap_seed": 20_260_903,
                "tasks": 14,
            }
        )
    validated = _validate_contrasts(rows, secondary=False)
    assert [row["support"] for row in validated] == ["positive", "negative", "negative"]
    broken = deepcopy(rows)
    broken[0]["support"] = "inconclusive"
    with pytest.raises(ValueError, match="Invalid corrected publication contrast"):
        _validate_contrasts(broken, secondary=False)


def test_bridge_validation_enforces_pooled_and_fold_support_rule() -> None:
    summary_rows = []
    association_rows = []
    for index, feature in enumerate(FEATURE_LABELS):
        reduction = 0.001 if index == 0 else -0.001
        summary_rows.append(
            {
                "feature": feature,
                "pooled_rows": 60,
                "folds_total": 4,
                "folds_improved": 3 if index == 0 else 2,
                "pooled_baseline_rmse": 0.03,
                "pooled_feature_rmse": 0.03 - reduction,
                "pooled_rmse_reduction": reduction,
                "predictively_useful": index == 0,
            }
        )
        association_rows.append(
            {
                "feature": feature,
                "rows": 60,
                "pearson_residual_association": 0.1,
                "spearman_residual_association": 0.2,
            }
        )
    assert _bridge_table(summary_rows, association_rows)[0]["supported"] is True
    summary_rows[0]["predictively_useful"] = False
    with pytest.raises(ValueError, match="support rule mismatch"):
        _bridge_table(summary_rows, association_rows)


def test_checked_in_publication_protocol_binds_current_sources_before_results() -> None:
    protocol = _load_publication_protocol(
        REPOSITORY / "configs/dense_no_packing_publication_protocol.json", REPOSITORY
    )
    visibility = protocol["visibility_at_implementation_freeze"]
    assert visibility["corrected_complete_runs"] == 0
    assert visibility["corrected_beir_outputs_visible"] is False
    assert visibility["corrected_publication_outputs_visible"] is False
    layout = protocol["layout_only_amendment"]
    assert layout["visibility_at_amendment"]["corrected_complete_runs"] == 2
    assert layout["visibility_at_amendment"]["corrected_beir_outputs_visible"] is False
    assert layout["scientific_claims_changed"] is False
    assert layout["main_text_tables"] == ["corrected all-rate retrieval inference"]
    assert len(layout["appendix_tables"]) == 2
    assert protocol["expected_outputs"] == {
        "standalone_markdown": ("reports/dense-no-packing-publication/corrected_dense_results.md"),
        "summary_manifest": "reports/dense-no-packing-publication/summary_manifest.json",
        "paper_latex": "paper/generated/corrected-no-packing.tex",
        "primary_contrasts": 3,
        "secondary_contrasts": 3,
        "optimizer_stage_rows": 15,
        "geometry_features": 9,
        "sensitivity_contrasts": 10,
        "sensitivity_rankings": 5,
    }


def test_publication_layout_migration_is_exact_and_non_scientific() -> None:
    migration = json.loads(
        (REPOSITORY / "configs/dense_no_packing_publication_layout_migration.json").read_text(
            encoding="utf-8"
        )
    )

    assert migration["from_contract_sha256"] == (
        "25eefbe52b4cc275600dae631860d2aa69a0734c1c143813b4e7e4a9190f3c13"
    )
    assert migration["to_contract_sha256"] == (
        "abb5973ab7247ec427195ab9afa6f60add579e9e33829e3cc08a61f4947d5a67"
    )
    assert migration["scientific_contract_changed"] is False
    assert migration["allowed_changed_sources"] == [
        "configs/dense_no_packing_publication_protocol.json"
    ]
    assert len(migration["required_unchanged_sources"]) == 9


def _write_csv(directory: Path, filename: str, rows: list[dict]) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_json(directory: Path, filename: str, payload: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _raw_contrasts(*, secondary: bool) -> list[dict]:
    rows = []
    for treatment, baseline, mean in (
        ("muon", "adamw", 0.02),
        ("normuon", "adamw", -0.03),
        ("normuon", "muon", -0.05),
    ):
        row = {
            "treatment": treatment,
            "baseline": baseline,
            "mean_delta_ndcg_at_10": mean,
            "across_task_standard_error": 0.004,
            "nominal_bootstrap_ci_95_lower": mean - 0.008,
            "nominal_bootstrap_ci_95_upper": mean + 0.008,
            "simultaneous_ci_95_lower": mean - 0.01,
            "simultaneous_ci_95_upper": mean + 0.01,
            "simultaneous_critical_value": 2.5,
            "support": "positive" if mean > 0 else "negative",
            "bootstrap_samples": 50_000,
            "bootstrap_seed": 20_260_903,
            "tasks": 14,
        }
        if secondary:
            row.update(
                {
                    "treatment_run_id": f"selected-{treatment}",
                    "baseline_run_id": f"selected-{baseline}",
                }
            )
        rows.append(row)
    return rows


def test_complete_publication_loader_uses_real_manifest_key_shapes(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    protocols = {
        name: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        for name, relative in {
            "analysis_protocol": "configs/dense_no_packing_analysis_protocol.json",
            "outcome_protocol": "configs/dense_no_packing_outcome_protocol.json",
            "bridge_protocol": "configs/dense_no_packing_bridge_implementation_protocol_v2.json",
            "sensitivity_protocol": "configs/dense_no_packing_sensitivity_implementation_protocol.json",
        }.items()
    }
    protocol = {"parent_bindings": {name: {"sha256": value} for name, value in protocols.items()}}

    outcomes = tmp_path / "outcomes"
    outcome_outputs = {
        "primary_summary": _write_csv(
            outcomes, "primary_summary.csv", _raw_contrasts(secondary=False)
        ),
        "secondary_summary": _write_csv(
            outcomes, "secondary_summary.csv", _raw_contrasts(secondary=True)
        ),
    }
    optimizer_stage = []
    for optimizer_index, optimizer in enumerate(OPTIMIZERS):
        for stage in range(1, 6):
            optimizer_stage.append(
                {
                    "optimizer": optimizer,
                    "stage": stage,
                    "progress_fraction": stage / 5,
                    "learning_rates": 4,
                    "mean_ndcg_at_10_across_rates": 0.5 + 0.01 * stage + 0.001 * optimizer_index,
                    "median_ndcg_at_10_across_rates": 0.5 + 0.01 * stage + 0.001 * optimizer_index,
                }
            )
    outcome_outputs["optimizer_stage_scores"] = _write_csv(
        outcomes, "optimizer_stage_scores.csv", optimizer_stage
    )
    validation = []
    selected = {}
    system_rows = []
    for optimizer_index, optimizer in enumerate(OPTIMIZERS):
        for dose in range(1, 5):
            run_id = f"selected-{optimizer}" if dose == 1 else f"{optimizer}-{dose}"
            if dose == 1:
                selected[optimizer] = run_id
            validation.append(
                {
                    "optimizer": optimizer,
                    "run_id": run_id,
                    "learning_rate": 10 ** (-6 + dose),
                    "contrastive_loss": 0.5 + 0.01 * dose,
                    "positive_margin": 0.1 + 0.01 * dose,
                }
            )
            system_row = {
                "optimizer": optimizer,
                "run_id": run_id,
                "world_size": 4,
            }
            for field_index, field in enumerate(SYSTEM_FIELDS):
                system_row[field] = 1 + optimizer_index + dose / 10 + field_index / 100
            system_rows.append(system_row)
    outcome_outputs["validation_run_metrics"] = _write_csv(
        outcomes, "validation_run_metrics.csv", validation
    )
    outcome_outputs["system_metrics"] = _write_csv(outcomes, "system_metrics.csv", system_rows)
    outcome_outputs["validation_recipe_selection"] = _write_json(
        outcomes,
        "validation_recipe_selection.json",
        {"schema_version": 1, "status": "complete", "selected_run_ids": selected},
    )
    _write_json(
        outcomes,
        "summary_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "coverage": {"task_units": 840},
            "protocol": {"sha256": protocols["outcome_protocol"]},
            "outputs": outcome_outputs,
        },
    )

    geometry = tmp_path / "geometry"
    geometry_rows = []
    for optimizer_index, optimizer in enumerate(OPTIMIZERS):
        for dose in range(1, 5):
            for stage in range(1, 6):
                geometry_rows.append(
                    {
                        "optimizer": optimizer,
                        "run_id": f"{optimizer}-{dose}",
                        "stage": stage,
                        "saved_segment_to_weight_ratio": 0.001 * (stage + dose),
                        "saved_segment_stable_rank_fraction_parameter_weighted": 0.1 + 0.01 * dose,
                        "saved_segment_sketch_effective_rank_fraction_parameter_weighted": 0.2
                        + 0.01 * dose,
                        "saved_segment_row_cv_parameter_weighted": 0.3 + 0.01 * optimizer_index,
                        "saved_segment_top_1pct_row_energy_parameter_weighted": 0.4 + 0.01 * stage,
                        "cumulative_displacement_to_weight_ratio": 0.002 * (stage + dose),
                        "cumulative_stable_rank_fraction_parameter_weighted": 0.15 + 0.01 * dose,
                    }
                )
    geometry_outputs = {
        "checkpoint_geometry.csv": _write_csv(geometry, "checkpoint_geometry.csv", geometry_rows)
    }
    _write_json(
        geometry,
        "summary_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "checkpoint_rows": 60,
            "run_pair_subspace_rows": 660,
            "protocol": {"sha256": protocols["analysis_protocol"]},
            "outputs": geometry_outputs,
        },
    )

    bridge = tmp_path / "bridge"
    feature_rows = []
    association_rows = []
    for index, feature in enumerate(FEATURE_LABELS):
        reduction = 0.001 if index == 0 else -0.001
        feature_rows.append(
            {
                "feature": feature,
                "pooled_rows": 60,
                "pooled_baseline_rmse": 0.03,
                "pooled_feature_rmse": 0.03 - reduction,
                "pooled_rmse_reduction": reduction,
                "folds_improved": 4 if index == 0 else 2,
                "folds_total": 4,
                "predictively_useful": index == 0,
            }
        )
        association_rows.append(
            {
                "feature": feature,
                "rows": 60,
                "pearson_residual_association": 0.1,
                "spearman_residual_association": 0.2,
            }
        )
    bridge_outputs = {
        "feature_prediction_summary.csv": _write_csv(
            bridge, "feature_prediction_summary.csv", feature_rows
        ),
        "residual_associations.csv": _write_csv(
            bridge, "residual_associations.csv", association_rows
        ),
    }
    _write_json(
        bridge,
        "summary_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "coverage": {
                "runs": 12,
                "stages": 5,
                "bridge_rows": 60,
                "features": 9,
                "leave_dose_fold_rows": 36,
            },
            "protocol": {"sha256": protocols["bridge_protocol"]},
            "outputs": bridge_outputs,
        },
    )

    sensitivity = tmp_path / "sensitivity"
    contrast_rows = []
    ranking_rows = []
    for stage in range(1, 6):
        for optimizer in ("muon", "normuon"):
            historical = 0.01
            corrected = -0.02
            contrast_rows.append(
                {
                    "optimizer": optimizer,
                    "stage": stage,
                    "historical_optimizer_minus_adamw": historical,
                    "corrected_optimizer_minus_adamw": corrected,
                    "corrected_minus_historical_contrast_shift": corrected - historical,
                }
            )
        ranking_rows.append(
            {
                "stage": stage,
                "historical_optimizer_order": "muon>normuon>adamw",
                "corrected_optimizer_order": "adamw>normuon>muon",
                "ranking_changed": True,
            }
        )
    sensitivity_outputs = {
        "optimizer_minus_adamw_sensitivity.csv": _write_csv(
            sensitivity, "optimizer_minus_adamw_sensitivity.csv", contrast_rows
        ),
        "stage_optimizer_rankings.csv": _write_csv(
            sensitivity, "stage_optimizer_rankings.csv", ranking_rows
        ),
    }
    _write_json(
        sensitivity,
        "summary_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "coverage": {"matched_rows": 60},
            "no_pooling": True,
            "protocol": {"sha256": protocols["sensitivity_protocol"]},
            "outputs": sensitivity_outputs,
        },
    )

    evidence = load_publication_evidence(
        argparse.Namespace(
            outcomes_dir=outcomes,
            geometry_dir=geometry,
            bridge_dir=bridge,
            sensitivity_dir=sensitivity,
        ),
        protocol,
    )
    assert len(evidence["primary"]) == 3
    assert len(evidence["optimizer_stage"]) == 15
    assert len(evidence["bridge"]) == 9
    assert len(evidence["sensitivity"]) == 10
