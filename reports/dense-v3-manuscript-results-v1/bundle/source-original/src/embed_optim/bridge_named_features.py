"""Exact-OLS bridge for explicit additional feature names, never relabelled legacy slots."""

from __future__ import annotations

from .bridge_exact_arithmetic import (
    Baseline,
    average_ranks,
    correlation,
    design_health,
    displayed,
    dot,
    rational,
)
from .bridge_numerics import error_columns, validate_panel
from .corrected_retrieval_bridge import _baseline_design

MISSING_POLICY = {
    "input_panel": "Authenticate the complete original bridge panel; additional columns keep their own explicit names and do not replace any legacy feature",
    "absent_row_or_column": "Error; missing coverage is not an undefined measurement",
    "undefined_measurement": "A legitimate None measurement retains every input row; all four augmented folds, complete pooled comparison and association for that feature are undefined",
    "baseline": "Retain the exact complete baseline predictions even when a feature is undefined",
    "partial_fit": "Never fit on available measurements, drop a stage/rate or synthesize a feature value",
}


def empty_health():
    return {
        key: None
        for key in (
            "training_feature_mean",
            "training_feature_scale",
            "training_feature_mean_exact",
            "training_feature_variance_exact",
            "augmented_numeric_rank",
            "rank_cutoff",
            "singular_values",
        )
    }


def evaluate_named(rows, features):
    """Pure numerical calculation; callers separately authenticate each feature family."""
    if (
        not isinstance(features, (tuple, list))
        or not features
        or any(not isinstance(name, str) or not name for name in features)
        or len(set(features)) != len(features)
        or set(features)
        & {
            "run_id",
            "optimizer",
            "stage",
            "dose_index",
            "learning_rate",
            "centered_log10_learning_rate",
            "mean_ndcg_at_10",
            "progress_fraction",
        }
    ):
        raise ValueError("Require distinct, explicitly named additional feature columns")
    rows = validate_panel(rows)  # Original columns are real and retained, not placeholder slots.
    for row in rows:
        for feature in features:
            if feature not in row:
                raise ValueError("Missing named feature column")
            if row[feature] is not None:
                rational(row[feature])
    design = [[rational(x) for x in row] for row in _baseline_design(rows).tolist()]
    outcome = [rational(row["mean_ndcg_at_10"]) for row in rows]
    contexts = {
        fold: Baseline(design, [i for i, row in enumerate(rows) if row["dose_index"] != fold])
        for fold in range(1, 5)
    }
    full = Baseline(design, range(60))
    _, outcome_residual = full.project(outcome)
    outcome_energy = dot(outcome_residual, outcome_residual)
    outcome_health = design_health(full.numeric, outcome)
    folds, summaries, associations, predictions, diagnostics = [], [], [], [], []
    for feature in features:
        missing = any(row[feature] is None for row in rows)
        values = None if missing else [rational(row[feature]) for row in rows]
        base_all, added_all = [None] * 60, [None] * 60
        valid, improved = 0, 0
        for fold, context in contexts.items():
            if missing:
                base, _ = context.project(outcome)
                result = {
                    "status": "undefined_feature_measurement",
                    "health": empty_health(),
                    "baseline_prediction": base,
                    "feature_prediction": None,
                    "exact_residual_energy": None,
                }
            else:
                result = context.feature(values, outcome)
            test = [i for i, row in enumerate(rows) if row["dose_index"] == fold]
            baseline = [result["baseline_prediction"][i] for i in test]
            added = result["feature_prediction"]
            selected = None if added is None else [added[i] for i in test]
            errors = error_columns([outcome[i] for i in test], baseline, selected)
            valid += int(selected is not None)
            improved += int(errors["feature_improves"] is True)
            health = result["health"]
            folds.append(
                {
                    "feature": feature,
                    "held_out_dose_index": fold,
                    "train_rows": 45,
                    "test_rows": 15,
                    "status": result["status"],
                    "training_feature_mean": health["training_feature_mean"],
                    "training_feature_scale": health["training_feature_scale"],
                    "augmented_numeric_rank": health["augmented_numeric_rank"],
                    "exact_residual_energy": result["exact_residual_energy"],
                    **errors,
                }
            )
            diagnostics.append({"feature": feature, "fold": fold, **health})
            for i in test:
                base_all[i] = result["baseline_prediction"][i]
                added_all[i] = None if added is None else added[i]
                predictions.append(
                    {
                        "feature": feature,
                        "run_id": rows[i]["run_id"],
                        "stage": rows[i]["stage"],
                        "held_out_dose_index": fold,
                        "status": result["status"],
                        "observed": float(outcome[i]),
                        "baseline_prediction": displayed(base_all[i]),
                        "feature_prediction": None if added is None else displayed(added[i]),
                        "baseline_prediction_exact": str(base_all[i]),
                        "feature_prediction_exact": None if added is None else str(added[i]),
                    }
                )
        pooled = error_columns(outcome, base_all, added_all if valid == 4 else None)
        summaries.append(
            {
                "feature": feature,
                "pooled_rows": 60,
                "pooled_baseline_rmse": pooled["baseline_rmse"],
                "pooled_feature_rmse": pooled["feature_rmse"],
                "pooled_rmse_reduction": pooled["rmse_reduction"],
                "pooled_baseline_mse_exact": pooled["baseline_mse_exact"],
                "pooled_feature_mse_exact": pooled["feature_mse_exact"],
                "pooled_mse_reduction_exact": pooled["mse_reduction_exact"],
                "folds_improved": improved,
                "folds_total": 4,
                "folds_defined": valid,
                "status": "complete" if valid == 4 else "undefined_fold_present",
                "predictively_useful": None
                if valid != 4
                else pooled["feature_improves"] and improved >= 3,
            }
        )
        if missing:
            residual, feature_energy, health, status = (
                None,
                None,
                empty_health(),
                "undefined_feature_measurement",
            )
        else:
            _, residual = full.project(values)
            feature_energy = dot(residual, residual)
            health = design_health(full.numeric, values)
            if feature_energy == 0:
                status = "feature_in_baseline_span"
            elif outcome_energy == 0:
                status = "outcome_in_baseline_span"
            elif health["augmented_numeric_rank"] != 9:
                status = "unresolved_feature_residual"
            elif outcome_health["augmented_numeric_rank"] != 9:
                status = "unresolved_outcome_residual"
            else:
                status = "resolved"
        associations.append(
            {
                "feature": feature,
                "rows": 60,
                "status": status,
                "feature_residual_energy_exact": None
                if feature_energy is None
                else str(feature_energy),
                "outcome_residual_energy_exact": str(outcome_energy),
                "pearson_residual_association": correlation(residual, outcome_residual)
                if status == "resolved"
                else None,
                "spearman_residual_association": correlation(
                    average_ranks(residual), average_ranks(outcome_residual)
                )
                if status == "resolved"
                else None,
            }
        )
        diagnostics.append({"feature": feature, "fold": "all", **health})
    return {
        "bridge_rows": rows,
        "leave_dose_fold_metrics": folds,
        "feature_prediction_summary": summaries,
        "residual_associations": associations,
        "held_out_predictions": predictions,
    }, {"feature_designs": diagnostics, "outcome_design": outcome_health}
