"""Complete original/exact comparison text; no model admission or manuscript writes."""

from collections import Counter

from . import primary_v3_exact_bridge as exact
from .corrected_publication import _tex_escape
from .primary_contract import require_same
from .primary_v3_publication_bridge import checked_bridge, delta_text

LABELS = {
    "exact_nonzero_saved_segment_stable_rank_fraction": "Segment stable rank",
    "full_spectrum_nonzero_saved_segment_entropy_rank_fraction": "Segment entropy rank",
    "exact_nonzero_cumulative_stable_rank_fraction": "Cumulative stable rank",
    "exact_mean_saved_segment_overlap_to_adamw": "Segment overlap to AdamW",
    "exact_mean_cumulative_overlap_to_adamw": "Cumulative overlap to AdamW",
}
INTERPRETATION = (
    "These are measurement-sensitivity comparisons, not uniformly pure approximation-error "
    "tests: entropy uses the full singular spectrum, and rank/entropy averages expose their "
    "nonzero-matrix denominators. Every declared correspondence and all four folds are retained. "
    "Undefined comparisons are not available-case averages. Predictive support is not causal "
    "mediation, statistical significance, equivalence, or evidence of useful embedding dimensions."
)
WRAPPER = (
    "\\let\\OriginalWeightGeometryFinding\\CorrectedGeometryBridgeFinding\n"
    "\\renewcommand{\\CorrectedGeometryBridgeFinding}{%\n"
    "\\OriginalWeightGeometryFinding{} \\ExactGeometrySensitivityFinding}\n"
    "\\let\\OriginalWeightGeometryTable\\CorrectedGeometryBridgeTable\n"
    "\\renewcommand{\\CorrectedGeometryBridgeTable}{%\n"
    "\\OriginalWeightGeometryTable\\ExactGeometrySensitivityTable}\n"
)


def checked_sensitivity(primary, original, checkpoints, pairs, stored):
    """Recompute the full original and exact families, including all 300 correspondences."""
    old = {row["feature"]: row for row in checked_bridge(original)}
    tables, diagnostics = exact.sensitivity_tables(primary, original, checkpoints, pairs)
    require_same(stored, tables)
    require_same(list(LABELS), list(exact.FEATURES))
    summaries = {row["feature"]: row for row in tables["feature_prediction_summary"]}
    comparisons = {row["exact_feature"]: row for row in tables["predictive_sensitivity_summary"]}
    associations = {row["feature"]: row for row in tables["residual_associations"]}
    rows = []
    for feature, label in LABELS.items():
        current = summaries[feature]
        statuses = Counter(
            row["status"] for row in tables["leave_dose_fold_metrics"] if row["feature"] == feature
        )
        useful = current["predictively_useful"]
        criterion = "undefined" if useful is None else "supported" if useful else "not supported"
        if statuses == {"baseline_equivalent": 4}:
            criterion = "baseline equivalent"
        comparison = comparisons[feature]
        rows.append(
            {
                "feature": feature,
                "label": label,
                "original_feature": comparison["original_feature"],
                "original": old[comparison["original_feature"]],
                "exact": {
                    **current,
                    "criterion": criterion,
                    "fold_status_counts": dict(sorted(statuses.items())),
                    "association": associations[feature],
                },
                "comparison": comparison,
            }
        )
    return {
        "rows": rows,
        "diagnostics": diagnostics,
        "interpretation": INTERPRETATION,
        "scientific_completion": False,
        "primary_admission_supplied": False,
        "manuscript_installed": False,
    }


def finding(rows):
    require_same([row["feature"] for row in rows], list(LABELS))
    supported = [row["label"] for row in rows if row["exact"]["predictively_useful"] is True]
    undefined = [row["label"] for row in rows if row["exact"]["predictively_useful"] is None]
    if len(undefined) == 5:
        text = "The predictive criterion is undefined for all five exact-measurement counterparts."
    elif supported:
        text = (
            "The exact-measurement sensitivity meets the declared predictive criterion for: "
            + ", ".join(supported)
            + "."
        )
    else:
        text = f"None of the {5 - len(undefined)} exact-measurement counterparts with a defined four-fold comparison meets the declared predictive criterion."
    if undefined and len(undefined) != 5:
        text += " It remains undefined for: " + ", ".join(undefined) + "."
    return (
        text
        + " These checks do not replace the original nine-feature analysis; all five paired comparisons are reported in the appendix."
    )


def render(summary):
    rows = summary["rows"]
    require_same([row["feature"] for row in rows], list(LABELS))
    require_same(summary["interpretation"], INTERPRETATION)
    body = []
    for row in rows:
        current, comparison = row["exact"], row["comparison"]
        difference = delta_text(
            {
                "pooled_rmse_reduction": comparison["original_minus_exact_feature_rmse"],
                "pooled_mse_reduction_exact": comparison[
                    "original_minus_exact_feature_mse_rational"
                ],
            }
        )
        body.append(
            f"{_tex_escape(row['label'])} & {_tex_escape(row['original']['criterion'])} & "
            f"{_tex_escape(current['criterion'])} & {_tex_escape(delta_text(current))} & "
            f"{current['folds_defined']}/4 & {_tex_escape(difference)} \\\\\n"
        )
    return (
        "% Generated exact-measurement sensitivity; no manuscript installation authorized.\n"
        "\\newcommand{\\ExactGeometrySensitivityFinding}{" + _tex_escape(finding(rows)) + "}\n"
        "\\newcommand{\\ExactGeometrySensitivityTable}{%\n"
        "\\begin{table*}[t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lllrrr}\\toprule\n"
        "Measurement & Original criterion & Exact criterion & Exact RMSE gain & Defined & Error difference \\\\\n"
        "\\midrule\n" + "".join(body) + "\\bottomrule\\end{tabular}\n"
        "\\caption{Every original/exact counterpart under the same four held-out learning-rate doses. "
        "RMSE gain compares the exact predictor with the nuisance-only baseline; error difference "
        "is original-predictor RMSE minus exact-predictor RMSE. Decisions use exact MSE, not rounded "
        "display values. Dashes denote undefined comparisons. "
        + _tex_escape(INTERPRETATION)
        + "}\n"
        "\\label{tab:exact-geometry-sensitivity}\n\\end{table*}%\n}\n"
    )


def combine(original, exact_latex):
    """Keep every original byte as a prefix, extending only its two invoked macros."""
    if type(original) is not bytes or type(exact_latex) is not bytes:
        raise ValueError("Require original and exact UTF-8 LaTeX bytes")
    text = original.decode("utf-8")
    for name in ("CorrectedGeometryBridgeFinding", "CorrectedGeometryBridgeTable"):
        if text.count("\\newcommand{\\" + name + "}") != 1:
            raise ValueError("Require the original complete geometry macro definitions")
    if any(
        name in text
        for name in ("OriginalWeightGeometry", "ExactGeometrySensitivity", "\\ResultPending")
    ):
        raise ValueError("Cannot append sensitivity to pending or already extended primary text")
    if not exact_latex.decode("utf-8").startswith("% Generated exact-measurement sensitivity;"):
        raise ValueError("Require generated exact sensitivity text")
    return original + b"\n" + exact_latex + WRAPPER.encode()
