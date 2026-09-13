"""Pure checked original-bridge rendering; not primary admission or publication.

Candidate component only. Complete outcome/geometry authoring, all manuscript
includes, source/runtime release and portable publication remain separate work.
"""

from collections import Counter
from fractions import Fraction

from embed_optim.bridge_numerics import evaluate
from embed_optim.corrected_publication import FEATURE_LABELS, _tex_escape
from embed_optim.primary_contract import require_same


def checked_bridge(tables):
    """Recompute every fold/prediction/association, not just a summary flag."""
    expected, _ = evaluate(tables["bridge_rows"])
    require_same(tables, expected)
    summaries = {row["feature"]: row for row in expected["feature_prediction_summary"]}
    associations = {row["feature"]: row for row in expected["residual_associations"]}
    rows = []
    for feature, label in FEATURE_LABELS.items():
        summary, association = summaries[feature], associations[feature]
        statuses = Counter(
            row["status"]
            for row in expected["leave_dose_fold_metrics"]
            if row["feature"] == feature
        )
        useful = summary["predictively_useful"]
        status = "undefined" if useful is None else "supported" if useful else "not supported"
        if statuses == {"baseline_equivalent": 4}:
            status = "baseline equivalent"
        rows.append(
            {
                **summary,
                "label": label,
                "criterion": status,
                "fold_status_counts": dict(sorted(statuses.items())),
                "association": association,
            }
        )
    return rows


def delta_text(row):
    """Exact signs govern decisions; a binary64 display zero is not exact equality."""
    value, exact = row["pooled_rmse_reduction"], row["pooled_mse_reduction_exact"]
    if exact is None:
        if value is not None:
            raise ValueError("An undefined comparison must not display a numeric delta")
        return "---"
    exact = Fraction(exact)
    if value is None:
        raise ValueError("A defined exact comparison is missing its display value")
    if value == 0 and exact:
        return ("positive" if exact > 0 else "negative") + " (below display range)"
    return f"{value:+.6g}"


def finding(rows):
    supported = [row["label"] for row in rows if row["predictively_useful"] is True]
    undefined = [row["label"] for row in rows if row["predictively_useful"] is None]
    if len(undefined) == 9:
        text = (
            "The held-out-dose predictive criterion is undefined for all nine weight-space "
            "features; these comparisons cannot establish either support or lack of support."
        )
    elif supported:
        text = (
            "The declared out-of-dose predictive criterion supports: " + ", ".join(supported) + "."
        )
    else:
        text = (
            f"None of the {9 - len(undefined)} weight-space features with a defined four-fold "
            "comparison meets the declared out-of-dose predictive criterion."
        )
    if undefined and len(undefined) != 9:
        text += " The criterion remains undefined for: " + ", ".join(undefined) + "."
    return text + (
        " All nine features and all four folds are retained. Undefined folds are not pooled "
        "as available cases. This is prediction outside a learning-rate dose on the same task "
        "suite, not causal mediation, statistical significance or evidence of equivalence."
    )


def render_bridge(tables):
    rows = checked_bridge(tables)
    body = []
    for row in rows:
        rho = row["association"]["spearman_residual_association"]
        rho = "---" if rho is None else f"{rho:+.3f}"
        body.append(
            f"{_tex_escape(row['label'])} & {_tex_escape(delta_text(row))} & "
            f"{row['folds_improved']}/4 & {row['folds_defined']}/4 & "
            f"{_tex_escape(row['criterion'])} & {rho} \\\\\n"
        )
    latex = (
        "% Candidate original-bridge rendering; no manuscript installation authorized.\n"
        "\\newcommand{\\CorrectedGeometryBridgeFinding}{" + _tex_escape(finding(rows)) + "}\n"
        "\\newcommand{\\CorrectedGeometryBridgeTable}{%\n"
        "\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{lrrrll}\\toprule\n"
        "Geometry feature & RMSE reduction & Improved & Defined & Criterion & Residual $\\rho$ \\\\\n"
        "\\midrule\n" + "".join(body) + "\\bottomrule\\end{tabular}\n"
        "\\caption{Every original weight-space predictor under four-fold leave-dose-index-out "
        "prediction. Decisions use exact MSE comparisons, not displayed RMSE. A missing "
        "correlation is undefined, not zero; baseline equivalence is distinct from an "
        "unidentified or numerically unresolved prediction.}\n"
        "\\label{tab:corrected-bridge}\n\\end{table*}%\n}\n"
    )
    return {
        "rows": rows,
        "latex": latex,
        "primary_admission_supplied": False,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
