"""Result-contingent text with distinct negative, undefined and rotation boundaries."""

from .dimension_publication import FEATURE_LABELS, OPTIMIZER_LABELS, _render_figure


def number(value):
    return "---" if value is None else f"{value:+.6g}"


def render(tables, decisions):
    primary = tables["primary_contrasts"]
    native = decisions["constructive_native_coordinate_use"]
    rotations = decisions["direction_stable_across_tested_rotations"]
    retention = decisions["random_50pct_relative_shortlist_ndcg"]
    lines = []
    for optimizer in ("muon", "normuon"):
        label = OPTIMIZER_LABELS[optimizer]
        if native[optimizer]:
            text = f"{label} meets the joint native-coordinate criterion relative to AdamW on this fixed probe."
        else:
            text = f"{label} does not meet the joint native-coordinate criterion relative to AdamW; this does not establish equivalence."
        text += (
            " The required mean directions hold under all three tested shared rotations."
            if rotations[optimizer]
            else " The required mean directions do not hold under every tested shared rotation."
        )
        lines.append(text)
    finding = " ".join(lines) + (
        " With 50\\% of coordinates removed, relative shortlist nDCG is "
        f"{retention['adamw']:.6g}/{retention['muon']:.6g}/{retention['normuon']:.6g} "
        "for AdamW/Muon/NorMuon. The figure reports all declared rates and stages."
    )
    bridge_lines, bridge_rows = [], []
    for row in tables["feature_prediction_summary"]:
        label = FEATURE_LABELS[row["feature"]].replace("%", r"\%")
        useful = row["predictively_useful"]
        if useful is None:
            status = "undefined"
            bridge_lines.append(
                f"The predictive criterion for {label} is undefined because at least one augmented fold is not identified or numerically resolved."
            )
        elif useful:
            status = "supported"
            bridge_lines.append(
                f"{label.capitalize()} meets the declared held-out-dose predictive criterion."
            )
        else:
            status = "not supported"
            bridge_lines.append(
                f"{label.capitalize()} does not meet the declared held-out-dose predictive criterion."
            )
        bridge_rows.append(
            f"{label} & {number(row['pooled_rmse_reduction'])} & {row['folds_improved']}/4 & {row['folds_defined']}/4 & {status} \\\\"
        )
    bridge_finding = (
        " ".join(bridge_lines)
        + " This tests prediction outside a learning-rate dose on the same task suite, not causal mediation or generalization to unseen tasks."
    )
    conclusion = (
        "Coordinate-level utility, its tested-basis sensitivity, and held-out-dose prediction "
        "are distinct results. Shared-rotation sign stability does not establish arbitrary-basis "
        "invariance or semantic independence of dimensions; these measurements alone do not "
        "identify a mediator of an optimizer effect."
    )
    rows = []
    for row in primary:
        label = FEATURE_LABELS[row["feature"]].replace("%", r"\%")
        contrast = OPTIMIZER_LABELS[row["treatment"]] + "--" + OPTIMIZER_LABELS[row["baseline"]]
        rows.append(
            f"{label} & {contrast} & {number(row['mean_difference'])} & [{number(row['simultaneous_ci_95_lower'])}, {number(row['simultaneous_ci_95_upper'])}] & {row['decision']} \\\\"
        )
    return (
        "% Prepared functional inference output; manuscript installation is separately gated.\n"
        + _render_figure(tables["figure_points"])
        + f"\\newcommand{{\\DimensionUtilizationFinding}}{{{finding}}}\n"
        + f"\\newcommand{{\\DimensionRetrievalBridgeFinding}}{{{bridge_finding}}}\n"
        + f"\\newcommand{{\\DimensionConclusionFinding}}{{{conclusion}}}\n"
        + "\\newcommand{\\DimensionUtilizationAppendixTable}{%\n"
        + "\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llrrl}\\toprule\n"
        + "Feature & Contrast & Effect & Simultaneous 95\\% CI & Decision \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{tabular}\n"
        + "\\caption{All nine final-stage co-primary contrasts, averaging every declared rate. One common-resample max-$T$ family covers all entries; task resampling does not estimate training-seed variation.}\n\\end{table*}\n"
        + "\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{lrrrl}\\toprule\n"
        + "Feature & RMSE reduction & Improved & Defined & Criterion \\\\\n\\midrule\n"
        + "\n".join(bridge_rows)
        + "\n\\bottomrule\\end{tabular}\n"
        + "\\caption{All four functional predictors beyond optimizer, stage and within-optimizer dose. Decisions use exact MSE comparisons; displayed RMSE is not a significance test. Undefined folds are retained, never pooled as available cases.}\n\\end{table*}%\n}\n"
    )
