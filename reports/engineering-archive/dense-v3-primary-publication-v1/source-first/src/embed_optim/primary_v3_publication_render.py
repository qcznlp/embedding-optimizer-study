"""Scientific manuscript bytes from complete checked v3 tables; no file writes."""

from .corrected_publication import (
    LABELS,
    _ci,
    _tex_escape,
    build_corrected_finding,
    build_weight_space_finding,
)
from .primary_v3_publication_bridge import finding as bridge_finding


def conclusion(evidence):
    selected = "; ".join(
        LABELS[row["treatment"]]
        + " versus "
        + LABELS[row["baseline"]]
        + " "
        + row["support"]
        + " at "
        + _ci(row)
        for row in evidence["secondary"]
    )
    return (
        build_corrected_finding(evidence)
        + " Validation-selected comparisons are secondary: "
        + selected
        + ". "
        + "These are one-seed, pinned-grid results; learning rates are not independent "
        "training seeds. An inconclusive interval is not evidence of equivalence."
    )


def render(evidence):
    """The author verifies all underlying bytes before calling this pure formatter."""
    primary_rows = "\n".join(
        f"{_tex_escape(LABELS[row['treatment']] + ' - ' + LABELS[row['baseline']])} & "
        f"{row['mean']:+.4f} & [{row['lower']:+.4f}, {row['upper']:+.4f}] & "
        f"{_tex_escape(row['support'])} \\\\"
        for row in evidence["primary"]
    )
    finding = _tex_escape(build_corrected_finding(evidence))
    weight = _tex_escape(build_weight_space_finding(evidence))
    return (
        "% Generated v3 primary evidence; no manuscript installation authorized.\n"
        "\\newcommand{\\CorrectedAbstractFinding}{" + finding + "}\n"
        "\\newcommand{\\CorrectedConclusionFinding}{" + finding + "}\n"
        "\\newcommand{\\CorrectedWeightSpaceFinding}{"
        + weight
        + "}\n"
        + evidence["bridge_latex"]
        + "\\newcommand{\\CorrectedMainSection}{%\n"
        "\\section{Optimizer Effects on Dense Retrieval}\n"
        "\\label{sec:optimizer-retrieval}\n"
        "The primary estimand averages all four declared rates within optimizer from a common "
        "model, dataset, training schedule, and evaluation suite.\n\n"
        "\\begin{table}[t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lrrl}\\toprule\n"
        "Contrast & Mean & 95\\% CI & Decision \\\\\n\\midrule\n"
        + primary_rows
        + "\n\\bottomrule\\end{tabular}\n"
        "\\caption{Final-stage retrieval over 14 decontaminated BEIR tasks, with "
        "simultaneous paired-task intervals for all three optimizer contrasts.}\n"
        "\\label{tab:optimizer-primary}\n\\end{table}\n"
        "\\paragraph{Primary comparison.} " + _tex_escape(conclusion(evidence)) + "\n}\n"
    )


def findings(evidence):
    return {
        "primary": build_corrected_finding(evidence),
        "primary_and_secondary": conclusion(evidence),
        "weight_states": build_weight_space_finding(evidence),
        "geometry_prediction": bridge_finding(evidence["bridge"]),
    }
