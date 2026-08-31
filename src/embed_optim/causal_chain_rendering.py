"""Deterministic Markdown and LaTeX views of strict causal-chain evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

PRIMARY_PREDICTOR = "update_tail_energy_fraction"
CAUSAL_HEADLINE_PREFIX = " Frozen causal-chain tests:"
CAUSAL_MAIN_BOUNDARY_SUBSTRINGS = (
    "In neither case do we interpret the chain as a formally identified mediation effect.",
    (
        "If every frozen component passes, the evidence supports a spectral-component chain "
        "across randomized branches, fixed-state interventions, and held-run prediction, but "
        "still does not identify formal mediation of the final BEIR contrast."
    ),
    (
        "Even joint passage supports causal-chain triangulation, not formal mediation of the "
        "full-training retrieval effect."
    ),
    "even complete support is explicitly not a formal causal mediation estimate.",
)
CAUSAL_MAIN_MACROS = (
    r"\input{generated/causal-chain}",
    r"\CausalChainSummaryTable",
    r"\CausalChainDiagnostics",
)
CAUSAL_MAIN_FORBIDDEN_OVERCLAIMS = (
    "prove formal mediation",
    "proves formal mediation",
    "proved formal mediation",
)
OUTCOME_LABELS = {
    "validation_loss_p95": "validation loss p95",
    "unseen_margin_p05": "unseen margin p05",
}
TEMPORAL_CRITERION_LABELS = {
    "treatment_shift": "treatment shift",
    "outcome_shift": "outcome shift",
    "held_out_prediction": "held-out prediction",
    "negative_control": "norm controls",
    "coefficient_behavior": "coefficient behavior",
}
DOSE_CRITERION_LABELS = {
    "loss_dose_monotone": "loss dose",
    "margin_dose_monotone": "margin dose",
    "tail_band_best_both_metrics": "tail band",
    "basis_swap_negative_control": "basis control",
}
PREDICTOR_LABELS = {
    "update_tail_energy_fraction": "tail energy",
    "update_stable_rank_fraction": "stable rank",
    "update_entropy_rank_fraction": "entropy rank",
    "update_head_energy_fraction": "head energy",
    "update_middle_energy_fraction": "middle energy",
    "update_row_norm_cv": "row CV",
    "update_frobenius_norm": "update norm",
    "weight_frobenius_norm": "weight norm",
    "baseline": "baseline",
    "spectrum_loss": "spectrum loss",
    "spectrum_margin": "spectrum margin",
    "basis_loss": "basis loss",
    "basis_margin": "basis margin",
}
PREDICTOR_KIND_LABELS = {
    "mechanism": "mech.",
    "negative_control": "norm control",
    "baseline": "baseline",
    "spectrum": "spectrum",
    "basis_negative_control": "basis control",
}


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite causal-chain display value: {value!r}")
    return result


def _number(value: Any, *, signed: bool = False) -> str:
    result = _finite(value)
    if result == 0:
        return "0"
    return f"{result:+.6g}" if signed else f"{result:.6g}"


def _margin_number(value: Any) -> str:
    """Keep strict decision margins visible even when rounded operands tie."""

    return f"{_finite(value):+.6e}"


def _latex_margin_number(value: Any) -> str:
    result = _finite(value)
    return "0.00e+00" if result == 0 else f"{result:+.2e}"


def _decision(value: Any) -> str:
    if type(value) is not bool:
        raise ValueError(f"Causal-chain decision is not boolean: {value!r}")
    return "pass" if value else "fail"


def _verdict(branch: dict[str, Any]) -> str:
    if not branch.get("complete"):
        return "pending, not claimable"
    return "supported" if branch.get("supported") is True else "claimable negative"


def _validate_shape(evidence: dict[str, Any]) -> None:
    temporal = evidence.get("temporal_short_branch")
    dose = evidence.get("dose_band")
    if not isinstance(temporal, dict) or not isinstance(dose, dict):
        raise ValueError("Causal-chain evidence lacks temporal or dose branches")
    if evidence.get("complete") is not True:
        return
    expected = {
        "temporal paired rows": (temporal.get("paired_rows"), 6),
        "temporal criteria": (temporal.get("criteria_rows"), 5),
        "temporal estimates": (temporal.get("estimate_rows"), 16),
        "temporal LOSO rows": (temporal.get("loso_rows"), 96),
        "dose criteria": (dose.get("criteria_rows"), 4),
        "dose anchors": (dose.get("anchor_rows"), 10),
        "dose RMSE rows": (dose.get("rmse_rows"), 5),
        "dose bridge rows": (dose.get("bridge_rows"), 2),
        "dose held-out rows": (dose.get("heldout_rows"), 84),
        "causal source tables": (evidence.get("source_table_records"), 5),
    }
    failures = {
        label: len(rows) if isinstance(rows, list) else type(rows).__name__
        for label, (rows, count) in expected.items()
        if not isinstance(rows, list) or len(rows) != count
    }
    if failures:
        raise ValueError(f"Causal-chain display coverage differs: {failures}")


def render_causal_chain_headline_fragment(evidence: dict[str, Any]) -> str:
    """Render the evidence-bound causal suffix for ``InterventionHeadline``."""
    temporal = evidence["temporal_short_branch"]
    dose = evidence["dose_band"]
    primary = {
        row["outcome"]: row
        for row in temporal["estimate_rows"]
        if row["predictor"] == PRIMARY_PREDICTOR
    }
    if set(primary) != set(OUTCOME_LABELS):
        raise ValueError("Causal headline lacks both primary temporal outcomes")
    rmse = {row["predictor"]: row for row in dose["rmse_rows"]}
    if set(rmse) != {
        "baseline",
        "spectrum_loss",
        "spectrum_margin",
        "basis_loss",
        "basis_margin",
    }:
        raise ValueError("Causal headline lacks the five frozen forward models")
    counts = dose["decision_counts"]
    required_counts = {
        "loss_dose_monotone_anchors",
        "margin_dose_monotone_anchors",
        "tail_band_anchors",
        "basis_control_anchors",
    }
    if not required_counts.issubset(counts):
        raise ValueError("Causal headline lacks frozen anchor decision counts")

    bridge = {row["spectrum_predictor"]: row for row in dose["bridge_rows"]}
    if set(bridge) != {"spectrum_loss", "spectrum_margin"}:
        raise ValueError("Causal headline lacks both frozen matched-control bridges")

    def number(value: Any) -> str:
        return _number(value, signed=True)

    temporal_passes = sum(row["passed"] for row in temporal["criteria_rows"])
    overall = "supported" if evidence["overall_verdict"] == "supported" else "claimable negative"
    return (
        f"{CAUSAL_HEADLINE_PREFIX} temporal {temporal_passes}/5, primary LOSO relative "
        "RMSE gains for loss/margin "
        f"{number(primary['validation_loss_p95']['relative_rmse_improvement'])}/"
        f"{number(primary['unseen_margin_p05']['relative_rmse_improvement'])}; fixed-state "
        "loss/margin/tail/basis support "
        f"{counts['loss_dose_monotone_anchors']}/"
        f"{counts['margin_dose_monotone_anchors']}/"
        f"{counts['tail_band_anchors']}/"
        f"{counts['basis_control_anchors']} of 10; 84-row spectrum loss/margin RMSE gains "
        f"{number(rmse['spectrum_loss']['rmse_improvement'])}/"
        f"{number(rmse['spectrum_margin']['rmse_improvement'])} versus matched basis "
        f"{number(rmse['basis_loss']['rmse_improvement'])}/"
        f"{number(rmse['basis_margin']['rmse_improvement'])}; matched decision gaps loss/margin "
        f"{_margin_number(min(bridge['spectrum_loss']['spectrum_rmse_improvement'], bridge['spectrum_loss']['spectrum_rmse_improvement'] - bridge['spectrum_loss']['matched_basis_rmse_improvement']))}/"
        f"{_margin_number(min(bridge['spectrum_margin']['spectrum_rmse_improvement'], bridge['spectrum_margin']['spectrum_rmse_improvement'] - bridge['spectrum_margin']['matched_basis_rmse_improvement']))}; "
        f"joint result {overall}. These "
        "tests constrain a spectral-component explanation but do not identify formal mediation "
        "of full-training BEIR gains."
    )


def causal_chain_paper_contract() -> dict[str, tuple[str, ...]]:
    """Return the frozen causal macro and claim-boundary contract for ``main.tex``."""
    return {
        "required_once": CAUSAL_MAIN_MACROS,
        "required_boundary_substrings": CAUSAL_MAIN_BOUNDARY_SUBSTRINGS,
        "forbidden_overclaim_substrings": CAUSAL_MAIN_FORBIDDEN_OVERCLAIMS,
    }


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def _temporal_primary_rows(temporal: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {
        row["outcome"]: row
        for row in temporal["estimate_rows"]
        if row["predictor"] == PRIMARY_PREDICTOR
    }
    if set(rows) != set(OUTCOME_LABELS):
        raise ValueError("Temporal display lacks both primary-predictor outcomes")
    return rows


def _temporal_criterion_details(temporal: dict[str, Any]) -> dict[str, str]:
    paired = temporal["paired_rows"]
    primary = _temporal_primary_rows(temporal)
    estimates = {(row["outcome"], row["predictor"]): row for row in temporal["estimate_rows"]}
    treatment = "/".join(
        f"{challenger}={sum(row['challenger'] == challenger and row[f'delta_{PRIMARY_PREDICTOR}'] > 0 for row in paired)}/3"
        for challenger in ("muon", "normuon")
    )
    outcome = "/".join(
        f"{challenger}={sum(row['challenger'] == challenger and row['delta_validation_loss_p95'] < 0 and row['delta_unseen_margin_p05'] > 0 for row in paired)}/3"
        for challenger in ("muon", "normuon")
    )
    held_out = "; ".join(
        f"{OUTCOME_LABELS[name]}="
        f"{_number(primary[name]['relative_rmse_improvement'], signed=True)} "
        f"(decision gap {_margin_number(primary[name]['relative_rmse_improvement'])})"
        for name in OUTCOME_LABELS
    )
    controls = []
    for outcome_name in OUTCOME_LABELS:
        primary_gain = primary[outcome_name]["relative_rmse_improvement"]
        values = "/".join(
            _number(estimates[(outcome_name, control)]["relative_rmse_improvement"], signed=True)
            for control in ("update_frobenius_norm", "weight_frobenius_norm")
        )
        gaps = "/".join(
            _margin_number(
                primary_gain - estimates[(outcome_name, control)]["relative_rmse_improvement"]
            )
            for control in ("update_frobenius_norm", "weight_frobenius_norm")
        )
        controls.append(
            f"{OUTCOME_LABELS[outcome_name]} primary={_number(primary_gain, signed=True)}, "
            f"update/weight={values}, decision gaps={gaps}"
        )
    coefficients = []
    for outcome_name in OUTCOME_LABELS:
        row = primary[outcome_name]
        coefficients.append(
            f"{OUTCOME_LABELS[outcome_name]} "
            f"muon abs(beta)={_number(abs(row['muon_coefficient_label_only']))} to "
            f"{_number(abs(row['muon_coefficient_with_predictor']))} "
            f"(gap {_margin_number(abs(row['muon_coefficient_label_only']) - abs(row['muon_coefficient_with_predictor']))}); "
            f"normuon abs(beta)={_number(abs(row['normuon_coefficient_label_only']))} to "
            f"{_number(abs(row['normuon_coefficient_with_predictor']))} "
            f"(gap {_margin_number(abs(row['normuon_coefficient_label_only']) - abs(row['normuon_coefficient_with_predictor']))})"
        )
    return {
        "treatment_shift": treatment,
        "outcome_shift": outcome,
        "held_out_prediction": held_out,
        "negative_control": "; ".join(controls),
        "coefficient_behavior": "; ".join(coefficients),
    }


def _anchor_decision_margins(row: dict[str, Any]) -> tuple[float, float, float, float]:
    loss_gap = min(
        row[f"loss_lambda_{left:.2f}"] - row[f"loss_lambda_{right:.2f}"]
        for left, right in zip((0.0, 0.25, 0.5, 0.75), (0.25, 0.5, 0.75, 1.0), strict=True)
    )
    margin_gap = min(
        row[f"margin_lambda_{right:.2f}"] - row[f"margin_lambda_{left:.2f}"]
        for left, right in zip((0.0, 0.25, 0.5, 0.75), (0.25, 0.5, 0.75, 1.0), strict=True)
    )
    tail_gap = min(
        row["loss_band_head"] - row["loss_band_tail"],
        row["loss_band_middle"] - row["loss_band_tail"],
        row["margin_band_tail"] - row["margin_band_head"],
        row["margin_band_tail"] - row["margin_band_middle"],
    )
    basis_gap = min(row["basis_loss_decision_gap"], row["basis_margin_decision_gap"])
    return loss_gap, margin_gap, tail_gap, basis_gap


def render_causal_chain_markdown(
    evidence: dict[str, Any], *, detailed: bool, heading_level: int = 3
) -> str:
    """Render a complete numerical causal-chain view or an explicit pending receipt."""

    _validate_shape(evidence)
    if heading_level < 1:
        raise ValueError("Markdown heading level must be positive")
    h = "#" * heading_level
    temporal = evidence["temporal_short_branch"]
    dose = evidence["dose_band"]
    lines = [f"{h} Frozen causal-chain numerical tests", ""]
    if evidence.get("complete") is not True:
        for label, branch in (("Temporal shared-start", temporal), ("Dose/band bridge", dose)):
            if branch.get("complete") is True:
                boundary = str(branch["claim_boundary"]).rstrip(".")
                lines.append(f"- **{label}:** {_verdict(branch)} — {boundary}.")
            else:
                reason = branch.get("pending_reason") or "required evidence is incomplete"
                lines.append(f"- **{label}:** pending, not claimable — {reason}.")
        lines += [
            "",
            "No joint causal-chain claim is permitted until both strict manifests and all five "
            "source tables are complete.",
        ]
        return "\n".join(lines)

    overall = "supported" if evidence["overall_verdict"] == "supported" else "claimable negative"
    lines += [
        f"Overall frozen chain: **{overall}**. Temporal: **{_verdict(temporal)}**; "
        f"dose/band/forward bridge: **{_verdict(dose)}**.",
        "",
        f"{h}# Shared-start temporal decision",
        "",
    ]
    details = _temporal_criterion_details(temporal)
    lines.append(
        _markdown_table(
            ("Criterion", "Decision", "Audited numerical evidence"),
            [
                (row["criterion"], _decision(row["passed"]), details[row["criterion"]])
                for row in temporal["criteria_rows"]
            ],
        )
    )
    lines += ["", "The decision is all-required: failure of any row is a complete negative result."]
    if detailed:
        lines += ["", f"{h}# Six randomized paired contrasts", ""]
        lines.append(
            _markdown_table(
                (
                    "Seed",
                    "Challenger",
                    "Δ early tail energy",
                    "Δ final loss p95",
                    "Δ final margin p05",
                ),
                [
                    (
                        row["seed"],
                        row["challenger"],
                        _number(row[f"delta_{PRIMARY_PREDICTOR}"], signed=True),
                        _number(row["delta_validation_loss_p95"], signed=True),
                        _number(row["delta_unseen_margin_p05"], signed=True),
                    )
                    for row in temporal["paired_rows"]
                ],
            )
        )
        lines += ["", f"{h}# All 16 temporal predictor estimates", ""]
        lines.append(
            _markdown_table(
                (
                    "Outcome",
                    "Predictor",
                    "Kind",
                    "Label RMSE",
                    "Predictor RMSE",
                    "Relative improvement",
                    "Muon β label→with (shrink)",
                    "NorMuon β label→with (shrink)",
                ),
                [
                    (
                        OUTCOME_LABELS[row["outcome"]],
                        row["predictor"],
                        row["predictor_kind"],
                        _number(row["label_only_rmse"]),
                        _number(row["mediator_rmse"]),
                        _number(row["relative_rmse_improvement"], signed=True),
                        f"{_number(row['muon_coefficient_label_only'], signed=True)}→"
                        f"{_number(row['muon_coefficient_with_predictor'], signed=True)} "
                        f"({_number(row['muon_absolute_coefficient_shrinkage'], signed=True)})",
                        f"{_number(row['normuon_coefficient_label_only'], signed=True)}→"
                        f"{_number(row['normuon_coefficient_with_predictor'], signed=True)} "
                        f"({_number(row['normuon_absolute_coefficient_shrinkage'], signed=True)})",
                    )
                    for row in temporal["estimate_rows"]
                ],
            )
        )

    lines += ["", f"{h}# Fixed-state dose, band, and basis tests", ""]
    lines.append(
        _markdown_table(
            ("Criterion", "Supporting anchors", "Threshold", "Decision"),
            [
                (
                    row["criterion"],
                    f"{row['supporting_anchors']}/{row['anchors']}",
                    row["threshold"],
                    _decision(row["passed"]),
                )
                for row in dose["criteria_rows"]
            ],
        )
    )
    if detailed:
        lines += ["", f"{h}# All 10 fixed-state anchors", ""]
        lines.append(
            _markdown_table(
                (
                    "Anchor",
                    "Loss dose λ=0/.25/.5/.75/1",
                    "Margin dose λ=0/.25/.5/.75/1",
                    "Loss band H/M/T",
                    "Margin band H/M/T",
                    "Dose L/M",
                    "Tail",
                    "Basis",
                    "All",
                    "Decision gaps L/M/T/B",
                ),
                [
                    (
                        row["anchor"],
                        "/".join(
                            _number(row[f"loss_lambda_{dose_value:.2f}"], signed=True)
                            for dose_value in (0.0, 0.25, 0.5, 0.75, 1.0)
                        ),
                        "/".join(
                            _number(row[f"margin_lambda_{dose_value:.2f}"], signed=True)
                            for dose_value in (0.0, 0.25, 0.5, 0.75, 1.0)
                        ),
                        "/".join(
                            _number(row[f"loss_band_{band}"], signed=True)
                            for band in ("head", "middle", "tail")
                        ),
                        "/".join(
                            _number(row[f"margin_band_{band}"], signed=True)
                            for band in ("head", "middle", "tail")
                        ),
                        f"{_decision(row['loss_dose_monotone'])}/"
                        f"{_decision(row['margin_dose_monotone'])}",
                        _decision(row["tail_band_best_both_metrics"]),
                        _decision(row["basis_swap_negative_control"]),
                        _decision(row["anchor_passed"]),
                        "/".join(map(_margin_number, _anchor_decision_margins(row))),
                    )
                    for row in dose["anchor_rows"]
                ],
            )
        )

    lines += [
        "",
        f"{h}# Held-run retrieval bridge (84 rows)",
        "",
        _markdown_table(
            ("Predictor", "Kind", "RMSE", "Improvement", "Matched control"),
            [
                (
                    row["predictor"],
                    row["predictor_kind"],
                    _number(row["rmse"]),
                    _number(row["rmse_improvement"], signed=True),
                    row["matched_control"] or "—",
                )
                for row in dose["rmse_rows"]
            ],
        ),
        "",
        _markdown_table(
            (
                "Spectrum predictor",
                "ΔRMSE",
                "Matched basis",
                "Basis ΔRMSE",
                "Baseline gap",
                "Control gap",
                "Decision",
            ),
            [
                (
                    row["spectrum_predictor"],
                    _number(row["spectrum_rmse_improvement"], signed=True),
                    row["matched_basis_control"],
                    _number(row["matched_basis_rmse_improvement"], signed=True),
                    _margin_number(row["spectrum_rmse_improvement"]),
                    _margin_number(
                        row["spectrum_rmse_improvement"] - row["matched_basis_rmse_improvement"]
                    ),
                    _decision(row["passed"]),
                )
                for row in dose["bridge_rows"]
            ],
        ),
    ]
    if detailed:
        lines += ["", f"{h}# All 84 held-run predictions", ""]
        lines.append(
            _markdown_table(
                (
                    "Held-out run",
                    "Task",
                    "Transition",
                    "Observed",
                    "Baseline",
                    "Spectrum loss",
                    "Spectrum margin",
                    "Basis loss",
                    "Basis margin",
                ),
                [
                    (
                        row["held_out_run"],
                        row["task"],
                        row["transition"],
                        _number(row["observed_increment"], signed=True),
                        _number(row["baseline_prediction"], signed=True),
                        _number(row["spectrum_loss_prediction"], signed=True),
                        _number(row["spectrum_margin_prediction"], signed=True),
                        _number(row["basis_loss_prediction"], signed=True),
                        _number(row["basis_margin_prediction"], signed=True),
                    )
                    for row in dose["heldout_rows"]
                ],
            )
        )
    lines += [
        "",
        f"> Temporal boundary: {str(temporal['claim_boundary']).rstrip('.')}.",
        "",
        f"> Dose/bridge boundary: {str(dose['claim_boundary']).rstrip('.')}.",
    ]
    return "\n".join(lines)


def _latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _latex_rows(rows: list[tuple[Any, ...]]) -> str:
    return "\n".join(" & ".join(_latex_escape(cell) for cell in row) + r" \\" for row in rows)


def _latex_anchor_label(value: str) -> str:
    label = value.removeprefix("dense/")
    if label == "pretrained":
        return "pretrained"
    run, checkpoint = label.split("/checkpoint-", 1)
    run_labels = {
        "adamw-lr1e-5": "AdamW 1e-5",
        "muon-lr1e-3": "Muon 1e-3",
        "normuon-lr1e-3": "NorMuon 1e-3",
    }
    if run not in run_labels:
        raise ValueError(f"Unexpected causal-chain anchor label: {value}")
    return f"{run_labels[run]} at {checkpoint}"


def render_causal_chain_latex(evidence: dict[str, Any]) -> str:
    """Render the one-table main result plus complete appendix diagnostics."""

    _validate_shape(evidence)
    if evidence.get("complete") is not True:
        raise ValueError("Final causal-chain LaTeX requires complete claimable evidence")
    temporal = evidence["temporal_short_branch"]
    dose = evidence["dose_band"]
    primary = _temporal_primary_rows(temporal)
    details = _temporal_criterion_details(temporal)
    temporal_passes = sum(row["passed"] for row in temporal["criteria_rows"])
    counts = dose["decision_counts"]
    spectrum_improvements = {
        row["predictor"]: row["rmse_improvement"]
        for row in dose["rmse_rows"]
        if row["predictor"].startswith("spectrum_")
    }
    basis_improvements = {
        row["predictor"]: row["rmse_improvement"]
        for row in dose["rmse_rows"]
        if row["predictor"].startswith("basis_")
    }
    bridge_by_predictor = {row["spectrum_predictor"]: row for row in dose["bridge_rows"]}
    local_verdict = "supported" if dose["local_supported"] else "claimable negative"
    summary_rows = [
        (
            "Shared-start temporal",
            "LOSO versus optimizer labels and two norm controls",
            f"{temporal_passes}/5; relative RMSE gain: loss "
            f"{_number(primary['validation_loss_p95']['relative_rmse_improvement'], signed=True)}; "
            f"margin {_number(primary['unseen_margin_p05']['relative_rmse_improvement'], signed=True)}; "
            f"{_verdict(temporal)}",
            (
                "Accumulated spectral bridge supported, not formal mediation"
                if temporal["supported"]
                else "Temporal spectral bridge rejected"
            ),
        ),
        (
            "Fixed-state component",
            "Five doses, tail localization, spectrum-versus-basis",
            f"support of 10: loss {counts['loss_dose_monotone_anchors']}; margin "
            f"{counts['margin_dose_monotone_anchors']}; tail {counts['tail_band_anchors']}; "
            f"basis {counts['basis_control_anchors']}; {local_verdict}",
            (
                "Fixed-weight component attribution supported"
                if dose["local_supported"]
                else "Frozen component account rejected"
            ),
        ),
        (
            "Forward retrieval",
            "84 leave-source-run-out task-transition rows",
            f"spectrum dRMSE: loss {_number(spectrum_improvements['spectrum_loss'], signed=True)}; "
            f"margin {_number(spectrum_improvements['spectrum_margin'], signed=True)}; basis: "
            f"{_number(basis_improvements['basis_loss'], signed=True)}; "
            f"{_number(basis_improvements['basis_margin'], signed=True)}; decision gaps "
            f"loss {_margin_number(min(bridge_by_predictor['spectrum_loss']['spectrum_rmse_improvement'], bridge_by_predictor['spectrum_loss']['spectrum_rmse_improvement'] - bridge_by_predictor['spectrum_loss']['matched_basis_rmse_improvement']))}; "
            f"margin {_margin_number(min(bridge_by_predictor['spectrum_margin']['spectrum_rmse_improvement'], bridge_by_predictor['spectrum_margin']['spectrum_rmse_improvement'] - bridge_by_predictor['spectrum_margin']['matched_basis_rmse_improvement']))}; "
            f"{_decision(dose['forward_bridge_supported'])}",
            (
                "Out-of-run retrieval bridge supported, not formal mediation"
                if dose["forward_bridge_supported"]
                else "No forward bridge; fixed-state conclusion only"
            ),
        ),
    ]
    temporal_criteria_rows = [
        (
            TEMPORAL_CRITERION_LABELS[row["criterion"]],
            details[row["criterion"]],
            _decision(row["passed"]),
        )
        for row in temporal["criteria_rows"]
    ] + [("Joint decision", "All five conditions are required", _verdict(temporal))]
    estimate_rows = [
        (
            OUTCOME_LABELS[row["outcome"]],
            PREDICTOR_LABELS[row["predictor"]],
            PREDICTOR_KIND_LABELS[row["predictor_kind"]],
            _number(row["label_only_rmse"]),
            _number(row["mediator_rmse"]),
            _number(row["relative_rmse_improvement"], signed=True),
            f"{_number(row['muon_coefficient_label_only'], signed=True)} to "
            f"{_number(row['muon_coefficient_with_predictor'], signed=True)} "
            f"({_number(row['muon_absolute_coefficient_shrinkage'], signed=True)})",
            f"{_number(row['normuon_coefficient_label_only'], signed=True)} to "
            f"{_number(row['normuon_coefficient_with_predictor'], signed=True)} "
            f"({_number(row['normuon_absolute_coefficient_shrinkage'], signed=True)})",
        )
        for row in temporal["estimate_rows"]
    ]
    paired_rows = [
        (
            row["seed"],
            row["challenger"],
            _number(row[f"delta_{PRIMARY_PREDICTOR}"], signed=True),
            _number(row["delta_validation_loss_p95"], signed=True),
            _number(row["delta_unseen_margin_p05"], signed=True),
        )
        for row in temporal["paired_rows"]
    ]
    dose_criteria_rows = [
        (
            DOSE_CRITERION_LABELS[row["criterion"]],
            f"{row['supporting_anchors']}/{row['anchors']}",
            row["threshold"],
            _decision(row["passed"]),
        )
        for row in dose["criteria_rows"]
    ] + [
        (
            "held-run bridge",
            "84 rows",
            "spectrum gain > 0 and > matched basis gain",
            _decision(dose["forward_bridge_supported"]),
        ),
        ("joint decision", "local plus forward", "all required", _verdict(dose)),
    ]
    anchor_rows = [
        (
            _latex_anchor_label(row["anchor"]),
            " / ".join(
                _number(row[f"loss_lambda_{dose_value:.2f}"], signed=True)
                for dose_value in (0.0, 0.25, 0.5, 0.75, 1.0)
            ),
            " / ".join(
                _number(row[f"margin_lambda_{dose_value:.2f}"], signed=True)
                for dose_value in (0.0, 0.25, 0.5, 0.75, 1.0)
            ),
            " / ".join(
                _number(row[f"loss_band_{band}"], signed=True)
                for band in ("head", "middle", "tail")
            ),
            " / ".join(
                _number(row[f"margin_band_{band}"], signed=True)
                for band in ("head", "middle", "tail")
            ),
            " / ".join(
                "P" if row[field] else "F"
                for field in (
                    "loss_dose_monotone",
                    "margin_dose_monotone",
                    "tail_band_best_both_metrics",
                    "basis_swap_negative_control",
                    "anchor_passed",
                )
            ),
            " / ".join(map(_latex_margin_number, _anchor_decision_margins(row))),
        )
        for row in dose["anchor_rows"]
    ]
    rmse_rows = [
        (
            PREDICTOR_LABELS[row["predictor"]],
            PREDICTOR_KIND_LABELS[row["predictor_kind"]],
            _number(row["rmse"]),
            _number(row["rmse_improvement"], signed=True),
            PREDICTOR_LABELS[row["matched_control"]] if row["matched_control"] else "--",
        )
        for row in dose["rmse_rows"]
    ]
    return (
        "% Generated from strict causal-chain evidence; do not edit.\n"
        "\\newcommand{\\CausalChainSummaryTable}{%\n"
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{p{0.16\\linewidth}p{0.25\\linewidth}p{0.32\\linewidth}p{0.20\\linewidth}}\n"
        "\\toprule\nTest & Frozen comparison & Audited numerical result & Permitted inference \\\\\n\n"
        "\\midrule\n" + _latex_rows(summary_rows) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Frozen causal-chain stress tests. Every decision and its numerical basis is "
        "reported whether supported or negative; joint passage is not formal mediation.}\n"
        "\\label{tab:causal-chain-summary}\n\\end{table*}%\n}\n\n"
        "\\newcommand{\\CausalChainDiagnostics}{%\n"
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{p{0.18\\linewidth}p{0.62\\linewidth}p{0.12\\linewidth}}\n"
        "\\toprule\nTemporal criterion & Audited numerical evidence & Decision \\\\\n\n\\midrule\n"
        + _latex_rows(temporal_criteria_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{All five frozen temporal criteria and their all-required decision.}\n"
        "\\label{tab:causal-temporal-diagnostics}\n\\end{table*}\n"
        "\\begin{table*}[t]\n\\centering\n\\tiny\n\\setlength{\\tabcolsep}{1.2pt}\n"
        "\\begin{tabular}{p{0.10\\linewidth}p{0.12\\linewidth}p{0.07\\linewidth}p{0.09\\linewidth}p{0.09\\linewidth}p{0.09\\linewidth}p{0.16\\linewidth}p{0.16\\linewidth}}\n"
        "\\toprule\nOutcome & Predictor & Kind & Label RMSE & Pred. RMSE & Rel. gain & "
        "Muon $\\beta$ before$\\rightarrow$after (shrink) & NorMuon $\\beta$ before$\\rightarrow$after (shrink) \\\\\n\n\\midrule\n"
        + _latex_rows(estimate_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{All 16 frozen temporal predictor estimates, including both norm controls.}\n"
        "\\label{tab:causal-temporal-estimates}\n\\end{table*}\n"
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
        "\\begin{tabular}{llrrr}\n\\toprule\nSeed & Challenger & $\\Delta$ tail energy & "
        "$\\Delta$ loss p95 & $\\Delta$ margin p05 \\\\\n\n\\midrule\n"
        + _latex_rows(paired_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{All six randomized shared-start paired contrasts.}\n"
        "\\label{tab:causal-temporal-pairs}\n\\end{table*}\n"
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{p{0.22\\linewidth}p{0.16\\linewidth}p{0.40\\linewidth}p{0.12\\linewidth}}\n"
        "\\toprule\nComponent criterion & Support & Threshold & Decision \\\\\n\n\\midrule\n"
        + _latex_rows(dose_criteria_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Dose, band, basis, and held-run bridge decisions.}\n"
        "\\label{tab:causal-dose-diagnostics}\n\\end{table*}\n"
        "\\begin{table*}[t]\n\\centering\n\\tiny\n\\setlength{\\tabcolsep}{1.2pt}\n"
        "\\begin{tabular}{p{0.13\\linewidth}p{0.17\\linewidth}p{0.17\\linewidth}p{0.11\\linewidth}p{0.11\\linewidth}p{0.08\\linewidth}p{0.16\\linewidth}}\n"
        "\\toprule\nAnchor & Loss doses (0, .25, .5, .75, 1) & "
        "Margin doses (0, .25, .5, .75, 1) & "
        "Loss H/M/T & Margin H/M/T & Tests & Min gaps L/M/T/B \\\\\n\n\\midrule\n"
        + _latex_rows(anchor_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{All ten fixed-state dose, band, and basis-control anchors. Test order is "
        "loss-dose, margin-dose, tail, basis, and all; signed minimum gaps make strict ties "
        "auditable.}\n"
        "\\label{tab:causal-dose-anchors}\n\\end{table*}\n"
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{p{0.18\\linewidth}p{0.16\\linewidth}p{0.10\\linewidth}p{0.12\\linewidth}p{0.22\\linewidth}}\n"
        "\\toprule\nPredictor & Kind & RMSE & $\\Delta$RMSE & Matched control \\\\\n\n\\midrule\n"
        + _latex_rows(rmse_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Five models over all 84 held-run retrieval rows. Positive $\\Delta$RMSE is improvement.}\n"
        "\\label{tab:causal-forward-rmse}\n\\end{table*}%\n}\n"
    )


def causal_chain_display_contract(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return row cardinalities and hashes that strict consumers can rederive."""

    _validate_shape(evidence)
    temporal = evidence["temporal_short_branch"]
    dose = evidence["dose_band"]
    complete = evidence.get("complete") is True
    repository_root = Path(str(evidence.get("repository_root", "")))
    if not repository_root.is_absolute():
        raise ValueError("Causal evidence lacks an absolute repository root")
    repository_root = repository_root.resolve()
    source_tables = []
    for record in evidence.get("source_table_records", []):
        path = Path(record["path"]).resolve()
        try:
            relative = path.relative_to(repository_root)
        except ValueError as error:
            raise ValueError(f"Causal source table is outside repository root: {path}") from error
        if not relative.parts or relative.parts[0] != "reports":
            raise ValueError(f"Causal source table is outside reports/: {path}")
        source_tables.append(
            {
                "path": relative.as_posix(),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )
    canonical = {
        "overall_verdict": evidence.get("overall_verdict"),
        "temporal_status": temporal.get("status"),
        "dose_status": dose.get("status"),
        "temporal_criteria": temporal.get("criteria_rows", []),
        "temporal_paired": temporal.get("paired_rows", []),
        "temporal_estimates": temporal.get("estimate_rows", []),
        "temporal_loso": temporal.get("loso_rows", []),
        "dose_criteria": dose.get("criteria_rows", []),
        "dose_anchors": dose.get("anchor_rows", []),
        "dose_rmse": dose.get("rmse_rows", []),
        "dose_bridge": dose.get("bridge_rows", []),
        "dose_heldout": dose.get("heldout_rows", []),
        "source_tables": source_tables,
        "claim_boundaries": evidence.get("claim_boundaries"),
    }
    canonical_bytes = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    full_markdown = render_causal_chain_markdown(evidence, detailed=True)
    compact_markdown = render_causal_chain_markdown(evidence, detailed=False)
    result = {
        "complete": complete,
        "overall_verdict": evidence.get("overall_verdict"),
        "temporal_criteria": len(temporal.get("criteria_rows", [])),
        "temporal_paired_rows": len(temporal.get("paired_rows", [])),
        "temporal_estimates": len(temporal.get("estimate_rows", [])),
        "temporal_loso_rows": len(temporal.get("loso_rows", [])),
        "dose_criteria": len(dose.get("criteria_rows", [])),
        "dose_anchors": len(dose.get("anchor_rows", [])),
        "dose_rmse_rows": len(dose.get("rmse_rows", [])),
        "dose_bridge_rows": len(dose.get("bridge_rows", [])),
        "dose_heldout_rows": len(dose.get("heldout_rows", [])),
        "source_tables": len(evidence.get("source_table_records", [])),
        "evidence_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "full_markdown_sha256": hashlib.sha256(full_markdown.encode()).hexdigest(),
        "compact_markdown_sha256": hashlib.sha256(compact_markdown.encode()).hexdigest(),
    }
    if complete:
        result["latex_sha256"] = hashlib.sha256(
            render_causal_chain_latex(evidence).encode()
        ).hexdigest()
    return result


__all__ = [
    "CAUSAL_HEADLINE_PREFIX",
    "causal_chain_paper_contract",
    "causal_chain_display_contract",
    "render_causal_chain_headline_fragment",
    "render_causal_chain_latex",
    "render_causal_chain_markdown",
]
