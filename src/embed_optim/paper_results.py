from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .decontamination import DECONTAMINATED_TASK_NAMES
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .mechanism_report import (
    OPTIMIZER_LABELS,
    _atomic_text,
    _basis_rows,
    _bridge_rows,
    _common_state_rows,
    _read_declared_csv,
    _retrieval_rows,
    _spectrum_rows,
)
from .outcome_report import (
    _confirmation_rows,
    _functional_rows,
    _hybrid_rows,
    _short_branch_rows,
)
from .paper_audit import (
    HEADLINE_MACROS,
    PAPER_RESULT_TABLE_PATHS,
    _macros,
    audit_paper,
    expected_constant_macros,
    load_paper_claim_protocol,
)

FAMILY_LABELS = {"dense": "DenseOn", "late": "LateOn"}
FAMILIES = ("dense", "late")
OPTIMIZERS = ("adamw", "muon", "normuon")
CONTRAST_LABELS = ("Muon - AdamW", "NorMuon - AdamW", "NorMuon - Muon")


class IncompletePaperEvidenceError(ValueError):
    """Raised when the frozen evidence tiers are not all available yet."""


def _finite(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid paper result value: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite paper result value: {value!r}")
    return parsed


def _indexed(rows: list[list[str]], *, context: str) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if len(row) < 2 or (row[0], row[1]) in result:
            raise ValueError(f"Invalid or duplicate {context} row: {row}")
        result[(row[0], row[1])] = row
    return result


def _discovery_final_medians(
    retrieval_dir: Path,
    manifest: dict[str, Any],
) -> tuple[dict[tuple[str, str], float], Path]:
    repository_root = retrieval_dir.resolve().parents[1]
    rows, table = _read_declared_csv(
        repository_root,
        manifest,
        "checkpoint_dynamics",
        required_fields={"model_family", "optimizer", "stage", "mean_ndcg_at_10"},
    )
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        family = row["model_family"]
        optimizer = row["optimizer"]
        try:
            stage = int(row["stage"])
        except ValueError as error:
            raise ValueError(f"Invalid discovery checkpoint stage: {row}") from error
        if family not in FAMILIES or optimizer not in OPTIMIZERS or not 1 <= stage <= 5:
            raise ValueError(f"Invalid discovery checkpoint identity: {row}")
        if stage == 5:
            grouped.setdefault((family, optimizer), []).append(_finite(row["mean_ndcg_at_10"]))
    expected = {(family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS}
    if (
        len(rows) != 120
        or set(grouped) != expected
        or any(len(values) != 4 for values in grouped.values())
    ):
        raise ValueError("Discovery headline requires four final points for all six family groups")
    return {key: float(statistics.median(values)) for key, values in grouped.items()}, table


def _discovery_task_rows(
    retrieval_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[list[str]], Path]:
    repository_root = retrieval_dir.resolve().parents[1]
    required = {
        "model_family",
        "task",
        "adamw",
        "muon",
        "normuon",
        "muon_minus_adamw",
        "normuon_minus_adamw",
    }
    rows, table = _read_declared_csv(
        repository_root,
        manifest,
        "best_config_task_comparison",
        required_fields=required,
    )
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        identity = (row.get("model_family", ""), row.get("task", ""))
        if identity in indexed:
            raise ValueError(f"Duplicate discovery per-task identity: {identity}")
        indexed[identity] = row
    expected = {(family, task) for family in FAMILIES for task in DECONTAMINATED_TASK_NAMES}
    if len(rows) != 28 or set(indexed) != expected:
        raise ValueError("Discovery per-task table requires both families and all 14 tasks")

    output: list[list[str]] = []
    for family in FAMILIES:
        for task in DECONTAMINATED_TASK_NAMES:
            row = indexed[(family, task)]
            adamw = _finite(row["adamw"])
            muon = _finite(row["muon"])
            normuon = _finite(row["normuon"])
            muon_delta = _finite(row["muon_minus_adamw"])
            normuon_delta = _finite(row["normuon_minus_adamw"])
            if (
                not all(0 <= score <= 1 for score in (adamw, muon, normuon))
                or abs((muon - adamw) - muon_delta) > 5e-12
                or abs((normuon - adamw) - normuon_delta) > 5e-12
            ):
                raise ValueError(f"Invalid discovery per-task values: {(family, task)}")
            output.append(
                [
                    FAMILY_LABELS[family],
                    task,
                    f"{adamw:.4f}",
                    f"{muon:.4f}",
                    f"{normuon:.4f}",
                    f"{muon_delta:+.4f}",
                    f"{normuon_delta:+.4f}",
                ]
            )
    return output, table


def _discovery_task_stability_rows(
    retrieval_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[list[str]], Path]:
    repository_root = retrieval_dir.resolve().parents[1]
    required = {
        "model_family",
        "optimizer",
        "baseline",
        "first_stage",
        "second_stage",
        "first_fraction",
        "second_fraction",
        "tasks",
        "same_direction_tasks",
        "pearson_correlation",
        "spearman_correlation",
    }
    rows, table = _read_declared_csv(
        repository_root,
        manifest,
        "task_delta_stability",
        required_fields=required,
    )
    indexed: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for row in rows:
        try:
            identity = (
                row["model_family"],
                row["optimizer"],
                int(row["first_stage"]),
                int(row["second_stage"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid discovery task-stability identity: {row}") from error
        if identity in indexed:
            raise ValueError(f"Duplicate discovery task-stability identity: {identity}")
        indexed[identity] = row
    expected = {
        (family, optimizer, stage, stage + 1)
        for family in FAMILIES
        for optimizer in ("muon", "normuon")
        for stage in range(1, 5)
    }
    if len(rows) != 16 or set(indexed) != expected:
        raise ValueError("Discovery task-stability table requires all 16 adjacent-stage contrasts")

    output = []
    for family, optimizer, first_stage, second_stage in sorted(expected):
        row = indexed[(family, optimizer, first_stage, second_stage)]
        tasks = int(row["tasks"])
        stable = int(row["same_direction_tasks"])
        first_fraction = _finite(row["first_fraction"])
        second_fraction = _finite(row["second_fraction"])
        pearson = _finite(row["pearson_correlation"])
        spearman = _finite(row["spearman_correlation"])
        if (
            row["baseline"] != "adamw"
            or tasks != len(DECONTAMINATED_TASK_NAMES)
            or not 0 <= stable <= tasks
            or abs(first_fraction - first_stage / 5) > 1e-12
            or abs(second_fraction - second_stage / 5) > 1e-12
            or not -1 <= pearson <= 1
            or not -1 <= spearman <= 1
        ):
            raise ValueError(f"Invalid discovery task-stability values: {row}")
        output.append(
            [
                FAMILY_LABELS[family],
                f"{OPTIMIZER_LABELS[optimizer]} - AdamW",
                f"{int(round(first_fraction * 100))}--{int(round(second_fraction * 100))}%",
                f"{stable}/{tasks}",
                f"{pearson:.3f}",
                f"{spearman:.3f}",
            ]
        )
    return output, table


def _ci_classification(cell: str) -> str:
    if not (cell.startswith("[") and cell.endswith("]") and "," in cell):
        raise ValueError(f"Invalid confirmatory interval cell: {cell!r}")
    lower_text, upper_text = cell[1:-1].split(",", 1)
    lower = _finite(lower_text.strip())
    upper = _finite(upper_text.strip())
    if lower > upper:
        raise ValueError(f"Reversed confirmatory interval: {cell!r}")
    if lower > 0:
        return "positive"
    if upper < 0:
        return "negative"
    return "inconclusive"


def build_headline_macros(
    *,
    final_medians: dict[tuple[str, str], float],
    retrieval_rows: list[list[str]],
    common_rows: list[list[str]],
    spectrum_rows: list[list[str]],
    representation_rows: list[list[str]],
    correlation_rows: list[list[str]],
    functional_rows: list[list[str]],
    hybrid_rows: list[list[str]],
    short_rows: list[list[str]],
    confirmation_rows: list[list[str]],
) -> dict[str, str]:
    retrieval = _indexed(retrieval_rows, context="retrieval")
    common = _indexed(common_rows, context="common-state")
    spectra = _indexed(spectrum_rows, context="spectrum")
    representation = _indexed(representation_rows, context="representation")
    functional = {(row[0], row[1], row[2]): row for row in functional_rows if len(row) >= 8}
    short = _indexed(short_rows, context="short-branch")
    confirmation = _indexed(confirmation_rows, context="confirmation")

    optimizer_labels = tuple(OPTIMIZER_LABELS[name] for name in OPTIMIZERS)
    discovery_parts = []
    for family in FAMILIES:
        family_label = FAMILY_LABELS[family]
        medians = "/".join(f"{final_medians[(family, optimizer)]:.4f}" for optimizer in OPTIMIZERS)
        reached = "/".join(
            retrieval[(family_label, optimizer_label)][3].split("/", 1)[0]
            for optimizer_label in optimizer_labels
        )
        discovery_parts.append(
            f"{family_label} median final nDCG@10 was {medians}, with {reached} of four "
            "learning rates reaching the frozen AdamW reference"
        )
    discovery = "For AdamW/Muon/NorMuon respectively, " + "; ".join(discovery_parts) + "."

    common_parts = []
    for family_label in FAMILY_LABELS.values():
        row_cv = "/".join(common[(family_label, operator)][2] for operator in ("Muon", "NorMuon"))
        stable = "/".join(spectra[(family_label, operator)][2] for operator in ("Muon", "NorMuon"))
        common_parts.append(
            f"{family_label} Muon/NorMuon row-CV ratios to AdamW were {row_cv} and their "
            f"normalized exact stable ranks were {stable}"
        )
    common_headline = "At shared-gradient common states, " + "; ".join(common_parts) + "."

    correlation = {(row[0], row[1], row[2]): row for row in correlation_rows if len(row) >= 5}
    representation_parts = []
    rho_parts = []
    for family_label in FAMILY_LABELS.values():
        margins = "/".join(
            representation[(family_label, operator)][3] for operator in optimizer_labels
        )
        representation_parts.append(f"{family_label} unseen margins were {margins}")
        rho_parts.append(correlation[(family_label, "unseen margin", "mean BEIR nDCG@10")][4])
    late_coverage = "/".join(
        representation[("LateOn", operator)][6] for operator in optimizer_labels
    )
    representation_headline = (
        "For AdamW/Muon/NorMuon respectively, "
        + "; ".join(representation_parts)
        + f", and LateOn document-token coverage was {late_coverage}. The descriptive within-run "
        f"margin-to-BEIR Spearman rho was {rho_parts[0]}/{rho_parts[1]} for DenseOn/LateOn."
    )

    intervention_parts = []
    for family_label in FAMILY_LABELS.values():
        margins = "/".join(
            functional[(family_label, operator, "descent")][4] for operator in optimizer_labels
        )
        muon_margin = short[(family_label, "Muon - AdamW")][3]
        normuon_margin = short[(family_label, "NorMuon - Muon")][3]
        hybrid_deltas = [
            _finite(row[4]) for row in hybrid_rows if len(row) >= 6 and row[0] == family_label
        ]
        if len(hybrid_deltas) != 4:
            raise ValueError(f"Hybrid headline requires four learning rates for {family_label}")
        intervention_parts.append(
            f"{family_label} matched-step margin changes were {margins}, final shared-start "
            f"Muon--AdamW/NorMuon--Muon margin contrasts were {muon_margin}/{normuon_margin}, "
            f"and mean hybrid-routing AdamW change was {statistics.mean(hybrid_deltas):.4f}"
        )
    intervention_headline = (
        "For AdamW/Muon/NorMuon respectively, " + "; ".join(intervention_parts) + "."
    )

    confirmation_parts = []
    for family_label in FAMILY_LABELS.values():
        cells = []
        for contrast in CONTRAST_LABELS:
            row = confirmation[(family_label, contrast)]
            cells.append(
                f"{contrast.replace(' - ', '--')} {row[2]} {row[4]} ({_ci_classification(row[4])})"
            )
        confirmation_parts.append(f"{family_label}: " + ", ".join(cells))
    confirmation_headline = (
        "The validation-frozen three-seed paired nDCG@10 contrasts were "
        + "; ".join(confirmation_parts)
        + "."
    )

    result = {
        "DiscoveryHeadline": discovery,
        "CommonStateHeadline": common_headline,
        "RepresentationHeadline": representation_headline,
        "InterventionHeadline": intervention_headline,
        "ConfirmationHeadline": confirmation_headline,
    }
    if set(result) != set(HEADLINE_MACROS) or any("\n" in value for value in result.values()):
        raise ValueError("Generated paper headlines violate the single-line macro contract")
    return result


def _latex_escape(value: object) -> str:
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


def _latex_table(
    *,
    environment: str,
    columns: str,
    headers: tuple[str, ...],
    rows: list[tuple[object, ...]],
    caption: str,
    label: str,
) -> str:
    if environment not in {"table", "table*"} or len(columns) != len(headers):
        raise ValueError("Invalid generated LaTeX table shape")
    if not rows or any(len(row) != len(headers) for row in rows):
        raise ValueError("Generated LaTeX table rows do not match their headers")
    body = [" & ".join(_latex_escape(cell) for cell in row) + r" \\" for row in rows]
    return "\n".join(
        [
            "% Generated atomically by embed-optim-render-paper-results.",
            f"\\begin{{{environment}}}[t]",
            r"\centering",
            r"\small",
            f"\\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\end{{{environment}}}",
            "",
        ]
    )


def build_result_tables(
    *,
    final_medians: dict[tuple[str, str], float],
    retrieval_rows: list[list[str]],
    task_rows: list[list[str]],
    task_stability_rows: list[list[str]],
    common_rows: list[list[str]],
    basis_rows: list[list[str]],
    spectrum_rows: list[list[str]],
    representation_rows: list[list[str]],
    functional_rows: list[list[str]],
    hybrid_rows: list[list[str]],
    short_rows: list[list[str]],
    confirmation_rows: list[list[str]],
) -> dict[str, str]:
    retrieval = _indexed(retrieval_rows, context="retrieval table")
    common = _indexed(common_rows, context="common-state table")
    basis = _indexed(basis_rows, context="basis-sensitivity table")
    spectra = _indexed(spectrum_rows, context="spectrum table")
    representation = _indexed(representation_rows, context="representation table")
    functional = {(row[0], row[1], row[2]): row for row in functional_rows if len(row) >= 8}
    short = _indexed(short_rows, context="short-branch table")
    confirmation = _indexed(confirmation_rows, context="confirmation table")
    optimizer_labels = tuple(OPTIMIZER_LABELS[name] for name in OPTIMIZERS)

    expected_task_identities = {
        (family_label, task)
        for family_label in FAMILY_LABELS.values()
        for task in DECONTAMINATED_TASK_NAMES
    }
    indexed_tasks = {(row[0], row[1]): row for row in task_rows if len(row) == 7}
    if len(task_rows) != 28 or set(indexed_tasks) != expected_task_identities:
        raise ValueError("Paper per-task table requires both families and all 14 tasks")
    expected_stability = {
        (family_label, f"{optimizer} - AdamW", f"{first * 20}--{(first + 1) * 20}%")
        for family_label in FAMILY_LABELS.values()
        for optimizer in ("Muon", "NorMuon")
        for first in range(1, 5)
    }
    indexed_stability = {
        (row[0], row[1], row[2]): row for row in task_stability_rows if len(row) == 6
    }
    if len(task_stability_rows) != 16 or set(indexed_stability) != expected_stability:
        raise ValueError("Paper task-stability table requires all adjacent-stage contrasts")

    discovery_rows = [
        (
            FAMILY_LABELS[family],
            OPTIMIZER_LABELS[optimizer],
            f"{final_medians[(family, optimizer)]:.4f}",
            retrieval[(FAMILY_LABELS[family], OPTIMIZER_LABELS[optimizer])][3],
        )
        for family in FAMILIES
        for optimizer in OPTIMIZERS
    ]
    common_table_rows = [
        (
            family_label,
            operator,
            common[(family_label, operator)][2],
            spectra[(family_label, operator)][2],
        )
        for family_label in FAMILY_LABELS.values()
        for operator in ("Muon", "NorMuon")
    ]
    basis_table_rows = [
        tuple(basis[(family_label, optimizer)])
        for family_label in FAMILY_LABELS.values()
        for optimizer in optimizer_labels
    ]
    representation_table_rows = [
        (
            family_label,
            optimizer,
            representation[(family_label, optimizer)][3],
            representation[(family_label, optimizer)][5],
            representation[(family_label, optimizer)][6] if family_label == "LateOn" else "--",
        )
        for family_label in FAMILY_LABELS.values()
        for optimizer in optimizer_labels
    ]
    intervention_rows = []
    for family_label in FAMILY_LABELS.values():
        matched = "/".join(
            functional[(family_label, optimizer, "descent")][4] for optimizer in optimizer_labels
        )
        shared = "/".join(
            (
                short[(family_label, "Muon - AdamW")][3],
                short[(family_label, "NorMuon - Muon")][3],
            )
        )
        hybrid_deltas = [
            _finite(row[4]) for row in hybrid_rows if len(row) >= 6 and row[0] == family_label
        ]
        if len(hybrid_deltas) != 4:
            raise ValueError(f"Intervention table requires four hybrid rows for {family_label}")
        intervention_rows.append(
            (family_label, matched, shared, f"{statistics.mean(hybrid_deltas):.4f}")
        )
    confirmation_table_rows = [
        (
            family_label,
            contrast,
            *confirmation[(family_label, contrast)][2:7],
        )
        for family_label in FAMILY_LABELS.values()
        for contrast in CONTRAST_LABELS
    ]

    per_task_tables = "\n".join(
        [
            *[
                _latex_table(
                    environment="table*",
                    columns="lccccc",
                    headers=(
                        "Task",
                        "AdamW",
                        "Muon",
                        "NorMuon",
                        "Muon $-$ AdamW",
                        "NorMuon $-$ AdamW",
                    ),
                    rows=[
                        tuple(indexed_tasks[(family_label, task)][1:])
                        for task in DECONTAMINATED_TASK_NAMES
                    ],
                    caption=(
                        f"{family_label} discovery per-task final nDCG@10 for each optimizer's "
                        "best learning-rate point on this same BEIR suite. These are exploratory, "
                        "test-selected comparisons rather than an unbiased recipe estimate."
                    ),
                    label=f"tab:{family_label.lower()}-per-task-results",
                )
                for family_label in FAMILY_LABELS.values()
            ],
            _latex_table(
                environment="table*",
                columns="lllccc",
                headers=(
                    "Model",
                    "Comparison",
                    "Stages",
                    "Same direction",
                    "Pearson $r$",
                    "Spearman $\\rho$",
                ),
                rows=[tuple(row) for row in task_stability_rows],
                caption=(
                    "Post-hoc adjacent-checkpoint stability of the 14 task effects for each "
                    "final-score-selected optimizer/LR run against the selected AdamW run. This "
                    "exploratory diagnostic was added after heterogeneous LateOn directions "
                    "became visible and is outside the confirmatory family."
                ),
                label="tab:task-delta-stability",
            ),
        ]
    )

    common_tables = "\n".join(
        (
            _latex_table(
                environment="table*",
                columns="llcc",
                headers=("Model", "Rule", "Row-CV / AdamW", "Normalized stable rank"),
                rows=common_table_rows,
                caption="Same-state update fingerprints under shared gradients.",
                label="tab:common-state-results",
            ),
            _latex_table(
                environment="table*",
                columns="llccccc",
                headers=(
                    "Model",
                    "Rule",
                    "Mapped cosine",
                    "Direction error",
                    "Norm-ratio error",
                    "Descent error",
                    "Q/K spectrum error",
                ),
                rows=basis_table_rows,
                caption=(
                    "Function-preserving basis sensitivity under the frozen RoPE-commuting Q/K "
                    "rotation grid. Values are medians over 90 comparisons per model and rule; "
                    "this coordinate diagnostic is not a retrieval intervention."
                ),
                label="tab:basis-sensitivity-results",
            ),
        )
    )

    table_contents = (
        _latex_table(
            environment="table*",
            columns="llcc",
            headers=("Model", "Optimizer", "Final nDCG@10", "Rates reaching AdamW target"),
            rows=discovery_rows,
            caption=(
                "Discovery retrieval outcomes. Final scores are medians over four learning-rate "
                "points; target passage uses the frozen within-family AdamW reference."
            ),
            label="tab:discovery-results",
        ),
        per_task_tables,
        common_tables,
        _latex_table(
            environment="table*",
            columns="llccc",
            headers=(
                "Model",
                "Optimizer",
                "Unseen margin",
                "Top-1 agreement",
                "Late token coverage",
            ),
            rows=representation_table_rows,
            caption=(
                "Final-stage representation and score geometry, aggregated without selecting a "
                "BEIR winner."
            ),
            label="tab:representation-results",
        ),
        _latex_table(
            environment="table*",
            columns="lccc",
            headers=(
                "Model",
                r"Matched-step margin $\Delta$ (A/M/N)",
                r"Shared-start margin $\Delta$ (M--A/N--M)",
                r"Hybrid AdamW $\Delta$",
            ),
            rows=intervention_rows,
            caption="Immediate, accumulated, and routing-matched causal controls.",
            label="tab:intervention-results",
        ),
        _latex_table(
            environment="table*",
            columns="llccccc",
            headers=(
                "Model",
                "Contrast",
                r"Mean $\Delta$ nDCG@10",
                r"Nominal 95\% CI",
                r"FWER 95\% CI",
                "Seed W/T/L",
                "Task W/T/L",
            ),
            rows=confirmation_table_rows,
            caption=(
                "Validation-frozen, three-seed confirmatory retrieval contrasts. The FWER "
                "interval applies a Bonferroni correction over all six prespecified comparisons "
                "and governs headline sign language."
            ),
            label="tab:confirmation-results",
        ),
    )
    if len(table_contents) != len(PAPER_RESULT_TABLE_PATHS):
        raise ValueError("Generated paper table count differs from the frozen path contract")
    return {
        path.as_posix(): content
        for path, content in zip(PAPER_RESULT_TABLE_PATHS, table_contents, strict=True)
    }


def _replace_headlines(text: str, headlines: dict[str, str]) -> str:
    lines = text.splitlines()
    replaced = set()
    for index, line in enumerate(lines):
        for name, value in headlines.items():
            if line.startswith(f"\\newcommand{{\\{name}}}"):
                if name in replaced:
                    raise ValueError(f"Duplicate paper headline macro: {name}")
                lines[index] = f"\\newcommand{{\\{name}}}{{{value}}}"
                replaced.add(name)
    missing = set(HEADLINE_MACROS) - replaced
    if missing:
        raise ValueError(f"Paper headline macros are absent: {sorted(missing)}")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def render_paper_results(
    *,
    repo_root: Path = Path("."),
    results_path: Path = Path("paper/results.tex"),
    output_manifest: Path = Path("reports/paper-results.manifest.json"),
) -> dict[str, Any]:
    root = repo_root.resolve()
    current_audit = audit_paper(repo_root=root)
    if current_audit["incomplete_evidence"]:
        raise IncompletePaperEvidenceError(
            "Paper headlines require every frozen evidence tier: "
            f"{current_audit['incomplete_evidence']}"
        )
    claim_path, claim_protocol, _claim_sources = load_paper_claim_protocol(repo_root=root)
    paper_results = (root / results_path).resolve()
    current_macros = _macros(paper_results)
    expected_constants, _ = expected_constant_macros(
        root / "configs/experiment.yaml",
        root / "reports/weight-space",
        root / "reports/training-dynamics",
        repo_root=root,
    )
    mismatches = {
        name: (expected, current_macros.get(name))
        for name, expected in expected_constants.items()
        if current_macros.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Paper constants differ before headline rendering: {mismatches}")

    retrieval_dir = root / "reports/retrieval-dynamics"
    retrieval_rows, retrieval_manifest, retrieval_table, _retrieval_figure = _retrieval_rows(
        retrieval_dir
    )
    final_medians, checkpoint_table = _discovery_final_medians(retrieval_dir, retrieval_manifest)
    task_rows, task_table = _discovery_task_rows(retrieval_dir, retrieval_manifest)
    task_stability_rows, task_stability_table = _discovery_task_stability_rows(
        retrieval_dir, retrieval_manifest
    )
    common_rows, _common_manifest, common_table = _common_state_rows(root / "reports/common-state")
    basis_rows, _basis_manifest, basis_table = _basis_rows(root / "reports/basis-sensitivity")
    spectrum_rows, _spectrum_manifest, spectrum_table = _spectrum_rows(
        root / "results/common-state-spectra/summary"
    )
    representation_rows, correlation_rows, _bridge_manifest, bridge_tables = _bridge_rows(
        root / "reports/mechanism-bridge"
    )
    functional_rows, functional_table, _functional_manifest = _functional_rows(
        root / "reports/functional-intervention"
    )
    hybrid_rows, hybrid_table, _hybrid_manifest = _hybrid_rows(root / "reports/hybrid-adamw")
    short_rows, short_table, _short_manifest = _short_branch_rows(root / "reports/short-branch")
    confirmation_rows, confirmation_table, _confirmation_manifest = _confirmation_rows(
        root / "reports/confirmatory"
    )
    headlines = build_headline_macros(
        final_medians=final_medians,
        retrieval_rows=retrieval_rows,
        common_rows=common_rows,
        spectrum_rows=spectrum_rows,
        representation_rows=representation_rows,
        correlation_rows=correlation_rows,
        functional_rows=functional_rows,
        hybrid_rows=hybrid_rows,
        short_rows=short_rows,
        confirmation_rows=confirmation_rows,
    )
    result_tables = build_result_tables(
        final_medians=final_medians,
        retrieval_rows=retrieval_rows,
        task_rows=task_rows,
        task_stability_rows=task_stability_rows,
        common_rows=common_rows,
        basis_rows=basis_rows,
        spectrum_rows=spectrum_rows,
        representation_rows=representation_rows,
        functional_rows=functional_rows,
        hybrid_rows=hybrid_rows,
        short_rows=short_rows,
        confirmation_rows=confirmation_rows,
    )
    for relative, content in result_tables.items():
        _atomic_text(root / relative, content)
    _atomic_text(
        paper_results,
        _replace_headlines(paper_results.read_text(encoding="utf-8"), headlines),
    )
    if any(_macros(paper_results).get(name) != value for name, value in headlines.items()):
        raise ValueError("Rendered paper headline macros do not round-trip")
    if any(
        (root / relative).read_text(encoding="utf-8") != content
        for relative, content in result_tables.items()
    ):
        raise ValueError("Rendered paper result tables do not round-trip")

    evidence_paths = sorted(
        {
            Path(item["path"]).resolve()
            for items in current_audit["evidence"].values()
            for item in items
        }
    )
    source_tables = [
        checkpoint_table,
        retrieval_table,
        task_table,
        task_stability_table,
        common_table,
        basis_table,
        spectrum_table,
        *bridge_tables,
        functional_table,
        hybrid_table,
        short_table,
        confirmation_table,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "claim_protocol": {
            **_source(claim_path),
            "status": claim_protocol["status"],
            "frozen_at": claim_protocol["frozen_at"],
        },
        "evidence_manifests": [_source(path) for path in evidence_paths],
        "source_tables": [_source(path) for path in source_tables],
        "result_tables": [_source(root / path) for path in PAPER_RESULT_TABLE_PATHS],
        "headlines": headlines,
        "results_tex": _source(paper_results),
        "claim_boundary": (
            "These macros report the complete prespecified contrasts and interval classifications; "
            "they do not convert descriptive checkpoint associations into causal evidence."
        ),
    }
    _atomic_json((root / output_manifest).resolve(), manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render audited final evidence into the frozen NAACL headline macros"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--results", type=Path, default=Path("paper/results.tex"))
    parser.add_argument("--output", type=Path, default=Path("reports/paper-results.manifest.json"))
    parser.add_argument(
        "--if-ready",
        action="store_true",
        help="Exit successfully without mutation when frozen evidence is still incomplete",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        manifest = render_paper_results(
            repo_root=args.repo_root,
            results_path=args.results,
            output_manifest=args.output,
        )
    except IncompletePaperEvidenceError as error:
        if not args.if_ready:
            raise
        print(f"Paper evidence is incomplete; retaining audited draft headlines: {error}")
        return
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
