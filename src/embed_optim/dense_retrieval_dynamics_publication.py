"""Publication views for the descriptive full-length Dense retrieval dynamics."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .geometry import _sha256

DYNAMICS_EXTENSION_MARKERS = (
    "<!-- DENSE-RETRIEVAL-DYNAMICS:BEGIN -->",
    "<!-- DENSE-RETRIEVAL-DYNAMICS:END -->",
)
DYNAMICS_EXTENSION_MANIFEST = Path("reports/dense-retrieval-dynamics/summary_manifest.json")
DYNAMICS_EXTENSION_CSV = Path("reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.csv")
DYNAMICS_EXTENSION_SVG = Path("reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.svg")
DYNAMICS_EXTENSION_PDF = Path("reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.pdf")
DYNAMICS_EXTENSION_TEX = Path("paper/generated/retrieval-dynamics-extension.tex")
PENDING_DYNAMICS_EXTENSION = "strict 13-run, five-stage retrieval dynamics"

GROUPS = (
    ("hybrid", "hybrid_adamw", "Hybrid AdamW", 4),
    ("confirmatory", "adamw", "Confirmatory AdamW", 3),
    ("confirmatory", "muon", "Confirmatory Muon", 3),
    ("confirmatory", "normuon", "Confirmatory NorMuon", 3),
)


def _declared_path(root: Path, record: Any) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("Dense retrieval-dynamics output record is malformed")
    declared = Path(record["path"])
    return declared.resolve() if declared.is_absolute() else (root / declared).resolve()


def _validate_output(root: Path, record: Any, expected: Path, *, rows: int | None = None) -> Path:
    path = _declared_path(root, record)
    if (
        path != (root / expected).resolve()
        or not path.is_file()
        or isinstance(record.get("bytes"), bool)
        or record.get("bytes") != path.stat().st_size
        or record.get("sha256") != _sha256(path)
        or (rows is not None and record.get("rows") != rows)
        or (rows is None and "rows" in record)
    ):
        raise ValueError(f"Dense retrieval-dynamics output differs: {expected}")
    return path


def load_publication_rows(root: str | Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load the source-bound 65-row publication table and validate its three outputs."""

    repository = Path(root).resolve()
    manifest_path = repository / DYNAMICS_EXTENSION_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Dense retrieval-dynamics publication manifest is unavailable") from error
    outputs = manifest.get("outputs")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("complete") is not True
        or manifest.get("families") != ["dense"]
        or manifest.get("coverage")
        != {
            "runs": 13,
            "stages_per_run": 5,
            "trajectory_rows": 65,
            "tasks_per_stage": 14,
            "task_units": 910,
            "dynamics_units": 728,
            "formal_stage5_units": 182,
        }
        or not isinstance(outputs, dict)
        or set(outputs) != {"trajectory_csv", "figure_svg", "figure_pdf"}
    ):
        raise ValueError("Dense retrieval-dynamics publication manifest differs")
    boundary = manifest.get("inference_boundary", {})
    if (
        boundary.get("dynamics_stages") != [1, 2, 3, 4]
        or boundary.get("formal_inference_stage") != 5
        or boundary.get("formal_inference_reads_joined_outputs") is not False
        or not isinstance(boundary.get("interpretation"), str)
        or "descriptive trajectory artifacts only" not in boundary["interpretation"]
    ):
        raise ValueError("Dense retrieval-dynamics inference boundary differs")
    csv_path = _validate_output(
        repository, outputs["trajectory_csv"], DYNAMICS_EXTENSION_CSV, rows=65
    )
    _validate_output(repository, outputs["figure_svg"], DYNAMICS_EXTENSION_SVG)
    _validate_output(repository, outputs["figure_pdf"], DYNAMICS_EXTENSION_PDF)
    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("Dense retrieval-dynamics publication CSV is unreadable") from error
    required = {
        "suite",
        "model_family",
        "optimizer",
        "run_id",
        "training_seed",
        "stage",
        "fraction",
        "tasks_completed",
        "mean_ndcg_at_10",
        "source_partition",
        "formal_source_stage5",
        "joined_summary_role",
        "joined_summary_used_for_formal_inference",
    }
    if len(rows) != 65 or not rows or not required.issubset(rows[0]):
        raise ValueError("Dense retrieval-dynamics publication CSV has invalid coverage/schema")
    return rows, manifest


def summarize_publication_rows(rows: Sequence[dict[str, str]]) -> list[list[str]]:
    """Aggregate four descriptive panels without changing the frozen inference inputs."""

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    identities: set[tuple[str, str, int, str, int]] = set()
    group_members: dict[tuple[str, str], set[tuple[int, str]]] = defaultdict(set)
    for row in rows:
        try:
            suite = row["suite"]
            optimizer = row["optimizer"]
            run_id = row["run_id"]
            training_seed = int(row["training_seed"])
            stage = int(row["stage"])
            fraction = float(row["fraction"])
            score = float(row["mean_ndcg_at_10"])
            tasks = int(row["tasks_completed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid Dense retrieval-dynamics publication row: {row}") from error
        identity = (suite, optimizer, training_seed, run_id, stage)
        if (
            identity in identities
            or row["model_family"] != "dense"
            or stage not in range(1, 6)
            or abs(fraction - stage / 5) > 1e-12
            or not math.isfinite(score)
            or not 0 <= score <= 1
            or tasks != 14
            or row["source_partition"] != ("formal-stage5" if stage == 5 else "dynamics-stage1-4")
            or row["formal_source_stage5"].lower() != ("true" if stage == 5 else "false")
            or row["joined_summary_role"] != "descriptive-only"
            or row["joined_summary_used_for_formal_inference"].lower() != "false"
        ):
            raise ValueError(
                f"Dense retrieval-dynamics row violates its descriptive boundary: {row}"
            )
        identities.add(identity)
        group_members[(suite, optimizer)].add((training_seed, run_id))
        grouped[(suite, optimizer, stage)].append(score)

    output = []
    for suite, optimizer, label, runs in GROUPS:
        members = group_members.get((suite, optimizer), set())
        if len(members) != runs or (
            suite == "confirmatory" and len({seed for seed, _run_id in members}) != 3
        ):
            raise ValueError(
                f"Dense retrieval-dynamics group lacks exact distinct runs/seeds: "
                f"{suite}/{optimizer}"
            )
        stage_scores = []
        for stage in range(1, 6):
            values = grouped.get((suite, optimizer, stage), [])
            if len(values) != runs:
                raise ValueError(
                    f"Dense retrieval-dynamics group lacks {runs} runs at stage {stage}: "
                    f"{suite}/{optimizer}"
                )
            stage_scores.append(f"{statistics.mean(values):.4f}")
        output.append([label, str(runs), *stage_scores])
    if len(rows) != 65 or len(identities) != 65:
        raise ValueError("Dense retrieval-dynamics publication requires exactly 65 run-stage rows")
    return output


def render_publication_markdown(summary_rows: Sequence[Sequence[str]]) -> str:
    header = ("Series", "Runs", "20%", "40%", "60%", "80%", "100%")
    lines = [
        "The completed full-length extension contains **13 runs × 5 stages = 65 trajectory rows "
        "and 910 decontaminated BEIR task units**. Stages 1–4 come from isolated dynamics roots; "
        "stage 5 is joined from the pre-existing formal roots.",
        "",
        "[Download the source-bound 65-row CSV](../reports/dense-retrieval-dynamics/"
        "five_stage_retrieval_dynamics.csv)",
        "",
        "![Five-stage hybrid and confirmatory retrieval dynamics](../reports/"
        "dense-retrieval-dynamics/five_stage_retrieval_dynamics.svg)",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" if index < 2 else "---:" for index in range(len(header))) + " |",
        *("| " + " | ".join(row) + " |" for row in summary_rows),
        "",
        "**Inference boundary:** these joined curves are descriptive training dynamics only. "
        "The hybrid-routing and confirmatory comparisons continue to read only their disjoint, "
        "pre-existing stage-5 result roots; neither the CSV nor either figure is an inference input.",
    ]
    return "\n".join(lines)


def render_publication_latex(summary_rows: Sequence[Sequence[str]]) -> str:
    body = [" & ".join(row) + r" \\" for row in summary_rows]
    return "\n".join(
        (
            "% Generated atomically by embed-optim-render-paper-results.",
            r"\begin{figure*}[t]",
            r"\centering",
            r"\includegraphics[width=\textwidth]{../reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.pdf}",
            r"\caption{Descriptive five-stage decontaminated-BEIR trajectories for the four routing-matched hybrid runs and nine validation-frozen confirmatory runs. Stages 1--4 come from isolated dynamics roots; stage 5 is joined from the pre-existing formal roots. This figure is not an inference input.}",
            r"\label{fig:extended-retrieval-dynamics}",
            r"\end{figure*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Series & Runs & 20\% & 40\% & 60\% & 80\% & 100\% \\ ",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Mean nDCG@10 across the four hybrid learning-rate runs or three new seeds per optimizer. These are descriptive aggregates of the source-bound 65-row CSV, not additional formal contrasts.}",
            r"\label{tab:extended-retrieval-dynamics}",
            r"\end{table*}",
            "",
        )
    )
