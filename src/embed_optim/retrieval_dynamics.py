from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .aggregate import (
    DECONTAMINATED_TASK_NAMES,
    _checkpoint_summaries,
    _optimizer_summaries,
    _task_comparison,
    collect_evaluations,
)
from .config import RunConfig, load_matrix, resolve_matrix_path
from .confirmatory_summary import _atomic_csv
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .scope import ALL_FAMILIES, resolve_scope, select_family_configs
from .training_dynamics_plot import _atomic_figure

FAMILIES = ALL_FAMILIES
OPTIMIZERS = ("adamw", "muon", "normuon")
FAMILY_LABELS = {"dense": "DenseOn", "late": "LateOn"}
OPTIMIZER_LABELS = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}
TASK_STABILITY_MARKERS = (
    "<!-- TASK-DELTA-STABILITY:BEGIN -->",
    "<!-- TASK-DELTA-STABILITY:END -->",
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_retrieval_dynamics_protocol(
    path: str | Path = "configs/retrieval_dynamics_protocol.json",
) -> tuple[Path, dict[str, Any]]:
    protocol_path = Path(path).resolve()
    protocol = _json(protocol_path)
    root = protocol_path.parent.parent
    matrix = protocol.get("matrix", {})
    training = protocol.get("training_summary", {})
    matrix_path = (root / str(matrix.get("path", ""))).resolve()
    training_path = (root / str(training.get("path", ""))).resolve()
    freeze_context = protocol.get("freeze_context", {})
    reference = protocol.get("reference_target", {})
    first_passage = protocol.get("first_passage", {})
    wall_time = protocol.get("wall_time", {})
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("status") != "prospective_completion_lock"
        or freeze_context.get("strict_beir_valid_units") != 160
        or freeze_context.get("strict_beir_expected_units") != 1_680
        or freeze_context.get("complete_retrieval_matrix_visible") is not False
        or freeze_context.get("retrieval_dynamics_output_visible") is not False
        or reference.get("scope") != "within_model_family"
        or reference.get("definition")
        != "median_final_ndcg_at_10_of_four_adamw_learning_rate_points"
        or reference.get("uses_muon_or_normuon_outcomes") is not False
        or reference.get("uses_confirmation_outcomes") is not False
        or first_passage.get("observed_stages") != [0.2, 0.4, 0.6, 0.8, 1.0]
        or first_passage.get("interpolation") != "none"
        or first_passage.get("non_reaching_runs") != "right_censored_after_stage_5"
        or first_passage.get("report_all_learning_rate_points") is not True
        or wall_time.get("definition")
        != "audited_terminal_useful_wall_time_hours_times_checkpoint_step_over_3907"
        or wall_time.get("checkpoint_timestamp_claim") is not False
        or wall_time.get("gpu_hours_multiplier") != 4
        or "not a preregistration" not in str(protocol.get("claim_boundary", ""))
        or not matrix_path.is_file()
        or _sha256(matrix_path) != matrix.get("sha256")
        or not training_path.is_file()
        or _sha256(training_path) != training.get("sha256")
    ):
        raise ValueError("Retrieval-dynamics protocol differs from its frozen contract")
    return protocol_path, protocol


def _portable_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        return str(resolved)


def _finite(row: dict[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} value in retrieval dynamics: {row}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} value in retrieval dynamics: {row}")
    return value


def _strict_coverage(path: Path) -> dict[str, Any]:
    payload = _json(path)
    if (
        payload.get("complete") is not True
        or payload.get("evaluation_complete") is not True
        or payload.get("training_complete") is not True
        or payload.get("observed_results") != 1_680
        or payload.get("expected_results") != 1_680
        or payload.get("observed_checkpoint_summaries") != 120
        or payload.get("expected_checkpoint_summaries") != 120
        or payload.get("missing") != []
        or payload.get("unexpected") != []
    ):
        raise ValueError("Retrieval dynamics requires the strict complete 1,680-unit coverage")
    return payload


def _training_runs(training_dir: Path, repository_root: Path) -> tuple[list[dict[str, str]], Path]:
    manifest_path = training_dir / "summary_manifest.json"
    manifest = _json(manifest_path)
    coverage = manifest.get("coverage", {})
    item = manifest.get("outputs", {}).get("runs", {})
    declared = Path(item.get("path", ""))
    table = declared.resolve() if declared.is_absolute() else (repository_root / declared).resolve()
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or coverage.get("runs") != 24
        or coverage.get("checkpoints") != 120
        or item.get("rows") != 24
        or not table.is_file()
        or table.stat().st_size != item.get("bytes")
        or _sha256(table) != item.get("sha256")
    ):
        raise ValueError("Retrieval dynamics training source failed its strict manifest contract")
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise ValueError("Retrieval dynamics requires exactly 24 training-system rows")
    return rows, table


def _expected_identities(configs: list[RunConfig]) -> set[tuple[str, str]]:
    return {(config.model_family, config.run_id) for config in configs}


def summarize_retrieval_dynamics(
    checkpoint_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    *,
    configs: list[RunConfig] | None = None,
    families: tuple[str, ...] = FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    indexed_runs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in run_rows:
        identity = (str(row.get("model_family", "")), str(row.get("run_id", "")))
        optimizer = str(row.get("optimizer", ""))
        if (
            identity[0] not in families
            or optimizer not in OPTIMIZERS
            or not identity[1]
            or identity in indexed_runs
            or _finite(row, "wall_time_hours") <= 0
            or int(row.get("world_size", 0)) != 4
        ):
            raise ValueError(f"Invalid or duplicate training-system row: {identity}")
        indexed_runs[identity] = row
    expected_runs = 12 * len(families)
    if len(indexed_runs) != expected_runs:
        raise ValueError(f"Retrieval dynamics requires {expected_runs} unique training runs")
    if configs is not None and set(indexed_runs) != _expected_identities(configs):
        raise ValueError("Retrieval dynamics training identities differ from the frozen matrix")

    indexed_checkpoints: dict[tuple[str, str, int], dict[str, Any]] = {}
    by_run: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoint_rows:
        family = str(row.get("model_family", ""))
        run_id = str(row.get("run_id", ""))
        optimizer = str(row.get("optimizer", ""))
        try:
            stage = int(row.get("stage", 0))
            checkpoint_step = int(row.get("checkpoint_step", 0))
            tasks = int(row.get("tasks_completed", 0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid checkpoint identity: {row}") from error
        identity = (family, run_id, stage)
        run_identity = (family, run_id)
        if (
            family not in families
            or optimizer not in OPTIMIZERS
            or run_identity not in indexed_runs
            or optimizer != indexed_runs[run_identity].get("optimizer")
            or not 1 <= stage <= 5
            or checkpoint_step <= 0
            or tasks != len(DECONTAMINATED_TASK_NAMES)
            or identity in indexed_checkpoints
            or abs(_finite(row, "fraction") - stage / 5) > 1e-12
            or not 0 <= _finite(row, "mean_ndcg_at_10") <= 1
        ):
            raise ValueError(f"Invalid or duplicate checkpoint row: {identity}")
        indexed_checkpoints[identity] = row
        by_run[run_identity].append(row)
    if (
        len(checkpoint_rows) != expected_runs * 5
        or set(by_run) != set(indexed_runs)
        or any({int(row["stage"]) for row in rows} != {1, 2, 3, 4, 5} for rows in by_run.values())
    ):
        raise ValueError(
            f"Retrieval dynamics requires {expected_runs} complete five-checkpoint trajectories"
        )

    targets = {}
    for family in families:
        values = [
            _finite(row, "mean_ndcg_at_10")
            for row in checkpoint_rows
            if row["model_family"] == family
            and row["optimizer"] == "adamw"
            and int(row["stage"]) == 5
        ]
        if len(values) != 4:
            raise ValueError(f"AdamW reference target requires four final points for {family}")
        targets[family] = statistics.median(values)

    dynamics = []
    first_passage = []
    for identity in sorted(by_run):
        family, run_id = identity
        system = indexed_runs[identity]
        ordered = sorted(by_run[identity], key=lambda row: int(row["stage"]))
        final_step = int(ordered[-1]["checkpoint_step"])
        if final_step != 3_907:
            raise ValueError(f"Unexpected terminal step for {identity}: {final_step}")
        target = targets[family]
        run_dynamics = []
        for row in ordered:
            step = int(row["checkpoint_step"])
            time_hours = _finite(system, "wall_time_hours") * step / final_step
            score = _finite(row, "mean_ndcg_at_10")
            record = {
                **row,
                "adamw_median_final_target": target,
                "target_reached": score >= target,
                "estimated_useful_wall_time_hours": time_hours,
                "estimated_gpu_hours": time_hours * int(system["world_size"]),
                "wall_time_estimation": "step_proportional_total_useful_wall_time",
            }
            dynamics.append(record)
            run_dynamics.append(record)
        reached = next((row for row in run_dynamics if row["target_reached"]), None)
        final_score = _finite(run_dynamics[-1], "mean_ndcg_at_10")
        first_passage.append(
            {
                "model_family": family,
                "optimizer": system["optimizer"],
                "learning_rate": _finite(system, "learning_rate"),
                "run_id": run_id,
                "adamw_median_final_target": target,
                "target_reached": reached is not None,
                "first_observed_stage": int(reached["stage"]) if reached else None,
                "first_observed_fraction": _finite(reached, "fraction") if reached else None,
                "first_observed_checkpoint_step": (
                    int(reached["checkpoint_step"]) if reached else None
                ),
                "first_observed_score": (_finite(reached, "mean_ndcg_at_10") if reached else None),
                "first_observed_estimated_useful_wall_time_hours": (
                    _finite(reached, "estimated_useful_wall_time_hours") if reached else None
                ),
                "final_score": final_score,
                "final_score_minus_target": final_score - target,
                "censoring": "observed" if reached else "right-censored-after-stage-5",
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in first_passage:
        grouped[(str(row["model_family"]), str(row["optimizer"]))].append(row)
    expected_groups = {(family, optimizer) for family in families for optimizer in OPTIMIZERS}
    if set(grouped) != expected_groups or any(len(rows) != 4 for rows in grouped.values()):
        raise ValueError("First-passage summary does not cover four sweep points per group")
    group_summary = []
    for (family, optimizer), rows in sorted(grouped.items()):
        observed = [
            float(row["first_observed_estimated_useful_wall_time_hours"])
            for row in rows
            if row["target_reached"]
        ]
        fractions = [float(row["first_observed_fraction"]) for row in rows if row["target_reached"]]
        group_summary.append(
            {
                "model_family": family,
                "optimizer": optimizer,
                "learning_rate_points": len(rows),
                "adamw_median_final_target": targets[family],
                "points_reaching_target": len(observed),
                "points_right_censored": len(rows) - len(observed),
                "reach_rate_across_lr_points": len(observed) / len(rows),
                "fastest_observed_useful_wall_time_hours": min(observed) if observed else None,
                "median_observed_useful_wall_time_hours": (
                    statistics.median(observed) if observed else None
                ),
                "slowest_observed_useful_wall_time_hours": max(observed) if observed else None,
                "median_first_observed_fraction": (
                    statistics.median(fractions) if fractions else None
                ),
                "target_definition": "within-family-median-of-four-adamw-final-points",
                "interpolation": "none-five-observed-checkpoints-only",
            }
        )
    return dynamics, first_passage, group_summary


def _average_ranks(values: list[float]) -> list[float]:
    """Return one-based average ranks with deterministic tie handling."""

    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = ((start + 1) + end) / 2
        for index, _value in ordered[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _finite_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(left, right)


def task_delta_dynamics(
    evaluation_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    families: tuple[str, ...] = FAMILIES,
) -> list[dict[str, Any]]:
    """Collect post-hoc task deltas for runs selected by final score on the same BEIR suite."""

    best_runs = {
        (str(row["model_family"]), str(row["optimizer"])): str(row["best_run_id"])
        for row in optimizer_rows
    }
    expected_groups = {(family, optimizer) for family in families for optimizer in OPTIMIZERS}
    if set(best_runs) != expected_groups:
        raise ValueError("Task-delta dynamics requires every optimizer-family best run in scope")
    lookup = {
        (str(row["model_family"]), str(row["run_id"]), int(row["stage"]), str(row["task"])): row
        for row in evaluation_rows
    }
    output = []
    for family in families:
        baseline_run = best_runs[(family, "adamw")]
        for optimizer in ("muon", "normuon"):
            treatment_run = best_runs[(family, optimizer)]
            for stage in range(1, 6):
                for task in DECONTAMINATED_TASK_NAMES:
                    treatment = lookup[(family, treatment_run, stage, task)]
                    baseline = lookup[(family, baseline_run, stage, task)]
                    treatment_fraction = _finite(treatment, "fraction")
                    baseline_fraction = _finite(baseline, "fraction")
                    if (
                        abs(treatment_fraction - stage / 5) > 1e-12
                        or abs(baseline_fraction - treatment_fraction) > 1e-12
                    ):
                        raise ValueError("Task-delta checkpoint fractions do not align")
                    treatment_score = _finite(treatment, "ndcg_at_10")
                    baseline_score = _finite(baseline, "ndcg_at_10")
                    output.append(
                        {
                            "model_family": family,
                            "optimizer": optimizer,
                            "baseline": "adamw",
                            "treatment_run_id": treatment_run,
                            "baseline_run_id": baseline_run,
                            "stage": stage,
                            "fraction": treatment_fraction,
                            "task": task,
                            "treatment_ndcg_at_10": treatment_score,
                            "baseline_ndcg_at_10": baseline_score,
                            "delta": treatment_score - baseline_score,
                        }
                    )
    expected_rows = len(families) * 2 * 5 * len(DECONTAMINATED_TASK_NAMES)
    if len(output) != expected_rows:
        raise ValueError("Task-delta dynamics has incomplete best-run checkpoint coverage")
    return output


def summarize_task_delta_stability(
    delta_rows: list[dict[str, Any]],
    families: tuple[str, ...] = FAMILIES,
) -> list[dict[str, Any]]:
    """Summarize adjacent-checkpoint persistence of post-hoc per-task effects."""

    grouped: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in delta_rows:
        identity = (str(row["model_family"]), str(row["optimizer"]), int(row["stage"]))
        task = str(row["task"])
        if task in grouped[identity]:
            raise ValueError(f"Duplicate task-delta identity: {identity}/{task}")
        grouped[identity][task] = _finite(row, "delta")

    output = []
    for family in families:
        for optimizer in ("muon", "normuon"):
            for first_stage in range(1, 5):
                second_stage = first_stage + 1
                first = grouped.get((family, optimizer, first_stage), {})
                second = grouped.get((family, optimizer, second_stage), {})
                if set(first) != set(DECONTAMINATED_TASK_NAMES) or set(second) != set(first):
                    raise ValueError(
                        f"Task-delta stages do not align for {family}/{optimizer}: "
                        f"{first_stage} versus {second_stage}"
                    )
                tasks = sorted(first)
                left = [first[task] for task in tasks]
                right = [second[task] for task in tasks]
                tolerance = 1e-12

                def direction(value: float) -> int:
                    return 1 if value > tolerance else -1 if value < -tolerance else 0

                stable = sum(direction(a) == direction(b) for a, b in zip(left, right))
                output.append(
                    {
                        "model_family": family,
                        "optimizer": optimizer,
                        "baseline": "adamw",
                        "first_stage": first_stage,
                        "second_stage": second_stage,
                        "first_fraction": first_stage / 5,
                        "second_fraction": second_stage / 5,
                        "tasks": len(tasks),
                        "same_direction_tasks": stable,
                        "direction_stability_fraction": stable / len(tasks),
                        "pearson_correlation": _finite_correlation(left, right),
                        "spearman_correlation": _finite_correlation(
                            _average_ranks(left), _average_ranks(right)
                        ),
                        "mean_absolute_delta_change": statistics.mean(
                            abs(a - b) for a, b in zip(left, right)
                        ),
                    }
                )
    if len(output) != len(families) * 2 * 4:
        raise ValueError("Task-delta stability summary has incomplete adjacent-stage coverage")
    return output


def _task_stability_markdown(
    rows: list[dict[str, Any]], families: tuple[str, ...] = FAMILIES
) -> str:
    lines = [
        "### Exploratory task-effect stability across checkpoints",
        "",
        "| Family | Comparison | Stages | Same direction | Pearson r | Spearman rho |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        pearson = row["pearson_correlation"]
        spearman = row["spearman_correlation"]
        pearson_text = "n/a" if pearson is None else f"{float(pearson):.3f}"
        spearman_text = "n/a" if spearman is None else f"{float(spearman):.3f}"
        lines.append(
            "| "
            f"{FAMILY_LABELS[str(row['model_family'])]} | "
            f"{OPTIMIZER_LABELS[str(row['optimizer'])]} − AdamW | "
            f"{int(round(float(row['first_fraction']) * 100))}%→"
            f"{int(round(float(row['second_fraction']) * 100))}% | "
            f"{row['same_direction_tasks']}/{row['tasks']} | "
            f"{pearson_text} | {spearman_text} |"
        )
    if families == FAMILIES:
        interpretation = (
            "This post-hoc diagnostic applies each optimizer's learning-rate run selected by final "
            "nDCG@10 on this same BEIR suite "
            "at every checkpoint and correlates its 14 paired task deltas against AdamW across "
            "adjacent stages. It was added after heterogeneous LateOn task directions became "
            "visible. It does not alter run selection, the primary aggregate, or the frozen "
            "confirmatory family, and it carries no causal interpretation."
        )
    else:
        interpretation = (
            "This DenseOn-only post-hoc diagnostic applies each optimizer's learning-rate run "
            "selected by final nDCG@10 on this same BEIR suite at every checkpoint and correlates "
            "its 14 paired task deltas against "
            "AdamW across adjacent stages. It is a filtered exploratory view of the already audited "
            "discovery matrix. It does not alter run selection, the primary aggregate, or the frozen "
            "Dense confirmatory family, and it carries no causal interpretation."
        )
    lines.extend(["", interpretation])
    return "\n".join(lines)


def render_task_stability_blog(
    path: Path,
    rows: list[dict[str, Any]],
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, str | int]:
    text = path.read_text(encoding="utf-8")
    begin, end = TASK_STABILITY_MARKERS
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Blog requires exactly one task-delta stability marker pair")
    before, remainder = text.split(begin)
    _old, after = remainder.split(end)
    content = _task_stability_markdown(rows, families)
    rendered = f"{before}{begin}\n\n{content}\n\n{end}{after}"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)
    written = path.read_text(encoding="utf-8")
    if written.count(begin) != 1 or written.count(end) != 1:
        raise ValueError("Written blog has an invalid task-delta stability marker pair")
    start = written.index(begin)
    stop = written.index(end, start) + len(end)
    observed = written[start:stop].encode("utf-8")
    expected = f"{begin}\n\n{content}\n\n{end}".encode("utf-8")
    if observed != expected:
        raise ValueError("Written task-delta stability marker block differs after render")
    return {
        "begin_marker": begin,
        "end_marker": end,
        "encoding": "utf-8",
        "bytes": len(observed),
        "sha256": hashlib.sha256(observed).hexdigest(),
    }


def _quality_figure(
    rows: list[dict[str, Any]],
    path: Path,
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "embedding-optimizer-retrieval-wall-time-v1"
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(families),
        len(OPTIMIZERS),
        figsize=(12.5, 7.0 if len(families) == 2 else 3.8),
        sharey="row",
        squeeze=False,
    )
    for row_index, family in enumerate(families):
        for column_index, optimizer in enumerate(OPTIMIZERS):
            axis = axes[row_index][column_index]
            subset = [
                row
                for row in rows
                if row["model_family"] == family and row["optimizer"] == optimizer
            ]
            by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in subset:
                by_run[str(row["run_id"])].append(row)
            ordered_runs = sorted(
                by_run.items(), key=lambda item: _finite(item[1][0], "learning_rate")
            )
            colors = plt.get_cmap("viridis")([0.12, 0.38, 0.64, 0.90])
            for color, (_run_id, values) in zip(colors, ordered_runs, strict=True):
                ordered = sorted(values, key=lambda row: int(row["stage"]))
                axis.plot(
                    [_finite(row, "estimated_useful_wall_time_hours") for row in ordered],
                    [_finite(row, "mean_ndcg_at_10") for row in ordered],
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    color=color,
                    label=f"{_finite(ordered[0], 'learning_rate'):.0e}",
                )
            target = _finite(subset[0], "adamw_median_final_target")
            axis.axhline(target, color="#B22222", linestyle="--", linewidth=1.2, alpha=0.75)
            axis.set_title(f"{FAMILY_LABELS[family]} — {OPTIMIZER_LABELS[optimizer]}")
            axis.grid(alpha=0.22)
            if row_index == len(families) - 1:
                axis.set_xlabel("Estimated useful wall time (hours)")
            if column_index == 0:
                axis.set_ylabel("Mean decontaminated BEIR nDCG@10")
            axis.legend(title="Configured LR", fontsize=7.5, title_fontsize=8, frameon=False)
    figure.suptitle(
        "Retrieval quality versus step-proportional useful wall time",
        fontsize=13,
    )
    figure.tight_layout()
    record = _atomic_figure(figure, path)
    plt.close(figure)
    return record


def _read_frozen_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"Missing frozen retrieval-dynamics table: {path}") from error
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value == "":
                row[key] = None
            elif value in {"True", "False"}:
                row[key] = value == "True"
            else:
                try:
                    row[key] = int(value)
                except ValueError:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
        rows.append(row)
    return rows


def _manifest_path(record: dict[str, Any], repository_root: Path) -> Path:
    declared = Path(str(record.get("path", "")))
    return declared.resolve() if declared.is_absolute() else (repository_root / declared).resolve()


def _validate_manifest_file(
    record: dict[str, Any],
    repository_root: Path,
    *,
    expected_rows: int | None = None,
) -> Path:
    path = _manifest_path(record, repository_root)
    if (
        not path.is_file()
        or path.stat().st_size != record.get("bytes")
        or _sha256(path) != record.get("sha256")
        or (expected_rows is not None and record.get("rows") != expected_rows)
    ):
        raise ValueError(f"Frozen retrieval-dynamics source differs: {path}")
    return path


def _strict_full_retrieval_report(
    repository_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Validate the complete historical report without reinterpreting its runtime lock."""

    manifest_path = repository_root / "reports/retrieval-dynamics/summary_manifest.json"
    manifest = _json(manifest_path)
    expected_coverage = {
        "runs": 24,
        "checkpoints": 120,
        "tasks": len(DECONTAMINATED_TASK_NAMES),
        "evaluation_units": 1_680,
        "optimizer_family_groups": 6,
        "best_config_task_delta_rows": 280,
        "adjacent_stage_task_stability_rows": 16,
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("coverage") != expected_coverage
    ):
        raise ValueError("Dense retrieval dynamics requires the complete historical report")

    expected_outputs = {
        "checkpoint_dynamics": 120,
        "run_first_passage": 24,
        "optimizer_first_passage": 6,
        "best_config_task_comparison": 28,
        "best_config_task_delta_dynamics": 280,
        "task_delta_stability": 16,
    }
    output_records = manifest.get("outputs", {})
    if set(output_records) != {*expected_outputs, "quality_vs_useful_wall_time"}:
        raise ValueError("Frozen retrieval-dynamics output ledger is incomplete")
    tables = {}
    for label, row_count in expected_outputs.items():
        table_path = _validate_manifest_file(
            output_records[label], repository_root, expected_rows=row_count
        )
        rows = _read_frozen_csv(table_path)
        if len(rows) != row_count or {row.get("model_family") for row in rows} != set(FAMILIES):
            raise ValueError(f"Frozen retrieval-dynamics table has invalid coverage: {label}")
        tables[label] = rows
    _validate_manifest_file(output_records["quality_vs_useful_wall_time"], repository_root)

    sources = manifest.get("sources", {})
    expected_source_keys = {
        "frozen_protocol",
        "matrix",
        "strict_coverage",
        "training_summary",
        "training_run_table",
        "evaluation_results",
    }
    if not isinstance(sources, dict) or set(sources) != expected_source_keys:
        raise ValueError("Frozen retrieval-dynamics source ledger is incomplete")
    for label in expected_source_keys - {"evaluation_results"}:
        _validate_manifest_file(sources[label], repository_root)
    result_records = sources["evaluation_results"]
    if not isinstance(result_records, list) or len(result_records) != 1_680:
        raise ValueError("Frozen retrieval-dynamics evaluation ledger is incomplete")
    result_paths = [record.get("path") for record in result_records if isinstance(record, dict)]
    if (
        len(result_paths) != 1_680
        or len(set(result_paths)) != 1_680
        or any(
            set(record) != {"path", "bytes", "sha256"}
            or not isinstance(record["path"], str)
            or not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] <= 0
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
            for record in result_records
        )
    ):
        raise ValueError("Frozen retrieval-dynamics evaluation ledger has invalid identities")
    return manifest_path, manifest, tables


def _build_scoped_retrieval_from_report(
    *,
    repository_root: Path,
    coverage_file: Path,
    output_dir: str | Path,
    blog_path: str | Path | None,
    families: tuple[str, ...],
    scope: dict[str, Any],
) -> dict[str, Any]:
    """Filter the immutable complete report into a separately bound Dense artifact."""

    source_manifest_path, source_manifest, tables = _strict_full_retrieval_report(repository_root)
    requested = set(families)
    selected = {
        label: [row for row in rows if row["model_family"] in requested]
        for label, rows in tables.items()
    }
    expected_rows = {
        "checkpoint_dynamics": 60 * len(families),
        "run_first_passage": 12 * len(families),
        "optimizer_first_passage": 3 * len(families),
        "best_config_task_comparison": 14 * len(families),
        "best_config_task_delta_dynamics": 140 * len(families),
        "task_delta_stability": 8 * len(families),
    }
    if any(len(selected[label]) != count for label, count in expected_rows.items()):
        raise ValueError("Scoped retrieval-dynamics filtering produced incomplete tables")

    output = Path(output_dir).resolve()
    canonical_output = (repository_root / "reports/retrieval-dynamics").resolve()
    if output == canonical_output:
        output = canonical_output.with_name("retrieval-dynamics-dense")
    outputs = {
        label: _atomic_csv(output / f"{label}.csv", selected[label]) for label in expected_rows
    }
    outputs["quality_vs_useful_wall_time"] = _quality_figure(
        selected["checkpoint_dynamics"],
        output / "quality_vs_useful_wall_time.svg",
        families,
    )
    for record in outputs.values():
        record["path"] = _portable_path(Path(record["path"]), repository_root)

    interpretation = (
        "The four learning rates are sweep points rather than random seeds. This DenseOn-only view "
        "was filtered from the strictly complete 24-run, 1,680-unit discovery report after the "
        "user-directed scope amendment; it does not require or impute a new historical evaluation. "
        "First passage and adjacent-checkpoint task stability remain exploratory and do not replace "
        "the frozen three-seed Dense confirmatory comparison or carry a causal interpretation."
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "families": list(families),
        "scope_amendment": scope,
        "coverage": {
            "runs": expected_rows["run_first_passage"],
            "checkpoints": expected_rows["checkpoint_dynamics"],
            "tasks": len(DECONTAMINATED_TASK_NAMES),
            "evaluation_units": 840 * len(families),
            "optimizer_family_groups": expected_rows["optimizer_first_passage"],
            "best_config_task_delta_rows": expected_rows["best_config_task_delta_dynamics"],
            "adjacent_stage_task_stability_rows": expected_rows["task_delta_stability"],
        },
        "target_definition": source_manifest["target_definition"],
        "wall_time_definition": source_manifest["wall_time_definition"],
        "sources": {
            "full_retrieval_dynamics": {
                "path": _portable_path(source_manifest_path, repository_root),
                "bytes": source_manifest_path.stat().st_size,
                "sha256": _sha256(source_manifest_path),
            },
            "strict_coverage": {
                "path": _portable_path(coverage_file, repository_root),
                "bytes": coverage_file.stat().st_size,
                "sha256": _sha256(coverage_file),
                "observed_results": 1_680,
            },
        },
        "source_full_discovery": {
            "complete": True,
            **source_manifest["coverage"],
        },
        "outputs": outputs,
        "interpretation": interpretation,
        "blog_marker_blocks": None,
    }
    if blog_path is not None:
        resolved_blog = Path(blog_path).resolve()
        block = render_task_stability_blog(
            resolved_blog, selected["task_delta_stability"], families
        )
        manifest["blog_marker_blocks"] = {
            "schema_version": 1,
            "path": _portable_path(resolved_blog, repository_root),
            "blocks": {"task_delta_stability": block},
        }
    _atomic_json(output / "summary_manifest.json", manifest)
    return manifest


def build_retrieval_dynamics(
    matrix: str | Path = "configs/experiment.yaml",
    results_root: str | Path = "results/decontaminated-beir",
    coverage_path: str | Path = "reports/coverage.json",
    training_dir: str | Path = "reports/training-dynamics",
    output_dir: str | Path = "reports/retrieval-dynamics",
    protocol_path: str | Path = "configs/retrieval_dynamics_protocol.json",
    blog_path: str | Path | None = None,
    *,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    families, scope = resolve_scope(families, scope_amendment)
    matrix_path = resolve_matrix_path(matrix).resolve()
    repository_root = matrix_path.parent.parent
    frozen_protocol_path, protocol = load_retrieval_dynamics_protocol(protocol_path)
    frozen_matrix = protocol["matrix"]
    if (
        matrix_path != (repository_root / frozen_matrix["path"]).resolve()
        or _sha256(matrix_path) != frozen_matrix["sha256"]
    ):
        raise ValueError("Requested matrix differs from the frozen retrieval-dynamics protocol")
    configs = load_matrix(matrix_path)
    if len(configs) != 24:
        raise ValueError("Retrieval dynamics requires the frozen 24-run discovery matrix")
    select_family_configs(configs, families)
    coverage_file = Path(coverage_path).resolve()
    coverage = _strict_coverage(coverage_file)
    if families != FAMILIES:
        return _build_scoped_retrieval_from_report(
            repository_root=repository_root,
            coverage_file=coverage_file,
            output_dir=output_dir,
            blog_path=blog_path,
            families=families,
            scope=scope,
        )
    evaluation_rows = collect_evaluations(Path(results_root), configs)
    if len(evaluation_rows) != 1_680:
        raise ValueError("Retrieval dynamics requires exactly 1,680 provenance-valid results")
    checkpoint_rows = _checkpoint_summaries(evaluation_rows)
    optimizer_rows, _best_dynamics = _optimizer_summaries(checkpoint_rows)
    task_rows = _task_comparison(evaluation_rows, optimizer_rows)
    if len(task_rows) != len(FAMILIES) * len(DECONTAMINATED_TASK_NAMES):
        raise ValueError("Retrieval dynamics requires 28 best-config per-task rows")
    task_delta_rows = task_delta_dynamics(evaluation_rows, optimizer_rows)
    task_stability_rows = summarize_task_delta_stability(task_delta_rows)
    training_root = Path(training_dir).resolve()
    frozen_training = protocol["training_summary"]
    frozen_training_path = (repository_root / frozen_training["path"]).resolve()
    if (
        training_root / "summary_manifest.json" != frozen_training_path
        or _sha256(frozen_training_path) != frozen_training["sha256"]
    ):
        raise ValueError(
            "Requested training summary differs from the frozen retrieval-dynamics protocol"
        )
    run_rows, training_table = _training_runs(training_root, repository_root)
    dynamics, first_passage, group_summary = summarize_retrieval_dynamics(
        checkpoint_rows, run_rows, configs=configs
    )
    output = Path(output_dir).resolve()
    outputs = {
        "checkpoint_dynamics": _atomic_csv(output / "checkpoint_dynamics.csv", dynamics),
        "run_first_passage": _atomic_csv(output / "run_first_passage.csv", first_passage),
        "optimizer_first_passage": _atomic_csv(
            output / "optimizer_first_passage.csv", group_summary
        ),
        "best_config_task_comparison": _atomic_csv(
            output / "best_config_task_comparison.csv", task_rows
        ),
        "best_config_task_delta_dynamics": _atomic_csv(
            output / "best_config_task_delta_dynamics.csv", task_delta_rows
        ),
        "task_delta_stability": _atomic_csv(
            output / "task_delta_stability.csv", task_stability_rows
        ),
        "quality_vs_useful_wall_time": _quality_figure(
            dynamics, output / "quality_vs_useful_wall_time.svg"
        ),
    }
    for record in outputs.values():
        record["path"] = _portable_path(Path(record["path"]), repository_root)
    result_paths = sorted({Path(row["result_path"]).resolve() for row in evaluation_rows})
    if len(result_paths) != 1_680 or any(not path.is_file() for path in result_paths):
        raise ValueError("Retrieval dynamics result-source identities are incomplete")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "coverage": {
            "runs": len(first_passage),
            "checkpoints": len(dynamics),
            "tasks": len(DECONTAMINATED_TASK_NAMES),
            "evaluation_units": len(evaluation_rows),
            "optimizer_family_groups": len(group_summary),
            "best_config_task_delta_rows": len(task_delta_rows),
            "adjacent_stage_task_stability_rows": len(task_stability_rows),
        },
        "target_definition": (
            "Within each model family, use the median final nDCG@10 of the four AdamW learning-"
            "rate points; report the first of five observed checkpoints that reaches it without "
            "interpolation, retaining non-reaching runs as right-censored."
        ),
        "wall_time_definition": (
            "Checkpoint useful wall time is estimated as terminal audited useful wall time times "
            "checkpoint_step/3907; it excludes pauses but assumes constant average step cost."
        ),
        "sources": {
            "frozen_protocol": {
                "path": _portable_path(frozen_protocol_path, repository_root),
                "bytes": frozen_protocol_path.stat().st_size,
                "sha256": _sha256(frozen_protocol_path),
                "frozen_at": protocol["frozen_at"],
            },
            "matrix": {
                "path": _portable_path(matrix_path, repository_root),
                "bytes": matrix_path.stat().st_size,
                "sha256": _sha256(matrix_path),
            },
            "strict_coverage": {
                "path": _portable_path(coverage_file, repository_root),
                "bytes": coverage_file.stat().st_size,
                "sha256": _sha256(coverage_file),
                "observed_results": coverage["observed_results"],
            },
            "training_summary": {
                "path": _portable_path(training_root / "summary_manifest.json", repository_root),
                "bytes": (training_root / "summary_manifest.json").stat().st_size,
                "sha256": _sha256(training_root / "summary_manifest.json"),
            },
            "training_run_table": {
                "path": _portable_path(training_table, repository_root),
                "bytes": training_table.stat().st_size,
                "sha256": _sha256(training_table),
            },
            "evaluation_results": [
                {
                    "path": _portable_path(path, repository_root),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in result_paths
            ],
        },
        "outputs": outputs,
        "interpretation": (
            "The four learning rates are sweep points rather than random seeds. First passage is "
            "an exploratory time-to-AdamW-reference diagnostic and does not replace the frozen "
            "three-seed confirmatory comparison. The rule was locked after 160 of 1,680 discovery "
            "units were visible and is a prospective completion analysis, not a preregistration. "
            "The adjacent-checkpoint task-delta stability table is a separate post-hoc exploratory "
            "diagnostic added after heterogeneous LateOn task directions became visible; it does "
            "not alter selection or the confirmatory family and has no causal interpretation."
        ),
    }
    _atomic_json(output / "summary_manifest.json", manifest)
    if blog_path is not None:
        render_task_stability_blog(Path(blog_path).resolve(), task_stability_rows)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict discovery retrieval-quality versus useful-time dynamics"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--results-root", type=Path, default=Path("results/decontaminated-beir"))
    parser.add_argument("--coverage", type=Path, default=Path("reports/coverage.json"))
    parser.add_argument("--training-dir", type=Path, default=Path("reports/training-dynamics"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/retrieval-dynamics"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/retrieval_dynamics_protocol.json")
    )
    parser.add_argument("--blog", type=Path, default=Path("docs/blog.md"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_retrieval_dynamics(
        args.matrix,
        args.results_root,
        args.coverage,
        args.training_dir,
        args.output_dir,
        args.protocol,
        args.blog,
        families=tuple(args.families),
        scope_amendment=args.scope_amendment,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
