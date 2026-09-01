from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .aggregate import DECONTAMINATED_TASK_NAMES, collect_evaluations
from .candidate_breadth_release import (
    RELEASE_STEP_NAMES,
    UPSTREAM_FINALIZATION_STEP_NAMES,
)
from .config import RunConfig, load_matrix
from .dense_completion_pipeline import CORE_STEP_NAMES
from .geometry import _atomic_json

SCHEMA_VERSION = 1
CONFIRMATORY_SEEDS = (314159, 271828, 161803)
DENSE_BEIR_EXPECTED_UNITS = 1_750


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def evaluation_progress(matrix: str | Path, results_root: str | Path) -> dict[str, Any]:
    """Collect only provenance-valid evaluation units and summarize exact coverage."""

    configs = load_matrix(matrix)
    rows = collect_evaluations(Path(results_root), configs)
    observed = {
        (row["model_family"], row["run_id"], int(row["stage"]), row["task"]) for row in rows
    }
    expected = {
        (config.model_family, config.run_id, stage, task)
        for config in configs
        for stage in range(1, 6)
        for task in DECONTAMINATED_TASK_NAMES
    }
    missing = expected - observed
    unexpected = observed - expected
    family_counts = Counter(item[0] for item in observed)
    task_counts = Counter(item[3] for item in observed)
    run_counts = Counter(f"{item[0]}/{item[1]}" for item in observed)
    return {
        "schema_version": SCHEMA_VERSION,
        "audited_at": _timestamp(),
        "complete": not missing and not unexpected,
        "error": None,
        "valid_units": len(observed),
        "expected_units": len(expected),
        "missing_units": len(missing),
        "unexpected_units": len(unexpected),
        "families": dict(sorted(family_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
        "runs": dict(sorted(run_counts.items())),
    }


def _repository_configs(repository: Path, matrix: str | Path) -> list[RunConfig]:
    """Load Dense configs with output roots anchored to an explicit checkout."""

    configs = [
        config for config in load_matrix(repository / matrix) if config.model_family == "dense"
    ]
    return [
        dataclasses.replace(
            config,
            output_root=str((repository / config.output_root).resolve()),
        )
        for config in configs
    ]


def _partition_progress(
    repository: Path,
    *,
    matrix: str | Path,
    results_root: str | Path,
    stages: Sequence[int],
) -> dict[str, Any]:
    """Audit one disjoint BEIR result partition against its frozen run/stage grid."""

    configs = _repository_configs(repository, matrix)
    if not configs:
        raise ValueError(f"Dense study matrix has no DenseOn runs: {matrix}")
    stage_set = set(stages)
    if not stage_set or any(stage not in range(1, 6) for stage in stage_set):
        raise ValueError(f"Dense study partition has invalid stages: {stages}")
    rows = collect_evaluations((repository / results_root).resolve(), configs)
    observed = {
        (str(row["model_family"]), str(row["run_id"]), int(row["stage"]), str(row["task"]))
        for row in rows
    }
    expected = {
        (config.model_family, config.run_id, stage, task)
        for config in configs
        for stage in stage_set
        for task in DECONTAMINATED_TASK_NAMES
    }
    missing = expected - observed
    unexpected = observed - expected
    return {
        "complete": not missing and not unexpected,
        "valid_units": len(observed & expected),
        "expected_units": len(expected),
        "missing_units": len(missing),
        "unexpected_units": len(unexpected),
        "runs": len(configs),
        "stages": sorted(stage_set),
        "results_root": str(Path(results_root)),
    }


def _combine_partitions(partitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "complete": all(partition["complete"] for partition in partitions.values()),
        "valid_units": sum(int(partition["valid_units"]) for partition in partitions.values()),
        "expected_units": sum(
            int(partition["expected_units"]) for partition in partitions.values()
        ),
        "missing_units": sum(int(partition["missing_units"]) for partition in partitions.values()),
        "unexpected_units": sum(
            int(partition["unexpected_units"]) for partition in partitions.values()
        ),
        "partitions": partitions,
    }


def _ledger_progress(path: Path, expected_names: Sequence[str]) -> dict[str, Any]:
    """Report controller progress without treating it as a substitute for its strict audit."""

    if not path.is_file():
        return {
            "status": "pending",
            "complete": False,
            "completed_steps": 0,
            "expected_steps": len(expected_names),
            "current_step": None,
            "path": str(path),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "invalid",
            "complete": False,
            "completed_steps": 0,
            "expected_steps": len(expected_names),
            "current_step": None,
            "path": str(path),
            "error": f"{type(error).__name__}: {error}",
        }
    steps = payload.get("steps")
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        return {
            "status": "invalid",
            "complete": False,
            "completed_steps": 0,
            "expected_steps": len(expected_names),
            "current_step": None,
            "path": str(path),
            "error": "ledger steps are missing or malformed",
        }
    names = [step.get("name") for step in steps]
    prefix_valid = names == list(expected_names[: len(names)])
    completed = sum(step.get("complete") is True for step in steps)
    current = next((step.get("name") for step in steps if step.get("complete") is not True), None)
    exact_complete = bool(
        payload.get("complete") is True
        and prefix_valid
        and names == list(expected_names)
        and completed == len(expected_names)
    )
    return {
        "status": "complete" if exact_complete else ("running" if prefix_valid else "invalid"),
        "complete": exact_complete,
        "completed_steps": completed,
        "materialized_steps": len(steps),
        "expected_steps": len(expected_names),
        "current_step": current,
        "path": str(path),
        "prefix_valid": prefix_valid,
    }


def study_evaluation_progress(repository: str | Path) -> dict[str, Any]:
    """Audit the exact Dense-only 1,750-unit BEIR design and publication controllers."""

    root = Path(repository).resolve()
    discovery = _partition_progress(
        root,
        matrix="configs/experiment.yaml",
        results_root="results/decontaminated-beir",
        stages=range(1, 6),
    )
    hybrid = _combine_partitions(
        {
            "dynamics_stage1_4": _partition_progress(
                root,
                matrix="configs/hybrid_adamw.yaml",
                results_root="results/hybrid-adamw-beir-dynamics",
                stages=range(1, 5),
            ),
            "formal_stage5": _partition_progress(
                root,
                matrix="configs/hybrid_adamw.yaml",
                results_root="results/hybrid-adamw-beir",
                stages=(5,),
            ),
        }
    )
    confirmatory_partitions: dict[str, dict[str, Any]] = {}
    for seed in CONFIRMATORY_SEEDS:
        matrix = f"configs/generated/confirmatory/seed{seed}.yaml"
        confirmatory_partitions[f"seed{seed}_dynamics_stage1_4"] = _partition_progress(
            root,
            matrix=matrix,
            results_root=f"results/confirmatory-beir-dynamics/seed{seed}",
            stages=range(1, 5),
        )
        confirmatory_partitions[f"seed{seed}_formal_stage5"] = _partition_progress(
            root,
            matrix=matrix,
            results_root=f"results/confirmatory-beir/seed{seed}",
            stages=(5,),
        )
    confirmatory = _combine_partitions(confirmatory_partitions)
    suites = {"discovery": discovery, "hybrid": hybrid, "confirmatory": confirmatory}
    valid_units = sum(int(suite["valid_units"]) for suite in suites.values())
    expected_units = sum(int(suite["expected_units"]) for suite in suites.values())
    if expected_units != DENSE_BEIR_EXPECTED_UNITS:
        raise ValueError(
            f"Dense study BEIR contract is {expected_units}, expected {DENSE_BEIR_EXPECTED_UNITS}"
        )

    controllers = {
        "completion": _ledger_progress(
            root / "logs/dense-completion-pipeline/pipeline-ledger.json",
            CORE_STEP_NAMES,
        ),
        "finalization": _ledger_progress(
            root / "logs/dense-finalization-pipeline/pipeline-ledger.json",
            UPSTREAM_FINALIZATION_STEP_NAMES,
        ),
        "candidate_breadth_release": _ledger_progress(
            root / "logs/candidate-breadth-release/pipeline-ledger.json",
            RELEASE_STEP_NAMES,
        ),
    }
    beir_complete = all(suite["complete"] for suite in suites.values())
    controllers_complete = all(controller["complete"] for controller in controllers.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "audited_at": _timestamp(),
        "complete": beir_complete and controllers_complete,
        "error": None,
        "beir": {
            "complete": beir_complete,
            "valid_units": valid_units,
            "expected_units": expected_units,
            "completion_fraction": valid_units / expected_units,
            "missing_units": sum(int(suite["missing_units"]) for suite in suites.values()),
            "unexpected_units": sum(int(suite["unexpected_units"]) for suite in suites.values()),
            "suites": suites,
        },
        "shared_start_probes": {
            "included_in_beir_total": False,
            "full_corpus_beir": False,
            "query_disjoint_checkpoint_jobs": 45,
            "unseen_probe_jobs": 46,
            "explanation": (
                "The shared-start mechanism branch uses two frozen functional probes and is not "
                "an additional 126-unit full-corpus BEIR panel."
            ),
        },
        "controllers": controllers,
        "status_boundary": (
            "Controller counts are progress indicators only; each canonical controller's own "
            "content-addressed audits remain the publication authority."
        ),
    }


def write_study_progress(repository: str | Path, output: str | Path) -> dict[str, Any]:
    snapshot = study_evaluation_progress(repository)
    _atomic_json(Path(output), snapshot)
    return snapshot


def write_progress(
    matrix: str | Path,
    results_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    snapshot = evaluation_progress(matrix, results_root)
    _atomic_json(Path(output), snapshot)
    return snapshot


def watch_progress(
    matrix: str | Path,
    results_root: str | Path,
    output: str | Path,
    *,
    interval_seconds: float,
    sleeper=time.sleep,
) -> dict[str, Any]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    while True:
        try:
            snapshot = write_progress(matrix, results_root, output)
        except Exception as error:
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "audited_at": _timestamp(),
                "complete": False,
                "error": f"{type(error).__name__}: {error}",
            }
            _atomic_json(Path(output), snapshot)
        print(json.dumps(snapshot, sort_keys=True), flush=True)
        if snapshot["complete"]:
            return snapshot
        sleeper(interval_seconds)


def watch_study_progress(
    repository: str | Path,
    output: str | Path,
    *,
    interval_seconds: float,
    sleeper=time.sleep,
) -> dict[str, Any]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    while True:
        try:
            snapshot = write_study_progress(repository, output)
        except Exception as error:
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "audited_at": _timestamp(),
                "complete": False,
                "error": f"{type(error).__name__}: {error}",
            }
            _atomic_json(Path(output), snapshot)
        print(json.dumps(snapshot, sort_keys=True), flush=True)
        if snapshot["complete"]:
            return snapshot
        sleeper(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write strict, provenance-validated evaluation coverage snapshots"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--results-root", type=Path, default=Path("results/decontaminated-beir"))
    parser.add_argument(
        "--study-root",
        type=Path,
        help=(
            "Audit the exact Dense-only 1,750-unit BEIR design and the three publication "
            "controllers instead of one matrix/root pair"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Refresh until complete; zero writes one snapshot and exits",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.watch_seconds < 0:
        raise SystemExit("--watch-seconds must be non-negative")
    if args.study_root is not None:
        output = args.output or args.study_root / "logs/evaluation/study-live-audit.json"
        if args.watch_seconds:
            watch_study_progress(
                args.study_root,
                output,
                interval_seconds=args.watch_seconds,
            )
        else:
            print(json.dumps(write_study_progress(args.study_root, output), sort_keys=True))
        return
    output = args.output or Path("logs/evaluation/live-audit.json")
    if args.watch_seconds:
        watch_progress(
            args.matrix,
            args.results_root,
            output,
            interval_seconds=args.watch_seconds,
        )
    else:
        print(json.dumps(write_progress(args.matrix, args.results_root, output), sort_keys=True))


if __name__ == "__main__":
    main()
