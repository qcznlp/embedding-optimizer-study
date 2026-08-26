from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aggregate import DECONTAMINATED_TASK_NAMES, collect_evaluations
from .config import load_matrix
from .geometry import _atomic_json

SCHEMA_VERSION = 1


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write strict, provenance-validated evaluation coverage snapshots"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--results-root", type=Path, default=Path("results/decontaminated-beir"))
    parser.add_argument("--output", type=Path, default=Path("logs/evaluation/live-audit.json"))
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
    if args.watch_seconds:
        watch_progress(
            args.matrix,
            args.results_root,
            args.output,
            interval_seconds=args.watch_seconds,
        )
    else:
        print(
            json.dumps(write_progress(args.matrix, args.results_root, args.output), sort_keys=True)
        )


if __name__ == "__main__":
    main()
