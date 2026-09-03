"""Artifact-only progress report for the corrected Dense no-packing matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .config import load_matrix
from .matrix import _checkpoint_is_resumable, _run_is_complete

PROGRESS_PATTERN = re.compile(r"(\d+)/(\d+)")
ERROR_PATTERNS = {
    "cuda_oom": re.compile(r"CUDA out of memory", re.IGNORECASE),
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "nccl_error": re.compile(r"NCCL[^\n]*(?:error|Error)"),
    "non_finite": re.compile(r"(?:loss|grad_norm)['\"]?:\s*['\"]?(?:nan|inf)", re.IGNORECASE),
}


def _log_progress(path: Path) -> tuple[int, int | None, dict[str, int]]:
    if not path.is_file():
        return 0, None, {name: 0 for name in ERROR_PATTERNS}
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = [(int(step), int(total)) for step, total in PROGRESS_PATTERN.findall(text)]
    # Logs also contain model-loading bars such as 134/134. The training
    # horizon is the largest declared total; select progress within that bar.
    step, total = max(matches, default=(0, None), key=lambda item: (item[1], item[0]))
    return (
        step,
        total,
        {name: len(pattern.findall(text)) for name, pattern in ERROR_PATTERNS.items()},
    )


def build_progress(
    matrix: str | Path = "configs/dense_no_packing_retrain.yaml",
    log_dir: str | Path = "logs/dense-no-packing-v1",
) -> dict:
    configs = load_matrix(matrix)
    log_dir = Path(log_dir)
    runs = []
    for config in configs:
        log_path = log_dir / f"dense-{config.run_id}.log"
        log_step, log_total, errors = _log_progress(log_path)
        schedule_path = config.output_dir / "checkpoint_schedule.json"
        if schedule_path.is_file():
            try:
                schedule = sorted(
                    int(value)
                    for value in json.loads(schedule_path.read_text(encoding="utf-8"))["steps"]
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                schedule = []
        else:
            schedule = []
        resumable = [
            step
            for step in schedule
            if _checkpoint_is_resumable(config.output_dir / f"checkpoint-{step}")
        ]
        complete = _run_is_complete(config)
        if complete:
            state = "complete"
        elif log_path.is_file() or schedule_path.is_file():
            state = "started_artifacts_present"
        else:
            state = "pending"
        runs.append(
            {
                "run_id": config.run_id,
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "state": state,
                "latest_log_step": log_step,
                "declared_total_steps": log_total,
                "resumable_checkpoint_steps": resumable,
                "error_markers": errors,
            }
        )
    complete = sum(run["state"] == "complete" for run in runs)
    started = sum(run["state"] == "started_artifacts_present" for run in runs)
    return {
        "schema_version": 1,
        "scope": "corrected_dense_no_packing",
        "observation": "artifact_only_no_process_inspection",
        "complete_runs": complete,
        "started_incomplete_runs": started,
        "pending_runs": len(runs) - complete - started,
        "resumable_checkpoints": sum(len(run["resumable_checkpoint_steps"]) for run in runs),
        "error_markers": {
            name: sum(run["error_markers"][name] for run in runs) for name in ERROR_PATTERNS
        },
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs/dense-no-packing-v1"))
    args = parser.parse_args(argv)
    print(json.dumps(build_progress(args.matrix, args.log_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
