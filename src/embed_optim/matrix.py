from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig, load_matrix


@dataclass
class Running:
    config: RunConfig
    process: subprocess.Popen
    log_handle: object
    started: float


@dataclass(frozen=True)
class Pool:
    gpu_ids: str
    master_port: int
    preferred_family: str


def _pop_next(
    pool: Pool,
    queues: dict[str, list[RunConfig]],
    running: dict[str, Running],
) -> RunConfig | None:
    """Prefer a pool's model family, then steal work after that family drains."""

    preferred = pool.preferred_family
    if queues[preferred]:
        return queues[preferred].pop(0)
    if any(job.config.model_family == preferred for job in running.values()):
        return None
    alternate = "late" if preferred == "dense" else "dense"
    return queues[alternate].pop(0) if queues[alternate] else None


def _checkpoint_is_resumable(path: Path, world_size: int = 4) -> bool:
    """Reject a checkpoint directory interrupted partway through its write."""

    try:
        step = int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return False
    required = (
        path / "config.json",
        path / "optimizer.pt",
        path / "scheduler.pt",
        path / "trainer_state.json",
        path / "training_args.bin",
    )
    if any(not item.is_file() or item.stat().st_size == 0 for item in required):
        return False
    if not any(item.stat().st_size > 0 for item in path.rglob("*.safetensors")):
        return False
    rng_states = sorted(path.glob("rng_state_*.pth"))
    if len(rng_states) != world_size or any(item.stat().st_size == 0 for item in rng_states):
        return False
    try:
        state = json.loads((path / "trainer_state.json").read_text())
        return int(state["global_step"]) == step
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False


def _latest_resumable_checkpoint(config: RunConfig) -> Path | None:
    checkpoints = sorted(
        (path for path in config.output_dir.glob("checkpoint-*") if _checkpoint_is_resumable(path)),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
        reverse=True,
    )
    if not checkpoints:
        return None

    # Older/synthetic output directories may not have the schedule. Formal runs
    # write it before their first checkpoint; when present, use the same deep
    # payload and runtime-contract audit that gates evaluation.
    schedule_path = config.output_dir / "checkpoint_schedule.json"
    if not schedule_path.is_file():
        return checkpoints[0]
    try:
        schedule = json.loads(schedule_path.read_text())
        steps = sorted(int(step) for step in schedule["steps"])
        if len(steps) != 5 or len(set(steps)) != 5:
            raise ValueError(f"expected five unique steps, got {steps}")
        final_step = steps[-1]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError, IndexError) as error:
        raise RuntimeError(f"Invalid checkpoint schedule {schedule_path}: {error}") from error

    from .aggregate import _deep_checkpoint_problems

    rejected: list[str] = []
    for checkpoint in checkpoints:
        step = int(checkpoint.name.rsplit("-", 1)[1])
        if step not in steps:
            rejected.append(f"{checkpoint.name}: step is outside the declared schedule")
            continue
        problems = _deep_checkpoint_problems(
            checkpoint,
            step,
            world_size=4,
            config=config,
            final_step=final_step,
        )
        if not problems:
            return checkpoint
        rejected.append(f"{checkpoint.name}: {'; '.join(problems)}")
    raise RuntimeError(
        f"No deeply resumable checkpoint remains in {config.output_dir}: " + " | ".join(rejected)
    )


def _run_is_complete(config: RunConfig) -> bool:
    """Only skip a run whose terminal marker and all resumable artifacts agree."""

    output = config.output_dir
    try:
        completed = json.loads((output / "completed.json").read_text())
        schedule = json.loads((output / "checkpoint_schedule.json").read_text())
        steps = [int(step) for step in schedule["steps"]]
        final_state = json.loads((output / "trainer_state_final.json").read_text())
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    if len(steps) != 5 or steps != sorted(set(steps)):
        return False
    try:
        completion_steps = sorted(int(step) for step in completed.get("checkpoints", []))
        completed_step = int(completed.get("global_step", -1))
        final_step = int(final_state.get("global_step", -1))
    except (TypeError, ValueError):
        return False
    if (
        completed.get("run_id") != config.run_id
        or completed.get("model_family") != config.model_family
        or completed_step != steps[-1]
        or completion_steps != steps
        or final_step != steps[-1]
    ):
        return False
    accepted_summary = completed.get("accepted_timing")
    if accepted_summary is not None:
        timing_path = output / "accepted_timing.json"
        try:
            timing = json.loads(timing_path.read_text())
            segments = timing["segments"]
            recorded_total = float(timing["total_wall_time_seconds_max_rank"])
            summary_total = float(accepted_summary["total_wall_time_seconds_max_rank"])
            total = sum(float(segment["wall_time_seconds_max_rank"]) for segment in segments)
            timing_steps = [
                (
                    int(segment["start_step_exclusive"]),
                    int(segment["end_step_inclusive"]),
                )
                for segment in segments
            ]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return False
        if (
            timing.get("schema_version") != 1
            or not timing_steps
            or any(end <= start for start, end in timing_steps)
            or any(
                previous[1] != current[0]
                for previous, current in zip(timing_steps, timing_steps[1:])
            )
            or timing_steps[-1][1] != steps[-1]
            or not math.isfinite(total)
            or total <= 0
            or not math.isclose(recorded_total, total, rel_tol=1e-9, abs_tol=1e-6)
            or accepted_summary.get("schema_version") != 1
            or accepted_summary.get("segments") != len(segments)
            or not math.isclose(summary_total, total, rel_tol=1e-9, abs_tol=1e-6)
        ):
            return False
    if not all(_checkpoint_is_resumable(output / f"checkpoint-{step}") for step in steps):
        return False
    final_dir = output / "final"
    return final_dir.is_dir() and any(
        path.stat().st_size > 0 for path in final_dir.rglob("*.safetensors")
    )


def _launch(
    config: RunConfig,
    matrix_path: Path,
    gpu_ids: str,
    master_port: int,
    log_dir: Path,
    dry_run: bool,
) -> Running | None:
    log_path = log_dir / f"{config.model_family}-{config.run_id}.log"
    command = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=4",
        f"--master-port={master_port}",
        "-m",
        "embed_optim.train",
        "--matrix",
        str(matrix_path),
        "--model-family",
        config.model_family,
        "--run-id",
        config.run_id,
    ]
    latest = _latest_resumable_checkpoint(config)
    if latest is not None:
        command.extend(["--resume-from-checkpoint", str(latest.resolve())])
    print(f"[{gpu_ids}] {' '.join(command)}", flush=True)
    if dry_run:
        return None
    log_handle = log_path.open("a")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu_ids,
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return Running(config, process, log_handle, time.monotonic())


def _complete(config: RunConfig) -> bool:
    return _run_is_complete(config)


def run_matrix(args: argparse.Namespace) -> int:
    matrix_path = Path(args.matrix).resolve()
    configs = [
        config
        for config in load_matrix(matrix_path)
        if config.model_family in args.families
        and (not args.run_ids or config.run_id in args.run_ids)
    ]
    queues = {
        "dense": [
            config for config in configs if config.model_family == "dense" and not _complete(config)
        ],
        "late": [
            config for config in configs if config.model_family == "late" and not _complete(config)
        ],
    }
    pools = {
        "a": Pool(args.gpus_a, args.port_a, "dense"),
        "b": Pool(args.gpus_b, args.port_b, "late"),
    }
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, Running] = {}
    failures = 0
    while any(queues.values()) or running:
        for pool_name, job in list(running.items()):
            return_code = job.process.poll()
            if return_code is None:
                continue
            job.log_handle.close()
            elapsed = time.monotonic() - job.started
            print(
                f"pool-{pool_name} {job.config.model_family}/{job.config.run_id} "
                f"exited {return_code} after {elapsed / 60:.1f} min",
                flush=True,
            )
            del running[pool_name]
            if return_code != 0:
                failures += 1
                if args.fail_fast:
                    for other in running.values():
                        other.process.terminate()
                    return failures
                # A failed distributed job is normally recoverable from its
                # latest complete checkpoint.  Put it back at the front of its
                # family queue so a transient CUDA/NCCL failure does not leave
                # the run incomplete until every later configuration finishes.
                if not _complete(job.config):
                    queues[job.config.model_family].insert(0, job.config)

        for pool_name, pool in pools.items():
            if pool_name in running:
                continue
            config = _pop_next(pool, queues, running)
            if config is None:
                continue
            job = _launch(
                config,
                matrix_path,
                pool.gpu_ids,
                pool.master_port,
                log_dir,
                args.dry_run,
            )
            if job is not None:
                running[pool_name] = job
        if args.dry_run:
            continue
        time.sleep(5)
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dense and late matrices on two four-GPU pools"
    )
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument(
        "--families", nargs="+", choices=["dense", "late"], default=["dense", "late"]
    )
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--gpus-a", default="0,1,2,3")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--port-a", type=int, default=29510)
    parser.add_argument("--port-b", type=int, default=29520)
    parser.add_argument("--log-dir", default="logs/training")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    failures = run_matrix(parse_args(argv))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
