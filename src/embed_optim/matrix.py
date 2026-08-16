from __future__ import annotations

import argparse
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
    checkpoints = [
        path
        for path in config.output_dir.glob("checkpoint-*")
        if (path / "trainer_state.json").is_file()
    ]
    if checkpoints:
        latest = max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[1]))
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
    return (config.output_dir / "completed.json").is_file()


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
        "dense": (args.gpus_a, args.port_a),
        "late": (args.gpus_b, args.port_b),
    }
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[str, Running] = {}
    failures = 0
    while any(queues.values()) or running:
        for family, job in list(running.items()):
            return_code = job.process.poll()
            if return_code is None:
                continue
            job.log_handle.close()
            elapsed = time.monotonic() - job.started
            print(
                f"{family}/{job.config.run_id} exited {return_code} after {elapsed / 60:.1f} min",
                flush=True,
            )
            del running[family]
            if return_code != 0:
                failures += 1
                if args.fail_fast:
                    for other in running.values():
                        other.process.terminate()
                    return failures

        for family in ("dense", "late"):
            if family in running or not queues[family]:
                continue
            gpu_ids, port = pools[family]
            job = _launch(
                queues[family].pop(0),
                matrix_path,
                gpu_ids,
                port,
                log_dir,
                args.dry_run,
            )
            if job is not None:
                running[family] = job
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
