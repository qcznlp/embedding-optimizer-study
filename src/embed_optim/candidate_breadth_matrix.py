from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .candidate_breadth_data import (
    audit_candidate_breadth_data,
    load_candidate_breadth_protocol,
)
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus) or any(not item.isdigit() for item in gpus):
        raise ValueError("GPUs must be a non-empty comma-separated list of unique integers")
    return gpus


def candidate_breadth_jobs(
    protocol_path: str | Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    protocol_path, protocol = load_candidate_breadth_protocol(protocol_path)
    root = protocol_path.parent.parent.resolve()
    evaluation = protocol["evaluation"]
    step = int(evaluation["checkpoint_step"])
    jobs = []
    for run_id in evaluation["run_ids"]:
        checkpoint = root / evaluation["checkpoint_root"] / run_id / f"checkpoint-{step}"
        jobs.append(
            {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "output_dir": root / evaluation["results_root"] / run_id,
            }
        )
    if len(jobs) != 12 or len({job["run_id"] for job in jobs}) != 12:
        raise ValueError("Candidate-breadth matrix must contain exactly 12 unique discovery runs")
    return protocol_path, protocol, jobs


def _preflight_candidate_breadth_jobs(jobs: list[dict[str, Any]], *, baseline_root: Path) -> None:
    missing: list[Path] = []
    for job in jobs:
        checkpoint = Path(job["checkpoint"])
        if not checkpoint.is_dir():
            missing.append(checkpoint)
        baseline = baseline_root / str(job["run_id"]) / "sample_metrics.jsonl"
        if not baseline.is_file():
            missing.append(baseline)
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Candidate-breadth matrix required inputs are missing before GPU launch: " + rendered
        )


def _verified_source_audit_receipt(
    path: str | Path,
    *,
    root: Path,
    protocol_path: Path,
    data_audit: dict[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "Candidate-breadth source audit receipt must be under the study root"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = {**data_audit, "upstream_reconstruction_verified": True}
    if receipt != expected or receipt.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("Candidate-breadth source audit receipt does not match current data")
    return {
        "path": str(relative),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "audit": receipt,
    }


def _run_job(
    job: dict[str, Any],
    *,
    gpu: str,
    root: Path,
    protocol_path: Path,
    data_root: Path,
    baseline_root: Path,
    log_dir: Path,
    python: str,
    retries: int,
) -> dict[str, Any]:
    checkpoint = Path(job["checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(job["output_dir"])
    log_path = log_dir / f"{job['run_id']}.log"
    command = [
        python,
        "-m",
        "embed_optim.candidate_breadth_evaluation",
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--protocol",
        str(protocol_path),
        "--device",
        "cuda:0",
        "--baseline-root",
        str(baseline_root),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    attempts = []
    for attempt in range(1, retries + 2):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== attempt {attempt}; gpu {gpu} ===\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        attempts.append({"attempt": attempt, "returncode": completed.returncode})
        if completed.returncode == 0:
            break
    else:
        raise RuntimeError(f"Candidate-breadth evaluation failed: {job['run_id']}")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("negative_widths") != [
        7,
        10,
        32,
        128,
        512,
        2048,
    ]:
        raise ValueError(f"Candidate-breadth output is incomplete: {job['run_id']}")
    return {
        "run_id": job["run_id"],
        "gpu": gpu,
        "attempts": attempts,
        "checkpoint": str(checkpoint),
        "manifest": {
            "path": str(manifest_path.relative_to(root)),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "baseline_maximum_absolute_error": manifest.get("baseline_reproduction", {}).get(
            "maximum_absolute_error"
        ),
    }


def run_candidate_breadth_matrix(
    protocol_path: str | Path,
    *,
    gpus: str,
    source_audit_receipt: str | Path = "reports/candidate-breadth/data-audit.json",
    python: str = sys.executable,
    retries: int = 1,
) -> dict[str, Any]:
    protocol_path, protocol, jobs = candidate_breadth_jobs(protocol_path)
    root = protocol_path.parent.parent.resolve()
    evaluation = protocol["evaluation"]
    data_root = (root / evaluation["data_output"]).resolve()
    baseline_root = (root / evaluation["baseline_root"]).resolve()
    _preflight_candidate_breadth_jobs(jobs, baseline_root=baseline_root)
    # The immediately preceding release step performs the independent pinned-source
    # reconstruction.  The matrix repeats the complete local semantic audit without
    # rescanning the source parquet files before each group of checkpoint jobs.
    data_audit = audit_candidate_breadth_data(protocol_path, data_root, verify_source=False)
    source_audit = _verified_source_audit_receipt(
        source_audit_receipt,
        root=root,
        protocol_path=protocol_path,
        data_audit=data_audit,
    )
    gpu_values = _parse_gpus(gpus)
    log_dir = root / "logs" / "candidate-breadth"
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in gpu_values:
        gpu_queue.put(gpu)

    def execute(job: dict[str, Any]) -> dict[str, Any]:
        gpu = gpu_queue.get()
        try:
            return _run_job(
                job,
                gpu=gpu,
                root=root,
                protocol_path=protocol_path,
                data_root=data_root,
                baseline_root=baseline_root,
                log_dir=log_dir,
                python=python,
                retries=retries,
            )
        finally:
            gpu_queue.put(gpu)

    completed_jobs = []
    with ThreadPoolExecutor(max_workers=min(len(gpu_values), len(jobs))) as executor:
        futures = {executor.submit(execute, job): job["run_id"] for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            completed_jobs.append(result)
            print(f"Completed candidate breadth: {result['run_id']}", flush=True)
    completed_jobs.sort(key=lambda item: item["run_id"])
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": {
            "path": str(protocol_path.relative_to(root)),
            "bytes": protocol_path.stat().st_size,
            "sha256": _sha256(protocol_path),
        },
        "data_audit": data_audit,
        "source_audit": source_audit,
        "gpus": gpu_values,
        "jobs": completed_jobs,
    }
    receipt_path = root / evaluation["results_root"] / "matrix-receipt.json"
    _atomic_json(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the 12 discovery checkpoints over nested candidate widths"
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/candidate_breadth_probe.json")
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--source-audit-receipt",
        type=Path,
        default=Path("reports/candidate-breadth/data-audit.json"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_candidate_breadth_matrix(
        args.protocol,
        gpus=args.gpus,
        source_audit_receipt=args.source_audit_receipt,
        python=args.python,
        retries=args.retries,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
