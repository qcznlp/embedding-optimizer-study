"""Run one two-operator factorial training wave under the frozen implementation lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from . import matrix
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens
from .probe_export import _checkpoint_inputs
from .state_operator_factorial import (
    CALIBRATION_ROOT,
    MATRIX_ROOT,
    SCIENTIFIC_PROTOCOL,
    audit_branch_data,
)
from .state_operator_factorial_contract import require_factorial_implementation
from .state_operator_factorial_evaluation import load_cell_configs
from .supplemental_training_audit import audit_derived_training_artifacts

GPU_LOCK_ROOT = Path("logs/dense-only-runtime/gpu-leases")
LOG_ROOT = Path("logs/state-operator-factorial/training")
RECEIPT_ROOT = Path("reports/state-operator-factorial/training")


def run(args: argparse.Namespace) -> dict[str, Any]:
    implementation_path, implementation = require_factorial_implementation()
    protocol_path, _, matrix_path, configs = load_cell_configs(
        args.state,
        args.seed,
        protocol_path=args.protocol,
        matrix_root=args.matrix_root,
        calibration_root=args.calibration_root,
    )
    raw = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if (
        raw.get("provenance", {}).get("implementation_protocol_sha256")
        != _sha256(implementation_path)
        or raw.get("provenance", {}).get("optimizer_state_at_branch_start") != "reset"
    ):
        raise ValueError("Factorial training matrix is outside the implementation lock")
    direction_manifest_path = (
        args.calibration_root.resolve() / args.state / "directions/manifest.json"
    )
    direction_manifest = json.loads(direction_manifest_path.read_text(encoding="utf-8"))
    source_checkpoint = Path(configs[0].model_name).resolve()
    if (
        {Path(config.model_name).resolve() for config in configs} != {source_checkpoint}
        or direction_manifest.get("checkpoint", {}).get("path") != str(source_checkpoint)
        or direction_manifest.get("checkpoint", {}).get("inputs")
        != _checkpoint_inputs(source_checkpoint)
    ):
        raise ValueError("Factorial source checkpoint content changed after scale calibration")
    dataset = audit_branch_data(protocol_path, deep=not args.dry_run)
    log_dir = args.log_root.resolve() / args.state / f"seed{args.seed}"
    runner_args = argparse.Namespace(
        matrix=str(matrix_path),
        families=["dense"],
        run_ids=[],
        gpus_a=args.gpus_a,
        gpus_b=args.gpus_b,
        port_a=args.port_a,
        port_b=args.port_b,
        log_dir=str(log_dir),
        max_retries=args.max_retries,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        failures = matrix.run_matrix(runner_args)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run",
            "state": args.state,
            "seed": args.seed,
            "matrix": str(matrix_path),
            "runs": len(configs),
            "failures": failures,
        }

    tokens_a = parse_gpu_tokens(args.gpus_a)
    tokens_b = parse_gpu_tokens(args.gpus_b)
    tokens = tuple(sorted({*tokens_a, *tokens_b}, key=int))
    if len(tokens_a) != 4 or len(tokens_b) != 4 or set(tokens_a).intersection(tokens_b):
        raise ValueError("Factorial training requires two disjoint four-GPU pools")
    with acquire_gpu_lease(
        tokens,
        lock_dir=args.gpu_lock_dir.resolve(),
        timeout_seconds=args.gpu_lock_timeout_seconds,
        purpose=f"state-operator-training:{args.state}:seed{args.seed}",
        ledger_path=log_dir / f"gpu-lease-{os.getpid()}.json",
    ):
        failures = matrix.run_matrix(runner_args)
    if failures:
        raise RuntimeError(f"Factorial training wave exhausted {failures} run retries")
    audit = audit_derived_training_artifacts(configs, dataset, deep=True)
    if audit.get("complete") is not True or audit.get("errors"):
        details = "; ".join(str(item) for item in audit.get("errors", [])[:10])
        raise RuntimeError(f"Factorial training completion audit failed: {details or 'incomplete'}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "state": args.state,
        "seed": args.seed,
        "implementation_commit": implementation["implementation_commit"],
        "implementation_protocol_sha256": _sha256(implementation_path),
        "scientific_protocol_sha256": _sha256(protocol_path),
        "matrix_sha256": _sha256(matrix_path),
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "verified_runs": audit["verified_runs"],
        "verified_checkpoints": audit["verified_checkpoints"],
        "errors": [],
    }
    receipt_path = args.receipt_root.resolve() / f"{args.state}-seed{args.seed}.json"
    _atomic_json(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=("adamw_state", "muon_state"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=SCIENTIFIC_PROTOCOL)
    parser.add_argument("--matrix-root", type=Path, default=MATRIX_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=CALIBRATION_ROOT)
    parser.add_argument("--gpus-a", default="0,1,2,3")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--port-a", type=int, default=29710)
    parser.add_argument("--port-b", type=int, default=29720)
    parser.add_argument("--gpu-lock-dir", type=Path, default=GPU_LOCK_ROOT)
    parser.add_argument("--gpu-lock-timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT)
    parser.add_argument("--receipt-root", type=Path, default=RECEIPT_ROOT)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_retries < 0 or args.gpu_lock_timeout_seconds <= 0:
        parser.error("--max-retries must be non-negative and lease timeout positive")
    return args


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
