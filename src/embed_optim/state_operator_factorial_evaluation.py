"""Run final-checkpoint full BEIR evaluation for one factorial source-state/seed matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import RunConfig, load_matrix
from .corrected_beir_evaluation import audit_requested_results
from .decontamination import DECONTAMINATED_TASK_NAMES
from .evaluate_matrix import (
    _record_evaluation_inputs,
    _record_runtime,
    _validate_formal_runtime,
    _validate_worker_runtime,
    _worker_python,
    checkpoint_paths,
)
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens
from .state_operator_factorial import (
    MATRIX_ROOT,
    SCIENTIFIC_PROTOCOL,
    audit_branch_data,
    audit_factorial_matrices,
    load_factorial_protocol,
)
from .state_operator_factorial_contract import require_factorial_implementation
from .supplemental_training_audit import audit_derived_training_artifacts

RESULTS_ROOT = Path("results/state-operator-factorial/full-beir")
LOG_ROOT = Path("logs/state-operator-factorial/full-beir")
GPU_LOCK_ROOT = Path("logs/dense-only-runtime/gpu-leases")


def matrix_path(state: str, seed: int, root: str | Path = MATRIX_ROOT) -> Path:
    return Path(root).resolve() / f"{state}-seed{seed}.yaml"


def load_cell_configs(
    state: str,
    seed: int,
    *,
    protocol_path: str | Path = SCIENTIFIC_PROTOCOL,
    matrix_root: str | Path = MATRIX_ROOT,
    calibration_root: str | Path | None = None,
    audit_matrices: bool = True,
) -> tuple[Path, dict[str, Any], Path, list[RunConfig]]:
    resolved, protocol = load_factorial_protocol(protocol_path)
    states = {item["label"] for item in protocol["source_states"]["states"]}
    seeds = {int(value) for value in protocol["branch_data"]["order_seeds"]}
    if state not in states or seed not in seeds:
        raise ValueError(f"Unknown factorial source-state/seed cell: {state}/seed{seed}")
    if audit_matrices:
        kwargs: dict[str, Any] = {
            "matrix_root": matrix_root,
            "deep_data_audit": False,
        }
        if calibration_root is not None:
            kwargs["calibration_root"] = calibration_root
        audit_factorial_matrices(resolved, **kwargs)
    path = matrix_path(state, seed, matrix_root)
    configs = load_matrix(path)
    expected_run_ids = {
        f"{state.replace('_', '-')}__adamw-reset",
        f"{state.replace('_', '-')}__muon-reset",
    }
    if (
        len(configs) != 2
        or {config.run_id for config in configs} != expected_run_ids
        or {config.seed for config in configs} != {seed}
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs is not False for config in configs)
    ):
        raise ValueError(f"Factorial matrix differs for {state}/seed{seed}")
    return resolved, protocol, path, configs


def _deep_training_audit(
    configs: list[RunConfig], protocol_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = audit_branch_data(protocol_path, deep=True)
    audit = audit_derived_training_artifacts(configs, dataset, deep=True)
    if audit.get("complete") is not True or audit.get("errors"):
        details = "; ".join(str(item) for item in audit.get("errors", [])[:10])
        raise RuntimeError(f"Factorial training preflight failed: {details or 'incomplete'}")
    return dataset, audit


def _source_manifest(repository: Path) -> dict[str, dict[str, int | str]]:
    package = Path(__file__).resolve().parent
    paths = {
        "src/embed_optim/state_operator_factorial_evaluation.py": Path(__file__).resolve(),
        "src/embed_optim/state_operator_factorial.py": package / "state_operator_factorial.py",
        "src/embed_optim/corrected_beir_evaluation.py": package / "corrected_beir_evaluation.py",
        "src/embed_optim/corrected_input_execution.py": package / "corrected_input_execution.py",
        "src/embed_optim/evaluate_matrix.py": package / "evaluate_matrix.py",
        "src/embed_optim/evaluation_utils.py": package / "evaluation_utils.py",
        "src/embed_optim/decontamination.py": package / "decontamination.py",
        "src/embed_optim/aggregate.py": package / "aggregate.py",
        "src/embed_optim/config.py": package / "config.py",
        "src/embed_optim/gpu_lease.py": package / "gpu_lease.py",
        "src/embed_optim/supplemental_training_audit.py": package
        / "supplemental_training_audit.py",
        "scripts/eval/dense_no_packing_parallel.py": repository
        / "scripts/eval/dense_no_packing_parallel.py",
        "scripts/eval/dense_parallel.py": repository / "scripts/eval/dense_parallel.py",
        "scripts/eval/dense_sequential.py": repository / "scripts/eval/dense_sequential.py",
    }
    return {
        label: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for label, path in paths.items()
    }


def _record_execution_contract(
    results_root: Path,
    *,
    repository: Path,
    protocol_path: Path,
    matrix: Path,
    matrix_manifest: Path,
    state: str,
    seed: int,
    checkpoints: list[Path],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    path = results_root / "factorial_execution.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "locked",
        "scientific_protocol": {
            "path": str(protocol_path.relative_to(repository)),
            "sha256": _sha256(protocol_path),
        },
        "matrix_manifest": {
            "path": str(matrix_manifest.relative_to(repository)),
            "sha256": _sha256(matrix_manifest),
        },
        "matrix": {
            "path": str(matrix.relative_to(repository)),
            "sha256": _sha256(matrix),
        },
        "source_state": state,
        "order_seed": seed,
        "stages": [5],
        "checkpoints": [str(item.resolve()) for item in checkpoints],
        "input_execution": {
            "mode": "independently_padded",
            "sentence_transformers_can_flatten_inputs": False,
        },
        "tasks": list(DECONTAMINATED_TASK_NAMES),
        "expected_task_units": 28,
        "source_files": source_files,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"Factorial evaluation contract changed under {results_root}")
        return payload
    if any((results_root / "dense").rglob("*Decontaminated.json")):
        raise RuntimeError("Refusing factorial results without a pre-existing execution contract")
    _atomic_json(path, payload)
    return payload


def _command(
    repository: Path,
    python: str,
    checkpoints: list[Path],
    results_root: Path,
    log_dir: Path,
    gpus: str,
) -> list[str]:
    return [
        python,
        str(repository / "scripts/eval/dense_no_packing_parallel.py"),
        "--gpus",
        gpus,
        "--results_folder",
        str(results_root / "dense"),
        "--models",
        *(str(path) for path in checkpoints),
        "--tasks",
        *DECONTAMINATED_TASK_NAMES,
        "--log_dir",
        str(log_dir),
        "--bf16",
        "--fa2",
        "--local",
        "--decontaminated",
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_factorial_implementation()
    repository = Path(__file__).resolve().parents[2]
    protocol_path, _, selected_matrix, configs = load_cell_configs(
        args.state,
        args.seed,
        protocol_path=args.protocol,
        matrix_root=args.matrix_root,
        calibration_root=args.calibration_root,
    )
    dataset, training_audit = _deep_training_audit(configs, protocol_path)
    checkpoints = [checkpoint for config in configs for checkpoint in checkpoint_paths(config, [5])]
    if len(checkpoints) != 2:
        raise AssertionError("One factorial matrix must contribute exactly two final checkpoints")
    python = _worker_python(args.python)
    _validate_formal_runtime(python, selected_matrix)
    versions = _validate_worker_runtime(python, {"dense": checkpoints})
    sources = _source_manifest(repository)
    results_root = args.results_root.resolve() / args.state / f"seed{args.seed}"
    log_dir = args.log_root.resolve() / args.state / f"seed{args.seed}"
    matrix_manifest = args.matrix_root.resolve() / "manifest.json"
    contract = _record_execution_contract(
        results_root,
        repository=repository,
        protocol_path=protocol_path,
        matrix=selected_matrix,
        matrix_manifest=matrix_manifest,
        state=args.state,
        seed=args.seed,
        checkpoints=checkpoints,
        source_files=sources,
    )
    _record_runtime(results_root, python, versions, sources)
    _record_evaluation_inputs(results_root, {"dense": checkpoints})
    command = _command(
        repository,
        python,
        checkpoints,
        results_root,
        log_dir,
        args.gpus,
    )
    if args.dry_run:
        return {
            "status": "dry_run",
            "source_state": args.state,
            "order_seed": args.seed,
            "checkpoints": len(checkpoints),
            "command": command,
            "contract": contract,
            "training_audit": {
                "verified_runs": training_audit["verified_runs"],
                "verified_checkpoints": training_audit["verified_checkpoints"],
                "dataset_manifest_sha256": dataset["manifest_sha256"],
            },
        }
    if not args.audit_only:
        gpu_tokens = tuple(sorted(parse_gpu_tokens(args.gpus), key=int))
        lease_path = log_dir / f"gpu-lease-{os.getpid()}.json"
        with acquire_gpu_lease(
            gpu_tokens,
            lock_dir=args.gpu_lock_dir.resolve(),
            timeout_seconds=args.gpu_lock_timeout_seconds,
            purpose=f"state-operator-full-beir:{args.state}:seed{args.seed}",
            ledger_path=lease_path,
        ):
            result = subprocess.run(command, cwd=repository, check=False)
        if result.returncode:
            raise RuntimeError(f"Factorial Dense evaluator exited {result.returncode}")
    audit = audit_requested_results(results_root, checkpoints)
    if audit != {
        "checkpoints": 2,
        "tasks": 14,
        "task_units": 28,
        "result_files": 28,
    }:
        raise RuntimeError(f"Factorial full-BEIR coverage differs: {audit}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source_state": args.state,
        "order_seed": args.seed,
        "scientific_protocol_sha256": _sha256(protocol_path),
        "matrix_sha256": _sha256(selected_matrix),
        "execution_contract_sha256": _sha256(results_root / "factorial_execution.json"),
        **audit,
    }
    _atomic_json(results_root / "completion.json", receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=("adamw_state", "muon_state"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=SCIENTIFIC_PROTOCOL)
    parser.add_argument("--matrix-root", type=Path, default=MATRIX_ROOT)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("results/dense-no-packing-state-operator/calibration"),
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--gpu-lock-dir", type=Path, default=GPU_LOCK_ROOT)
    parser.add_argument("--gpu-lock-timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--python")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu_lock_timeout_seconds <= 0:
        parser.error("--gpu-lock-timeout-seconds must be positive")
    if args.dry_run and args.audit_only:
        parser.error("--dry-run and --audit-only are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
