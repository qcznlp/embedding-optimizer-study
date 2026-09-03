"""Evaluate corrected Dense checkpoints on pinned decontaminated BEIR."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .aggregate import audit_dataset_artifacts, audit_training_artifacts
from .config import RunConfig, load_matrix, resolve_matrix_path
from .corrected_input_execution import (
    PADDED_DENSE_RECEIPT,
    require_corrected_training_receipt,
)
from .decontamination import DECONTAMINATED_TASK_NAMES, get_decontaminated_task
from .evaluate_matrix import (
    _record_evaluation_inputs,
    _record_runtime,
    _validate_formal_runtime,
    _validate_worker_runtime,
    _worker_python,
    checkpoint_paths,
)
from .evaluation_utils import task_result_remaining


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(repository: Path) -> dict[str, dict[str, int | str]]:
    package = Path(__file__).resolve().parent
    paths = {
        "src/embed_optim/corrected_beir_evaluation.py": Path(__file__).resolve(),
        "src/embed_optim/corrected_input_execution.py": package / "corrected_input_execution.py",
        "src/embed_optim/evaluate_matrix.py": package / "evaluate_matrix.py",
        "src/embed_optim/evaluation_utils.py": package / "evaluation_utils.py",
        "src/embed_optim/decontamination.py": package / "decontamination.py",
        "src/embed_optim/aggregate.py": package / "aggregate.py",
        "src/embed_optim/config.py": package / "config.py",
        "scripts/eval/dense_no_packing_parallel.py": repository
        / "scripts/eval/dense_no_packing_parallel.py",
        "scripts/eval/dense_parallel.py": repository / "scripts/eval/dense_parallel.py",
        "scripts/eval/dense_sequential.py": repository / "scripts/eval/dense_sequential.py",
    }
    return {
        label: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for label, path in paths.items()
    }


def _selected_configs(
    matrix: str | Path,
    run_ids: list[str],
) -> tuple[Path, list[RunConfig]]:
    matrix_path = resolve_matrix_path(matrix).resolve()
    configs = load_matrix(matrix_path)
    if (
        len(configs) != 12
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
    ):
        raise ValueError("Corrected BEIR evaluation requires the 12-run padded Dense matrix")
    requested = set(run_ids)
    unknown = requested - {config.run_id for config in configs}
    if unknown:
        raise ValueError(f"Unknown corrected run IDs: {sorted(unknown)}")
    selected = [config for config in configs if not requested or config.run_id in requested]
    for config in selected:
        completion = json.loads((config.output_dir / "completed.json").read_text(encoding="utf-8"))
        require_corrected_training_receipt(completion)
    return matrix_path, selected


def _validate_training(configs: list[RunConfig]) -> None:
    dataset = audit_dataset_artifacts(configs)
    if not dataset["complete"]:
        raise RuntimeError(
            "Corrected BEIR training-data audit failed: " + "; ".join(dataset["errors"][:5])
        )
    training = audit_training_artifacts(
        configs,
        deep=True,
        expected_dataset_fingerprint=dataset.get("training_view_fingerprint"),
    )
    if not training["complete"]:
        raise RuntimeError(
            "Corrected BEIR checkpoint audit failed: " + "; ".join(training["errors"][:5])
        )


def _model_folder(checkpoint: Path) -> str:
    resolved = checkpoint.resolve()
    return f"{resolved.parent.name}__{resolved.name}"


def _task_contracts() -> dict[str, tuple[str, list[str], list[str]]]:
    output = {}
    for name in DECONTAMINATED_TASK_NAMES:
        task = get_decontaminated_task(name)
        splits = list(task.metadata.eval_splits)
        if "test" in splits and len(splits) > 1:
            splits = ["test"]
        output[name] = (task.metadata.name, list(task.hf_subsets), splits)
    return output


def audit_requested_results(results_root: Path, checkpoints: list[Path]) -> dict:
    contracts = _task_contracts()
    missing = []
    result_files = []
    for checkpoint in checkpoints:
        model_root = results_root / "dense" / _model_folder(checkpoint)
        for base_name, (task_name, subsets, splits) in contracts.items():
            if task_result_remaining(model_root, task_name, subsets, splits):
                missing.append(f"{checkpoint.parent.name}/{checkpoint.name}/{base_name}")
            else:
                matches = sorted(model_root.rglob(f"{task_name}.json"))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one result for {checkpoint}/{task_name}, found {len(matches)}"
                    )
                result_files.append(matches[0])
    if missing:
        raise RuntimeError(f"Corrected BEIR results are incomplete ({len(missing)}): {missing[:5]}")
    return {
        "checkpoints": len(checkpoints),
        "tasks": len(contracts),
        "task_units": len(checkpoints) * len(contracts),
        "result_files": len(result_files),
    }


def _record_execution_contract(
    results_root: Path,
    *,
    repository: Path,
    matrix_path: Path,
    protocol_path: Path,
    source_files: dict,
) -> None:
    path = results_root / "corrected_execution.json"
    payload = {
        "schema_version": 1,
        "status": "locked",
        "matrix": {
            "path": str(matrix_path.relative_to(repository)),
            "sha256": _sha256(matrix_path),
        },
        "protocol": {
            "path": str(protocol_path.relative_to(repository)),
            "sha256": _sha256(protocol_path),
        },
        "input_execution": dict(PADDED_DENSE_RECEIPT),
        "tasks": list(DECONTAMINATED_TASK_NAMES),
        "source_files": source_files,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"Corrected evaluation contract changed under {results_root}")
        return
    if any((results_root / "dense").rglob("*Decontaminated.json")):
        raise RuntimeError("Refusing corrected results without a pre-existing execution contract")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command(
    repository: Path,
    python: str,
    checkpoints: list[Path],
    results_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        python,
        str(repository / "scripts/eval/dense_no_packing_parallel.py"),
        "--gpus",
        args.gpus,
        "--results_folder",
        str(results_root / "dense"),
        "--models",
        *(str(path) for path in checkpoints),
        "--tasks",
        *DECONTAMINATED_TASK_NAMES,
        "--log_dir",
        str(args.log_dir.resolve()),
        "--bf16",
        "--fa2",
        "--local",
        "--decontaminated",
    ]


def run(args: argparse.Namespace) -> dict:
    repository = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol.resolve()
    matrix_path, configs = _selected_configs(args.matrix, args.run_ids)
    _validate_training(configs)
    checkpoints = [
        checkpoint for config in configs for checkpoint in checkpoint_paths(config, args.stages)
    ]
    python = _worker_python(args.python)
    _validate_formal_runtime(python, matrix_path)
    versions = _validate_worker_runtime(python, {"dense": checkpoints})
    sources = _source_manifest(repository)
    results_root = args.results_root.resolve()
    _record_execution_contract(
        results_root,
        repository=repository,
        matrix_path=matrix_path,
        protocol_path=protocol_path,
        source_files=sources,
    )
    _record_runtime(results_root, python, versions, sources)
    _record_evaluation_inputs(results_root, {"dense": checkpoints})
    command = _command(repository, python, checkpoints, results_root, args)
    if args.dry_run:
        return {"status": "dry_run", "command": command, "checkpoints": len(checkpoints)}
    if not args.audit_only:
        result = subprocess.run(command, cwd=repository, check=False)
        if result.returncode:
            raise RuntimeError(f"Corrected Dense evaluator exited {result.returncode}")
    return {"status": "complete", **audit_requested_results(results_root, checkpoints)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/dense_no_packing_execution_protocol.json")
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/dense-no-packing-beir"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/dense-no-packing-beir"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--stages", nargs="*", type=int)
    parser.add_argument("--python")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.stages and (
        len(set(args.stages)) != len(args.stages)
        or any(not 1 <= value <= 5 for value in args.stages)
    ):
        parser.error("--stages must contain unique values in [1, 5]")
    return args


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
