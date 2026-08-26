from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .aggregate import collect_evaluations
from .confirmatory_data import load_confirmatory_protocol
from .confirmatory_matrix import audit_confirmatory_matrices
from .decontamination import DECONTAMINATED_TASK_NAMES
from .evaluate_matrix import run_evaluation
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256


def _matrix_paths(protocol: dict[str, Any], matrix_dir: Path) -> dict[int, Path]:
    return {
        int(seed): (matrix_dir / f"seed{int(seed)}.yaml").resolve()
        for seed in protocol["confirmatory_data"]["seeds"]
    }


def audit_confirmatory_evaluations(
    protocol_path: str | Path = "configs/confirmatory_protocol.json",
    *,
    experiment_matrix: str | Path = "configs/experiment.yaml",
    validation_spec: str | Path = "configs/validation_probe.json",
    matrix_dir: str | Path | None = None,
    results_root: str | Path = "results/confirmatory-beir",
    verify_results: bool = True,
) -> dict[str, Any]:
    resolved_protocol, protocol = load_confirmatory_protocol(protocol_path)
    generated = Path(matrix_dir or protocol["training"]["matrix_output_dir"]).resolve()
    matrix_audit = audit_confirmatory_matrices(
        resolved_protocol,
        experiment_matrix=experiment_matrix,
        validation_spec=validation_spec,
        output_dir=generated,
    )
    root = Path(results_root).resolve()
    matrices = _matrix_paths(protocol, generated)
    per_seed: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    total = 0
    for seed, matrix_path in matrices.items():
        from .config import load_matrix

        configs = load_matrix(matrix_path)
        seed_root = root / f"seed{seed}"
        try:
            rows = collect_evaluations(seed_root, configs)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if verify_results:
                raise
            rows = []
            per_seed[str(seed)] = {
                "complete": False,
                "units": 0,
                "error": f"{type(error).__name__}: {error}",
            }
            continue
        expected = {
            (config.model_family, config.run_id, task)
            for config in configs
            for task in DECONTAMINATED_TASK_NAMES
        }
        observed = {
            (str(row["model_family"]), str(row["run_id"]), str(row["task"]))
            for row in rows
            if int(row["stage"]) == 5
        }
        only_final = len(rows) == len(observed) and all(int(row["stage"]) == 5 for row in rows)
        complete = observed == expected and only_final and len(rows) == 84
        if verify_results and not complete:
            raise ValueError(
                f"Seed {seed}: confirmatory evaluation coverage is {len(rows)}/84 final units"
            )
        for row in rows:
            path = Path(row["result_path"]).resolve()
            sources.append(
                {
                    "seed": seed,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        per_seed[str(seed)] = {"complete": complete, "units": len(rows)}
        total += len(rows)
    expected_total = int(protocol["training"]["expected_beir_units"])
    complete = total == expected_total and all(item["complete"] for item in per_seed.values())
    if verify_results and not complete:
        raise ValueError(f"Confirmatory evaluation coverage is {total}/{expected_total}")
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "protocol_sha256": _sha256(resolved_protocol),
        "matrix_manifest_sha256": matrix_audit["manifest_sha256"],
        "expected_units": expected_total,
        "valid_units": total,
        "per_seed": per_seed,
        "result_sources": sorted(sources, key=lambda item: (item["seed"], item["path"])),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or strictly audit final-only BEIR evaluation for all confirmatory seeds"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/confirmatory_protocol.json"))
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/confirmatory-beir"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/confirmatory-evaluation"))
    parser.add_argument("--gpus-a", default="0,1,2,3")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--late-port-a", type=int, default=29710)
    parser.add_argument("--late-port", type=int, default=29720)
    parser.add_argument("--worker-python", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--receipt", type=Path, default=Path("reports/confirmatory/evaluation-receipt.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol_path, protocol = load_confirmatory_protocol(args.protocol)
    generated = Path(args.matrix_dir or protocol["training"]["matrix_output_dir"]).resolve()
    audit_confirmatory_matrices(
        protocol_path,
        experiment_matrix=args.experiment_matrix,
        validation_spec=args.validation_spec,
        output_dir=generated,
    )
    if not args.audit_only:
        for seed, matrix_path in _matrix_paths(protocol, generated).items():
            worker_args = argparse.Namespace(
                matrix=str(matrix_path),
                families=["dense", "late"],
                run_ids=[],
                stages=[5],
                tasks=list(DECONTAMINATED_TASK_NAMES),
                gpus_a=args.gpus_a,
                gpus_b=args.gpus_b,
                late_port_a=args.late_port_a,
                late_port=args.late_port,
                results_root=str((args.results_root / f"seed{seed}").resolve()),
                log_dir=str((args.log_dir / f"seed{seed}").resolve()),
                worker_python=args.worker_python or sys.executable,
            )
            if failures := run_evaluation(worker_args):
                raise RuntimeError(f"Seed {seed}: {failures} evaluator subprocesses failed")
    receipt = audit_confirmatory_evaluations(
        protocol_path,
        experiment_matrix=args.experiment_matrix,
        validation_spec=args.validation_spec,
        matrix_dir=generated,
        results_root=args.results_root,
    )
    _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
