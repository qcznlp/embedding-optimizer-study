"""Evaluate and audit DenseOn stage-1--4 BEIR retrieval dynamics.

The formal hybrid and confirmatory comparisons intentionally remain final-only.
This extension writes the four earlier stages to disjoint result roots, reuses the
resumable/content-addressed matrix evaluator, and rejects any stage-5 result in
the dynamics roots.  Consequently these rows can describe trajectories without
entering the frozen stage-5 confirmatory inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .aggregate import collect_evaluations
from .config import RunConfig, load_matrix
from .confirmatory_data import load_confirmatory_protocol
from .confirmatory_evaluation import (
    _matrix_paths,
    audit_confirmatory_training,
)
from .confirmatory_matrix import audit_confirmatory_matrices
from .decontamination import DECONTAMINATED_TASK_NAMES
from .evaluate_matrix import _checkpoint_input_identity, checkpoint_paths
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .hybrid_control import LEARNING_RATES, audit_hybrid_training
from .scope import resolve_scope
from .supplemental_training_audit import run_evaluation_after_specialized_audit

DYNAMICS_STAGES = (1, 2, 3, 4)
FINAL_INFERENCE_STAGE = 5
SUITES = ("hybrid", "confirmatory")
EXPECTED_RUNS = {"hybrid": 4, "confirmatory": 9}
EXPECTED_UNITS = {
    suite: runs * len(DYNAMICS_STAGES) * len(DECONTAMINATED_TASK_NAMES)
    for suite, runs in EXPECTED_RUNS.items()
}
TOTAL_EXPECTED_UNITS = sum(EXPECTED_UNITS.values())
CONTRACT_STATUS = "user_directed_dense_retrieval_dynamics_extension"


@dataclass(frozen=True)
class DynamicsContract:
    path: Path
    repository: Path
    payload: dict[str, Any]

    def source_path(self, name: str) -> Path:
        return (self.repository / self.payload["source_bindings"][name]["path"]).resolve()

    def result_root(self, suite: str) -> Path:
        return (self.repository / self.payload["suites"][suite]["results_root"]).resolve()

    def formal_result_root(self, suite: str) -> Path:
        return (self.repository / self.payload["suites"][suite]["formal_results_root"]).resolve()

    def log_root(self, suite: str) -> Path:
        return (self.repository / self.payload["suites"][suite]["log_root"]).resolve()


def _repository_for(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent.resolve()
    raise ValueError(f"Cannot locate repository root for dynamics contract: {path}")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _relative(path: Path, repository: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(repository).as_posix()
        if _inside(resolved, repository)
        else str(resolved)
    )


def load_dynamics_contract(
    path: str | Path = "configs/dense_retrieval_dynamics_extension.json",
) -> DynamicsContract:
    """Load the immutable stage-1--4 contract and verify all source bindings."""

    resolved = Path(path).resolve()
    repository = _repository_for(resolved)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Missing/invalid Dense retrieval-dynamics contract: {resolved}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Dense retrieval-dynamics contract must be a JSON object")

    evaluation = payload.get("evaluation") or {}
    decision_timing = payload.get("decision_timing") or {}
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != CONTRACT_STATUS
        or decision_timing.get("discovery_scores_visible") is not True
        or decision_timing.get("hybrid_training_complete") is not True
        or decision_timing.get("hybrid_beir_scores_visible") is not False
        or decision_timing.get("confirmatory_terminal_runs_visible") is not False
        or decision_timing.get("confirmatory_beir_scores_visible") is not False
        or decision_timing.get("extension_rule_result_contingent") is not False
        or not isinstance(decision_timing.get("reason"), str)
        or not decision_timing["reason"].strip()
        or evaluation.get("families") != ["dense"]
        or evaluation.get("tasks") != list(DECONTAMINATED_TASK_NAMES)
        or evaluation.get("dynamics_stages") != list(DYNAMICS_STAGES)
        or evaluation.get("formal_inference_stage") != FINAL_INFERENCE_STAGE
        or evaluation.get("expected_additional_units") != TOTAL_EXPECTED_UNITS
        or evaluation.get("formal_inference_uses_dynamics_rows") is not False
    ):
        raise ValueError("Dense retrieval-dynamics evaluation contract differs")

    suites = payload.get("suites") or {}
    if set(suites) != set(SUITES):
        raise ValueError("Dense retrieval-dynamics contract has an incomplete suite map")
    formal_roots: set[Path] = set()
    dynamics_roots: set[Path] = set()
    for suite in SUITES:
        item = suites[suite]
        if (
            item.get("runs") != EXPECTED_RUNS[suite]
            or item.get("stages") != list(DYNAMICS_STAGES)
            or item.get("tasks") != len(DECONTAMINATED_TASK_NAMES)
            or item.get("expected_units") != EXPECTED_UNITS[suite]
            or not isinstance(item.get("results_root"), str)
            or not isinstance(item.get("formal_results_root"), str)
            or not isinstance(item.get("log_root"), str)
        ):
            raise ValueError(f"Dense retrieval-dynamics {suite} contract differs")
        dynamics = (repository / item["results_root"]).resolve()
        formal = (repository / item["formal_results_root"]).resolve()
        log_root = (repository / item["log_root"]).resolve()
        if not all(_inside(candidate, repository) for candidate in (dynamics, formal, log_root)):
            raise ValueError(f"Dense retrieval-dynamics {suite} path escapes the repository")
        if dynamics == formal or _inside(dynamics, formal) or _inside(formal, dynamics):
            raise ValueError(f"Dense retrieval-dynamics {suite} and formal roots overlap")
        dynamics_roots.add(dynamics)
        formal_roots.add(formal)
    all_roots = [*dynamics_roots, *formal_roots]
    roots_overlap = any(
        left != right and (_inside(left, right) or _inside(right, left))
        for index, left in enumerate(all_roots)
        for right in all_roots[index + 1 :]
    )
    if (
        len(dynamics_roots) != len(SUITES)
        or len(formal_roots) != len(SUITES)
        or dynamics_roots & formal_roots
        or roots_overlap
    ):
        raise ValueError("Dense retrieval-dynamics result roots are not pairwise isolated")

    bindings = payload.get("source_bindings") or {}
    expected_bindings = {
        "implementation",
        "scope_amendment",
        "hybrid_matrix",
        "hybrid_control",
        "confirmatory_protocol",
        "confirmatory_matrix_manifest",
    }
    if set(bindings) != expected_bindings:
        raise ValueError("Dense retrieval-dynamics source ledger is incomplete")
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"Dense retrieval-dynamics source binding is invalid: {name}")
        source = (repository / str(binding["path"])).resolve()
        if (
            not _inside(source, repository)
            or not source.is_file()
            or _sha256(source) != binding["sha256"]
        ):
            raise ValueError(f"Dense retrieval-dynamics source differs: {name} ({source})")
    return DynamicsContract(resolved, repository, payload)


def coverage_problems(
    rows: Sequence[dict[str, Any]],
    configs: Sequence[RunConfig],
    *,
    stages: Sequence[int] = DYNAMICS_STAGES,
    tasks: Sequence[str] = DECONTAMINATED_TASK_NAMES,
) -> list[str]:
    """Return exact-coverage problems for one isolated dynamics result root."""

    expected = {
        (config.model_family, config.run_id, stage, task)
        for config in configs
        for stage in stages
        for task in tasks
    }
    identities: list[tuple[str, str, int, str]] = []
    problems: list[str] = []
    for row in rows:
        try:
            identity = (
                str(row["model_family"]),
                str(row["run_id"]),
                int(row["stage"]),
                str(row["task"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"invalid dynamics row identity ({error})")
            continue
        identities.append(identity)
        if identity[2] == FINAL_INFERENCE_STAGE:
            problems.append(f"formal stage 5 leaked into dynamics results: {identity}")
    counts = Counter(identities)
    duplicates = sorted(identity for identity, count in counts.items() if count != 1)
    missing = sorted(expected - set(counts))
    unexpected = sorted(set(counts) - expected)
    if duplicates:
        problems.append(f"duplicate dynamics identities: {duplicates[:3]}")
    if missing:
        problems.append(f"missing dynamics identities: {missing[:3]}")
    if unexpected:
        problems.append(f"unexpected dynamics identities: {unexpected[:3]}")
    if len(rows) != len(expected):
        problems.append(f"dynamics coverage is {len(rows)}/{len(expected)} units")
    return problems


def _audit_input_manifest(
    root: Path,
    configs: Sequence[RunConfig],
    *,
    repository: Path,
    stages: Sequence[int] = DYNAMICS_STAGES,
) -> dict[str, Any]:
    """Re-hash every expected checkpoint and match the evaluator's cache ledger."""

    path = root / "evaluation_inputs.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing/invalid dynamics input manifest: {path}") from error
    if payload.get("schema_version") != 1 or not isinstance(payload.get("checkpoints"), dict):
        raise ValueError(f"Invalid dynamics input manifest schema: {path}")
    checkpoints = [
        checkpoint for config in configs for checkpoint in checkpoint_paths(config, list(stages))
    ]
    expected_keys = {str(checkpoint.resolve()) for checkpoint in checkpoints}
    recorded = payload["checkpoints"]
    if set(recorded) != expected_keys:
        raise ValueError(
            f"Dynamics input manifest covers {len(recorded)}/{len(expected_keys)} checkpoints: {path}"
        )
    for checkpoint in checkpoints:
        key = str(checkpoint.resolve())
        if recorded[key] != _checkpoint_input_identity(checkpoint):
            raise ValueError(f"Dynamics checkpoint content differs from input manifest: {key}")
    return {
        "path": _relative(path, repository),
        "sha256": _sha256(path),
        "checkpoints": len(checkpoints),
    }


def _audit_result_root(
    root: Path,
    configs: Sequence[RunConfig],
    *,
    label: str,
    repository: Path,
    verify_results: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    input_manifest: dict[str, Any] | None = None
    runtime_manifest: dict[str, Any] | None = None
    try:
        rows = collect_evaluations(root, list(configs))
        errors.extend(coverage_problems(rows, configs))
        candidates = {path.resolve() for path in root.rglob("*Decontaminated.json")}
        selected = {Path(str(row["result_path"])).resolve() for row in rows}
        if candidates != selected:
            errors.append(
                f"unrecognized dynamics result files: observed={len(candidates)} selected={len(selected)}"
            )
        input_manifest = _audit_input_manifest(root, configs, repository=repository)
        runtime_path = root / "evaluation_runtime.json"
        runtime_manifest = {
            "path": _relative(runtime_path, repository),
            "sha256": _sha256(runtime_path),
            "bytes": runtime_path.stat().st_size,
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    if verify_results and errors:
        raise ValueError(f"{label} dynamics audit failed: {'; '.join(errors[:10])}")
    sources = []
    if not errors:
        for row in rows:
            path = Path(str(row["result_path"])).resolve()
            sources.append(
                {
                    "path": _relative(path, repository),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {
        "complete": not errors,
        "valid_units": len(rows) if not errors else 0,
        "expected_units": len(configs) * len(DYNAMICS_STAGES) * len(DECONTAMINATED_TASK_NAMES),
        "errors": errors,
        "input_manifest": input_manifest,
        "runtime_manifest": runtime_manifest,
        "result_sources": sorted(sources, key=lambda item: item["path"]),
    }


def _hybrid_configs(contract: DynamicsContract) -> list[RunConfig]:
    configs = [
        config
        for config in load_matrix(contract.source_path("hybrid_matrix"))
        if config.model_family == "dense"
    ]
    if (
        len(configs) != EXPECTED_RUNS["hybrid"]
        or {config.optimizer.name for config in configs} != {"hybrid_adamw"}
        or {config.optimizer.lr for config in configs} != set(LEARNING_RATES)
    ):
        raise ValueError("Hybrid dynamics selection differs from the DenseOn 4-run control")
    return configs


def _confirmatory_context(
    contract: DynamicsContract,
) -> tuple[Path, dict[str, Any], dict[int, Path], str]:
    protocol_path, protocol = load_confirmatory_protocol(
        contract.source_path("confirmatory_protocol")
    )
    matrix_dir = contract.source_path("confirmatory_matrix_manifest").parent
    matrix_audit = audit_confirmatory_matrices(
        protocol_path,
        experiment_matrix=contract.repository / "configs/experiment.yaml",
        validation_spec=contract.repository / "configs/validation_probe.json",
        output_dir=matrix_dir,
    )
    return (
        protocol_path,
        protocol,
        _matrix_paths(protocol, matrix_dir),
        matrix_audit["manifest_sha256"],
    )


def _confirmatory_configs(matrix_path: Path) -> list[RunConfig]:
    configs = [config for config in load_matrix(matrix_path) if config.model_family == "dense"]
    if len(configs) != 3 or {config.optimizer.name for config in configs} != {
        "adamw",
        "muon",
        "normuon",
    }:
        raise ValueError(f"Confirmatory dynamics selection differs: {matrix_path}")
    return configs


def _selected_suites(value: str | Iterable[str]) -> tuple[str, ...]:
    selected = (
        SUITES if value == "all" else tuple(value) if not isinstance(value, str) else (value,)
    )
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(SUITES):
        raise ValueError(f"Invalid dynamics suite selection: {selected}")
    return tuple(suite for suite in SUITES if suite in selected)


def audit_dense_retrieval_dynamics(
    contract_path: str | Path = "configs/dense_retrieval_dynamics_extension.json",
    *,
    suites: str | Iterable[str] = "all",
    verify_results: bool = True,
) -> dict[str, Any]:
    """Strictly audit stage-1--4 coverage, provenance, and inference isolation."""

    contract = load_dynamics_contract(contract_path)
    selected = _selected_suites(suites)
    families, scope = resolve_scope(("dense",), contract.source_path("scope_amendment"))
    if families != ("dense",):
        raise ValueError("Dense retrieval dynamics unexpectedly escaped DenseOn-only scope")

    suite_audits: dict[str, Any] = {}
    matrix_manifest_sha256: str | None = None
    if "hybrid" in selected:
        configs = _hybrid_configs(contract)
        suite_audits["hybrid"] = _audit_result_root(
            contract.result_root("hybrid"),
            configs,
            label="hybrid",
            repository=contract.repository,
            verify_results=verify_results,
        )
    if "confirmatory" in selected:
        _, _, matrix_paths, matrix_manifest_sha256 = _confirmatory_context(contract)
        per_seed: dict[str, Any] = {}
        result_sources: list[dict[str, Any]] = []
        input_manifests: list[dict[str, Any]] = []
        valid_units = 0
        errors: list[str] = []
        for seed, matrix_path in matrix_paths.items():
            audit = _audit_result_root(
                contract.result_root("confirmatory") / f"seed{seed}",
                _confirmatory_configs(matrix_path),
                label=f"confirmatory seed {seed}",
                repository=contract.repository,
                verify_results=verify_results,
            )
            per_seed[str(seed)] = audit
            valid_units += int(audit["valid_units"])
            errors.extend(audit["errors"])
            result_sources.extend({"seed": seed, **item} for item in audit["result_sources"])
            if audit["input_manifest"] is not None:
                input_manifests.append({"seed": seed, **audit["input_manifest"]})
        complete = (
            not errors
            and valid_units == EXPECTED_UNITS["confirmatory"]
            and len(per_seed) == 3
            and all(item["complete"] for item in per_seed.values())
        )
        if verify_results and not complete:
            raise ValueError(
                f"Confirmatory dynamics coverage is {valid_units}/{EXPECTED_UNITS['confirmatory']}"
            )
        suite_audits["confirmatory"] = {
            "complete": complete,
            "valid_units": valid_units,
            "expected_units": EXPECTED_UNITS["confirmatory"],
            "errors": errors,
            "per_seed": per_seed,
            "input_manifests": input_manifests,
            "result_sources": sorted(result_sources, key=lambda item: (item["seed"], item["path"])),
        }

    valid_units = sum(int(suite_audits[suite]["valid_units"]) for suite in selected)
    expected_units = sum(EXPECTED_UNITS[suite] for suite in selected)
    complete = valid_units == expected_units and all(
        suite_audits[suite]["complete"] for suite in selected
    )
    if verify_results and not complete:
        raise ValueError(f"Dense retrieval-dynamics coverage is {valid_units}/{expected_units}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "families": ["dense"],
        "selected_suites": list(selected),
        "tasks": list(DECONTAMINATED_TASK_NAMES),
        "dynamics_stages": list(DYNAMICS_STAGES),
        "formal_inference_stage": FINAL_INFERENCE_STAGE,
        "formal_inference_uses_dynamics_rows": False,
        "expected_units": expected_units,
        "valid_units": valid_units,
        "contract": {
            "path": _relative(contract.path, contract.repository),
            "sha256": _sha256(contract.path),
        },
        "scope_amendment": scope,
        "confirmatory_matrix_manifest_sha256": matrix_manifest_sha256,
        "suites": suite_audits,
    }


def _worker_args(
    args: argparse.Namespace,
    contract: DynamicsContract,
    *,
    matrix: Path,
    results_root: Path,
    log_dir: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        matrix=str(matrix.resolve()),
        families=["dense"],
        run_ids=[],
        stages=list(DYNAMICS_STAGES),
        tasks=list(DECONTAMINATED_TASK_NAMES),
        gpus_a=args.gpus_a,
        gpus_b=args.gpus_b,
        late_port_a=29710,
        late_port=29720,
        results_root=str(results_root.resolve()),
        log_dir=str(log_dir.resolve()),
        worker_python=args.worker_python or sys.executable,
        scope_amendment=contract.source_path("scope_amendment"),
        gpu_lock_dir=args.gpu_lock_dir,
        gpu_lock_timeout_seconds=args.gpu_lock_timeout_seconds,
    )


def _run_hybrid(args: argparse.Namespace, contract: DynamicsContract) -> None:
    configs = _hybrid_configs(contract)
    training_audit = audit_hybrid_training(configs, ("dense",))
    worker_args = _worker_args(
        args,
        contract,
        matrix=contract.source_path("hybrid_matrix"),
        results_root=contract.result_root("hybrid"),
        log_dir=contract.log_root("hybrid"),
    )
    failures = run_evaluation_after_specialized_audit(
        worker_args,
        training_audit,
        label="DenseOn hybrid stage-1--4 dynamics",
    )
    if failures:
        raise RuntimeError(f"Hybrid dynamics evaluator reported {failures} failed subprocesses")


def _run_confirmatory(args: argparse.Namespace, contract: DynamicsContract) -> None:
    protocol_path, _, matrix_paths, _ = _confirmatory_context(contract)
    for seed, matrix_path in matrix_paths.items():
        configs = _confirmatory_configs(matrix_path)
        training_audit = audit_confirmatory_training(
            protocol_path,
            seed,
            configs,
            ("dense",),
        )
        worker_args = _worker_args(
            args,
            contract,
            matrix=matrix_path,
            results_root=contract.result_root("confirmatory") / f"seed{seed}",
            log_dir=contract.log_root("confirmatory") / f"seed{seed}",
        )
        failures = run_evaluation_after_specialized_audit(
            worker_args,
            training_audit,
            label=f"DenseOn confirmatory seed {seed} stage-1--4 dynamics",
        )
        if failures:
            raise RuntimeError(
                f"Confirmatory seed {seed} dynamics evaluator reported "
                f"{failures} failed subprocesses"
            )


def _default_receipt(repository: Path, suite: str) -> Path:
    name = "evaluation-receipt.json" if suite == "all" else f"{suite}-evaluation-receipt.json"
    return repository / "reports/dense-retrieval-dynamics" / name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/dense_retrieval_dynamics_extension.json"),
    )
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    parser.add_argument("--gpus-a", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--worker-python", default=None)
    parser.add_argument(
        "--gpu-lock-dir",
        type=Path,
        default=Path("logs/dense-only-runtime/gpu-leases"),
    )
    parser.add_argument("--gpu-lock-timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.gpu_lock_timeout_seconds <= 0:
        parser.error("--gpu-lock-timeout-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    contract = load_dynamics_contract(args.contract)
    selected = _selected_suites(args.suite)
    if not args.audit_only:
        if "hybrid" in selected:
            _run_hybrid(args, contract)
        if "confirmatory" in selected:
            _run_confirmatory(args, contract)
    receipt = audit_dense_retrieval_dynamics(
        contract.path,
        suites=selected,
        verify_results=True,
    )
    receipt_path = (
        args.receipt.resolve()
        if args.receipt is not None
        else _default_receipt(contract.repository, args.suite)
    )
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
