from __future__ import annotations

import gc
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from .aggregate import (
    _deep_checkpoint_problems,
    _scheduler_contract_problem,
    _training_arguments_problem,
    audit_training_artifacts,
)
from .config import RunConfig
from .geometry import _sha256
from .hybrid_control import _hybrid_optimizer_contract_problem


def run_evaluation_after_specialized_audit(
    args: Namespace,
    audit: dict[str, Any],
    *,
    label: str,
) -> int:
    """Run the locked evaluator after a stronger specialized training audit.

    The locked evaluator owns all formal-runtime, worker-source, package-version,
    checkpoint-selection, and subprocess checks. Its generic training preflight
    predates hybrid AdamW and the derived-dataset manifest schema, so a caller
    may replace *only* that preflight after proving the corresponding specialized
    audit complete. The original validator is restored even if evaluation fails.
    """

    if audit.get("complete") is not True or audit.get("errors"):
        details = "; ".join(str(error) for error in audit.get("errors", [])[:10])
        raise RuntimeError(f"{label} training preflight failed: {details or 'incomplete audit'}")

    from . import evaluate_matrix

    original = evaluate_matrix._validate_training_inputs
    preflight_consumed = False

    def specialized_preflight(selected_args: Namespace) -> None:
        nonlocal preflight_consumed
        if selected_args is not args:
            raise RuntimeError(f"{label} evaluator arguments changed after training audit")
        if preflight_consumed:
            raise RuntimeError(f"{label} specialized training preflight was consumed twice")
        preflight_consumed = True
        print(
            f"{label} specialized training preflight: "
            f"{audit.get('verified_runs', 0)} runs / "
            f"{audit.get('verified_checkpoints', 0)} checkpoints deep-validated",
            flush=True,
        )

    evaluate_matrix._validate_training_inputs = specialized_preflight
    try:
        result = evaluate_matrix.run_evaluation(args)
    finally:
        evaluate_matrix._validate_training_inputs = original
    if not preflight_consumed:
        raise RuntimeError(f"{label} evaluator did not consume its specialized training preflight")
    return result


def _load_json(path: Path, *, context: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{context}: missing/invalid JSON ({error})")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{context}: expected a JSON object")
        return None
    return payload


def _audit_hybrid_checkpoints(config: RunConfig) -> tuple[int, list[str]]:
    """Deep-audit one hybrid-routed AdamW run with its three-group contract."""

    label = f"{config.model_family}/{config.run_id}"
    schedule_path = config.output_dir / "checkpoint_schedule.json"
    try:
        steps = [int(step) for step in json.loads(schedule_path.read_text())["steps"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return 0, [f"{label}: invalid checkpoint schedule for hybrid audit ({error})"]
    if len(steps) != 5 or steps != sorted(set(steps)):
        return 0, [f"{label}: hybrid audit requires five increasing checkpoint steps"]

    errors: list[str] = []
    verified = 0
    final_step = steps[-1]
    for step in steps:
        checkpoint = config.output_dir / f"checkpoint-{step}"
        problems = hybrid_checkpoint_problems(
            checkpoint,
            config,
            step,
            final_step,
            world_size=4,
        )
        checkpoint_label = f"{label}/checkpoint-{step}"
        if problems:
            errors.extend(f"{checkpoint_label}: {problem}" for problem in problems)
        else:
            verified += 1
    return verified, errors


def hybrid_checkpoint_problems(
    checkpoint: Path,
    config: RunConfig,
    expected_step: int,
    final_step: int,
    *,
    world_size: int,
) -> list[str]:
    """Deep-audit one hybrid AdamW checkpoint with its three-group contract."""

    problems = _deep_checkpoint_problems(checkpoint, expected_step, world_size)
    optimizer = None
    try:
        optimizer = torch.load(
            checkpoint / "optimizer.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if problem := _hybrid_optimizer_contract_problem(
            optimizer, config, expected_step, final_step
        ):
            problems.append(problem)
    except Exception as error:  # noqa: BLE001
        problems.append(f"hybrid optimizer contract load failed ({type(error).__name__}: {error})")
    finally:
        del optimizer
        gc.collect()
    try:
        scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
        if problem := _scheduler_contract_problem(scheduler, config, expected_step, final_step):
            problems.append(problem)
    except Exception as error:  # noqa: BLE001
        problems.append(f"scheduler contract load failed ({type(error).__name__}: {error})")
    if problem := _training_arguments_problem(
        checkpoint / "training_args.bin", config, world_size, final_step
    ):
        problems.append(problem)
    return problems


def _append_unique(target: list[str], additions: list[str]) -> None:
    seen = set(target)
    for error in additions:
        if error not in seen:
            target.append(error)
            seen.add(error)


def audit_derived_training_artifacts(
    configs: list[RunConfig],
    dataset_receipt: dict[str, Any],
    *,
    deep: bool = True,
) -> dict[str, Any]:
    """Bind deep training validation to a specialized derived-dataset receipt.

    Confirmatory views and the 50K shared-start branch intentionally use a
    provenance-rich manifest schema with ``rows`` instead of the discovery
    manifest's ``total_queries``. The shared checkpoint auditor predates that
    schema and therefore emits one predictable row-count error per otherwise
    valid run. This adapter removes only that schema-only error after independently
    verifying the derived manifest hash, copied manifest, exact row count, and
    training-view fingerprint.
    """

    errors: list[str] = []
    rows = dataset_receipt.get("rows")
    fingerprint = dataset_receipt.get("training_view_fingerprint")
    manifest_sha256 = dataset_receipt.get("manifest_sha256")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        errors.append("derived dataset receipt has an invalid row count")
    if not isinstance(fingerprint, str) or not fingerprint:
        errors.append("derived dataset receipt has an invalid training-view fingerprint")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        errors.append("derived dataset receipt has an invalid manifest SHA-256")
    if errors:
        return {
            "complete": False,
            "verified_runs": 0,
            "expected_runs": len(configs),
            "verified_checkpoints": 0,
            "expected_checkpoints": len(configs) * 5,
            "deep_validation": deep,
            "errors": errors,
        }

    generic = audit_training_artifacts(
        configs,
        deep=False,
        expected_dataset_fingerprint=fingerprint,
    )
    remaining = list(generic["errors"])
    verified_checkpoints = int(generic.get("verified_checkpoints", 0))
    if deep:
        verified_checkpoints = 0
        native = [config for config in configs if config.optimizer.name != "hybrid_adamw"]
        if native:
            native_audit = audit_training_artifacts(
                native,
                deep=True,
                expected_dataset_fingerprint=fingerprint,
            )
            _append_unique(remaining, list(native_audit["errors"]))
            verified_checkpoints += int(native_audit.get("verified_checkpoints", 0))
        for config in configs:
            if config.optimizer.name == "hybrid_adamw":
                verified, hybrid_errors = _audit_hybrid_checkpoints(config)
                verified_checkpoints += verified
                _append_unique(remaining, hybrid_errors)
    for config in configs:
        label = f"{config.model_family}/{config.run_id}"
        source_manifest_path = Path(config.dataset_path) / "manifest.json"
        copied_manifest_path = config.output_dir / "dataset_manifest.json"
        completion_path = config.output_dir / "completed.json"
        local_errors: list[str] = []
        source = _load_json(
            source_manifest_path,
            context=f"{label}: derived source manifest",
            errors=local_errors,
        )
        copied = _load_json(
            copied_manifest_path,
            context=f"{label}: copied derived manifest",
            errors=local_errors,
        )
        completed = _load_json(
            completion_path,
            context=f"{label}: completion receipt",
            errors=local_errors,
        )
        if source is not None:
            if _sha256(source_manifest_path) != manifest_sha256:
                local_errors.append(f"{label}: derived source manifest differs from its receipt")
            if source.get("rows") != rows or "total_queries" in source:
                local_errors.append(f"{label}: derived source manifest row schema differs")
        if source is not None and copied is not None and copied != source:
            local_errors.append(f"{label}: copied derived manifest differs from source")
        if completed is not None:
            if completed.get("dataset_rows") != rows:
                local_errors.append(f"{label}: completion row count differs from derived dataset")
            if completed.get("dataset_fingerprint") != fingerprint:
                local_errors.append(
                    f"{label}: completion training-view fingerprint differs from derived dataset"
                )

        schema_error = f"{label}: completion dataset row count does not match manifest"
        if not local_errors and schema_error in remaining:
            remaining.remove(schema_error)
        errors.extend(local_errors)

    errors = [*remaining, *errors]
    expected_checkpoints = len(configs) * 5
    if verified_checkpoints != expected_checkpoints and not errors:
        errors.append(
            f"derived training deep audit verified {verified_checkpoints} checkpoints, "
            f"expected {expected_checkpoints}"
        )
    errored_runs = {
        f"{config.model_family}/{config.run_id}"
        for config in configs
        if any(
            error.startswith(f"{config.model_family}/{config.run_id}:")
            or error.startswith(f"{config.model_family}/{config.run_id}/")
            for error in errors
        )
    }
    result = dict(generic)
    result["errors"] = errors
    result["complete"] = not errors
    result["verified_runs"] = max(0, len(configs) - len(errored_runs))
    result["verified_checkpoints"] = verified_checkpoints
    result["expected_checkpoints"] = expected_checkpoints
    result["deep_validation"] = deep
    result["derived_dataset_rows"] = rows
    result["derived_training_view_fingerprint"] = fingerprint
    return result
