"""Aggregate checkpoint-level MTEB results and render the final study report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .collators import TEXT_COLUMNS
from .config import MUON_NS_IMPLEMENTATION, RunConfig, load_matrix, resolve_matrix_path
from .data import SOURCE_REPO, SOURCE_REVISION, SPLITS
from .decontamination import DECONTAMINATED_BEIR, DECONTAMINATED_TASK_NAMES
from .scope import ALL_FAMILIES, resolve_scope, select_family_configs

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
EVALUATION_PACKAGES = {
    "mteb",
    "torch",
    "sentence-transformers",
    "flash-attn",
    "transformers",
    "pylate",
    "fast-plaid",
    "late-interaction-kernels",
}
EXPECTED_SWEEP = {
    "adamw": {
        "adamw-lr1e-6": 1e-6,
        "adamw-lr3e-6": 3e-6,
        "adamw-lr1e-5": 1e-5,
        "adamw-lr3e-5": 3e-5,
    },
    "muon": {
        "muon-lr1e-4": 1e-4,
        "muon-lr3e-4": 3e-4,
        "muon-lr1e-3": 1e-3,
        "muon-lr3e-3": 3e-3,
    },
    "normuon": {
        "normuon-lr1e-4": 1e-4,
        "normuon-lr3e-4": 3e-4,
        "normuon-lr1e-3": 1e-3,
        "normuon-lr3e-3": 3e-3,
    },
}
EXPECTED_MODELS = {
    "dense": (
        "lightonai/DenseOn-unsupervised",
        "0edbd55684eb782bce55ee74c95b25c97cbe7f43",
        0.02,
    ),
    "late": (
        "lightonai/LateOn-unsupervised",
        "1047071849a708b9b3ee4dccdc60186c185224a7",
        0.001,
    ),
}


def audit_experiment_contract(configs: list[RunConfig]) -> dict:
    """Verify that the matrix still represents the complete, frozen user request."""

    errors: list[str] = []
    expected_identities = {
        (family, run_id)
        for family in EXPECTED_MODELS
        for runs in EXPECTED_SWEEP.values()
        for run_id in runs
    }
    observed_identities = [(config.model_family, config.run_id) for config in configs]
    if len(observed_identities) != len(set(observed_identities)):
        errors.append("matrix contains duplicate family/run identities")
    if set(observed_identities) != expected_identities:
        errors.append("matrix does not contain the exact planned 24 family/run identities")

    common_expected = {
        "dataset_path": "data/denseon-sft-500k-seed42",
        "seed": 42,
        "epochs": 1.0,
        "global_batch_size": 128,
        "micro_batch_size": 8,
        "max_length": 8192,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "flash_attention": True,
        "wandb_project": "embedding-optimizer-study",
        "wandb_entity": "stevezenguom",
        "checkpoint_fractions": (0.2, 0.4, 0.6, 0.8, 1.0),
    }
    for config in configs:
        label = f"{config.model_family}/{config.run_id}"
        if config.model_family not in EXPECTED_MODELS:
            errors.append(f"{label}: unexpected model family")
            continue
        expected_model, expected_revision, expected_temperature = EXPECTED_MODELS[
            config.model_family
        ]
        if config.model_name != expected_model or config.model_revision != expected_revision:
            errors.append(f"{label}: base model identity/revision differs from frozen contract")
        if config.resolved_temperature != expected_temperature:
            errors.append(f"{label}: contrastive temperature differs from frozen contract")
        for field, expected in common_expected.items():
            if getattr(config, field) != expected:
                errors.append(f"{label}: {field} differs from frozen contract")

        optimizer = config.optimizer
        expected_runs = EXPECTED_SWEEP.get(optimizer.name)
        if expected_runs is None or expected_runs.get(config.run_id) != optimizer.lr:
            errors.append(f"{label}: optimizer name/learning rate differs from frozen sweep")
        if optimizer.weight_decay != 0.01:
            errors.append(f"{label}: weight decay differs from frozen contract")
        if optimizer.name == "adamw":
            if (optimizer.beta1, optimizer.beta2, optimizer.eps) != (0.9, 0.999, 1e-8):
                errors.append(f"{label}: AdamW moments/epsilon differ from frozen contract")
        elif optimizer.name in {"muon", "normuon"}:
            if (
                optimizer.aux_lr != 3e-6
                or (optimizer.aux_beta1, optimizer.aux_beta2, optimizer.aux_eps)
                != (0.9, 0.999, 1e-8)
                or optimizer.momentum != 0.95
                or optimizer.ns_steps != 5
            ):
                errors.append(f"{label}: matrix/auxiliary optimizer settings differ")
            if optimizer.name == "muon" and optimizer.adjust_lr_fn != "original":
                errors.append(f"{label}: Muon LR adjustment differs from frozen contract")
            if optimizer.name == "normuon" and optimizer.normuon_beta2 != 0.95:
                errors.append(f"{label}: NorMuon beta2 differs from frozen contract")
    return {
        "complete": not errors,
        "observed_runs": len(configs),
        "expected_runs": 24,
        "errors": errors,
    }


def _dataset_rows_audit(path: Path, manifest: dict) -> dict:
    """Stream and validate the canonical 500k row manifest without loading texts."""

    errors: list[str] = []
    checksum = hashlib.sha256()
    source_counts: Counter[str] = Counter()
    seen_queries: set[tuple[str, int]] = set()
    row_count = 0

    def record(message: str) -> None:
        if len(errors) < 25:
            errors.append(message)

    try:
        handle = path.open()
    except OSError as error:
        return {"errors": [f"missing/unreadable rows.jsonl ({error})"], "rows": 0}
    with handle:
        for index, line in enumerate(handle):
            row_count = index + 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                record(f"row {index}: invalid JSON ({error})")
                continue
            checksum.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
            checksum.update(b"\n")
            expected_keys = {
                "sample_id",
                "source",
                "query_id",
                "positive_id",
                "negative_ids",
                "negative_pool_indices",
            }
            if not isinstance(row, dict) or set(row) != expected_keys:
                record(f"row {index}: unexpected fields")
                continue
            source = row["source"]
            query_id = row["query_id"]
            if (
                isinstance(row["sample_id"], bool)
                or not isinstance(row["sample_id"], int)
                or row["sample_id"] != index
            ):
                record(f"row {index}: sample_id is not its canonical position")
            if source not in SPLITS or isinstance(query_id, bool) or not isinstance(query_id, int):
                record(f"row {index}: invalid source/query identity")
            else:
                identity = (source, query_id)
                if identity in seen_queries:
                    record(f"row {index}: duplicate source/query identity")
                seen_queries.add(identity)
                source_counts[source] += 1
            negatives = row["negative_ids"]
            pool_indices = row["negative_pool_indices"]
            if (
                not isinstance(negatives, list)
                or len(negatives) != 7
                or any(isinstance(value, bool) or not isinstance(value, int) for value in negatives)
                or len(set(negatives)) != 7
                or isinstance(row["positive_id"], bool)
                or not isinstance(row["positive_id"], int)
                or row["positive_id"] in negatives
            ):
                record(f"row {index}: expected seven distinct negatives excluding the positive")
            if (
                not isinstance(pool_indices, list)
                or len(pool_indices) != 7
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 10
                    for value in pool_indices
                )
                or pool_indices != sorted(set(pool_indices))
            ):
                record(f"row {index}: invalid seven-of-ten negative sample indices")
    digest = checksum.hexdigest()
    if row_count != manifest.get("total_queries"):
        record(f"row count is {row_count}, expected {manifest.get('total_queries')}")
    if dict(source_counts) != manifest.get("quotas"):
        record("observed source counts differ from quotas")
    if digest != manifest.get("row_manifest_sha256"):
        record("canonical row-manifest SHA-256 differs")
    return {
        "errors": errors,
        "rows": row_count,
        "unique_source_queries": len(seen_queries),
        "source_counts": dict(source_counts),
        "row_manifest_sha256": digest,
    }


def audit_dataset_artifacts(configs: list[RunConfig]) -> dict:
    """Prove that every run points at one valid, pinned 500k training dataset."""

    roots = sorted({Path(config.dataset_path).resolve() for config in configs})
    errors: list[str] = []
    if len(roots) != 1:
        errors.append(f"expected one shared dataset path, found {len(roots)}")
        return {"complete": False, "verified_rows": 0, "errors": errors}
    root = roots[0]
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {
            "complete": False,
            "dataset_path": str(root),
            "verified_rows": 0,
            "errors": [f"missing/invalid source manifest ({error})"],
        }

    expected_scalars = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "seed": 42,
        "total_queries": 500_000,
        "nv_threshold": 0.95,
        "negative_pool_size": 10,
        "sampled_negatives": 7,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            errors.append(f"manifest {key} is {manifest.get(key)!r}, expected {expected!r}")
    quotas = manifest.get("quotas")
    scorable_counts = manifest.get("scorable_query_counts")
    if (
        not isinstance(quotas, dict)
        or set(quotas) != set(SPLITS)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in quotas.values())
        or sum(quotas.values()) != 500_000
    ):
        errors.append("manifest quotas do not cover exactly 500,000 rows across seven sources")
    if (
        not isinstance(scorable_counts, dict)
        or set(scorable_counts) != set(SPLITS)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in scorable_counts.values()
        )
    ):
        errors.append("manifest scorable query counts do not cover the seven sources")
    elif (
        isinstance(quotas, dict)
        and set(quotas) == set(SPLITS)
        and any(
            isinstance(quotas[source], int) and quotas[source] > scorable_counts[source]
            for source in SPLITS
        )
    ):
        errors.append("a source quota exceeds its scorable query count")
    for key in (
        "row_manifest_sha256",
        "materialized_dataset_fingerprint",
        "dataset_fingerprint",
    ):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            errors.append(f"manifest {key} is missing/invalid")

    row_audit = _dataset_rows_audit(root / "rows.jsonl", manifest)
    errors.extend(row_audit["errors"])
    dataset_path = root / "dataset"
    training_view_fingerprint = None
    try:
        from datasets import Dataset

        dataset = Dataset.load_from_disk(str(dataset_path))
    except Exception as error:  # noqa: BLE001
        errors.append(f"missing/invalid materialized Dataset ({error})")
    else:
        expected_columns = {
            "sample_id",
            "source",
            "query_id",
            "positive_id",
            "query",
            "positive",
            "length",
            *(f"negative_{index}" for index in range(7)),
            *(f"negative_{index}_id" for index in range(7)),
        }
        if len(dataset) != 500_000:
            errors.append(f"materialized Dataset has {len(dataset)} rows, expected 500000")
        if set(dataset.column_names) != expected_columns:
            errors.append("materialized Dataset has unexpected columns")
        if dataset._fingerprint != manifest.get("dataset_fingerprint"):
            errors.append("materialized Dataset fingerprint differs from manifest")
        try:
            training_view_fingerprint = dataset.select_columns(
                [*TEXT_COLUMNS, "length"]
            )._fingerprint
        except Exception as error:  # noqa: BLE001
            errors.append(f"could not construct the fixed training dataset view ({error})")
    return {
        "complete": not errors,
        "dataset_path": str(root),
        "verified_rows": row_audit["rows"] if not row_audit["errors"] else 0,
        "row_manifest_sha256": row_audit.get("row_manifest_sha256"),
        "training_view_fingerprint": training_view_fingerprint,
        "errors": errors,
    }


def _trainer_state_problem(path: Path, expected_step: int) -> str | None:
    try:
        state = json.loads(path.read_text())
        global_step = int(state["global_step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return f"invalid Trainer state ({error})"
    if global_step != expected_step:
        return f"Trainer state step is {global_step}, expected {expected_step}"

    history = [item for item in state.get("log_history", []) if "loss" in item]
    if not history:
        return "Trainer state has no loss history"
    try:
        history_steps = [int(item["step"]) for item in history]
    except (KeyError, TypeError, ValueError) as error:
        return f"invalid loss-history step ({error})"
    if history_steps != sorted(set(history_steps)):
        return "loss-history steps are duplicated or non-monotonic"
    if history_steps[-1] > expected_step:
        return f"loss history extends past checkpoint step {expected_step}"
    for item in history:
        for key in ("loss", "grad_norm", "learning_rate", "epoch"):
            if key not in item:
                continue
            try:
                value = float(item[key])
            except (TypeError, ValueError):
                return f"loss-history {key} is not numeric at step {item['step']}"
            if not math.isfinite(value):
                return f"loss-history {key} is non-finite at step {item['step']}"
    return None


def _completion_system_problems(completed: dict, steps: list[int]) -> list[str]:
    problems: list[str] = []
    metrics = completed.get("system_metrics")
    if not isinstance(metrics, dict):
        return ["missing/invalid completion system metrics"]
    for key in (
        "wall_time_seconds_max_rank",
        "peak_allocated_bytes_max_rank",
        "peak_reserved_bytes_max_rank",
    ):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"system metric {key} is missing/non-numeric")
        elif not math.isfinite(value) or value <= 0:
            problems.append(f"system metric {key} is non-finite/non-positive")

    trainer = metrics.get("trainer")
    if not isinstance(trainer, dict):
        problems.append("missing/invalid Trainer performance metrics")
    else:
        for key in (
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "epoch",
        ):
            value = trainer.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"Trainer performance metric {key} is missing/non-numeric")
            elif not math.isfinite(value) or (key != "train_loss" and value <= 0):
                problems.append(f"Trainer performance metric {key} is non-finite/non-positive")

    checkpoint_names = {f"checkpoint-{step}" for step in steps}
    for key in ("checkpoint_bytes", "optimizer_state_bytes"):
        sizes = metrics.get(key)
        if not isinstance(sizes, dict) or set(sizes) != checkpoint_names:
            problems.append(f"system metric {key} does not cover all five checkpoints")
        elif any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in sizes.values()
        ):
            problems.append(f"system metric {key} contains a non-positive/invalid size")

    if not isinstance(metrics.get("gpu_name"), str) or not metrics["gpu_name"].strip():
        problems.append("completion GPU name is missing/invalid")
    expected_versions = {
        "torch",
        "transformers",
        "sentence-transformers",
        "pylate",
        "late-interaction-kernels",
    }
    versions = completed.get("versions")
    if not isinstance(versions, dict) or set(versions) != expected_versions:
        problems.append("completion package versions are missing/incomplete")
    elif any(not isinstance(value, str) or not value.strip() for value in versions.values()):
        problems.append("completion package versions contain an invalid value")
    return problems


def _safetensors_problem(root: Path) -> str | None:
    """Validate every safetensors tensor, including finite floating-point values."""

    from safetensors import safe_open

    files = sorted(root.rglob("*.safetensors"))
    if not files:
        return "missing safetensors model"
    tensor_count = 0
    try:
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if not keys:
                    return f"empty safetensors payload {path.name}"
                for key in keys:
                    shape = handle.get_slice(key).get_shape()
                    if any(not isinstance(size, int) or size < 0 for size in shape):
                        return f"invalid tensor shape in {path.name}:{key}"
                    tensor = handle.get_tensor(key)
                    if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
                        tensor.isfinite().all()
                    ):
                        return f"non-finite tensor in {path.name}:{key}"
                    del tensor
                tensor_count += len(keys)
    except Exception as error:  # noqa: BLE001
        return f"invalid safetensors payload ({type(error).__name__}: {error})"
    return None if tensor_count else "safetensors model has no tensors"


def _safetensors_digest(root: Path) -> str:
    """Hash the complete model payload so repeated checkpoint weights are detectable."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.safetensors")):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _optimizer_state_problem(optimizer: object) -> str | None:
    """Validate optimizer topology and reject silently non-finite state."""

    import torch

    if (
        not isinstance(optimizer, dict)
        or set(optimizer) != {"state", "param_groups"}
        or not isinstance(optimizer["state"], dict)
        or not optimizer["state"]
        or not isinstance(optimizer["param_groups"], list)
        or not optimizer["param_groups"]
    ):
        return "optimizer state has an invalid structure"

    parameter_ids: list[object] = []
    for group in optimizer["param_groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            return "optimizer parameter groups have an invalid structure"
        parameter_ids.extend(group["params"])
    try:
        grouped_parameters = set(parameter_ids)
    except TypeError:
        return "optimizer parameter groups contain an invalid parameter identifier"
    if len(grouped_parameters) != len(parameter_ids):
        return "optimizer parameter groups contain duplicate parameters"
    if not set(optimizer["state"]).issubset(grouped_parameters):
        return "optimizer state contains a parameter outside its groups"

    for parameter_id, state in optimizer["state"].items():
        if not isinstance(state, dict):
            return f"optimizer state for parameter {parameter_id!r} is not a mapping"
        for state_name, value in state.items():
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                if not bool(torch.isfinite(value).all()):
                    return (
                        "optimizer state contains a non-finite tensor at "
                        f"parameter {parameter_id!r}:{state_name}"
                    )
            elif isinstance(value, float) and not math.isfinite(value):
                return (
                    "optimizer state contains a non-finite scalar at "
                    f"parameter {parameter_id!r}:{state_name}"
                )
    return None


def _linear_schedule_multiplier(step: int, final_step: int, warmup_ratio: float) -> float:
    """Match Transformers' linear warmup/decay multiplier at a saved optimizer step."""

    warmup_steps = math.ceil(final_step * warmup_ratio)
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (final_step - step) / max(1, final_step - warmup_steps))


def _optimizer_contract_problem(
    optimizer: object,
    config: RunConfig,
    expected_step: int,
    final_step: int,
) -> str | None:
    """Prove that a readable state still has the intended mixed-optimizer topology."""

    if not isinstance(optimizer, dict):
        return "optimizer state has an invalid structure"
    state = optimizer.get("state")
    groups = optimizer.get("param_groups")
    if not isinstance(state, dict) or not isinstance(groups, list):
        return "optimizer state has an invalid structure"

    optimizer_config = config.optimizer
    if optimizer_config.name == "adamw":
        expected_groups = [
            {
                "algorithm": "adamw",
                "base_lr": optimizer_config.lr,
                "weight_decay": optimizer_config.weight_decay,
                "betas": (optimizer_config.beta1, optimizer_config.beta2),
                "eps": optimizer_config.eps,
            },
            {
                "algorithm": "adamw",
                "base_lr": optimizer_config.lr,
                "weight_decay": 0.0,
                "betas": (optimizer_config.beta1, optimizer_config.beta2),
                "eps": optimizer_config.eps,
            },
        ]
    else:
        expected_groups = [
            {
                "algorithm": optimizer_config.name,
                "base_lr": optimizer_config.lr,
                "weight_decay": optimizer_config.weight_decay,
                "momentum": optimizer_config.momentum,
                "beta2": optimizer_config.normuon_beta2,
                "ns_steps": optimizer_config.ns_steps,
                "ns_implementation": MUON_NS_IMPLEMENTATION,
                "adjust_lr_fn": optimizer_config.adjust_lr_fn,
            },
            {
                "algorithm": "adamw",
                "base_lr": optimizer_config.aux_lr,
                "weight_decay": optimizer_config.weight_decay,
                "betas": (optimizer_config.aux_beta1, optimizer_config.aux_beta2),
                "eps": optimizer_config.aux_eps,
            },
            {
                "algorithm": "adamw",
                "base_lr": optimizer_config.aux_lr,
                "weight_decay": 0.0,
                "betas": (optimizer_config.aux_beta1, optimizer_config.aux_beta2),
                "eps": optimizer_config.aux_eps,
            },
        ]

    if len(groups) != len(expected_groups):
        return f"optimizer has {len(groups)} parameter groups, expected {len(expected_groups)}"
    multiplier = _linear_schedule_multiplier(expected_step, final_step, config.warmup_ratio)
    grouped_ids: set[object] = set()
    for index, (group, expected) in enumerate(zip(groups, expected_groups)):
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, list) or not parameter_ids:
            return f"optimizer parameter group {index} is empty or invalid"
        grouped_ids.update(parameter_ids)
        for name, expected_value in expected.items():
            observed = group.get(name)
            if name == "base_lr":
                name = "lr"
                observed = group.get(name)
                expected_value = float(expected_value) * multiplier
            if isinstance(expected_value, float):
                try:
                    matches = math.isclose(
                        float(observed), expected_value, rel_tol=1e-12, abs_tol=1e-15
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = observed == expected_value
            if not matches:
                return (
                    f"optimizer parameter group {index} {name} is {observed!r}, "
                    f"expected {expected_value!r}"
                )

        algorithm = expected["algorithm"]
        expected_state_fields = {
            "adamw": {"step", "exp_avg", "exp_avg_sq"},
            "muon": {"momentum_buffer"},
            "normuon": {"momentum_buffer", "second_moment"},
        }[algorithm]
        for parameter_id in parameter_ids:
            parameter_state = state.get(parameter_id)
            if not isinstance(parameter_state, dict):
                return f"optimizer state is missing parameter {parameter_id!r}"
            if set(parameter_state) != expected_state_fields:
                return (
                    f"optimizer state fields for parameter {parameter_id!r} are "
                    f"{sorted(parameter_state)}, expected {sorted(expected_state_fields)}"
                )
            if algorithm == "adamw":
                try:
                    state_step = float(parameter_state["step"])
                except (TypeError, ValueError):
                    return f"AdamW state step for parameter {parameter_id!r} is invalid"
                if state_step != expected_step:
                    return (
                        f"AdamW state step for parameter {parameter_id!r} is {state_step}, "
                        f"expected {expected_step}"
                    )
                exp_avg_shape = getattr(parameter_state["exp_avg"], "shape", None)
                exp_avg_sq_shape = getattr(parameter_state["exp_avg_sq"], "shape", None)
                if exp_avg_shape is None or exp_avg_sq_shape is None:
                    return f"AdamW moments for parameter {parameter_id!r} are not tensors"
                if exp_avg_shape != exp_avg_sq_shape:
                    return f"AdamW moments for parameter {parameter_id!r} have different shapes"
            else:
                momentum_buffer = parameter_state["momentum_buffer"]
                if getattr(momentum_buffer, "ndim", None) != 2:
                    return f"{algorithm} momentum for parameter {parameter_id!r} is not 2-D"
                if algorithm == "normuon":
                    expected_shape = (*momentum_buffer.shape[:-1], 1)
                    observed_shape = getattr(parameter_state["second_moment"], "shape", None)
                    if observed_shape != expected_shape:
                        return (
                            f"NorMuon second moment for parameter {parameter_id!r} has shape "
                            f"{tuple(observed_shape) if observed_shape is not None else None}, expected "
                            f"{tuple(expected_shape)}"
                        )

    if set(state) != grouped_ids:
        return "optimizer state does not cover every grouped parameter"
    return None


def _scheduler_contract_problem(
    scheduler: object,
    config: RunConfig,
    expected_step: int,
    final_step: int,
) -> str | None:
    """Validate the LR scheduler fields that control a resumed trajectory."""

    if not isinstance(scheduler, dict):
        return "scheduler state has an invalid structure"
    try:
        last_epoch = int(scheduler.get("last_epoch", -1))
        step_count = int(scheduler.get("_step_count", -1))
    except (TypeError, ValueError):
        return "scheduler step fields are invalid"
    if last_epoch != expected_step:
        return "scheduler state does not match checkpoint step"
    if step_count != expected_step + 1:
        return f"scheduler step count is {step_count}, expected {expected_step + 1}"

    optimizer_config = config.optimizer
    expected_base_lrs = (
        [optimizer_config.lr, optimizer_config.lr]
        if optimizer_config.name == "adamw"
        else [optimizer_config.lr, optimizer_config.aux_lr, optimizer_config.aux_lr]
    )
    multiplier = _linear_schedule_multiplier(expected_step, final_step, config.warmup_ratio)
    expected_last_lrs = [base_lr * multiplier for base_lr in expected_base_lrs]
    for name, expected_values in (
        ("base_lrs", expected_base_lrs),
        ("_last_lr", expected_last_lrs),
    ):
        observed_values = scheduler.get(name)
        if not isinstance(observed_values, list) or len(observed_values) != len(expected_values):
            return f"scheduler {name} has an invalid group count"
        for index, (observed, expected) in enumerate(zip(observed_values, expected_values)):
            try:
                matches = math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-15)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                return f"scheduler {name}[{index}] is {observed!r}, expected {expected!r}"

    lr_lambdas = scheduler.get("lr_lambdas")
    if not isinstance(lr_lambdas, list) or len(lr_lambdas) != len(expected_base_lrs):
        return "scheduler LR lambda state has an invalid group count"
    return None


def _training_arguments_problem(
    path: Path, config: RunConfig, world_size: int, final_step: int
) -> str | None:
    """Prove the serialized runtime arguments match the frozen run contract."""

    import torch

    try:
        args = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001
        return f"invalid training arguments ({type(error).__name__}: {error})"

    micro_global = config.micro_batch_size * world_size
    if config.global_batch_size % micro_global:
        return "configured global batch is not divisible by micro batch times world size"
    expected = {
        "per_device_train_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.global_batch_size // micro_global,
        "num_train_epochs": config.epochs,
        "max_steps": -1,
        "learning_rate": config.optimizer.lr,
        "max_grad_norm": config.max_grad_norm,
        "bf16": True,
        "tf32": True,
        "fp16": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "gradient_checkpointing": config.gradient_checkpointing,
        "dataloader_num_workers": config.dataloader_workers,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": config.dataloader_workers > 0,
        "dataloader_prefetch_factor": 4 if config.dataloader_workers > 0 else None,
        "dataloader_drop_last": True,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "train_sampling_strategy": "group_by_length",
        "logging_steps": 10,
        "run_name": f"{config.model_family}-{config.run_id}",
        "project": config.wandb_project,
    }
    for name, expected_value in expected.items():
        try:
            observed = getattr(args, name)
        except AttributeError:
            return f"training arguments are missing {name}"
        if isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(observed), expected_value, rel_tol=1e-12, abs_tol=1e-15
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected_value
        if not matches:
            return f"training argument {name} is {observed!r}, expected {expected_value!r}"

    scheduler = getattr(args, "lr_scheduler_type", None)
    if getattr(scheduler, "value", scheduler) != "linear":
        return f"training argument lr_scheduler_type is {scheduler!r}, expected 'linear'"
    save_strategy = getattr(args, "save_strategy", None)
    if getattr(save_strategy, "value", save_strategy) != "no":
        return f"training argument save_strategy is {save_strategy!r}, expected 'no'"
    report_to = getattr(args, "report_to", None)
    if report_to != ["wandb"]:
        return f"training argument report_to is {report_to!r}, expected ['wandb']"

    warmup_value = getattr(args, "warmup_steps", None)
    try:
        warmup_value = float(warmup_value)
        observed_warmup = (
            int(warmup_value) if warmup_value >= 1 else math.ceil(final_step * warmup_value)
        )
    except (TypeError, ValueError):
        return f"training argument warmup_steps is invalid: {warmup_value!r}"
    expected_warmup = math.ceil(final_step * config.warmup_ratio)
    if observed_warmup != expected_warmup:
        return (
            f"resolved warmup is {observed_warmup} steps, expected {expected_warmup} "
            f"for terminal step {final_step}"
        )
    return None


def _deep_checkpoint_problems(
    checkpoint: Path,
    expected_step: int,
    world_size: int,
    config: RunConfig | None = None,
    final_step: int | None = None,
) -> list[str]:
    """Parse resumable payloads so non-empty but corrupt files cannot pass strict audit."""

    import gc
    import zipfile

    import torch

    problems: list[str] = []
    if problem := _safetensors_problem(checkpoint):
        problems.append(problem)

    optimizer = None
    try:
        optimizer = torch.load(
            checkpoint / "optimizer.pt", map_location="cpu", weights_only=True, mmap=True
        )
        if problem := _optimizer_state_problem(optimizer):
            problems.append(problem)
        elif config is not None and final_step is not None:
            if problem := _optimizer_contract_problem(optimizer, config, expected_step, final_step):
                problems.append(problem)
    except Exception as error:  # noqa: BLE001
        problems.append(f"invalid optimizer state ({type(error).__name__}: {error})")
    finally:
        del optimizer
        gc.collect()

    try:
        scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
        if config is not None and final_step is not None:
            if problem := _scheduler_contract_problem(scheduler, config, expected_step, final_step):
                problems.append(problem)
        elif (
            not isinstance(scheduler, dict) or int(scheduler.get("last_epoch", -1)) != expected_step
        ):
            problems.append("scheduler state does not match checkpoint step")
    except Exception as error:  # noqa: BLE001
        problems.append(f"invalid scheduler state ({type(error).__name__}: {error})")

    if config is not None:
        if final_step is None:
            problems.append("training argument audit is missing the terminal step")
        elif problem := _training_arguments_problem(
            checkpoint / "training_args.bin", config, world_size, final_step
        ):
            problems.append(problem)

    for rank in range(world_size):
        path = checkpoint / f"rng_state_{rank}.pth"
        try:
            if not zipfile.is_zipfile(path):
                problems.append(f"rank {rank} RNG state is not a PyTorch archive")
                continue
            with zipfile.ZipFile(path) as archive:
                corrupt_member = archive.testzip()
                if corrupt_member is not None:
                    problems.append(f"rank {rank} RNG state has corrupt member {corrupt_member}")
        except (OSError, zipfile.BadZipFile) as error:
            problems.append(f"invalid rank {rank} RNG state ({type(error).__name__}: {error})")
    return problems


def _accepted_timing_problems(
    path: Path,
    *,
    expected_start_step: int,
    expected_final_step: int | None = None,
) -> list[str]:
    """Validate the checkpoint-committed, non-overlapping timing ledger."""

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f"invalid accepted timing ledger ({error})"]
    if payload.get("schema_version") != 1:
        return ["accepted timing ledger has an unsupported schema"]
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        return ["accepted timing ledger has no segments"]

    problems: list[str] = []
    expected_start = expected_start_step
    total = 0.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            problems.append(f"accepted timing segment {index} is not an object")
            continue
        try:
            start = int(segment["start_step_exclusive"])
            end = int(segment["end_step_inclusive"])
            wall_time = float(segment["wall_time_seconds_max_rank"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"accepted timing segment {index} has invalid numeric fields")
            continue
        if start != expected_start:
            problems.append(
                f"accepted timing segment {index} starts at {start}, expected {expected_start}"
            )
        if end <= start:
            problems.append(f"accepted timing segment {index} has non-increasing steps")
        if not math.isfinite(wall_time) or wall_time <= 0:
            problems.append(f"accepted timing segment {index} has invalid wall time")
        for field in ("started_at_utc", "checkpoint_completed_at_utc"):
            if not isinstance(segment.get(field), str) or not segment[field]:
                problems.append(f"accepted timing segment {index} has invalid {field}")
        expected_start = end
        total += wall_time

    recorded_total = payload.get("total_wall_time_seconds_max_rank")
    if (
        not isinstance(recorded_total, (int, float))
        or not math.isfinite(float(recorded_total))
        or not math.isclose(float(recorded_total), total, rel_tol=1e-9, abs_tol=1e-6)
    ):
        problems.append("accepted timing total does not match its segments")
    if expected_final_step is not None and expected_start != expected_final_step:
        problems.append(
            f"accepted timing ledger ends at {expected_start}, expected {expected_final_step}"
        )
    return problems


def _timing_adjustment_problems(path: Path, checkpoint_steps: list[int]) -> list[str]:
    """Validate manually retained pre-ledger work against timestamp evidence."""

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f"invalid timing adjustment ({error})"]
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        return ["timing adjustment has no evidence segments"]

    problems: list[str] = []
    total = 0.0
    previous_end: datetime | None = None
    previous_step = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            problems.append(f"timing adjustment segment {index} is not an object")
            continue
        try:
            start = datetime.fromisoformat(str(segment["started_at_utc"]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(
                str(segment["checkpoint_completed_at_utc"]).replace("Z", "+00:00")
            )
            wall_time = float(segment["wall_time_seconds"])
            included_step = int(segment["included_through_checkpoint_step"])
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"timing adjustment segment {index} is invalid ({error})")
            continue
        if start.tzinfo is None or end.tzinfo is None:
            problems.append(f"timing adjustment segment {index} timestamps lack timezones")
        if end <= start:
            problems.append(f"timing adjustment segment {index} has non-positive duration")
        if previous_end is not None and start < previous_end:
            problems.append(f"timing adjustment segment {index} overlaps its predecessor")
        if included_step not in checkpoint_steps:
            problems.append(
                f"timing adjustment segment {index} ends at undeclared checkpoint {included_step}"
            )
        if included_step <= previous_step:
            problems.append(
                f"timing adjustment segment {index} checkpoint is not strictly increasing"
            )
        observed_duration = (end - start).total_seconds()
        if (
            not math.isfinite(wall_time)
            or wall_time <= 0
            or not math.isclose(wall_time, observed_duration, rel_tol=1e-9, abs_tol=1e-3)
        ):
            problems.append(
                f"timing adjustment segment {index} wall time differs from its timestamps"
            )
        total += wall_time
        previous_end = end
        previous_step = included_step

    recorded_total = payload.get("prior_training_wall_time_seconds")
    if (
        isinstance(recorded_total, bool)
        or not isinstance(recorded_total, (int, float))
        or not math.isfinite(float(recorded_total))
        or not math.isclose(float(recorded_total), total, rel_tol=1e-9, abs_tol=1e-6)
    ):
        problems.append("timing adjustment total does not match its evidence segments")
    if payload.get("included_through_checkpoint_step") != previous_step:
        problems.append("timing adjustment terminal checkpoint does not match its segments")
    for field in ("evidence", "reason"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            problems.append(f"timing adjustment {field} is missing")
    return problems


def audit_training_artifacts(
    configs: list[RunConfig],
    *,
    deep: bool = False,
    expected_dataset_fingerprint: str | None = None,
) -> dict:
    """Verify that every planned run has five complete, resumable checkpoints."""

    errors: list[str] = []
    verified_runs = 0
    verified_checkpoints = 0
    partition_by_family: dict[str, dict] = {}
    versions_reference: dict | None = None
    gpu_name_reference: str | None = None
    dataset_fingerprint_reference = expected_dataset_fingerprint
    for config in configs:
        label = f"{config.model_family}/{config.run_id}"
        output = config.output_dir
        schedule_path = output / "checkpoint_schedule.json"
        completed_path = output / "completed.json"
        final_state_path = output / "trainer_state_final.json"
        final_model_path = output / "final"
        run_config_path = output / "run_config.json"
        source_manifest_path = Path(config.dataset_path) / "manifest.json"
        run_manifest_path = output / "dataset_manifest.json"
        if not schedule_path.is_file():
            errors.append(f"{label}: missing checkpoint_schedule.json")
            continue
        try:
            schedule = json.loads(schedule_path.read_text())
            steps = [int(step) for step in schedule["steps"]]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{label}: invalid checkpoint schedule ({error})")
            continue
        if len(steps) != 5 or steps != sorted(set(steps)):
            errors.append(
                f"{label}: expected five strictly increasing checkpoint steps, got {steps}"
            )
            continue
        if not run_config_path.is_file():
            errors.append(f"{label}: missing run_config.json")
        else:
            try:
                run_config = json.loads(run_config_path.read_text())
            except json.JSONDecodeError as error:
                errors.append(f"{label}: invalid run_config.json ({error})")
            else:
                expected_config = json.loads(json.dumps(config.as_dict()))
                if run_config != expected_config:
                    errors.append(f"{label}: resolved run config differs from matrix")

        source_manifest: dict = {}
        if not source_manifest_path.is_file():
            errors.append(f"{label}: missing source dataset manifest")
        else:
            try:
                source_manifest = json.loads(source_manifest_path.read_text())
            except json.JSONDecodeError as error:
                errors.append(f"{label}: invalid source dataset manifest ({error})")
        if not run_manifest_path.is_file():
            errors.append(f"{label}: missing copied dataset manifest")
        else:
            try:
                run_manifest = json.loads(run_manifest_path.read_text())
            except json.JSONDecodeError as error:
                errors.append(f"{label}: invalid copied dataset manifest ({error})")
            else:
                if source_manifest and run_manifest != source_manifest:
                    errors.append(f"{label}: copied dataset manifest differs from source")

        completed: dict = {}
        completion_system_metrics: dict = {}
        if not completed_path.is_file():
            errors.append(f"{label}: missing completed.json")
        else:
            try:
                completed = json.loads(completed_path.read_text())
            except json.JSONDecodeError as error:
                errors.append(f"{label}: invalid completed.json ({error})")
            if completed and int(completed.get("global_step", -1)) != steps[-1]:
                errors.append(f"{label}: completion global_step does not match final checkpoint")
            if (
                completed
                and sorted(int(step) for step in completed.get("checkpoints", [])) != steps
            ):
                errors.append(f"{label}: completion checkpoint list does not match schedule")
            if completed and completed.get("run_id") != config.run_id:
                errors.append(f"{label}: completion run_id does not match matrix")
            if completed and completed.get("model_family") != config.model_family:
                errors.append(f"{label}: completion model family does not match matrix")
            if completed and source_manifest:
                if completed.get("dataset_rows") != source_manifest.get("total_queries"):
                    errors.append(f"{label}: completion dataset row count does not match manifest")
            if completed:
                dataset_fingerprint = completed.get("dataset_fingerprint")
                if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
                    errors.append(f"{label}: missing/invalid training dataset view fingerprint")
                elif (
                    expected_dataset_fingerprint is not None
                    and dataset_fingerprint != expected_dataset_fingerprint
                ):
                    errors.append(
                        f"{label}: training dataset view fingerprint differs from audited dataset"
                    )
                elif dataset_fingerprint_reference is None:
                    dataset_fingerprint_reference = dataset_fingerprint
                elif dataset_fingerprint != dataset_fingerprint_reference:
                    errors.append(f"{label}: training dataset view fingerprint differs across runs")
                raw_system_metrics = completed.get("system_metrics")
                if isinstance(raw_system_metrics, dict):
                    completion_system_metrics = raw_system_metrics
                partition = completed.get("optimizer_partition")
                if not isinstance(partition, dict) or set(partition) != {
                    "hidden",
                    "aux_decay",
                    "aux_no_decay",
                }:
                    errors.append(f"{label}: missing/invalid optimizer parameter partition")
                elif config.model_family not in partition_by_family:
                    partition_by_family[config.model_family] = partition
                elif partition != partition_by_family[config.model_family]:
                    errors.append(f"{label}: optimizer parameter partition differs within family")
                for problem in _completion_system_problems(completed, steps):
                    errors.append(f"{label}: {problem}")
                versions = completed.get("versions")
                if isinstance(versions, dict) and len(versions) == 5:
                    if versions_reference is None:
                        versions_reference = versions
                    elif versions != versions_reference:
                        errors.append(f"{label}: completion package versions differ across runs")
                gpu_name = completion_system_metrics.get("gpu_name")
                if isinstance(gpu_name, str) and gpu_name:
                    if gpu_name_reference is None:
                        gpu_name_reference = gpu_name
                    elif gpu_name != gpu_name_reference:
                        errors.append(f"{label}: completion GPU name differs across runs")

        accepted_timing_path = output / "accepted_timing.json"
        if accepted_timing_path.is_file():
            timing_adjustment_path = output / "timing_adjustment.json"
            expected_timing_start = 0
            if timing_adjustment_path.is_file():
                adjustment_problems = _timing_adjustment_problems(timing_adjustment_path, steps)
                errors.extend(f"{label}: {problem}" for problem in adjustment_problems)
                try:
                    expected_timing_start = int(
                        json.loads(timing_adjustment_path.read_text())[
                            "included_through_checkpoint_step"
                        ]
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    errors.append(f"{label}: invalid timing adjustment ({error})")
            timing_problems = _accepted_timing_problems(
                accepted_timing_path,
                expected_start_step=expected_timing_start,
                expected_final_step=steps[-1] if completed else None,
            )
            errors.extend(f"{label}: {problem}" for problem in timing_problems)
            if completed and not timing_problems:
                timing_payload = json.loads(accepted_timing_path.read_text())
                expected_summary = {
                    "schema_version": 1,
                    "segments": len(timing_payload["segments"]),
                    "total_wall_time_seconds_max_rank": timing_payload[
                        "total_wall_time_seconds_max_rank"
                    ],
                }
                if completed.get("accepted_timing") != expected_summary:
                    errors.append(
                        f"{label}: completion accepted timing summary differs from ledger"
                    )
        if not final_state_path.is_file():
            errors.append(f"{label}: missing trainer_state_final.json")
        else:
            if problem := _trainer_state_problem(final_state_path, steps[-1]):
                errors.append(f"{label}: final {problem}")
        final_model_present = final_model_path.is_dir() and any(
            path.stat().st_size > 0 for path in final_model_path.rglob("*.safetensors")
        )
        if not final_model_present:
            errors.append(f"{label}: missing final safetensors model")
        elif deep and (problem := _safetensors_problem(final_model_path)):
            errors.append(f"{label}: final {problem}")

        recorded_world_size = completion_system_metrics.get("world_size")
        if completed and recorded_world_size != 4:
            errors.append(f"{label}: expected completion world_size 4, got {recorded_world_size}")
        world_size = 4
        run_checkpoint_errors = 0
        previous_model_digest: str | None = None
        for step in steps:
            checkpoint = output / f"checkpoint-{step}"
            required = (
                checkpoint / "config.json",
                checkpoint / "optimizer.pt",
                checkpoint / "scheduler.pt",
                checkpoint / "trainer_state.json",
                checkpoint / "training_args.bin",
            )
            missing = [
                path.name for path in required if not path.is_file() or path.stat().st_size == 0
            ]
            if missing:
                errors.append(f"{label}/checkpoint-{step}: missing/empty {', '.join(missing)}")
                run_checkpoint_errors += 1
                continue
            if not any(path.stat().st_size > 0 for path in checkpoint.rglob("*.safetensors")):
                errors.append(f"{label}/checkpoint-{step}: missing/empty safetensors model")
                run_checkpoint_errors += 1
                continue
            rng_states = sorted(checkpoint.glob("rng_state_*.pth"))
            if len(rng_states) != world_size or any(
                path.stat().st_size == 0 for path in rng_states
            ):
                errors.append(
                    f"{label}/checkpoint-{step}: expected {world_size} non-empty rank RNG states, "
                    f"found {len(rng_states)}"
                )
                run_checkpoint_errors += 1
                continue
            if problem := _trainer_state_problem(required[3], step):
                errors.append(f"{label}/checkpoint-{step}: {problem}")
                run_checkpoint_errors += 1
                continue
            if deep and (
                problems := _deep_checkpoint_problems(
                    checkpoint, step, world_size, config=config, final_step=steps[-1]
                )
            ):
                errors.extend(f"{label}/checkpoint-{step}: {problem}" for problem in problems)
                run_checkpoint_errors += 1
                continue
            if deep:
                model_digest = _safetensors_digest(checkpoint)
                if model_digest == previous_model_digest:
                    errors.append(
                        f"{label}/checkpoint-{step}: model payload is unchanged from the "
                        "previous checkpoint"
                    )
                    run_checkpoint_errors += 1
                    continue
                previous_model_digest = model_digest
            verified_checkpoints += 1
        if run_checkpoint_errors == 0 and not any(
            error.startswith(f"{label}:") for error in errors
        ):
            verified_runs += 1

    return {
        "complete": not errors,
        "verified_runs": verified_runs,
        "expected_runs": len(configs),
        "verified_checkpoints": verified_checkpoints,
        "expected_checkpoints": len(configs) * 5,
        "deep_validation": deep,
        "errors": errors,
    }


def _contains_run_id(path: Path, run_id: str) -> bool:
    """Match a run directory exactly, avoiding muon/normuon substring collisions."""

    return any(part == run_id or part.startswith(f"{run_id}__checkpoint-") for part in path.parts)


def _run_for_result(path: Path, configs: list[RunConfig]) -> RunConfig | None:
    matches = [
        config
        for config in configs
        if config.model_family in path.parts and _contains_run_id(path, config.run_id)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _base_task_name(name: str) -> str:
    suffix = "Decontaminated"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _run_settings_scope_matches(
    item: dict,
    expected_split: str,
    expected_subset: str,
    mteb_version: str,
) -> bool:
    """Match the versioned MTEB run-settings schema without accepting hybrids."""

    try:
        major, minor = (int(part) for part in mteb_version.split(".", 2)[:2])
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported MTEB run-settings version: {mteb_version!r}") from error
    if major != 2 or minor < 18:
        raise ValueError(f"Unsupported MTEB run-settings version: {mteb_version!r}")
    if minor == 18:
        return (
            item.get("split") == expected_split
            and item.get("subset") == expected_subset
            and "splits" not in item
            and "subsets" not in item
        )
    return (
        item.get("splits") == [expected_split]
        and item.get("subsets") == [expected_subset]
        and "split" not in item
        and "subset" not in item
    )


def _result_provenance(
    path: Path,
    payload: dict,
    config: RunConfig,
    step: int,
    task: str,
    runtime_versions: dict[str, str],
) -> dict[str, str]:
    """Validate one MTEB result and return its recorded package versions."""

    expected_task_name = f"{task}Decontaminated"
    expected_revision = DECONTAMINATED_BEIR[task][1]
    expected_split = "dev" if task == "MSMARCO" else "test"
    if payload.get("task_name") != expected_task_name:
        raise ValueError(f"Unexpected task identity in {path}")
    if payload.get("dataset_revision") != expected_revision:
        raise ValueError(f"Unexpected dataset revision in {path}")
    mteb_version = payload.get("mteb_version")
    if not isinstance(mteb_version, str) or not mteb_version:
        raise ValueError(f"Missing/invalid MTEB version in {path}")
    evaluation_time = payload.get("evaluation_time")
    if (
        isinstance(evaluation_time, bool)
        or not isinstance(evaluation_time, (int, float))
        or not math.isfinite(evaluation_time)
        or evaluation_time <= 0
    ):
        raise ValueError(f"Missing/invalid evaluation time in {path}")

    scores = payload.get("scores")
    if not isinstance(scores, dict) or set(scores) != {expected_split}:
        raise ValueError(f"Unexpected evaluation split coverage in {path}")
    split_rows = scores[expected_split]
    if (
        not isinstance(split_rows, list)
        or len(split_rows) != 1
        or not isinstance(split_rows[0], dict)
        or split_rows[0].get("hf_subset") != "default"
    ):
        raise ValueError(f"Unexpected evaluation subset coverage in {path}")
    score_row = split_rows[0]
    if score_row.get("mteb_version") != mteb_version:
        raise ValueError(f"Per-subset MTEB version differs in {path}")
    ndcg = score_row.get("ndcg_at_10")
    main_score = score_row.get("main_score")
    if (
        isinstance(ndcg, bool)
        or not isinstance(ndcg, (int, float))
        or not math.isfinite(ndcg)
        or isinstance(main_score, bool)
        or not isinstance(main_score, (int, float))
        or not math.isfinite(main_score)
        or not math.isclose(ndcg, main_score, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError(f"Missing/inconsistent nDCG@10 main score in {path}")

    meta_path = path.parent / "model_meta.json"
    settings_path = path.parent / "run_settings.jsonl"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing/invalid model metadata beside {path}") from error
    expected_model_name = f"{config.run_id}/checkpoint-{step}"
    if meta.get("name") != expected_model_name or meta.get("revision") != "local":
        raise ValueError(f"Unexpected model identity beside {path}")
    max_tokens = meta.get("max_tokens")
    expected_embedding = 768 if config.model_family == "dense" else 128
    expected_similarity = "cosine" if config.model_family == "dense" else "MaxSim"
    required_framework = "Sentence Transformers" if config.model_family == "dense" else "PyLate"
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, (int, float))
        or not math.isfinite(max_tokens)
        or max_tokens != config.max_length
        or meta.get("embed_dim") != expected_embedding
        or meta.get("similarity_fn_name") != expected_similarity
        or not isinstance(meta.get("framework"), list)
        or required_framework not in meta["framework"]
    ):
        raise ValueError(f"Unexpected model evaluation semantics beside {path}")

    try:
        settings = [
            json.loads(line) for line in settings_path.read_text().splitlines() if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing/invalid run settings beside {path}") from error
    if not settings or any(not isinstance(item, dict) for item in settings):
        raise ValueError(f"Missing/invalid run settings beside {path}")
    matching_settings = [
        item
        for item in settings
        if item.get("task") == expected_task_name
        and _run_settings_scope_matches(item, expected_split, "default", mteb_version)
    ]
    if len(matching_settings) != 1:
        raise ValueError(f"Missing/ambiguous run settings beside {path}")
    versions = matching_settings[0].get("version")
    expected_packages = {
        "mteb",
        "torch",
        "sentence-transformers",
        "flash-attn",
        "transformers",
    }
    if (
        not isinstance(versions, dict)
        or set(versions) != expected_packages
        or any(not isinstance(value, str) or not value for value in versions.values())
        or versions["mteb"] != mteb_version
    ):
        raise ValueError(f"Missing/inconsistent evaluation package versions beside {path}")
    if any(runtime_versions[package] != value for package, value in versions.items()):
        raise ValueError(f"MTEB settings differ from evaluation runtime beside {path}")

    completed_path = config.output_dir / "completed.json"
    try:
        training_versions = json.loads(completed_path.read_text())["versions"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Missing training versions for {config.model_family}/{config.run_id}"
        ) from error
    for package in (
        "torch",
        "transformers",
        "sentence-transformers",
        "pylate",
        "late-interaction-kernels",
    ):
        if runtime_versions[package] != training_versions.get(package):
            raise ValueError(f"Training/evaluation {package} versions differ for {path}")
    return versions


def _evaluation_runtime(results_root: Path) -> dict[str, str]:
    """Load the immutable evaluator manifest stored before any GPU work."""

    path = results_root / "evaluation_runtime.json"
    try:
        payload = json.loads(path.read_text())
        versions = payload["versions"]
        source_files = payload["source_files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing/invalid evaluation runtime manifest: {path}") from error
    if (
        payload.get("schema_version") != 2
        or not isinstance(payload.get("python"), str)
        or not payload["python"]
        or not isinstance(versions, dict)
        or set(versions) != EVALUATION_PACKAGES
        or any(not isinstance(value, str) or not value for value in versions.values())
        or not isinstance(source_files, dict)
        or not source_files
        or any(
            not isinstance(label, str)
            or not isinstance(identity, dict)
            or set(identity) != {"sha256", "bytes"}
            or not isinstance(identity["sha256"], str)
            or len(identity["sha256"]) != 64
            or not isinstance(identity["bytes"], int)
            or isinstance(identity["bytes"], bool)
            or identity["bytes"] <= 0
            for label, identity in source_files.items()
        )
    ):
        raise ValueError(f"Incomplete evaluation runtime manifest: {path}")
    from .evaluation_source_provenance import (
        EvaluationSourceProvenanceError,
        verify_evaluation_source_manifest,
    )

    try:
        verify_evaluation_source_manifest(
            source_files,
            repo_root=Path(__file__).resolve().parents[2],
        )
    except EvaluationSourceProvenanceError as error:
        raise ValueError(
            f"Unauthenticated evaluation source files in runtime manifest: {path}: {error}"
        ) from error
    return versions


def collect_evaluations(results_root: Path, configs: list[RunConfig]) -> list[dict]:
    indexed: dict[tuple, dict] = {}
    versions_reference: dict[str, str] | None = None
    runtime_versions: dict[str, str] | None = None
    for path in results_root.rglob("*Decontaminated.json"):
        config = _run_for_result(path, configs)
        match = CHECKPOINT_PATTERN.search(str(path))
        if config is None or match is None:
            continue
        step = int(match.group(1))
        schedule_path = config.output_dir / "checkpoint_schedule.json"
        if not schedule_path.is_file():
            continue
        steps = sorted(json.loads(schedule_path.read_text())["steps"])
        if step not in steps:
            continue
        if runtime_versions is None:
            runtime_versions = _evaluation_runtime(results_root)
        payload = json.loads(path.read_text())
        task = _base_task_name(payload["task_name"])
        if task not in DECONTAMINATED_BEIR:
            raise ValueError(f"Unexpected decontaminated task in {path}")
        versions = _result_provenance(path, payload, config, step, task, runtime_versions)
        if versions_reference is None:
            versions_reference = versions
        elif versions != versions_reference:
            raise ValueError(f"Evaluation package versions differ for {path}")
        split_rows = [item for values in payload["scores"].values() for item in values]
        scores = [float(item["ndcg_at_10"]) for item in split_rows]
        if not scores or not all(math.isfinite(score) for score in scores):
            raise ValueError(f"Missing/non-finite nDCG@10 in {path}")
        row = {
            "model_family": config.model_family,
            "optimizer": config.optimizer.name,
            "learning_rate": config.optimizer.lr,
            "aux_learning_rate": config.optimizer.aux_lr,
            "run_id": config.run_id,
            "stage": steps.index(step) + 1,
            "fraction": (steps.index(step) + 1) / 5,
            "checkpoint_step": step,
            "task": task,
            "ndcg_at_10": statistics.mean(scores),
            "subsets": len(scores),
            "result_path": str(path),
        }
        identity = (config.model_family, config.run_id, step, task)
        previous = indexed.get(identity)
        if previous and previous["ndcg_at_10"] != row["ndcg_at_10"]:
            raise ValueError(f"Conflicting duplicate evaluation for {identity}")
        indexed[identity] = row
    return sorted(indexed.values(), key=lambda row: tuple(str(value) for value in row.values()))


def collect_training_history(configs: list[RunConfig]) -> list[dict]:
    rows: list[dict] = []
    for config in configs:
        schedule_path = config.output_dir / "checkpoint_schedule.json"
        if not schedule_path.is_file():
            continue
        steps = sorted(json.loads(schedule_path.read_text())["steps"])
        state_path = config.output_dir / "trainer_state_final.json"
        if not state_path.is_file():
            state_path = config.output_dir / f"checkpoint-{steps[-1]}" / "trainer_state.json"
        if not state_path.is_file():
            continue
        for item in json.loads(state_path.read_text()).get("log_history", []):
            if "loss" not in item:
                continue
            rows.append(
                {
                    "model_family": config.model_family,
                    "optimizer": config.optimizer.name,
                    "learning_rate_config": config.optimizer.lr,
                    "run_id": config.run_id,
                    **item,
                }
            )
    return rows


def collect_system_metrics(configs: list[RunConfig]) -> list[dict]:
    rows = []
    for config in configs:
        path = config.output_dir / "completed.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        metrics = payload.get("system_metrics", {})
        adjustment_path = config.output_dir / "timing_adjustment.json"
        adjustment = json.loads(adjustment_path.read_text()) if adjustment_path.is_file() else {}
        accepted_timing_path = config.output_dir / "accepted_timing.json"
        if accepted_timing_path.is_file():
            accepted_timing = json.loads(accepted_timing_path.read_text())
            segment_wall_time = sum(
                float(segment["wall_time_seconds_max_rank"])
                for segment in accepted_timing.get("segments", [])
            )
        else:
            segment_wall_time = metrics.get("wall_time_seconds_max_rank", 0)
        prior_wall_time = adjustment.get("prior_training_wall_time_seconds", 0)
        total_wall_time = segment_wall_time + prior_wall_time
        trainer = metrics.get("trainer", {})
        checkpoint_sizes = metrics.get("checkpoint_bytes", {})
        state_sizes = metrics.get("optimizer_state_bytes", {})
        rows.append(
            {
                "model_family": config.model_family,
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "run_id": config.run_id,
                "wall_time_hours": total_wall_time / 3600,
                "recorded_segment_wall_time_hours": segment_wall_time / 3600,
                "prior_training_wall_time_hours": prior_wall_time / 3600,
                "timing_adjustment_path": str(adjustment_path) if adjustment else None,
                "accepted_timing_path": str(accepted_timing_path)
                if accepted_timing_path.is_file()
                else None,
                "samples_per_second": payload.get("dataset_rows", 0) / total_wall_time
                if total_wall_time
                else None,
                "steps_per_second": payload.get("global_step", 0) / total_wall_time
                if total_wall_time
                else None,
                "trainer_reported_samples_per_second": trainer.get("train_samples_per_second"),
                "trainer_reported_steps_per_second": trainer.get("train_steps_per_second"),
                "peak_allocated_gib": metrics.get("peak_allocated_bytes_max_rank", 0) / 2**30,
                "peak_reserved_gib": metrics.get("peak_reserved_bytes_max_rank", 0) / 2**30,
                "checkpoint_gib": max(checkpoint_sizes.values(), default=0) / 2**30,
                "optimizer_state_gib": max(state_sizes.values(), default=0) / 2**30,
                "gpu_name": metrics.get("gpu_name"),
                "world_size": metrics.get("world_size"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_summaries(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["model_family"],
            row["optimizer"],
            row["learning_rate"],
            row["run_id"],
            row["stage"],
            row["fraction"],
            row["checkpoint_step"],
        )
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items()):
        output.append(
            {
                "model_family": key[0],
                "optimizer": key[1],
                "learning_rate": key[2],
                "run_id": key[3],
                "stage": key[4],
                "fraction": key[5],
                "checkpoint_step": key[6],
                "mean_ndcg_at_10": statistics.mean(row["ndcg_at_10"] for row in values),
                "tasks_completed": len(values),
            }
        )
    return output


def _trajectory_auc(curve: list[dict]) -> float | None:
    """Normalized trapezoidal mean score over the observed 20%–100% window."""

    points = sorted((float(row["fraction"]), float(row["mean_ndcg_at_10"])) for row in curve)
    if len(points) != 5 or len({fraction for fraction, _ in points}) != 5:
        return None
    span = points[-1][0] - points[0][0]
    if span <= 0:
        return None
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2 for left, right in zip(points, points[1:])
    )
    return area / span


def _optimizer_summaries(summary: list[dict]) -> tuple[list[dict], list[dict]]:
    final = [row for row in summary if row["stage"] == 5]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in final:
        grouped[(row["model_family"], row["optimizer"])].append(row)

    optimizer_rows, best_dynamics = [], []
    for (family, optimizer), values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: row["learning_rate"])
        scores = [row["mean_ndcg_at_10"] for row in values]
        best = max(values, key=lambda row: row["mean_ndcg_at_10"])
        curves = [
            row
            for row in summary
            if row["model_family"] == family and row["run_id"] == best["run_id"]
        ]
        curves = sorted(curves, key=lambda row: row["stage"])
        best_dynamics.extend(curves)
        all_curves: dict[str, list[dict]] = defaultdict(list)
        for row in summary:
            if row["model_family"] == family and row["optimizer"] == optimizer:
                all_curves[row["run_id"]].append(row)
        trajectory_aucs = [
            auc for curve in all_curves.values() if (auc := _trajectory_auc(curve)) is not None
        ]
        best_auc = _trajectory_auc(curves)
        optimizer_rows.append(
            {
                "model_family": family,
                "optimizer": optimizer,
                "configurations": len(values),
                "best_run_id": best["run_id"],
                "best_learning_rate": best["learning_rate"],
                "best_final_ndcg_at_10": best["mean_ndcg_at_10"],
                "final_mean_across_lrs": statistics.mean(scores),
                "final_median_across_lrs": statistics.median(scores),
                "final_population_std_across_lrs": statistics.pstdev(scores),
                "final_min_across_lrs": min(scores),
                "final_max_across_lrs": max(scores),
                "best_config_observed_auc_ndcg_at_10": best_auc,
                "observed_auc_mean_across_lrs": (
                    statistics.mean(trajectory_aucs) if trajectory_aucs else None
                ),
                "observed_auc_population_std_across_lrs": (
                    statistics.pstdev(trajectory_aucs) if trajectory_aucs else None
                ),
                "best_config_mean_five_stage_ndcg_at_10": statistics.mean(
                    row["mean_ndcg_at_10"] for row in curves
                ),
            }
        )
    return optimizer_rows, best_dynamics


def _task_comparison(
    rows: list[dict],
    optimizer_rows: list[dict],
    families: tuple[str, ...] = ALL_FAMILIES,
) -> list[dict]:
    best_runs = {
        (row["model_family"], row["optimizer"]): row["best_run_id"] for row in optimizer_rows
    }
    lookup = {
        (row["model_family"], row["run_id"], row["stage"], row["task"]): row["ndcg_at_10"]
        for row in rows
    }
    output = []
    for family in families:
        for task in DECONTAMINATED_TASK_NAMES:
            values = {
                optimizer: lookup[(family, best_runs[(family, optimizer)], 5, task)]
                for optimizer in ("adamw", "muon", "normuon")
            }
            output.append(
                {
                    "model_family": family,
                    "task": task,
                    **values,
                    "muon_minus_adamw": values["muon"] - values["adamw"],
                    "normuon_minus_adamw": values["normuon"] - values["adamw"],
                }
            )
    return output


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _paired_comparisons(
    task_rows: list[dict],
    bootstrap_samples: int = 20_000,
    seed: int = 42,
    families: tuple[str, ...] = ALL_FAMILIES,
) -> list[dict]:
    """Summarize best-config task deltas with deterministic paired uncertainty."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    output = []
    for family in families:
        family_rows = [row for row in task_rows if row["model_family"] == family]
        if not family_rows:
            raise ValueError(f"No paired task rows for {family}")
        for optimizer in ("muon", "normuon"):
            deltas = [float(row[f"{optimizer}_minus_adamw"]) for row in family_rows]
            wins = sum(delta > 1e-12 for delta in deltas)
            losses = sum(delta < -1e-12 for delta in deltas)
            ties = len(deltas) - wins - losses
            nonties = wins + losses
            if nonties:
                tail = (
                    sum(math.comb(nonties, count) for count in range(min(wins, losses) + 1))
                    / 2**nonties
                )
                sign_p = min(1.0, 2 * tail)
            else:
                sign_p = 1.0

            identity = f"{seed}/{family}/{optimizer}".encode()
            stable_seed = int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "big")
            generator = random.Random(stable_seed)
            means = sorted(
                sum(deltas[generator.randrange(len(deltas))] for _ in deltas) / len(deltas)
                for _ in range(bootstrap_samples)
            )
            output.append(
                {
                    "model_family": family,
                    "optimizer": optimizer,
                    "baseline": "adamw",
                    "tasks": len(deltas),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "mean_delta": statistics.mean(deltas),
                    "median_delta": statistics.median(deltas),
                    "bootstrap_ci_95_lower": _percentile(means, 0.025),
                    "bootstrap_ci_95_upper": _percentile(means, 0.975),
                    "bootstrap_samples": bootstrap_samples,
                    "bootstrap_seed": seed,
                    "exact_sign_test_p_value": sign_p,
                }
            )
    ordered = sorted(enumerate(output), key=lambda item: item[1]["exact_sign_test_p_value"])
    running_adjusted = 0.0
    comparisons = len(ordered)
    for rank, (original_index, row) in enumerate(ordered):
        adjusted = min(1.0, (comparisons - rank) * row["exact_sign_test_p_value"])
        running_adjusted = max(running_adjusted, adjusted)
        output[original_index]["holm_sign_test_p_value"] = running_adjusted
    return output


def _system_summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_family"], row["optimizer"])].append(row)
    output = []
    for (family, optimizer), values in sorted(grouped.items()):
        numeric = (
            "wall_time_hours",
            "samples_per_second",
            "steps_per_second",
            "peak_allocated_gib",
            "peak_reserved_gib",
            "checkpoint_gib",
            "optimizer_state_gib",
        )
        output.append(
            {
                "model_family": family,
                "optimizer": optimizer,
                "runs": len(values),
                **{
                    f"median_{key}": statistics.median(
                        row[key] for row in values if row[key] is not None
                    )
                    for key in numeric
                },
                "gpu_name": values[0]["gpu_name"],
                "world_size": values[0]["world_size"],
            }
        )
    adamw_by_family = {row["model_family"]: row for row in output if row["optimizer"] == "adamw"}
    for row in output:
        baseline = adamw_by_family.get(row["model_family"])
        row["throughput_vs_adamw"] = (
            row["median_samples_per_second"] / baseline["median_samples_per_second"]
            if baseline and baseline["median_samples_per_second"] > 0
            else None
        )
        row["wall_time_speedup_vs_adamw"] = (
            baseline["median_wall_time_hours"] / row["median_wall_time_hours"]
            if baseline and row["median_wall_time_hours"] > 0
            else None
        )
    return output


def _plot(summary: list[dict], output_dir: Path) -> None:
    if not summary:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(summary)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for family in sorted(frame.model_family.unique()):
        subset = frame[frame.model_family == family]
        fig, axis = plt.subplots(figsize=(8, 5))
        for optimizer, values in subset.groupby("optimizer"):
            grouped = values.groupby("fraction")["mean_ndcg_at_10"]
            means = grouped.mean()
            std = grouped.std(ddof=0).fillna(0)
            axis.plot(means.index, means, marker="o", label=optimizer)
            axis.fill_between(means.index, means - std, means + std, alpha=0.15)
        axis.set(xlabel="Training fraction", ylabel="Mean decontaminated BEIR nDCG@10")
        axis.set_title(f"{family.capitalize()} training dynamics (mean ± LR-config SD)")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{family}-training-dynamics.png", dpi=180)
        plt.close(fig)

        optimizers = sorted(subset.optimizer.unique())
        fig, axes = plt.subplots(
            1,
            len(optimizers),
            figsize=(5 * len(optimizers), 4.5),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for axis, optimizer in zip(axes[0], optimizers):
            optimizer_rows = subset[subset.optimizer == optimizer]
            for (_run_id, learning_rate), values in optimizer_rows.groupby(
                ["run_id", "learning_rate"]
            ):
                ordered = values.sort_values("fraction")
                axis.plot(
                    ordered.fraction,
                    ordered.mean_ndcg_at_10,
                    marker="o",
                    label=f"LR {_format_lr(float(learning_rate))}",
                )
            axis.set_title({"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}[optimizer])
            axis.set_xlabel("Training fraction")
            axis.grid(alpha=0.25)
            axis.legend(fontsize="small")
        axes[0][0].set_ylabel("Mean decontaminated BEIR nDCG@10")
        fig.suptitle(f"{family.capitalize()} five-checkpoint dynamics for every LR run")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{family}-training-dynamics-by-run.png", dpi=180)
        plt.close(fig)

        final = subset[subset.stage == 5]
        fig, axis = plt.subplots(figsize=(8, 5))
        for optimizer, values in final.groupby("optimizer"):
            ordered = values.sort_values("learning_rate")
            axis.semilogx(
                ordered.learning_rate,
                ordered.mean_ndcg_at_10,
                marker="o",
                label=optimizer,
            )
        axis.set(xlabel="Hidden-matrix learning rate", ylabel="Final mean nDCG@10")
        axis.set_title(f"{family.capitalize()} learning-rate sensitivity")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{family}-lr-sensitivity.png", dpi=180)
        plt.close(fig)


def _markdown_table(headers: list[str], values: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if i < 2 else "---:" for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in values)
    return "\n".join(lines)


def _format_lr(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _render_results(
    optimizer_rows: list[dict],
    best_dynamics: list[dict],
    task_rows: list[dict],
    paired_rows: list[dict],
    families: tuple[str, ...] = ALL_FAMILIES,
    figure_prefix: str = "../reports/figures",
) -> str:
    final_table = _markdown_table(
        [
            "Family",
            "Optimizer",
            "Same-suite BEIR-best LR",
            "Exploratory BEIR-best final",
            "4-LR mean",
            "4-LR median",
            "SD",
            "Range",
            "4-LR trajectory AUC",
        ],
        [
            [
                row["model_family"],
                row["optimizer"],
                _format_lr(row["best_learning_rate"]),
                f"{row['best_final_ndcg_at_10']:.4f}",
                f"{row['final_mean_across_lrs']:.4f}",
                f"{row['final_median_across_lrs']:.4f}",
                f"{row['final_population_std_across_lrs']:.4f}",
                f"{row['final_min_across_lrs']:.4f}–{row['final_max_across_lrs']:.4f}",
                f"{row['observed_auc_mean_across_lrs']:.4f}",
            ]
            for row in optimizer_rows
        ],
    )

    dynamics_lookup = defaultdict(dict)
    for row in best_dynamics:
        dynamics_lookup[(row["model_family"], row["optimizer"])][row["stage"]] = row[
            "mean_ndcg_at_10"
        ]
    dynamics_table = _markdown_table(
        ["Family", "Optimizer", "20%", "40%", "60%", "80%", "100%", "AUC"],
        [
            [
                family,
                optimizer,
                *[f"{stages[stage]:.4f}" for stage in range(1, 6)],
                f"{next(row for row in optimizer_rows if row['model_family'] == family and row['optimizer'] == optimizer)['best_config_observed_auc_ndcg_at_10']:.4f}",
            ]
            for (family, optimizer), stages in sorted(dynamics_lookup.items())
        ],
    )

    paired_table = _markdown_table(
        [
            "Family",
            "Comparison",
            "W/T/L",
            "Mean Δ",
            "Paired bootstrap 95% CI",
            "Sign p",
            "Holm p",
        ],
        [
            [
                row["model_family"],
                f"{row['optimizer']} − AdamW",
                f"{row['wins']}/{row['ties']}/{row['losses']}",
                f"{row['mean_delta']:+.4f}",
                f"[{row['bootstrap_ci_95_lower']:+.4f}, {row['bootstrap_ci_95_upper']:+.4f}]",
                f"{row['exact_sign_test_p_value']:.4g}",
                f"{row['holm_sign_test_p_value']:.4g}",
            ]
            for row in paired_rows
        ],
    )

    winners = []
    for family in families:
        candidates = [row for row in optimizer_rows if row["model_family"] == family]
        best_tuned = max(candidates, key=lambda row: row["best_final_ndcg_at_10"])
        robust = max(candidates, key=lambda row: row["final_mean_across_lrs"])
        fastest_convergence = max(candidates, key=lambda row: row["observed_auc_mean_across_lrs"])
        family_tasks = [row for row in task_rows if row["model_family"] == family]
        paired = []
        for optimizer in ("muon", "normuon"):
            comparison = next(
                row
                for row in paired_rows
                if row["model_family"] == family and row["optimizer"] == optimizer
            )
            paired.append(
                f"{optimizer} beats AdamW on {comparison['wins']}/{len(family_tasks)} tasks"
                + (f" ({comparison['ties']} ties)" if comparison["ties"] else "")
                + f", mean Δ={comparison['mean_delta']:+.4f} "
                f"(95% CI [{comparison['bootstrap_ci_95_lower']:+.4f}, "
                f"{comparison['bootstrap_ci_95_upper']:+.4f}])"
            )
        winners.append(
            f"- **{family.capitalize()}:** best same-suite BEIR-selected final score is "
            f"{best_tuned['optimizer']} at {_format_lr(best_tuned['best_learning_rate'])} "
            f"({best_tuned['best_final_ndcg_at_10']:.4f}); the highest four-LR mean is "
            f"{robust['optimizer']} ({robust['final_mean_across_lrs']:.4f}); the highest mean "
            f"observed-window AUC is {fastest_convergence['optimizer']} "
            f"({fastest_convergence['observed_auc_mean_across_lrs']:.4f}). Same-suite "
            "BEIR-selected paired " + "; ".join(paired) + "."
        )

    per_task_sections = []
    for family in families:
        values = [row for row in task_rows if row["model_family"] == family]
        per_task_sections.append(
            f"#### {family.capitalize()} same-suite BEIR-selected discovery task scores\n"
        )
        per_task_sections.append(
            _markdown_table(
                ["Task", "AdamW", "Muon", "NorMuon", "Muon − AdamW", "NorMuon − AdamW"],
                [
                    [
                        row["task"],
                        f"{row['adamw']:.4f}",
                        f"{row['muon']:.4f}",
                        f"{row['normuon']:.4f}",
                        f"{row['muon_minus_adamw']:+.4f}",
                        f"{row['normuon_minus_adamw']:+.4f}",
                    ]
                    for row in values
                ],
            )
        )

    family_figures = "\n\n".join(
        f"![{'Dense' if family == 'dense' else 'Late-interaction'} training dynamics]"
        f"({figure_prefix}/{family}-training-dynamics.png)"
        for family in families
    )
    per_run_figures = "\n\n".join(
        f"![{'Dense' if family == 'dense' else 'Late-interaction'} per-run training dynamics]"
        f"({figure_prefix}/{family}-training-dynamics-by-run.png)"
        for family in families
    )
    sensitivity_figures = "\n\n".join(
        f"![{'Dense' if family == 'dense' else 'Late-interaction'} learning-rate sensitivity]"
        f"({figure_prefix}/{family}-lr-sensitivity.png)"
        for family in families
    )
    evaluation_units = 12 * 5 * len(DECONTAMINATED_TASK_NAMES) * len(families)
    holm_scope = (
        "the family of four reported sign tests"
        if families == ALL_FAMILIES
        else "the original four-comparison discovery family"
    )
    return "\n\n".join(
        [
            f"All {evaluation_units:,} planned task/checkpoint evaluations completed. Scores below "
            "are the unweighted mean nDCG@10 across the 14 tasks.",
            "### Final quality and learning-rate robustness\n\n" + final_table,
            "\n".join(winners),
            family_figures,
            "### Five-checkpoint dynamics for every learning-rate run\n\n"
            "Each panel below shows all four LR configurations rather than an optimizer-level "
            "average; every curve contains the formal 20%, 40%, 60%, 80%, and 100% checkpoints.\n\n"
            + per_run_figures,
            "### Dynamics after selecting each optimizer by final nDCG@10 on this same BEIR suite\n\n"
            + dynamics_table,
            "### Paired effects after same-suite BEIR selection\n\n" + paired_table,
            sensitivity_figures,
            "### Per-task final scores after same-suite BEIR selection",
            *per_task_sections,
            "The best-LR comparisons are selected on this same benchmark suite and should therefore "
            "be read as controlled exploratory results, not as an unbiased model-selection estimate. "
            "Paired intervals use 20,000 deterministic task-level bootstrap resamples; the sign-test "
            f"p-value is exact after excluding ties, and Holm p controls {holm_scope}. BEIR tasks "
            "are heterogeneous and not "
            "independent draws, so these are descriptive uncertainty summaries rather than "
            "population inference. The four-LR mean, spread, and complete per-task rows are included "
            "to expose sensitivity rather than reporting only the winning point. Trajectory AUC is "
            "the normalized trapezoidal mean nDCG@10 over the observed 20%–100% checkpoint window; "
            "it measures early-to-late quality, not time before the first checkpoint.",
        ]
    )


def _render_systems(
    rows: list[dict], system_metrics_path: str = "reports/system_metrics.csv"
) -> str:
    table = _markdown_table(
        [
            "Family",
            "Optimizer",
            "Median hours",
            "Samples/s",
            "Throughput vs AdamW",
            "Peak allocated GiB",
            "Optimizer state GiB",
            "Checkpoint GiB",
        ],
        [
            [
                row["model_family"],
                row["optimizer"],
                f"{row['median_wall_time_hours']:.2f}",
                f"{row['median_samples_per_second']:.2f}",
                f"{row['throughput_vs_adamw']:.2f}×",
                f"{row['median_peak_allocated_gib']:.2f}",
                f"{row['median_optimizer_state_gib']:.2f}",
                f"{row['median_checkpoint_gib']:.2f}",
            ]
            for row in rows
        ],
    )
    gpu = rows[0]["gpu_name"] if rows else "unknown GPU"
    world_size = rows[0]["world_size"] if rows else 4
    return (
        f"Every run used {world_size} × {gpu}. Values are medians over the four learning-rate "
        "configurations for that optimizer and family; CUDA memory is the maximum per rank, not "
        "the sum across ranks.\n\n"
        + table
        + "\n\nThe recorded wall time includes training and five full checkpoint writes. Peak CUDA memory "
        "comes from PyTorch allocator counters inside each training process, so the independent "
        "utilization guard process is excluded. For checkpoint-resumed runs, throughput is recomputed "
        "from the sum of non-overlapping useful training segments rather than Trainer's resume-local "
        "runtime; the segment adjustment and original Trainer fields remain in the audit table. "
        f"Exact per-run measurements are in `{system_metrics_path}`."
    )


def _coverage(
    rows: list[dict],
    summary: list[dict],
    configs: list[RunConfig],
    contract_audit: dict,
    dataset_audit: dict,
    training_audit: dict,
) -> dict:
    observed = {(row["model_family"], row["run_id"], row["stage"], row["task"]) for row in rows}
    expected = {
        (config.model_family, config.run_id, stage, task)
        for config in configs
        for stage in range(1, 6)
        for task in DECONTAMINATED_TASK_NAMES
    }
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    evaluation_complete = not missing and not unexpected
    training_complete = training_audit["complete"] and training_audit.get("deep_validation", False)
    return {
        "complete": evaluation_complete
        and contract_audit["complete"]
        and dataset_audit["complete"]
        and training_complete,
        "contract_complete": contract_audit["complete"],
        "dataset_complete": dataset_audit["complete"],
        "training_complete": training_complete,
        "deep_training_artifact_validation": training_audit.get("deep_validation", False),
        "evaluation_complete": evaluation_complete,
        "verified_experiment_runs": contract_audit["observed_runs"],
        "expected_experiment_runs": contract_audit["expected_runs"],
        "verified_training_examples": dataset_audit["verified_rows"],
        "expected_training_examples": 500_000,
        "training_row_manifest_sha256": dataset_audit.get("row_manifest_sha256"),
        "training_dataset_view_fingerprint": dataset_audit.get("training_view_fingerprint"),
        "verified_training_runs": training_audit["verified_runs"],
        "expected_training_runs": training_audit["expected_runs"],
        "verified_training_checkpoints": training_audit["verified_checkpoints"],
        "expected_training_checkpoints": training_audit["expected_checkpoints"],
        "observed_results": len(observed),
        "expected_results": len(expected),
        "observed_checkpoint_summaries": len(summary),
        "expected_checkpoint_summaries": len(configs) * 5,
        "missing": ["/".join(map(str, item)) for item in missing],
        "unexpected": ["/".join(map(str, item)) for item in unexpected],
        "contract_errors": contract_audit["errors"],
        "dataset_errors": dataset_audit["errors"],
        "training_errors": training_audit["errors"],
    }


def _read_report_csv(path: Path) -> list[dict]:
    """Read a frozen report table while restoring scalar CSV types."""

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"Missing frozen discovery report: {path}") from error
    rows = []
    integer = re.compile(r"^-?\d+$")
    for raw in raw_rows:
        row = {}
        for key, value in raw.items():
            if value == "":
                row[key] = None
            elif value in {"True", "False"}:
                row[key] = value == "True"
            elif integer.fullmatch(value):
                row[key] = int(value)
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
        rows.append(row)
    return rows


def _strict_discovery_report_source(
    report_root: Path, configs: list[RunConfig]
) -> tuple[dict, list[dict], list[dict], list[dict], dict[str, dict]]:
    """Validate the already-complete discovery report before making a scoped view."""

    coverage_path = report_root / "coverage.json"
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Dense discovery rendering requires the frozen full coverage report"
        ) from error
    required = {
        "complete": True,
        "contract_complete": True,
        "dataset_complete": True,
        "training_complete": True,
        "evaluation_complete": True,
        "verified_experiment_runs": 24,
        "expected_experiment_runs": 24,
        "verified_training_runs": 24,
        "expected_training_runs": 24,
        "verified_training_checkpoints": 120,
        "expected_training_checkpoints": 120,
        "observed_results": 1_680,
        "expected_results": 1_680,
        "observed_checkpoint_summaries": 120,
        "expected_checkpoint_summaries": 120,
        "missing": [],
        "unexpected": [],
    }
    if any(coverage.get(key) != value for key, value in required.items()):
        raise ValueError("Dense discovery rendering requires strict complete 1,680-unit coverage")

    evaluation_path = report_root / "evaluation_long.csv"
    system_path = report_root / "system_metrics.csv"
    history_path = report_root / "training_history.csv"
    rows = _read_report_csv(evaluation_path)
    observed = {
        (row.get("model_family"), row.get("run_id"), row.get("stage"), row.get("task"))
        for row in rows
    }
    expected = {
        (config.model_family, config.run_id, stage, task)
        for config in configs
        for stage in range(1, 6)
        for task in DECONTAMINATED_TASK_NAMES
    }
    if len(rows) != 1_680 or observed != expected:
        raise ValueError("Frozen discovery evaluation table has incomplete or duplicate identities")
    if any(
        not isinstance(row.get("ndcg_at_10"), (int, float))
        or isinstance(row.get("ndcg_at_10"), bool)
        or not math.isfinite(float(row["ndcg_at_10"]))
        for row in rows
    ):
        raise ValueError("Frozen discovery evaluation table contains an invalid score")

    system_rows = _read_report_csv(system_path)
    identities = {(row.get("model_family"), row.get("run_id")) for row in system_rows}
    expected_identities = {(config.model_family, config.run_id) for config in configs}
    if len(system_rows) != 24 or identities != expected_identities:
        raise ValueError("Frozen discovery system table does not cover all 24 runs")
    history_rows = _read_report_csv(history_path)
    if not history_rows or any(
        (row.get("model_family"), row.get("run_id")) not in expected_identities
        for row in history_rows
    ):
        raise ValueError("Frozen discovery training-history table has invalid run identities")

    sources = {}
    for label, path in {
        "coverage": coverage_path,
        "evaluation_long": evaluation_path,
        "system_metrics": system_path,
        "training_history": history_path,
    }.items():
        sources[label] = {
            "path": str(path.resolve().relative_to(report_root.parent.resolve())),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return coverage, rows, system_rows, history_rows, sources


def _aggregate_scoped_from_reports(
    args: argparse.Namespace,
    *,
    families: tuple[str, ...],
    scope: dict,
    matrix_path: Path,
    configs: list[RunConfig],
) -> None:
    """Create an isolated family view without rewriting the frozen full report."""

    repository_root = matrix_path.parent.parent
    source_coverage, all_rows, all_system_metrics, all_history, sources = (
        _strict_discovery_report_source(repository_root / "reports", configs)
    )
    requested = set(families)
    scoped_configs = select_family_configs(configs, families)
    rows = [row for row in all_rows if row["model_family"] in requested]
    summary = _checkpoint_summaries(rows)
    optimizer_rows, best_dynamics = _optimizer_summaries(summary)
    task_rows = _task_comparison(rows, optimizer_rows, families)
    # Preserve the original four-comparison discovery correction, then filter.
    all_summary = _checkpoint_summaries(all_rows)
    all_optimizer_rows, _ = _optimizer_summaries(all_summary)
    all_task_rows = _task_comparison(all_rows, all_optimizer_rows)
    paired_rows = [
        row for row in _paired_comparisons(all_task_rows) if row["model_family"] in requested
    ]
    system_metrics = [row for row in all_system_metrics if row["model_family"] in requested]
    system_rows = _system_summaries(system_metrics)
    history_rows = [row for row in all_history if row["model_family"] in requested]
    expected_units = len(scoped_configs) * 5 * len(DECONTAMINATED_TASK_NAMES)
    if (
        len(rows) != expected_units
        or len(summary) != len(scoped_configs) * 5
        or len(optimizer_rows) != len(families) * len(EXPECTED_SWEEP)
        or len(task_rows) != len(families) * len(DECONTAMINATED_TASK_NAMES)
        or len(paired_rows) != len(families) * 2
    ):
        raise ValueError("Dense discovery report filtering produced incomplete derived tables")

    coverage = {
        **source_coverage,
        "verified_experiment_runs": len(scoped_configs),
        "expected_experiment_runs": len(scoped_configs),
        "verified_training_runs": len(scoped_configs),
        "expected_training_runs": len(scoped_configs),
        "verified_training_checkpoints": len(scoped_configs) * 5,
        "expected_training_checkpoints": len(scoped_configs) * 5,
        "observed_results": len(rows),
        "expected_results": expected_units,
        "observed_checkpoint_summaries": len(summary),
        "expected_checkpoint_summaries": len(scoped_configs) * 5,
        "families": list(families),
        "scope_amendment": scope,
        "selected_experiment_runs": len(scoped_configs),
        "selected_training_checkpoints": len(scoped_configs) * 5,
        "outputs": None,
        "source_full_discovery": {
            "complete": True,
            "verified_experiment_runs": source_coverage["verified_experiment_runs"],
            "expected_experiment_runs": source_coverage["expected_experiment_runs"],
            "verified_training_runs": source_coverage["verified_training_runs"],
            "expected_training_runs": source_coverage["expected_training_runs"],
            "verified_training_checkpoints": source_coverage["verified_training_checkpoints"],
            "expected_training_checkpoints": source_coverage["expected_training_checkpoints"],
            "observed_results": source_coverage["observed_results"],
            "expected_results": source_coverage["expected_results"],
            "observed_checkpoint_summaries": source_coverage["observed_checkpoint_summaries"],
            "expected_checkpoint_summaries": source_coverage["expected_checkpoint_summaries"],
            "reports": sources,
        },
    }
    output = Path(args.output_dir) / "dense-discovery"
    tables = {
        "evaluation_long": rows,
        "checkpoint_summary": summary,
        "optimizer_summary": optimizer_rows,
        "best_config_dynamics": best_dynamics,
        "best_config_task_comparison": task_rows,
        "paired_comparison": paired_rows,
        "training_history": history_rows,
        "system_metrics": system_metrics,
        "system_summary": system_rows,
    }
    for name, table_rows in tables.items():
        _write_csv(output / f"{name}.csv", table_rows)
    _plot(summary, output)

    def identity(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
        resolved = path.resolve()
        try:
            portable = str(resolved.relative_to(repository_root.resolve()))
        except ValueError:
            portable = str(resolved)
        record: dict[str, Any] = {
            "path": portable,
            "bytes": resolved.stat().st_size,
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        }
        if row_count is not None:
            record["rows"] = row_count
        return record

    coverage["outputs"] = {
        **{
            name: identity(output / f"{name}.csv", row_count=len(table_rows))
            for name, table_rows in tables.items()
        },
        **{
            f"{family}_{suffix.replace('-', '_')}": identity(
                output / "figures" / f"{family}-{suffix}.png"
            )
            for family in families
            for suffix in (
                "training-dynamics",
                "training-dynamics-by-run",
                "lr-sensitivity",
            )
        },
    }
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in coverage.items() if key != "missing"}, indent=2))


def aggregate(args: argparse.Namespace) -> None:
    families, scope = resolve_scope(
        getattr(args, "families", ALL_FAMILIES),
        getattr(args, "scope_amendment", None),
    )
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    configs = load_matrix(matrix_path)
    if families != ALL_FAMILIES:
        _aggregate_scoped_from_reports(
            args,
            families=families,
            scope=scope,
            matrix_path=matrix_path,
            configs=configs,
        )
        return
    rows = collect_evaluations(Path(args.results_root), configs)
    summary = _checkpoint_summaries(rows)
    optimizer_rows, best_dynamics = _optimizer_summaries(summary)
    system_metrics = collect_system_metrics(configs)
    system_rows = _system_summaries(system_metrics)
    contract_audit = audit_experiment_contract(configs)
    dataset_audit = audit_dataset_artifacts(configs)
    all_training_markers_present = all(
        (config.output_dir / "completed.json").is_file() for config in configs
    )
    training_audit = audit_training_artifacts(
        configs,
        deep=args.strict or all_training_markers_present,
        expected_dataset_fingerprint=dataset_audit.get("training_view_fingerprint"),
    )
    coverage = _coverage(rows, summary, configs, contract_audit, dataset_audit, training_audit)
    task_rows = _task_comparison(rows, optimizer_rows) if coverage["complete"] else []
    paired_rows = _paired_comparisons(task_rows) if coverage["complete"] else []

    output = Path(args.output_dir)
    _write_csv(output / "evaluation_long.csv", rows)
    _write_csv(output / "checkpoint_summary.csv", summary)
    _write_csv(output / "optimizer_summary.csv", optimizer_rows)
    _write_csv(output / "best_config_dynamics.csv", best_dynamics)
    _write_csv(output / "best_config_task_comparison.csv", task_rows)
    _write_csv(output / "paired_comparison.csv", paired_rows)
    _write_csv(output / "training_history.csv", collect_training_history(configs))
    _write_csv(output / "system_metrics.csv", system_metrics)
    _write_csv(output / "system_summary.csv", system_rows)
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    _plot(summary, output)
    print(json.dumps({key: value for key, value in coverage.items() if key != "missing"}, indent=2))

    if args.strict and not coverage["complete"]:
        raise RuntimeError("Evaluation matrix is incomplete; see coverage.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--results-root", default="results/decontaminated-beir")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    aggregate(parse_args(argv))


if __name__ == "__main__":
    main()
