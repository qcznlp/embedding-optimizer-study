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
from pathlib import Path

from .collators import TEXT_COLUMNS
from .config import RunConfig, load_matrix
from .data import SOURCE_REPO, SOURCE_REVISION, SPLITS
from .decontamination import DECONTAMINATED_BEIR, DECONTAMINATED_TASK_NAMES

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
RESULTS_MARKERS = ("<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->")
SYSTEMS_MARKERS = ("<!-- SYSTEMS:BEGIN -->", "<!-- SYSTEMS:END -->")
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
    """Validate every safetensors header and tensor extent without materializing weights."""

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
                tensor_count += len(keys)
    except Exception as error:  # noqa: BLE001
        return f"invalid safetensors payload ({type(error).__name__}: {error})"
    return None if tensor_count else "safetensors model has no tensors"


def _deep_checkpoint_problems(checkpoint: Path, expected_step: int, world_size: int) -> list[str]:
    """Parse resumable payloads so non-empty but corrupt files cannot pass strict audit."""

    import gc
    import zipfile

    import torch

    problems: list[str] = []
    if problem := _safetensors_problem(checkpoint):
        problems.append(problem)

    optimizer = None
    try:
        optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
        if (
            not isinstance(optimizer, dict)
            or set(optimizer) != {"state", "param_groups"}
            or not isinstance(optimizer["state"], dict)
            or not optimizer["state"]
            or not isinstance(optimizer["param_groups"], list)
            or not optimizer["param_groups"]
        ):
            problems.append("optimizer state has an invalid structure")
    except Exception as error:  # noqa: BLE001
        problems.append(f"invalid optimizer state ({type(error).__name__}: {error})")
    finally:
        del optimizer
        gc.collect()

    try:
        scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
        if not isinstance(scheduler, dict) or int(scheduler.get("last_epoch", -1)) != expected_step:
            problems.append("scheduler state does not match checkpoint step")
    except Exception as error:  # noqa: BLE001
        problems.append(f"invalid scheduler state ({type(error).__name__}: {error})")

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
            if deep and (problems := _deep_checkpoint_problems(checkpoint, step, world_size)):
                errors.extend(f"{label}/checkpoint-{step}: {problem}" for problem in problems)
                run_checkpoint_errors += 1
                continue
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
    from .evaluate_matrix import _evaluation_source_manifest

    current_sources = _evaluation_source_manifest(Path(__file__).resolve().parents[2])
    if source_files != current_sources:
        raise ValueError(f"Evaluation source files differ from runtime manifest: {path}")
    return versions


def collect_evaluations(results_root: Path, configs: list[RunConfig]) -> list[dict]:
    indexed: dict[tuple, dict] = {}
    versions_reference: dict[str, str] | None = None
    runtime_versions: dict[str, str] | None = None
    for path in results_root.rglob("*Decontaminated.json"):
        if runtime_versions is None:
            runtime_versions = _evaluation_runtime(results_root)
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


def _task_comparison(rows: list[dict], optimizer_rows: list[dict]) -> list[dict]:
    best_runs = {
        (row["model_family"], row["optimizer"]): row["best_run_id"] for row in optimizer_rows
    }
    lookup = {
        (row["model_family"], row["run_id"], row["stage"], row["task"]): row["ndcg_at_10"]
        for row in rows
    }
    output = []
    for family in ("dense", "late"):
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
    task_rows: list[dict], bootstrap_samples: int = 20_000, seed: int = 42
) -> list[dict]:
    """Summarize best-config task deltas with deterministic paired uncertainty."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    output = []
    for family in ("dense", "late"):
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
) -> str:
    final_table = _markdown_table(
        [
            "Family",
            "Optimizer",
            "Best LR",
            "Best final",
            "4-LR mean",
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
    for family in ("dense", "late"):
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
            f"- **{family.capitalize()}:** best tuned final score is "
            f"{best_tuned['optimizer']} at {_format_lr(best_tuned['best_learning_rate'])} "
            f"({best_tuned['best_final_ndcg_at_10']:.4f}); the highest four-LR mean is "
            f"{robust['optimizer']} ({robust['final_mean_across_lrs']:.4f}); the highest mean "
            f"observed-window AUC is {fastest_convergence['optimizer']} "
            f"({fastest_convergence['observed_auc_mean_across_lrs']:.4f}). Best-config paired "
            + "; ".join(paired)
            + "."
        )

    per_task_sections = []
    for family in ("dense", "late"):
        values = [row for row in task_rows if row["model_family"] == family]
        per_task_sections.append(f"#### {family.capitalize()} best-config task scores\n")
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

    return "\n\n".join(
        [
            "All 1,680 planned task/checkpoint evaluations completed. Scores below are the "
            "unweighted mean nDCG@10 across the 14 tasks.",
            "### Final quality and learning-rate robustness\n\n" + final_table,
            "\n".join(winners),
            "![Dense training dynamics](../reports/figures/dense-training-dynamics.png)\n\n"
            "![Late-interaction training dynamics](../reports/figures/late-training-dynamics.png)",
            "### Dynamics of each optimizer's best final configuration\n\n" + dynamics_table,
            "### Paired best-config task effects\n\n" + paired_table,
            "![Dense learning-rate sensitivity](../reports/figures/dense-lr-sensitivity.png)\n\n"
            "![Late-interaction learning-rate sensitivity](../reports/figures/late-lr-sensitivity.png)",
            "### Per-task final scores for the best configuration of each optimizer",
            *per_task_sections,
            "The best-LR comparisons are selected on this same benchmark suite and should therefore "
            "be read as controlled exploratory results, not as an unbiased model-selection estimate. "
            "Paired intervals use 20,000 deterministic task-level bootstrap resamples; the sign-test "
            "p-value is exact after excluding ties, and Holm p controls the family of four reported "
            "sign tests. BEIR tasks are heterogeneous and not independent draws, so these are "
            "descriptive uncertainty summaries rather than population inference. "
            "The four-LR mean, spread, and complete per-task rows are included to expose sensitivity "
            "rather than reporting only the winning point. Trajectory AUC is the normalized "
            "trapezoidal mean nDCG@10 over the observed 20%–100% checkpoint window; it measures "
            "early-to-late quality, not time before the first checkpoint.",
        ]
    )


def _render_systems(rows: list[dict]) -> str:
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
        "Exact per-run measurements are in "
        "`reports/system_metrics.csv`."
    )


def _replace_marked(text: str, markers: tuple[str, str], content: str) -> str:
    begin, end = markers
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one marker pair {markers}")
    before, remainder = text.split(begin)
    _, after = remainder.split(end)
    return f"{before}{begin}\n\n{content}\n\n{end}{after}"


def render_blog(
    blog_path: Path,
    optimizer_rows: list[dict],
    best_dynamics: list[dict],
    task_rows: list[dict],
    paired_rows: list[dict],
    system_rows: list[dict],
) -> None:
    text = blog_path.read_text()
    text = _replace_marked(
        text,
        RESULTS_MARKERS,
        _render_results(optimizer_rows, best_dynamics, task_rows, paired_rows),
    )
    text = _replace_marked(text, SYSTEMS_MARKERS, _render_systems(system_rows))
    text = text.replace(
        "**Experiment status:** training matrix in progress. This document already records the frozen protocol;\n"
        "the results sections are populated only from the checked-in aggregation artifacts after coverage reaches\n"
        "1,680/1,680.",
        "**Experiment status:** complete — 24/24 training runs and 1,680/1,680 checkpoint/task evaluations.",
    )
    blog_path.write_text(text)


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


def aggregate(args: argparse.Namespace) -> None:
    configs = load_matrix(args.matrix)
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

    if coverage["complete"] and not args.no_render_blog:
        render_blog(
            Path(args.blog), optimizer_rows, best_dynamics, task_rows, paired_rows, system_rows
        )
    if args.strict and not coverage["complete"]:
        raise RuntimeError("Evaluation matrix is incomplete; see coverage.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--results-root", default="results/decontaminated-beir")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--blog", default="docs/blog.md")
    parser.add_argument("--no-render-blog", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    aggregate(parse_args(argv))


if __name__ == "__main__":
    main()
