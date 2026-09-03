"""Build the prospective corrected DenseOn state-by-operator factorial.

This module is deliberately separate from the active corrected-matrix controller.  It prepares
the two source-state calibrations and six two-run matrices without changing any source checkpoint
or inheriting optimizer state from the source trajectory.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml

from . import gradient_probe
from .config import OptimizerConfig, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, TensorStore, _atomic_json, _atomic_jsonl, _sha256
from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens
from .optimizers import parameter_partition_name
from .probe_export import _checkpoint_inputs
from .short_branch import audit_short_branch_subset
from .state_operator_factorial_contract import require_factorial_implementation
from .update_geometry import (
    UpdateOperatorConfig,
    _gradient_names,
    _resolve_gradient_shards,
    replay_update_directions,
)

SCIENTIFIC_PROTOCOL = Path("configs/dense_no_packing_state_operator_factorial_protocol.json")
SOURCE_MATRIX = Path("configs/dense_no_packing_retrain.yaml")
COMMON_STATE_SPEC = Path("configs/common_state_probe.json")
PROBE_SPEC = Path("configs/representation_probe.json")
PROBE_ROOT = Path("data/probes/training-1024-seed1729")
SHORT_BRANCH_PROTOCOL = Path("configs/short_branch_protocol.json")
CALIBRATION_ROOT = Path("results/dense-no-packing-state-operator/calibration")
MATRIX_ROOT = Path("configs/generated/dense-no-packing-state-operator")
OUTPUT_ROOT = Path("outputs/dense-no-packing-state-operator-v1")
CONTINUATION_OPERATORS = ("adamw", "muon")
EXPECTED_HIDDEN_TENSORS = 88
EXPECTED_HIDDEN_PARAMETERS = 110_297_088


@dataclasses.dataclass(frozen=True)
class FactorialJob:
    state: str
    source_run_id: str
    operator: str
    seed: int
    run_id: str
    matrix_path: Path


def _repo_path(value: str | Path) -> Path:
    return Path(value).resolve()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = _repo_path(path)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def load_factorial_protocol(
    path: str | Path = SCIENTIFIC_PROTOCOL,
) -> tuple[Path, dict[str, Any]]:
    """Load and fully validate the prospective scientific lock."""

    resolved = resolve_matrix_path(path).resolve()
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported state-by-operator protocol schema")
    if protocol.get("status") != "prospective_scientific_lock_pending_implementation":
        raise ValueError("State-by-operator scientific protocol is not prospectively locked")

    for label, binding in protocol.get("parent_bindings", {}).items():
        target = _repo_path(binding["path"])
        if not target.is_file() or _sha256(target) != binding.get("sha256"):
            raise ValueError(f"Scientific parent binding differs: {label}")

    states = protocol.get("source_states", {}).get("states", [])
    if [state.get("label") for state in states] != ["adamw_state", "muon_state"]:
        raise ValueError("Factorial requires the fixed AdamW and Muon source states")
    source_configs = {config.run_id: config for config in load_matrix(SOURCE_MATRIX)}
    checkpoint_step = protocol["source_states"].get("checkpoint_step")
    for state in states:
        config = source_configs.get(state.get("run_id"))
        if (
            config is None
            or config.model_family != "dense"
            or config.optimizer.name != state.get("optimizer")
            or config.optimizer.lr != state.get("learning_rate")
            or config.dense_can_flatten_inputs is not False
            or Path(state.get("checkpoint", ""))
            != config.output_dir / f"checkpoint-{checkpoint_step}"
        ):
            raise ValueError(f"Source-state binding differs: {state.get('label')}")

    design = protocol.get("factorial_design", {})
    factors = design.get("factors", {})
    seeds = protocol.get("branch_data", {}).get("order_seeds", [])
    if factors != {
        "weight_state": ["adamw_state", "muon_state"],
        "continuation_operator": list(CONTINUATION_OPERATORS),
    }:
        raise ValueError("Factorial coverage differs from the prospective lock")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Factorial requires exactly three fixed order seeds")
    expected_runs = len(states) * len(CONTINUATION_OPERATORS) * len(seeds)
    training = design.get("training", {})
    if (
        design.get("expected_runs") != expected_runs
        or design.get("optimizer_state_at_branch_start")
        != "reset to zero for both operators in every cell"
        or design.get("dense_input_execution")
        != {
            "mode": "independently_padded",
            "sentence_transformers_can_flatten_inputs": False,
        }
        or training.get("expected_optimizer_steps") != 391
        or training.get("expected_checkpoint_steps") != [79, 157, 235, 313, 391]
    ):
        raise ValueError("Factorial training contract differs from the prospective lock")
    calibration = protocol.get("scale_calibration", {})
    target = calibration.get("target_global_hidden_update_to_weight")
    if (
        calibration.get("gradient_groups") != 32
        or calibration.get("gradient_history_steps") != 8
        or calibration.get("weight_decay_included") is not False
        or not isinstance(target, (int, float))
        or not math.isfinite(target)
        or target <= 0
    ):
        raise ValueError("Factorial scale-calibration contract differs")
    return resolved, protocol


def audit_branch_data(
    protocol_path: str | Path = SCIENTIFIC_PROTOCOL,
    *,
    deep: bool = False,
) -> dict[str, Any]:
    """Verify the portable subset receipt, and the host data when requested."""

    resolved, protocol = load_factorial_protocol(protocol_path)
    branch = protocol["branch_data"]
    receipt_binding = branch["portable_receipt"]
    receipt_path = _repo_path(receipt_binding["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if _sha256(receipt_path) != receipt_binding["sha256"]:
        raise ValueError("Portable branch-data receipt differs from the scientific lock")
    expected = {
        "path": branch["path"],
        "rows": branch["rows"],
        "manifest_sha256": branch["manifest_sha256"],
        "selected_sample_ids_sha256": branch["selected_sample_ids_sha256"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Portable branch-data receipt contents differ")

    result = {
        "scientific_protocol_sha256": _sha256(resolved),
        "portable_receipt": _identity(receipt_path),
        "rows": branch["rows"],
        "selected_sample_ids_sha256": branch["selected_sample_ids_sha256"],
        "host_data_verified": False,
    }
    if not deep:
        return result

    root = _repo_path(branch["path"])
    manifest_path = root / "manifest.json"
    ledger_path = root / "rows.jsonl"
    if not manifest_path.is_file() or not ledger_path.is_file() or not (root / "dataset").is_dir():
        raise FileNotFoundError(f"Complete host branch data is required under {root}")
    if (
        _sha256(manifest_path) != branch["manifest_sha256"]
        or _sha256(ledger_path) != branch["row_ledger_sha256"]
    ):
        raise ValueError("Host branch-data identity differs from the scientific lock")
    historical_audit = audit_short_branch_subset(SHORT_BRANCH_PROTOCOL)
    if (
        historical_audit.get("rows") != branch["rows"]
        or historical_audit.get("manifest_sha256") != branch["manifest_sha256"]
        or historical_audit.get("selected_sample_ids_sha256")
        != branch["selected_sample_ids_sha256"]
    ):
        raise ValueError("Deep branch-data audit differs from the factorial binding")
    result.update(
        {
            "host_data_verified": True,
            "manifest_sha256": branch["manifest_sha256"],
            "manifest": _identity(manifest_path),
            "row_ledger": _identity(ledger_path),
            "dataset_fingerprint": historical_audit["dataset_fingerprint"],
            "training_view_fingerprint": historical_audit["training_view_fingerprint"],
        }
    )
    return result


def _state_index(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {state["label"]: state for state in protocol["source_states"]["states"]}


def _run_id(state: str, operator: str) -> str:
    return f"{state.replace('_', '-')}__{operator}-reset"


def factorial_jobs(
    protocol_path: str | Path = SCIENTIFIC_PROTOCOL,
    *,
    matrix_root: str | Path = MATRIX_ROOT,
) -> list[FactorialJob]:
    _, protocol = load_factorial_protocol(protocol_path)
    root = Path(matrix_root)
    jobs = [
        FactorialJob(
            state=state["label"],
            source_run_id=state["run_id"],
            operator=operator,
            seed=int(seed),
            run_id=_run_id(state["label"], operator),
            matrix_path=root / f"{state['label']}-seed{seed}.yaml",
        )
        for state, seed, operator in itertools.product(
            protocol["source_states"]["states"],
            protocol["branch_data"]["order_seeds"],
            CONTINUATION_OPERATORS,
        )
    ]
    labels = [(job.state, job.operator, job.seed, job.run_id) for job in jobs]
    if len(jobs) != protocol["factorial_design"]["expected_runs"] or len(labels) != len(
        set(labels)
    ):
        raise ValueError("Factorial job expansion is not exact and unique")
    return jobs


def _calibration_dir(state: str, root: str | Path = CALIBRATION_ROOT) -> Path:
    return Path(root) / state


def _gradient_request(
    protocol_path: Path,
    protocol: dict[str, Any],
    state: dict[str, Any],
    output_dir: Path,
    *,
    device: str,
) -> dict[str, Any]:
    common_spec = json.loads(COMMON_STATE_SPEC.read_text(encoding="utf-8"))
    return {
        "scientific_protocol": _identity(protocol_path),
        "state": state["label"],
        "source_run_id": state["run_id"],
        "checkpoint": str(_repo_path(state["checkpoint"])),
        "probe": str(PROBE_ROOT.resolve()),
        "probe_spec": _identity(PROBE_SPEC),
        "common_state_selection_spec": _identity(COMMON_STATE_SPEC),
        "expected_probe_manifest_sha256": common_spec["probe_manifest_sha256"],
        "expected_selection": common_spec["selection"],
        "output_dir": str(output_dir.resolve()),
        "gradient_config": {
            "family": "dense",
            "gradient_steps": protocol["scale_calibration"]["gradient_history_steps"],
            "examples_per_gradient": 4,
            "micro_batch_size": 1,
            "seed": common_spec["selection"]["seed"],
            "temperature": 0.02,
            "max_grad_norm": 1.0,
            "model_dtype": "float32",
            "forward_dtype": "bfloat16",
            "storage_dtype": "float32",
            "device": device,
            "flash_attention": True,
            "train_mode": True,
            "gradient_checkpointing": True,
            "weights_advanced": False,
        },
        "input_execution": {
            "mode": "independently_padded",
            "sentence_transformers_can_flatten_inputs": False,
        },
    }


def _load_receipt(path: Path, request: dict[str, Any]) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid padded gradient receipt: {path}") from error
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") not in {"in_progress", "model_verified", "complete"}
        or receipt.get("request") != request
    ):
        raise ValueError("Existing padded gradient receipt differs from the request")
    return receipt


@contextmanager
def _padded_dense_loader(receipt_path: Path, receipt: dict[str, Any]) -> Iterator[None]:
    """Patch only this calibration process and record the observed Dense execution mode."""

    original = gradient_probe._load_model

    def load(*args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        family = kwargs.get("family", args[0] if args else None)
        if family != "dense" or not hasattr(model, "_first_module"):
            raise TypeError("Factorial calibration requires a Dense SentenceTransformer model")
        first = model._first_module()
        if not hasattr(first, "can_flatten_inputs"):
            raise AttributeError("Dense transformer exposes no can_flatten_inputs control")
        first.can_flatten_inputs = False
        if bool(first.can_flatten_inputs):
            raise RuntimeError("Could not disable Dense flattened-input execution")
        receipt["status"] = "model_verified"
        receipt["observed_input_execution"] = {
            "mode": "independently_padded",
            "sentence_transformers_can_flatten_inputs": False,
        }
        _atomic_json(receipt_path, receipt)
        return model

    gradient_probe._load_model = load
    try:
        yield
    finally:
        gradient_probe._load_model = original


def _validate_gradient_manifest(
    manifest: dict[str, Any], request: dict[str, Any], output_dir: Path
) -> None:
    expected = request["expected_selection"]
    config = request["gradient_config"]
    expected_config = {
        "family": "dense",
        "gradient_steps": config["gradient_steps"],
        "examples_per_gradient": config["examples_per_gradient"],
        "micro_batch_size": config["micro_batch_size"],
        "seed": config["seed"],
        "temperature": config["temperature"],
        "max_grad_norm": config["max_grad_norm"],
        "model_dtype": config["model_dtype"],
        "forward_dtype": config["forward_dtype"],
        "storage_dtype": config["storage_dtype"],
        "device": config["device"],
        "flash_attention": config["flash_attention"],
        "model_mode": "train",
        "gradient_checkpointing": config["gradient_checkpointing"],
        "parameter_partition": "hidden",
        "weights_advanced": False,
    }
    selection = manifest.get("selection", [])
    counts = dict(sorted(Counter(row.get("source") for row in selection).items()))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or Path(manifest.get("checkpoint", {}).get("path", "")) != Path(request["checkpoint"])
        or manifest.get("config") != expected_config
        or manifest.get("common_state_spec") is not None
        or manifest.get("probe", {}).get("frozen_spec", {}).get("sha256")
        != request["probe_spec"]["sha256"]
        or manifest.get("probe", {}).get("manifest_sha256")
        != request["expected_probe_manifest_sha256"]
        or manifest.get("selection_sha256") != expected["expected_selection_sha256"]
        or [row.get("sample_id") for row in selection]
        != [item for shard in manifest.get("gradient_shards", []) for item in shard["sample_ids"]]
        or counts != expected["expected_source_counts"]
        or len(selection) != expected["count"]
        or len(manifest.get("gradient_shards", [])) != config["gradient_steps"]
        or manifest.get("partition_summary", {}).get("hidden")
        != {"tensors": EXPECTED_HIDDEN_TENSORS, "parameters": EXPECTED_HIDDEN_PARAMETERS}
    ):
        raise ValueError("Padded gradient manifest differs from the factorial calibration request")
    sample_digest = hashlib.sha256()
    for row in selection:
        sample_digest.update(f"{row['sample_id']}\n".encode())
    if sample_digest.hexdigest() != expected["expected_sample_ids_sha256"]:
        raise ValueError("Padded gradient sample identities differ from the common-state selection")
    gradient_probe._verify_shards(output_dir, manifest)


def export_padded_gradient_calibration(
    state_label: str,
    protocol_path: str | Path = SCIENTIFIC_PROTOCOL,
    *,
    calibration_root: str | Path = CALIBRATION_ROOT,
    device: str = "cuda",
) -> dict[str, Any]:
    """Export one fixed-state gradient history with independently padded Dense inputs."""

    require_factorial_implementation()
    resolved, protocol = load_factorial_protocol(protocol_path)
    state = _state_index(protocol).get(state_label)
    if state is None:
        raise ValueError(f"Unknown source state: {state_label}")
    checkpoint = _repo_path(state["checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    output_dir = _calibration_dir(state_label, calibration_root) / "gradients"
    receipt_path = output_dir / "padded-execution-receipt.json"
    manifest_path = output_dir / "manifest.json"
    request = _gradient_request(resolved, protocol, state, output_dir, device=device)

    if manifest_path.exists() and not receipt_path.is_file():
        raise FileExistsError(
            "Refusing an untagged gradient artifact without a padded-execution receipt"
        )
    if receipt_path.is_file():
        receipt = _load_receipt(receipt_path, request)
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Unmanifested calibration files exist under {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "in_progress",
            "request": request,
        }
        _atomic_json(receipt_path, receipt)

    config = request["gradient_config"]
    with _padded_dense_loader(receipt_path, receipt):
        manifest = gradient_probe.export_gradient_probe(
            checkpoint,
            PROBE_ROOT,
            output_dir,
            family="dense",
            probe_spec=PROBE_SPEC,
            common_state_spec=None,
            gradient_steps=config["gradient_steps"],
            examples_per_gradient=config["examples_per_gradient"],
            micro_batch_size=config["micro_batch_size"],
            seed=config["seed"],
            temperature=config["temperature"],
            max_grad_norm=config["max_grad_norm"],
            model_dtype=config["model_dtype"],
            forward_dtype=config["forward_dtype"],
            storage_dtype=config["storage_dtype"],
            device=config["device"],
            flash_attention=config["flash_attention"],
            train_mode=config["train_mode"],
            gradient_checkpointing=config["gradient_checkpointing"],
        )
    receipt = _load_receipt(receipt_path, request)
    if receipt.get("observed_input_execution") != request["input_execution"]:
        raise RuntimeError("Gradient export completed without observing padded Dense execution")
    _validate_gradient_manifest(manifest, request, output_dir)
    receipt.update(
        {
            "status": "complete",
            "gradient_manifest": _identity(manifest_path),
            "gradient_shards": len(manifest["gradient_shards"]),
            "hidden_partition": manifest["partition_summary"]["hidden"],
        }
    )
    _atomic_json(receipt_path, receipt)
    return receipt


def _calibration_metric_paths(state: str, root: str | Path) -> tuple[Path, Path]:
    update_root = _calibration_dir(state, root) / "directions"
    return update_root / "manifest.json", update_root / "metrics.jsonl"


def analyze_calibration_directions(
    state_label: str,
    protocol_path: str | Path = SCIENTIFIC_PROTOCOL,
    *,
    calibration_root: str | Path = CALIBRATION_ROOT,
    operator_device: str = "cuda",
) -> dict[str, Any]:
    """Compute only the direction norms needed for scale matching, without GB-size updates."""

    require_factorial_implementation()
    resolved, protocol = load_factorial_protocol(protocol_path)
    state = _state_index(protocol).get(state_label)
    if state is None:
        raise ValueError(f"Unknown source state: {state_label}")
    gradient_root = _calibration_dir(state_label, calibration_root) / "gradients"
    receipt_path = gradient_root / "padded-execution-receipt.json"
    gradient_manifest_path = gradient_root / "manifest.json"
    try:
        recorded_request = json.loads(receipt_path.read_text(encoding="utf-8"))["request"]
        gradient_device = recorded_request["gradient_config"]["device"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid padded gradient receipt: {receipt_path}") from error
    request = _gradient_request(
        resolved, protocol, state, gradient_root, device=str(gradient_device)
    )
    receipt = _load_receipt(receipt_path, request)
    if receipt.get("status") != "complete":
        raise ValueError("Padded gradient calibration is not complete")
    gradient_manifest = json.loads(gradient_manifest_path.read_text(encoding="utf-8"))
    _validate_gradient_manifest(gradient_manifest, request, gradient_root)

    checkpoint = _repo_path(state["checkpoint"])
    operator = UpdateOperatorConfig()
    analysis_config = {
        "operator": dataclasses.asdict(operator),
        "operator_device": operator_device,
        "weight_decay_included": False,
        "stored_matched_updates": False,
        "purpose": "global Frobenius scale calibration only",
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "scientific_protocol": _identity(resolved),
        "state": state_label,
        "checkpoint": {"path": str(checkpoint), "inputs": _checkpoint_inputs(checkpoint)},
        "gradient_manifest": _identity(gradient_manifest_path),
        "padded_execution_receipt": _identity(receipt_path),
        "analysis_config": analysis_config,
    }
    manifest_path, metrics_path = _calibration_metric_paths(state_label, calibration_root)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: existing.get(key) for key in identity} != identity:
            raise ValueError("Existing direction calibration differs from the request")
        output = existing.get("output", {})
        if (
            not metrics_path.is_file()
            or output.get("bytes") != metrics_path.stat().st_size
            or output.get("sha256") != _sha256(metrics_path)
        ):
            raise ValueError("Existing direction calibration metrics differ from its manifest")
        return existing
    if metrics_path.exists():
        raise FileExistsError(f"Unmanifested direction metrics exist: {metrics_path}")

    gradient_payload, shards = _resolve_gradient_shards(gradient_manifest_path)
    records: list[dict[str, Any]] = []
    with ExitStack() as stack:
        weights = stack.enter_context(TensorStore(checkpoint))
        names, handles = _gradient_names(stack, shards)
        parameters = sum(math.prod(weights.shape(name)) for name in names)
        if len(names) != EXPECTED_HIDDEN_TENSORS or parameters != EXPECTED_HIDDEN_PARAMETERS:
            raise ValueError("Calibration gradient partition differs from the fixed hidden routing")
        replay_device = torch.device(operator_device)
        for name in names:
            shape = weights.shape(name)
            if len(shape) != 2 or parameter_partition_name(name, len(shape)) != "hidden":
                raise ValueError(f"Calibration tensor is outside hidden routing: {name!r}")
            gradients = [handle.get_tensor(name) for handle in handles]
            if any(tuple(gradient.shape) != shape for gradient in gradients):
                raise ValueError(f"Calibration gradient shape differs for {name!r}")
            directions = replay_update_directions(gradients, operator, device=operator_device)
            weight = weights.tensor(name).to(device=replay_device, dtype=torch.float32)
            if not torch.isfinite(weight).all():
                raise ValueError(f"Calibration checkpoint tensor is non-finite: {name!r}")
            weight_norm = torch.linalg.vector_norm(weight)
            record = {
                "schema_version": SCHEMA_VERSION,
                "tensor": name,
                "shape": list(weight.shape),
                "parameters": math.prod(weight.shape),
                "gradient_steps": len(gradients),
                "weight_frobenius_norm": float(weight_norm.item()),
                "algorithms": {
                    algorithm: {
                        "frobenius_norm": float(
                            torch.linalg.vector_norm(directions[algorithm]).item()
                        )
                    }
                    for algorithm in CONTINUATION_OPERATORS
                },
            }
            if (
                not math.isfinite(float(weight_norm.item()))
                or weight_norm <= 0
                or not all(
                    math.isfinite(item["frobenius_norm"]) and item["frobenius_norm"] > 0
                    for item in record["algorithms"].values()
                )
            ):
                raise ValueError(f"Invalid calibration norm for hidden tensor {name!r}")
            records.append(record)
    _atomic_jsonl(metrics_path, records)
    manifest = {
        **identity,
        "status": "complete",
        "gradient_steps": len(shards),
        "tensors": len(records),
        "parameters": sum(record["parameters"] for record in records),
        "output": {
            "path": metrics_path.name,
            "bytes": metrics_path.stat().st_size,
            "sha256": _sha256(metrics_path),
        },
        "gradient_manifest_status": gradient_payload["status"],
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _load_calibration_metrics(
    state: str,
    calibration_root: str | Path = CALIBRATION_ROOT,
    *,
    protocol_path: str | Path = SCIENTIFIC_PROTOCOL,
    verify_provenance: bool = True,
) -> tuple[dict[str, Any], dict[str, float]]:
    manifest_path, metrics_path = _calibration_metric_paths(state, calibration_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest.get("output", {})
    if (
        manifest.get("status") != "complete"
        or manifest.get("state") != state
        or manifest.get("tensors") != EXPECTED_HIDDEN_TENSORS
        or manifest.get("parameters") != EXPECTED_HIDDEN_PARAMETERS
        or manifest.get("analysis_config", {}).get("weight_decay_included") is not False
        or output.get("path") != metrics_path.name
        or output.get("bytes") != metrics_path.stat().st_size
        or output.get("sha256") != _sha256(metrics_path)
    ):
        raise ValueError(f"{state}: direction calibration manifest is inconsistent")
    if verify_provenance:
        resolved, protocol = load_factorial_protocol(protocol_path)
        source = _state_index(protocol).get(state)
        if source is None:
            raise ValueError(f"{state}: source state is absent from the scientific protocol")
        gradient_identity = manifest.get("gradient_manifest", {})
        receipt_identity = manifest.get("padded_execution_receipt", {})
        gradient_path = Path(gradient_identity.get("path", ""))
        receipt_path = Path(receipt_identity.get("path", ""))
        expected_analysis = {
            "operator": dataclasses.asdict(UpdateOperatorConfig()),
            "operator_device": "cuda:0",
            "weight_decay_included": False,
            "stored_matched_updates": False,
            "purpose": "global Frobenius scale calibration only",
        }
        if (
            manifest.get("scientific_protocol", {}).get("sha256") != _sha256(resolved)
            or manifest.get("checkpoint", {}).get("path") != str(_repo_path(source["checkpoint"]))
            or manifest.get("analysis_config") != expected_analysis
            or manifest.get("gradient_steps")
            != protocol["scale_calibration"]["gradient_history_steps"]
            or not gradient_path.is_file()
            or gradient_identity.get("bytes") != gradient_path.stat().st_size
            or gradient_identity.get("sha256") != _sha256(gradient_path)
            or not receipt_path.is_file()
            or receipt_identity.get("bytes") != receipt_path.stat().st_size
            or receipt_identity.get("sha256") != _sha256(receipt_path)
        ):
            raise ValueError(f"{state}: direction calibration provenance differs")
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    if (
        len(rows) != EXPECTED_HIDDEN_TENSORS
        or len({row["tensor"] for row in rows}) != len(rows)
        or any(row.get("gradient_steps") not in {None, 8} for row in rows)
    ):
        raise ValueError(f"{state}: direction calibration tensor coverage differs")
    weight_sq = sum(float(row["weight_frobenius_norm"]) ** 2 for row in rows)
    ratios = {
        algorithm: math.sqrt(
            sum(float(row["algorithms"][algorithm]["frobenius_norm"]) ** 2 for row in rows)
            / weight_sq
        )
        for algorithm in CONTINUATION_OPERATORS
    }
    if weight_sq <= 0 or not all(math.isfinite(value) and value > 0 for value in ratios.values()):
        raise ValueError(f"{state}: invalid aggregate direction ratios")
    return {"manifest": _identity(manifest_path), "metrics": _identity(metrics_path)}, ratios


def _optimizer_payload(operator: str, lr: float, auxiliary_lr: float) -> dict[str, Any]:
    name = "hybrid_adamw" if operator == "adamw" else operator
    payload = dataclasses.asdict(OptimizerConfig(name=name, lr=lr, aux_lr=auxiliary_lr))
    if operator == "muon":
        payload["ns_implementation"] = "unfused-bfloat16-v1"
    return payload


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def generate_factorial_matrices(
    protocol_path: str | Path = SCIENTIFIC_PROTOCOL,
    *,
    matrix_root: str | Path = MATRIX_ROOT,
    calibration_root: str | Path = CALIBRATION_ROOT,
    deep_data_audit: bool = True,
) -> dict[str, Any]:
    """Generate six source-state/seed matrices containing two reset operators each."""

    require_factorial_implementation()
    resolved, protocol = load_factorial_protocol(protocol_path)
    subset = audit_branch_data(resolved, deep=deep_data_audit)
    root = Path(matrix_root).resolve()
    source_matrix = SOURCE_MATRIX.resolve()
    common = load_matrix(source_matrix)[0]
    states = _state_index(protocol)
    target = float(protocol["scale_calibration"]["target_global_hidden_update_to_weight"])
    calibration: dict[str, Any] = {}
    ratios: dict[str, dict[str, float]] = {}
    learning_rates: dict[str, dict[str, float]] = {}
    for label in states:
        calibration[label], ratios[label] = _load_calibration_metrics(
            label, calibration_root, protocol_path=resolved
        )
        learning_rates[label] = {
            operator: target / ratios[label][operator] for operator in CONTINUATION_OPERATORS
        }

    design = protocol["factorial_design"]
    training = design["training"]
    matrix_identities = []
    formal_runtime = SOURCE_MATRIX.parent / "formal_runtime.json"
    for label, seed in itertools.product(states, protocol["branch_data"]["order_seeds"]):
        state = states[label]
        path = root / f"{label}-seed{seed}.yaml"
        payload = {
            "formal_runtime": os.path.relpath(formal_runtime.resolve(), root),
            "common": {
                "dataset_path": protocol["branch_data"]["path"],
                "output_root": str(OUTPUT_ROOT / label / f"seed{seed}"),
                "seed": seed,
                "epochs": training["epochs"],
                "global_batch_size": training["global_batch_size"],
                "micro_batch_size": training["micro_batch_size"],
                "max_length": training["max_length"],
                "warmup_ratio": common.warmup_ratio,
                "max_grad_norm": common.max_grad_norm,
                "dataloader_workers": common.dataloader_workers,
                "gradient_checkpointing": common.gradient_checkpointing,
                "flash_attention": common.flash_attention,
                "dense_can_flatten_inputs": False,
                "wandb_project": common.wandb_project,
                "wandb_entity": common.wandb_entity,
                "checkpoint_fractions": training["checkpoint_fractions"],
            },
            "models": {
                "dense": {
                    "model_name": state["checkpoint"],
                    "temperature": 0.02,
                }
            },
            "runs": [
                {
                    "id": _run_id(label, operator),
                    "model_family": "dense",
                    "optimizer": _optimizer_payload(
                        operator,
                        learning_rates[label][operator],
                        design["auxiliary_adamw_learning_rate"],
                    ),
                }
                for operator in CONTINUATION_OPERATORS
            ],
            "provenance": {
                "scientific_protocol_sha256": _sha256(resolved),
                "source_state": label,
                "source_checkpoint": state["checkpoint"],
                "branch_subset_manifest_sha256": protocol["branch_data"]["manifest_sha256"],
                "calibration_metrics_sha256": calibration[label]["metrics"]["sha256"],
                "target_global_hidden_update_to_weight": target,
                "derived_hidden_learning_rates": learning_rates[label],
                "optimizer_state_at_branch_start": "reset",
            },
        }
        _atomic_yaml(path, payload)
        matrix_identities.append(_identity(path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scientific_protocol": _identity(resolved),
        "source_matrix": _identity(source_matrix),
        "subset": subset,
        "calibration": calibration,
        "global_per_unit_lr_update_to_weight": ratios,
        "target_global_hidden_update_to_weight": target,
        "derived_hidden_learning_rates": learning_rates,
        "order_seeds": protocol["branch_data"]["order_seeds"],
        "expected_runs": design["expected_runs"],
        "expected_matrices": len(states) * len(protocol["branch_data"]["order_seeds"]),
        "matrices": matrix_identities,
    }
    _atomic_json(root / "manifest.json", manifest)
    audit_factorial_matrices(
        resolved,
        matrix_root=root,
        calibration_root=calibration_root,
        deep_data_audit=deep_data_audit,
    )
    return manifest


def audit_factorial_matrices(
    protocol_path: str | Path = SCIENTIFIC_PROTOCOL,
    *,
    matrix_root: str | Path = MATRIX_ROOT,
    calibration_root: str | Path = CALIBRATION_ROOT,
    deep_data_audit: bool = False,
) -> dict[str, Any]:
    resolved, protocol = load_factorial_protocol(protocol_path)
    subset = audit_branch_data(resolved, deep=deep_data_audit)
    root = Path(matrix_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    states = _state_index(protocol)
    expected_paths = [
        root / f"{state}-seed{seed}.yaml"
        for state, seed in itertools.product(states, protocol["branch_data"]["order_seeds"])
    ]
    if (
        manifest.get("status") != "complete"
        or manifest.get("scientific_protocol", {}).get("sha256") != _sha256(resolved)
        or manifest.get("source_matrix", {}).get("sha256") != _sha256(SOURCE_MATRIX)
        or manifest.get("subset", {}).get("portable_receipt", {}).get("sha256")
        != subset["portable_receipt"]["sha256"]
        or manifest.get("matrices") != [_identity(path) for path in expected_paths]
        or manifest.get("expected_runs") != protocol["factorial_design"]["expected_runs"]
    ):
        raise ValueError("Factorial matrix manifest is inconsistent")

    target = float(protocol["scale_calibration"]["target_global_hidden_update_to_weight"])
    expected_lrs = {}
    for state in states:
        _, ratios = _load_calibration_metrics(state, calibration_root, protocol_path=resolved)
        expected_lrs[state] = {
            operator: target / ratios[operator] for operator in CONTINUATION_OPERATORS
        }
    if manifest.get("derived_hidden_learning_rates") != expected_lrs:
        raise ValueError("Factorial learning rates differ from the calibration metrics")

    observed = 0
    for state, seed, path in (
        (state, seed, root / f"{state}-seed{seed}.yaml")
        for state, seed in itertools.product(states, protocol["branch_data"]["order_seeds"])
    ):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        runs = load_matrix(path)
        indexed = {
            "adamw" if run.optimizer.name == "hybrid_adamw" else run.optimizer.name: run
            for run in runs
        }
        if (
            set(indexed) != set(CONTINUATION_OPERATORS)
            or len(runs) != 2
            or {run.seed for run in runs} != {seed}
            or {run.dataset_path for run in runs} != {protocol["branch_data"]["path"]}
            or {run.model_name for run in runs} != {states[state]["checkpoint"]}
            or any(run.model_revision is not None for run in runs)
            or any(run.dense_can_flatten_inputs is not False for run in runs)
            or raw.get("provenance", {}).get("scientific_protocol_sha256") != _sha256(resolved)
            or raw.get("provenance", {}).get("optimizer_state_at_branch_start") != "reset"
        ):
            raise ValueError(f"Factorial matrix coverage differs for {state}/seed{seed}")
        for operator, run in indexed.items():
            if (
                run.run_id != _run_id(state, operator)
                or not math.isclose(
                    run.optimizer.lr,
                    expected_lrs[state][operator],
                    rel_tol=1e-15,
                    abs_tol=0.0,
                )
                or run.optimizer.aux_lr
                != protocol["factorial_design"]["auxiliary_adamw_learning_rate"]
                or run.checkpoint_fractions
                != tuple(protocol["factorial_design"]["training"]["checkpoint_fractions"])
            ):
                raise ValueError(f"Factorial recipe differs for {state}/{operator}/seed{seed}")
        observed += len(runs)
    if observed != protocol["factorial_design"]["expected_runs"]:
        raise ValueError("Factorial run cardinality differs")
    return {
        "status": "complete",
        "manifest_sha256": _sha256(manifest_path),
        "matrices": len(expected_paths),
        "runs": observed,
        "derived_hidden_learning_rates": expected_lrs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "audit-protocol",
            "calibrate-gradients",
            "calibrate-directions",
            "generate",
            "audit",
        ),
    )
    parser.add_argument("--protocol", type=Path, default=SCIENTIFIC_PROTOCOL)
    parser.add_argument("--state", choices=("adamw_state", "muon_state"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--operator-device", default="cuda")
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--gpu-lock-dir",
        type=Path,
        default=Path("logs/dense-only-runtime/gpu-leases"),
    )
    parser.add_argument("--gpu-lock-timeout-seconds", type=float, default=86_400.0)
    parser.add_argument("--calibration-root", type=Path, default=CALIBRATION_ROOT)
    parser.add_argument("--matrix-root", type=Path, default=MATRIX_ROOT)
    parser.add_argument("--portable-data-audit", action="store_true")
    args = parser.parse_args(argv)
    if args.action.startswith("calibrate-") and args.state is None:
        parser.error(f"{args.action} requires --state")
    gpu_tokens = parse_gpu_tokens(args.gpus)
    if args.action.startswith("calibrate-") and len(gpu_tokens) != 1:
        parser.error("Calibration requires exactly one leased GPU token")
    if args.gpu_lock_timeout_seconds <= 0:
        parser.error("--gpu-lock-timeout-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.action == "audit-protocol":
        path, protocol = load_factorial_protocol(args.protocol)
        result = {
            "status": "locked",
            "protocol": _identity(path),
            "jobs": len(factorial_jobs(path, matrix_root=args.matrix_root)),
            "states": [state["label"] for state in protocol["source_states"]["states"]],
            "branch_data": audit_branch_data(path, deep=not args.portable_data_audit),
        }
    elif args.action == "calibrate-gradients":
        tokens = parse_gpu_tokens(args.gpus)
        with acquire_gpu_lease(
            tokens,
            lock_dir=args.gpu_lock_dir.resolve(),
            timeout_seconds=args.gpu_lock_timeout_seconds,
            purpose=f"state-operator-gradient-calibration:{args.state}",
            ledger_path=Path("logs/state-operator-factorial")
            / f"gradient-calibration-{args.state}-{os.getpid()}.json",
        ):
            result = export_padded_gradient_calibration(
                args.state,
                args.protocol,
                calibration_root=args.calibration_root,
                device=args.device,
            )
    elif args.action == "calibrate-directions":
        tokens = parse_gpu_tokens(args.gpus)
        with acquire_gpu_lease(
            tokens,
            lock_dir=args.gpu_lock_dir.resolve(),
            timeout_seconds=args.gpu_lock_timeout_seconds,
            purpose=f"state-operator-direction-calibration:{args.state}",
            ledger_path=Path("logs/state-operator-factorial")
            / f"direction-calibration-{args.state}-{os.getpid()}.json",
        ):
            result = analyze_calibration_directions(
                args.state,
                args.protocol,
                calibration_root=args.calibration_root,
                operator_device=args.operator_device,
            )
    elif args.action == "generate":
        result = generate_factorial_matrices(
            args.protocol,
            matrix_root=args.matrix_root,
            calibration_root=args.calibration_root,
            deep_data_audit=not args.portable_data_audit,
        )
    else:
        result = audit_factorial_matrices(
            args.protocol,
            matrix_root=args.matrix_root,
            calibration_root=args.calibration_root,
            deep_data_audit=not args.portable_data_audit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
