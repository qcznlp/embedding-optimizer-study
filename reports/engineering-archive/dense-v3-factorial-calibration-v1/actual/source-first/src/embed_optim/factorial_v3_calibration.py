"""Numerical calibration component for the two genuine v3 source states.

This module has no scheduler or GPU-acquisition entrypoint. Its GPU producer is
for a separately admitted, exclusively leased worker; it does not waive current
resource, source-release, primary-completion or formal-branch admission gates.
It reuses the original fixed-probe gradient and direction computations, with an
observed corrected model loader and an explicit new source/data provenance chain.
"""

from __future__ import annotations

import copy
import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from . import dense_numerical_contract as numerical
from . import factorial_v3_inputs, factorial_v3_optimizer
from . import gradient_probe, optimizers, update_geometry
from .factorial_v3_inputs import (
    ORDER_SEEDS, STATES, calibration_request,
    load_inputs, require_gradient_manifest,
)
from .factorial_v3_optimizer import denseon_shapes
from .primary_contract import digest, file_identity, read_json, require_same, verify_file

SCOPE = "dense-v3-factorial-calibration-v1"
SOURCES = {
    "optimizers.py": "7d7d4f200c410cb1b24e1773ae72811582f43a0c794a5d1b2af269c123e69b1e",
    "dense_numerical_contract.py": "207d8545f7d329979ebbe0e90317f95f6ca01bb310eeff040fe3de461956d8f4",
    "gradient_probe.py": "3e06f35eb3a6a294be0da6b78ca6ae6846357c6ff2aac70a54bd99e4d384ddac",
    "update_geometry.py": "93c295cad852fbeb3e065077442f7f94740d7b6981dd16e4c141fc4377c3fbb4",
}


def require_sources():
    result = {}
    for module in (optimizers, numerical, gradient_probe, update_geometry):
        path = Path(module.__file__)
        value = file_identity(path)
        if value["sha256"] != SOURCES[path.name]:
            raise ValueError("Calibration imported a different numerical source")
        result[path.name] = value
    for module in (factorial_v3_inputs, factorial_v3_optimizer):
        result[Path(module.__file__).name] = file_identity(module.__file__)
    result[Path(__file__).name] = file_identity(__file__)
    return result


def _exclusive_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _verify_loaded_weights(model, checkpoint):
    numerical.require_backward(model)
    if model.max_seq_length != 8192 or model[0].can_flatten_inputs is not False:
        raise ValueError("Loaded calibration model has different execution settings")
    parameters = dict(model.named_parameters())
    if {n: list(p.shape) for n, p in parameters.items()} != denseon_shapes():
        raise ValueError("Calibration model does not have the complete DenseOn topology")
    with safe_open(str(Path(checkpoint) / "model.safetensors"), framework="pt", device="cpu") as store:
        if {"0.model." + n for n in store.keys()} != set(parameters):
            raise ValueError("Saved and loaded calibration parameter names differ")
        for name in store.keys():
            parameter = parameters["0.model." + name]
            if (not parameter.requires_grad or parameter.dtype != torch.float32
                or not torch.equal(parameter.detach().cpu(), store.get_tensor(name))):
                raise ValueError("Loaded calibration weights were changed or cast")


def export_gradients(locations, state, output):
    """Actual GPU producer; call only inside an independently admitted GPU worker.

    No old output is resumed or overwritten. Failures retain the new request and
    partial shards. There is no implicit retry or CPU/SDPA numerical fallback.
    """
    source = require_sources()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Calibration worker requires exactly one isolated visible CUDA device")
    inputs = load_inputs(locations)
    request = calibration_request(inputs, state)
    from .runtime import verify_runtime_spec
    runtime = verify_runtime_spec(Path(locations.repository) / "configs/formal_runtime.json")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _exclusive_json(output / "request.json", request)
    original = gradient_probe._load_model
    observed = []

    def load(family, checkpoint, *, dtype, device, flash_attention):
        if (family != "dense" or str(checkpoint) != request["source"]["checkpoint"]
            or dtype != torch.float32 or device != "cuda:0" or flash_attention is not True):
            raise ValueError("Calibration loader received an unbound model request")
        model = original(family, checkpoint, dtype=dtype, device=device, flash_attention=True)
        _verify_loaded_weights(model, checkpoint)
        if model.device != torch.device("cuda:0"):
            raise ValueError("Calibration model was loaded on another device")
        if model[0].model.config._attn_implementation != "flash_attention_2":
            raise ValueError("Calibration did not load the specified attention backend")
        observed.append(model)
        return model

    gradient_probe._load_model = load
    try:
        manifest = gradient_probe.export_gradient_probe(
            request["source"]["checkpoint"], request["probe"], output / "gradients",
            family="dense", probe_spec=request["probe_spec"]["path"],
            common_state_spec=request["common_state_spec"]["path"],
            gradient_steps=8, examples_per_gradient=4, micro_batch_size=1,
            seed=2718, temperature=0.02, max_grad_norm=1.0,
            model_dtype="float32", forward_dtype="bfloat16", storage_dtype="float32",
            device="cuda:0", flash_attention=True, train_mode=True,
            gradient_checkpointing=True, overwrite=False,
        )
        if len(observed) != 1:
            raise ValueError("Calibration did not freshly load exactly one genuine model")
        _verify_loaded_weights(observed[0], request["source"]["checkpoint"])
        require_gradient_manifest(manifest, request, output / "gradients")
        require_same(source, require_sources())
        receipt = {
            "scope": SCOPE, "status": "gradients_complete", "request": request,
            "sources": source, "runtime": runtime,
            "manifest": file_identity(output / "gradients/manifest.json"),
            "loaded_weights_unchanged": True, "fresh_gpu_gradient_history": True,
            "historical_gradients_reused": False, "formal_branch_admission": False,
        }
        _exclusive_json(output / "gradient-receipt.json", receipt)
        return receipt
    finally:
        gradient_probe._load_model = original
        observed.clear()


def calibrated_rates(rows, expected_shapes):
    """The original global-hidden Frobenius rule, not a mean of tensor ratios."""
    if (not expected_shapes or len(rows) != len(expected_shapes)
        or len({r["tensor"] for r in rows}) != len(rows)
        or {r["tensor"] for r in rows} != set(expected_shapes)):
        raise ValueError("Calibration does not cover exactly the required hidden matrices")
    for row in rows:
        if (row["shape"] != expected_shapes[row["tensor"]]
            or type(row["gradient_steps"]) is not int or row["gradient_steps"] != 8
            or row["parameters"] != math.prod(row["shape"])
            or set(row["algorithms"]) != {"adamw", "muon"}):
            raise ValueError("Calibration matrix shape, history or operator set differs")
        values = [row["weight_frobenius_norm"], *[
            row["algorithms"][op]["frobenius_norm"] for op in ("adamw", "muon")]]
        if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Calibration requires positive finite weight/direction norms")
    # Retain the original sorted tensor order and scalar reduction, including
    # its FP32 per-tensor norms followed by Python double scalar aggregation.
    rows = sorted(rows, key=lambda r: r["tensor"])
    weight_sq = sum(float(r["weight_frobenius_norm"]) ** 2 for r in rows)
    ratios = {op: math.sqrt(sum(float(r["algorithms"][op]["frobenius_norm"]) ** 2
                                for r in rows) / weight_sq) for op in ("adamw", "muon")}
    rates = {op: 5e-4 / ratio for op, ratio in ratios.items()}
    if not all(math.isfinite(v) and v > 0 for v in (*ratios.values(), *rates.values())):
        raise ValueError("Invalid global calibration scale")
    return {"target_global_hidden_update_to_weight": 5e-4,
            "global_direction_to_weight": ratios, "hidden_learning_rates": rates,
            "weight_decay_included": False, "fixed_probe_not_first_training_batch": True}


def direction_rows(manifest, request, gradient_root, *, device):
    """Replay authentic hidden histories; a caller may use CPU for diagnostics only."""
    require_gradient_manifest(manifest, request, gradient_root)
    wanted = {r["checkpoint_name"]: r["shape"] for r in manifest["parameter_name_mapping"]}
    rows = []
    with ExitStack() as stack:
        weights = stack.enter_context(safe_open(request["source"]["checkpoint"] + "/model.safetensors",
                                                framework="pt", device="cpu"))
        shards = [stack.enter_context(safe_open(str(Path(gradient_root) / shard["path"]),
                                                framework="pt", device="cpu"))
                  for shard in manifest["gradient_shards"]]
        if any(set(s.keys()) != set(wanted) for s in shards):
            raise ValueError("Gradient shard hidden tensor names differ")
        for name in sorted(wanted):
            gradients = [shard.get_tensor(name) for shard in shards]
            weight = weights.get_tensor(name).to(device=device, dtype=torch.float32)
            if (list(weight.shape) != wanted[name] or not torch.isfinite(weight).all()
                or any(list(g.shape) != wanted[name] or g.dtype != torch.float32 for g in gradients)):
                raise ValueError("Calibration weight/gradient shape or storage dtype differs")
            directions = update_geometry.replay_update_directions(gradients, device=device)
            rows.append({"tensor": name, "shape": wanted[name], "parameters": weight.numel(),
                         "gradient_steps": 8, "weight_frobenius_norm": float(weight.norm().item()),
                         "algorithms": {op: {"frobenius_norm": float(directions[op].norm().item())}
                                        for op in ("adamw", "muon")}})
    return rows, calibrated_rates(rows, wanted)


def replay_calibration(locations, state, directory):
    """Actual CUDA replay of a new v3 gradient receipt; no model weights advance."""
    source = require_sources()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Direction calibration requires one isolated visible CUDA device")
    inputs = load_inputs(locations)
    request = calibration_request(inputs, state)
    root = Path(directory)
    receipt = read_json(root / "gradient-receipt.json")
    require_same(receipt["request"], request)
    require_same(receipt["sources"], source)
    if (receipt.get("scope") != SCOPE or receipt.get("status") != "gradients_complete"
        or receipt.get("fresh_gpu_gradient_history") is not True
        or receipt.get("loaded_weights_unchanged") is not True
        or receipt.get("historical_gradients_reused") is not False):
        raise ValueError("Require a fresh complete corrected calibration history")
    verify_file(root / "gradients/manifest.json", receipt["manifest"])
    output = root / "directions"
    output.mkdir(exist_ok=False)
    rows, rates = direction_rows(read_json(root / "gradients/manifest.json"), request,
                                root / "gradients", device="cuda:0")
    _exclusive_json(output / "rows.json", rows)
    result = {"scope": SCOPE, "status": "complete", "request": request,
              "sources": source,
              "gradient_receipt": file_identity(root / "gradient-receipt.json"),
              "rows": file_identity(output / "rows.json"), "calibration": rates,
              "operator_device": "cuda:0", "formal_branch_admission": False}
    _exclusive_json(output / "calibration.json", result)
    return result


def read_calibration(inputs, state, directory):
    """Verify the actual saved chain and recompute every rate from all norm rows.

    This is a file/norm readback, not fresh GPU gradient/direction computation.
    Missing v3 tags, stale source states and hand-entered calibrated rates fail.
    """
    request = calibration_request(inputs, state)
    root = Path(directory)
    result = read_json(root / "directions/calibration.json")
    receipt = read_json(root / "gradient-receipt.json")
    require_same(result["request"], request)
    require_same(receipt["request"], request)
    require_same(result["sources"], require_sources())
    require_same(receipt["sources"], result["sources"])
    require_same(read_json(root / "request.json"), request)
    if (result.get("scope") != SCOPE or result.get("status") != "complete"
        or result.get("operator_device") != "cuda:0"
        or result.get("formal_branch_admission") is not False
        or receipt.get("scope") != SCOPE or receipt.get("status") != "gradients_complete"
        or receipt.get("fresh_gpu_gradient_history") is not True
        or receipt.get("loaded_weights_unchanged") is not True
        or receipt.get("historical_gradients_reused") is not False
        or receipt.get("formal_branch_admission") is not False):
        raise ValueError("Require complete corrected gradient and direction receipts")
    verify_file(root / "gradient-receipt.json", result["gradient_receipt"])
    verify_file(root / "gradients/manifest.json", receipt["manifest"])
    verify_file(root / "directions/rows.json", result["rows"])
    manifest = read_json(root / "gradients/manifest.json")
    require_gradient_manifest(manifest, request, root / "gradients")
    shapes = {r["name"].removeprefix("0.model."): r["shape"]
              for r in request["source"]["named_layout"][0]["members"]}
    actual = calibrated_rates(read_json(root / "directions/rows.json"), shapes)
    require_same(result["calibration"], actual)
    return result


def load_calibrated_cells(locations, directories):
    """Read both complete calibration outputs before constructing any branch plan."""
    if set(directories) != set(STATES):
        raise ValueError("Supply exactly the two fixed calibration directories")
    inputs = load_inputs(locations)
    calibrated = {state: read_calibration(inputs, state, directories[state]) for state in STATES}
    return branch_cells(inputs, calibrated)


def branch_cells(inputs, calibrated):
    """Expand all four genuine calibrated cells across the three fixed order seeds."""
    if set(calibrated) != set(STATES):
        raise ValueError("Both fixed source-state calibrations are required")
    jobs = []
    for seed in ORDER_SEEDS:
        for state in STATES:
            calibration = calibrated[state]
            require_same(calibration["request"], calibration_request(inputs, state))
            if (calibration.get("scope") != SCOPE or calibration.get("status") != "complete"
                or calibration.get("operator_device") != "cuda:0"):
                raise ValueError("Only complete genuine CUDA calibration can set branch rates")
            for operator in ("adamw", "muon"):
                rate = calibration["calibration"]["hidden_learning_rates"][operator]
                if type(rate) is not float or not math.isfinite(rate) or rate <= 0:
                    raise ValueError("Invalid calibrated branch rate")
                jobs.append({"state": state, "operator": operator, "seed": seed,
                             "hidden_lr": rate, "auxiliary_lr": 3e-6,
                             "optimizer_state": "fresh_zero_state_after_calibration",
                             "calibration_sha256": digest(calibration),
                             "source": copy.deepcopy(inputs["sources"][state]),
                             "branch_data": copy.deepcopy(inputs["branch"]),
                             "steps": 391, "checkpoint_steps": [79, 157, 235, 313, 391],
                             "execution_authorized": False})
    return jobs
