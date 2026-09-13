"""Calibration tests: genuine recorded identities, explicitly synthetic gradients.

No case is a primary GPU calibration or a formally admitted branch. Synthetic
norm/shard fixtures test consumers; independent scalar/update references test
numerics separately. The real input read is a distinct archived command.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from embed_optim import factorial_v3_calibration as calibration
from embed_optim import factorial_v3_inputs as inputs_module
from embed_optim.optimizers import EmbeddingOptimizer
from embed_optim.primary_contract import file_identity

REPOSITORY = Path(__file__).resolve().parents[1]
ARCHIVE = REPOSITORY / inputs_module.EVIDENCE


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def recorded_inputs():
    """Load known metadata only: deliberately not invoking actual input admission."""
    data = json.loads((ARCHIVE / "data-first.json").read_text())
    states = json.loads((ARCHIVE / "states-first.json").read_text())["states"]
    return {
        "scope": inputs_module.SCOPE, "scientific_admission": False,
        "execution_authorized": False, "accepted_audit_digests": inputs_module.AUDITS.copy(),
        "branch": data["branch"],
        "calibration": {"probe": "/synthetic/probe", **data["calibration"]},
        "common_state_spec": {"path": "/synthetic/common-state.json", "bytes": 1, "sha256": "c" * 64},
        "probe_spec": {"path": "/synthetic/probe-spec.json", "bytes": 1, "sha256": "d" * 64},
        "sources": {r["label"]: {
            "run_id": r["genuine_v3_run_id"], "checkpoint_step": 2345,
            "checkpoint": r["checkpoint"], "native_checkpoint": r["native_checkpoint"],
            "named_layout": r["named_layout"],
        } for r in states},
    }


def gradient_fixture(tmp_path, state="adamw_state", *, numeric=False):
    inputs = recorded_inputs()
    request = inputs_module.calibration_request(inputs, state)
    root = tmp_path / state
    root.mkdir(parents=True)
    mapping = [{"model_name": r["name"], "checkpoint_name": r["name"].removeprefix("0.model."),
                "shape": r["shape"]} for r in request["source"]["named_layout"][0]["members"]]
    if numeric:
        # Full name cardinality but tiny synthetic shapes, not full-model numerics.
        for index, row in enumerate(mapping):
            row["shape"] = [[4, 2], [2, 4], [3, 3]][index % 3]
        request["source"]["named_layout"][0]["members"] = [
            {"name": r["model_name"], "shape": r["shape"], "dtype": "torch.float32"} for r in mapping]
        checkpoint = tmp_path / "tiny-checkpoint"
        checkpoint.mkdir()
        request["source"]["checkpoint"] = str(checkpoint)
        save_file({r["checkpoint_name"]: torch.full(r["shape"], 0.7) for r in mapping}, str(checkpoint / "model.safetensors"))
        request["source"]["native_checkpoint"]["files"] = [{"path": "model.safetensors", **file_identity(checkpoint / "model.safetensors")}]
    shards, gradients = [], []
    for step in range(8):
        path = root / f"gradient-{step:04d}.safetensors"
        if numeric:
            values = {r["checkpoint_name"]: torch.randn(r["shape"], generator=torch.Generator().manual_seed(1000 * step + index)) * (step + 1) * 0.003
                      for index, r in enumerate(mapping)}
            save_file(values, str(path))
            gradients.append(values)
        else:
            path.write_bytes(f"synthetic consumer shard {step}".encode())
        shards.append({"path": path.name, **file_identity(path), "step_index": step,
                       "sample_ids": [r["sample_id"] for r in request["selection"][4 * step:4 * step + 4]],
                       "mean_loss": 0.3, "pre_clip_grad_norm": 2.0,
                       "clip_coefficient": 1.0 / (2.0 + 1e-6)})
    manifest = {
        "schema_version": 1, "status": "complete", "config": request["config"],
        "checkpoint": {"path": request["source"]["checkpoint"],
                       "inputs": sorted([r for r in request["source"]["native_checkpoint"]["files"]
                                         if Path(r["path"]).suffix in {".json", ".safetensors"}], key=lambda r: r["path"])},
        "selection": request["selection"], "selection_sha256": request["selection_sha256"],
        "common_state_spec": request["common_state_spec"],
        "probe": {"path": request["probe"], "frozen_spec": request["probe_spec"],
                  "manifest_sha256": "40953eb60bb5dbfa02d9abde5e634bb3221ee934d7ca654c5e5a7e961b39eed2"},
        "parameter_name_mapping": mapping, "gradient_shards": shards,
        "partition_summary": {"hidden": {"tensors": 88, "parameters": 110297088}},
    }
    return inputs, request, manifest, root, gradients


@pytest.mark.parametrize("state", inputs_module.STATES)
def test_full_recorded_request_and_eight_file_wiring(tmp_path, state):
    _, request, manifest, root, _ = gradient_fixture(tmp_path, state)
    assert len(inputs_module.require_gradient_manifest(manifest, request, root)) == 8
    assert request["config"] == inputs_module.CALIBRATION_CONFIG
    assert request["source"]["run_id"] == inputs_module.STATES[state]
    assert len(request["selection"]) == 32


@pytest.mark.parametrize("key,value", [
    ("micro_batch_size", 4), ("gradient_steps", 7), ("examples_per_gradient", 8),
    ("seed", 314159), ("temperature", 0.03), ("max_grad_norm", 0),
    ("model_dtype", "bfloat16"), ("forward_dtype", "float32"),
    ("storage_dtype", "float16"), ("device", "cpu"), ("flash_attention", False),
    ("model_mode", "eval"), ("gradient_checkpointing", False), ("weights_advanced", True),
])
def test_each_changed_numerical_gradient_setting_refused(tmp_path, key, value):
    _, request, manifest, root, _ = gradient_fixture(tmp_path)
    manifest = copy.deepcopy(manifest)
    manifest["config"][key] = value
    with pytest.raises(ValueError):
        inputs_module.require_gradient_manifest(manifest, request, root)


@pytest.mark.parametrize("case", ["old_state", "selection_order", "source_weights", "mapping", "missing_shard", "shard_order", "shard_rows", "shard_bytes", "clipping", "nan_norm", "boolean_schema", "old_common_spec"])
def test_invalid_history_cannot_set_a_calibrated_rate(tmp_path, case):
    _, request, manifest, root, _ = gradient_fixture(tmp_path)
    manifest = copy.deepcopy(manifest)
    if case == "old_state": manifest["checkpoint"]["path"] = "/historical/checkpoint-2345"
    elif case == "selection_order": manifest["selection"].reverse()
    elif case == "source_weights": manifest["checkpoint"]["inputs"][0]["sha256"] = "0" * 64
    elif case == "mapping": manifest["parameter_name_mapping"][0]["shape"] = [2, 2]
    elif case == "missing_shard": manifest["gradient_shards"].pop()
    elif case == "shard_order": manifest["gradient_shards"].reverse()
    elif case == "shard_rows": manifest["gradient_shards"][0]["sample_ids"].reverse()
    elif case == "shard_bytes": (root / "gradient-0000.safetensors").write_bytes(b"modified")
    elif case == "clipping": manifest["gradient_shards"][0]["clip_coefficient"] = 1.0
    elif case == "nan_norm": manifest["gradient_shards"][0]["pre_clip_grad_norm"] = float("nan")
    elif case == "boolean_schema": manifest["schema_version"] = True
    elif case == "old_common_spec": manifest["common_state_spec"] = None
    with pytest.raises(ValueError):
        inputs_module.require_gradient_manifest(manifest, request, root)


def norm_rows():
    return [{"tensor": f"layers.{i}.weight", "shape": [2, 2], "parameters": 4,
             "gradient_steps": 8, "weight_frobenius_norm": w,
             "algorithms": {"adamw": {"frobenius_norm": a}, "muon": {"frobenius_norm": m}}}
            for i, (w, a, m) in enumerate(zip([2., 3., 5.], [7., 11., 13.], [3., 5., 17.], strict=True))]


def test_global_scale_independent_scalar_reference_and_no_mutation():
    rows = norm_rows()
    before = copy.deepcopy(rows)
    result = calibration.calibrated_rates(rows, {r["tensor"]: r["shape"] for r in rows})
    assert rows == before
    for op, squares in [("adamw", 49 + 121 + 169), ("muon", 9 + 25 + 289)]:
        expected = 0.0005 * math.sqrt(38 / squares)
        assert math.isclose(result["hidden_learning_rates"][op], expected, rel_tol=1e-15)
        wrong = 0.0005 / np.mean([r["algorithms"][op]["frobenius_norm"] / r["weight_frobenius_norm"] for r in rows])
        assert not math.isclose(expected, wrong, rel_tol=0.01)


@pytest.mark.parametrize("case", ["missing", "duplicate", "extra", "shape", "history", "parameters", "operator", "zero", "negative", "nan", "inf", "boolean"])
def test_bad_norm_rows_refused(case):
    rows = norm_rows()
    shapes = {r["tensor"]: r["shape"][:] for r in rows}
    if case == "missing": rows.pop()
    elif case == "duplicate": rows[1] = copy.deepcopy(rows[0])
    elif case == "extra": rows.append(copy.deepcopy(rows[0]))
    elif case == "shape": rows[0]["shape"] = [1, 4]
    elif case == "history": rows[0]["gradient_steps"] = 7
    elif case == "parameters": rows[0]["parameters"] = 1
    elif case == "operator": rows[0]["algorithms"]["normuon"] = {"frobenius_norm": 1.}
    else: rows[0]["weight_frobenius_norm"] = {"zero": 0., "negative": -1., "nan": float("nan"), "inf": float("inf"), "boolean": True}[case]
    with pytest.raises(ValueError):
        calibration.calibrated_rates(rows, shapes)


def test_actual_tiny_tensor_history_reader_against_independent_updates(tmp_path):
    _, request, manifest, root, gradients = gradient_fixture(tmp_path, numeric=True)
    rows, _ = calibration.direction_rows(manifest, request, root, device="cpu")
    for row in rows:
        history = [g[row["tensor"]] for g in gradients]
        # Adam oracle in independent NumPy FP64, not the replay helper.
        m = np.zeros(row["shape"], dtype=np.float64)
        v = m.copy()
        for gradient in history:
            g = gradient.numpy().astype(np.float64)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
        direction = (m / (1 - 0.9 ** 8)) / (np.sqrt(v / (1 - 0.999 ** 8)) + 1e-8)
        assert math.isclose(row["algorithms"]["adamw"]["frobenius_norm"], np.linalg.norm(direction), rel_tol=2e-6, abs_tol=1e-7)
        # Actual primary optimizer executes all eight state transitions. Reset
        # only cloned parameter values, never its moments or the source history.
        parameter = torch.nn.Parameter(torch.zeros(row["shape"]))
        optimizer = EmbeddingOptimizer([{"params": [parameter], "algorithm": "muon", "lr": 1.0,
                                        "weight_decay": 0.0, "momentum": 0.95,
                                        "ns_steps": 5, "adjust_lr_fn": "original"}])
        for gradient in history:
            with torch.no_grad(): parameter.zero_()
            parameter.grad = gradient.clone()
            optimizer.step()
        assert math.isclose(row["algorithms"]["muon"]["frobenius_norm"], float(parameter.norm().item()), rel_tol=2e-6, abs_tol=1e-7)


def test_missing_calibration_cannot_create_branch_grid(tmp_path):
    with pytest.raises(ValueError): calibration.branch_cells(recorded_inputs(), {})
    with pytest.raises(ValueError): calibration.load_calibrated_cells(None, {})


def test_declared_input_roles_never_fall_back_to_historical_host(tmp_path):
    roles = inputs_module.Locations(*(tmp_path / name for name in ("repo", "primary", "experiment", "data", "evidence")))
    relocated = roles.locate("/root/embedding-optimizer-story-refactor/" + inputs_module.EVIDENCE + "/data-first.json")
    assert relocated == tmp_path / "evidence/data-first.json"
    for path in ("/unbound/file", "/root/embedding-optimizer-study/not-data/file", "/root/embedding-optimizer-study/data/../file"):
        with pytest.raises(ValueError): roles.locate(path)


def test_symlink_role_refused(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    roles = inputs_module.Locations(link, tmp_path, tmp_path, tmp_path, tmp_path)
    with pytest.raises(ValueError): roles.locate("/root/embedding-optimizer-story-refactor/configs/common_state_probe.json")
