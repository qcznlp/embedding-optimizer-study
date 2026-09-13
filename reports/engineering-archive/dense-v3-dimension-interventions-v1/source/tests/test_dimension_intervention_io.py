import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from embed_optim.dimension_intervention_io import inspect_state, save_state
from embed_optim.dimension_interventions import compute_state
from embed_optim.primary_contract import file_identity, read_json

ROOT = Path(__file__).resolve().parents[1]


def fixture(step):
    return runpy.run_path(str(ROOT / "tests/test_dimension_interventions.py"))["state_fixture"](
        step
    )


@pytest.mark.parametrize("step", [782, 3907])
def test_state_bundles_recompute_every_table_array_and_empty_rotation_table(tmp_path, step):
    args = fixture(step)
    result = compute_state(*args)
    plan = {"scope": "engineering_synthetic_dimension_state", "scientific_completion": False}
    receipt = save_state(tmp_path / "state", plan, result)
    assert inspect_state(tmp_path / "state", plan, compute_state(*args)) == receipt
    assert receipt["fresh_numerics_verified"] is True
    assert receipt["scientific_completion"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "task_summary.csv",
        "random_removal.csv",
        "rotation_summary.csv",
        "coordinate_attribution.npz",
        "numerical_evidence.json",
    ],
)
def test_rehashing_a_changed_result_never_substitutes_for_recomputation(tmp_path, mutation):
    result = compute_state(*fixture(3907))
    plan = {"scope": "engineering_synthetic_dimension_state", "scientific_completion": False}
    output = tmp_path / "state"
    save_state(output, plan, result)
    path = output / mutation
    if mutation.endswith(".npz"):
        with np.load(path, allow_pickle=False) as src:
            arrays = {key: src[key] for key in src.files}
        arrays["margin_removal_gain"][0, 0] += 0.0001
        np.savez_compressed(path, **arrays)
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    manifest = read_json(output / "manifest.json")
    manifest["outputs"][mutation] = {"path": mutation, **file_identity(path)}
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fresh numerical"):
        inspect_state(output, plan, result)


def test_cached_complete_result_cannot_be_overwritten(tmp_path):
    result = compute_state(*fixture(782))
    output = tmp_path / "state"
    save_state(output, {}, result)
    with pytest.raises(ValueError, match="new state bundle"):
        save_state(output, {}, result)


@pytest.mark.parametrize(
    "mutation", ["plan", "extra", "path", "symlink", "array_dtype", "missing_rotation"]
)
def test_malformed_bundle_cannot_pass_admission(tmp_path, mutation):
    result = compute_state(*fixture(3907))
    output = tmp_path / "state"
    plan = {"scope": "engineering_synthetic_dimension_state", "scientific_completion": False}
    save_state(output, plan, result)
    manifest = read_json(output / "manifest.json")
    if mutation == "plan":
        manifest["plan"]["scientific_completion"] = True
    elif mutation == "extra":
        (output / "extra").write_text("unexpected")
    elif mutation == "path":
        manifest["outputs"]["task_summary.csv"]["path"] = "../task_summary.csv"
    elif mutation == "symlink":
        other = tmp_path / "alias"
        other.symlink_to(output, target_is_directory=True)
        output = other
    elif mutation == "array_dtype":
        path = output / "coordinate_attribution.npz"
        with np.load(path, allow_pickle=False) as src:
            arrays = {key: src[key] for key in src.files}
        arrays["margin_removal_gain"] = arrays["margin_removal_gain"].astype(np.float32)
        np.savez_compressed(path, **arrays)
        manifest["outputs"][path.name] = {"path": path.name, **file_identity(path)}
    else:
        result = copy.deepcopy(result)
        result["rotated_attributions"] = result["rotated_attributions"][:-1]
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        inspect_state(output, plan, result)
