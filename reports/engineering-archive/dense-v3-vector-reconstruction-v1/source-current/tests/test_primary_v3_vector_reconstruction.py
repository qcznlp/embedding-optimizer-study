"""Portable full-population wiring; all upstream authoring is explicitly synthetic."""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embed_optim import primary_v3_vector_reconstruction as replay
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import digest, file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_reconstruction_inputs import (
    ReconstructionInput,
    raw_vectors,
    running_sources,
    vector_admission,
)
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from scripts.audit_dense_v3_vector_reconstruction import CHILD, mutate_feature
from scripts.vector_reconstruction_fixture import admitted_fixture, make, reduced_kernel

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(autouse=True)
def isolated_archival_test_modules(monkeypatch):
    """The frozen old controller tests deliberately load both versions under aliases.

    They leave these two registrations in sys.modules. A genuine reconstruction
    must refuse that mixed-source process. Isolate this test module's positive
    fixtures only, restoring the registrations afterwards; do not filter or
    weaken the production source check. The explicit mixed-source test below
    retains the refusal requirement.
    """
    for side in ("before", "after"):
        name = f"embed_optim._main_resume_{side}"
        module = sys.modules.get(name)
        if module is not None:
            assert (
                Path(module.__file__)
                == ROOT
                / f"reports/engineering-archive/main-resume-v1/{side}/corrected_completion_pipeline.py"
            )
            monkeypatch.delitem(sys.modules, name)


@pytest.fixture(scope="module")
def contract():
    primary = PrimaryV3Contract.load(
        ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE
    )
    return FunctionalInferenceContract.load(ROOT / INFERENCE_PROTOCOL, primary)


@pytest.fixture(scope="module")
def archive(contract, tmp_path_factory):
    directory = tmp_path_factory.mktemp("explicit-synthetic-vector-reconstruction")
    root, anchor = make(directory / "fixture", contract, kernel=reduced_kernel)
    return root, anchor


def test_actual_frozen_admission_metadata_and_location_free_plans(contract):
    admitted, _ = admitted_fixture(contract)
    jobs = vector_admission(contract.dimensions, admitted)
    assert len(jobs) == 61 and all(set(job) == {"plan"} for job in jobs)
    assert all(row["simulated_admission"] for row in admitted["complete_runs"].values())
    assert len(admitted["probe"]["row_identities"]) == 224


@pytest.mark.parametrize(
    "case",
    [
        "missing_run",
        "extra_run",
        "identity",
        "unsealed",
        "step_order",
        "missing_stage",
        "state_order",
        "state_model",
        "probe_order",
        "probe_candidate",
        "reference_file",
        "reference_shape",
        "hidden_shape",
        "encoding",
        "extra_field",
    ],
)
def test_changed_admission_metadata_rejected(contract, case):
    value, _ = admitted_fixture(contract)
    value = copy.deepcopy(value)
    run = next(iter(value["complete_runs"].values()))
    if case == "missing_run":
        value["complete_runs"].pop(next(iter(value["complete_runs"])))
    elif case == "extra_run":
        value["complete_runs"]["historical"] = run
    elif case == "identity":
        run["run_identity_sha256"] = "a" * 64
    elif case == "unsealed":
        run["whole_run_artifacts_verified"] = False
    elif case == "step_order":
        run["steps"] = list(reversed(run["steps"]))
    elif case == "missing_stage":
        run["checkpoints"].pop()
    elif case == "state_order":
        value["states"].reverse()
    elif case == "state_model":
        value["states"][1]["model"]["complete_run_sha256"] = "a" * 64
    elif case == "probe_order":
        value["probe"]["row_identities"].reverse()
    elif case == "probe_candidate":
        value["probe"]["row_identities"][0]["candidate_ids_positive_first"].reverse()
    elif case == "reference_file":
        value["reference"]["files"][0]["sha256"] = "a" * 64
    elif case == "reference_shape":
        value["reference"]["all_shapes"].pop(next(iter(value["reference"]["all_shapes"])))
    elif case == "hidden_shape":
        value["reference"]["hidden_shapes"].pop(next(iter(value["reference"]["hidden_shapes"])))
    elif case == "encoding":
        value["encoding"]["storage_dtype"] = "float16"
    else:
        value["unexpected"] = True
    with pytest.raises((ValueError, KeyError, IndexError)):
        vector_admission(contract.dimensions, value)


def test_archived_inputs_do_not_require_original_producer_or_model_paths(archive):
    root, anchor = archive
    inputs = ReconstructionInput.load(root, anchor)
    assert not Path(inputs.evidence["producer_path_must_not_be_read"]).exists()
    assert inputs.evidence["upstream_primary_admission_simulated"] is True
    assert len(inputs.jobs) == 61
    inputs.recheck()


def test_mixed_archived_implementation_in_running_process_is_still_rejected(archive, monkeypatch):
    root, anchor = archive
    manifest = files.inspect(root, anchor)
    foreign = (
        ROOT / "reports/engineering-archive/main-resume-v1/after/corrected_completion_pipeline.py"
    )
    monkeypatch.setitem(
        sys.modules, "embed_optim._foreign_archival_control", SimpleNamespace(__file__=str(foreign))
    )
    with pytest.raises(ValueError, match="File content identity differs"):
        running_sources(root, manifest)


def inspect_raw(root, inputs, expected):
    return raw_vectors(
        root,
        inputs.jobs[0]["plan"],
        inputs.admitted["probe"]["row_identities"],
        inputs.admitted["reference"]["all_shapes"],
        expected,
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing_array",
        "extra_array",
        "fp16",
        "nan",
        "zero",
        "sample_order",
        "group_order",
        "loading_changed",
        "loading_dtype",
        "loading_sha",
        "loading_tensor",
    ],
)
def test_rehashed_raw_state_semantics_rejected(archive, tmp_path, case):
    inputs = ReconstructionInput.load(*archive)
    original = inputs.root / "vectors/states/pretrained"
    target = tmp_path / "changed-state"
    shutil.copytree(original, target)
    path = target / "manifest.json"
    value = read_json(path)
    if case.startswith("loading_"):
        before = value["observed_loading"]["before"]
        if case == "loading_changed":
            value["observed_loading"]["parameters_unchanged"] = False
        elif case == "loading_dtype":
            next(iter(before["state_tensors"].values()))["dtype"] = "torch.float32"
        elif case == "loading_sha":
            next(iter(before["state_tensors"].values()))["sha256"] = "invalid"
        else:
            before["state_tensors"].pop(next(iter(before["state_tensors"])))
        before["state_tensors_sha256"] = digest(before["state_tensors"])
        value["observed_loading"]["after"] = copy.deepcopy(before)
    else:
        with np.load(target / "vectors.npz", allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        if case == "missing_array":
            arrays.pop("query_embeddings")
        elif case == "extra_array":
            arrays["unrelated"] = np.zeros(1)
        elif case == "fp16":
            arrays["query_embeddings"] = arrays["query_embeddings"].astype(np.float16)
        elif case == "nan":
            arrays["query_embeddings"][0, 0] = np.nan
        elif case == "zero":
            arrays["document_embeddings"][0, 0] = 0
        elif case == "sample_order":
            arrays["sample_ids"] = arrays["sample_ids"][::-1]
        else:
            arrays["sample_groups"] = arrays["sample_groups"][::-1]
        with (target / "vectors.npz").open("wb") as stream:
            np.savez(stream, **arrays)
        value["output"].update(file_identity(target / "vectors.npz"))
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        inspect_raw(target, inputs, file_identity(path))


def test_complete_reconstruction_explicit_reduced_numerical_wiring(archive, tmp_path, monkeypatch):
    inputs = ReconstructionInput.load(*archive)
    monkeypatch.setattr(replay, "compute_state", reduced_kernel)
    progress = []
    value = replay.reconstruct(inputs, tmp_path / "recomputed", progress=progress.append)
    assert value["raw_vector_states_recomputed"] == 61 and len(progress) == 61
    assert value["upstream_admission_simulated"] is True
    assert value["comparison_tolerance"] is None and value["exact_numerics_verified"] is True
    for key in (
        "model_encoding_repeated",
        "checkpoint_tensors_revalidated",
        "raw_probe_text_revalidated",
        "outcome_geometry_primitives_recomputed",
        "publication_inference_recomputed",
        "primary_scientific_admission",
        "manuscript_installed",
        "scientific_completion",
    ):
        assert value[key] is False
    assert len(value["outputs"]) == 432
    files.inspect(*archive)


@pytest.mark.parametrize("case", ["existing_output", "inside_archive", "wrong_anchor"])
def test_fail_before_new_output(archive, tmp_path, monkeypatch, case):
    monkeypatch.setattr(replay, "compute_state", lambda *a: pytest.fail("must not compute"))
    output = tmp_path / "new"
    if case == "wrong_anchor":
        with pytest.raises(ValueError, match="external trusted anchor"):
            ReconstructionInput.load(archive[0], "0" * 64)
    else:
        inputs = ReconstructionInput.load(*archive)
        if case == "inside_archive":
            output = archive[0] / "forbidden-output"
        else:
            output.mkdir()
            (output / "keep.txt").write_text("preserve")
        with pytest.raises(ValueError, match="new reconstruction directory"):
            replay.reconstruct(inputs, output)
    if case == "existing_output":
        assert (output / "keep.txt").read_text() == "preserve"
    else:
        assert not output.exists()


@pytest.mark.parametrize("case", ["table", "attribution"])
def test_one_ulp_change_rehashed_through_every_parent_still_fails_numerically(
    archive, tmp_path, monkeypatch, case
):
    changed = tmp_path / "changed-archive"
    shutil.copytree(archive[0], changed)
    anchor = mutate_feature(changed, case)
    files.inspect(changed, anchor)
    inputs = ReconstructionInput.load(changed, anchor)
    monkeypatch.setattr(replay, "compute_state", reduced_kernel)
    output = tmp_path / "retained-failure"
    with pytest.raises(ValueError, match="differs from fresh numerical computation"):
        replay.reconstruct(inputs, output)
    assert (output / "states/pretrained/manifest.json").is_file()
    assert not (output / "reconstruction.json").exists()


@pytest.mark.parametrize("case", ["scope", "runtime"])
def test_self_rehashed_wrong_authoring_scope_or_runtime_is_not_compatible(archive, tmp_path, case):
    changed = tmp_path / "changed-archive"
    shutil.copytree(archive[0], changed)
    path = changed / "manifest.json"
    value = read_json(path)
    if case == "scope":
        value["metadata"]["scope"] = "engineering_source_only_not_completed_authoring"
    else:
        value["metadata"]["authoring_runtime"]["packages"]["numpy"] = "0.0.invalid"
    path.write_text(json.dumps(value))
    anchor = file_identity(path)["sha256"]
    files.inspect(changed, anchor)
    with pytest.raises(ValueError):
        ReconstructionInput.load(changed, anchor)


def test_cold_input_loader_only_archived_sources_without_network_or_producer(archive, tmp_path):
    root = tmp_path / "relocated"
    shutil.copytree(archive[0], root)
    forbidden = [str(archive[0]), str(archive[0].parent / "producer")]
    code = (
        CHILD.split("sys.argv =", 1)[0]
        + """
from embed_optim.primary_v3_reconstruction_inputs import ReconstructionInput
inputs = ReconstructionInput.load(root, anchor)
assert len(inputs.jobs) == 61
for n,m in sys.modules.items():
    if n.startswith('embed_optim') and getattr(m, '__file__', None):
        assert Path(m.__file__).is_relative_to(root/'source/src')
for event, args in [('socket.connect', (None,('127.0.0.1',1))), ('open',(forbidden[0]+'/manifest.json','r',0))]:
    try:
        sys.audit(event, *args)
    except RuntimeError as error:
        assert 'forbidden' in str(error)
    else:
        raise AssertionError('Expected active process guard')
print('actual_contracts_loaded_offline_from_archived_sources')
"""
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root / "source/src"),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(root),
            archive[1],
            str(tmp_path / "unused"),
            json.dumps(forbidden),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert "actual_contracts_loaded_offline_from_archived_sources" in result.stdout
    files.inspect(root, archive[1])
