"""Full-768D synthetic portable reconstruction in network/producer-blocked processes."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import digest, file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_reconstruction_authoring import build
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff

SOURCES = (
    "src/embed_optim/primary_v3_reconstruction_inputs.py",
    "src/embed_optim/primary_v3_vector_reconstruction.py",
    "scripts/vector_reconstruction_fixture.py",
    "tests/test_primary_v3_vector_reconstruction.py",
    "scripts/audit_dense_v3_vector_reconstruction.py",
)
CHILD = """
import json, os, runpy, sys
from pathlib import Path
root, anchor, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
forbidden = json.loads(sys.argv[4])
def guard(event, args):
    if event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo'}:
        raise RuntimeError('Network access forbidden in reconstruction')
    if event in {'open', 'os.listdir', 'os.scandir'} and args and isinstance(args[0], (str,bytes,os.PathLike)):
        name = os.path.abspath(os.fsdecode(args[0]))
        if any(name == p or name.startswith(p + os.sep) for p in forbidden):
            raise RuntimeError('Original producer/source path forbidden in reconstruction')
sys.addaudithook(guard)
sys.argv = ['primary_v3_vector_reconstruction', '--archive-root', str(root), '--expected-manifest-sha256', anchor, '--output', str(output)]
runpy.run_module('embed_optim.primary_v3_vector_reconstruction', run_name='__main__')
from embed_optim.reconstruction_files import inspect
inspect(root, anchor)
result = json.loads((output/'reconstruction.json').read_text())
modules = {n: str(Path(m.__file__).relative_to(root/'source/src')) for n,m in sys.modules.items() if n.startswith('embed_optim') and getattr(m,'__file__',None)}
assert modules and result['raw_vector_states_recomputed'] == 61
assert result['upstream_admission_simulated'] is True
assert result['scientific_completion'] is False and result['primary_scientific_admission'] is False
print('PORTABLE_CHILD_RESULT=' + json.dumps({'receipt':result,'imported_modules':modules,'network_and_original_paths_blocked':True},sort_keys=True),flush=True)
"""


def binding(path):
    return {"path": str(Path(path).absolute()), **file_identity(path)}


def child(work, payload, anchor, forbidden, *, expect_failure=False):
    work.mkdir()
    output = work / "recomputed"
    env = {
        **os.environ,
        "PYTHONPATH": str(payload / "source/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    command = [
        sys.executable,
        "-B",
        "-c",
        CHILD,
        str(payload),
        anchor,
        str(output),
        json.dumps(forbidden),
    ]
    with (work / "stdout.txt").open("x") as stdout, (work / "stderr.txt").open("x") as stderr:
        result = subprocess.run(
            command, cwd=work, env=env, stdout=stdout, stderr=stderr, timeout=7200
        )
    text = (work / "stdout.txt").read_text()
    if expect_failure:
        assert result.returncode != 0, "A changed reconstruction was accepted"
        assert not (output / "reconstruction.json").exists()
        return {
            "command": command,
            "exit_code": result.returncode,
            "success_receipt_absent": True,
            "partial_numerical_output_retained": output.exists(),
            "stderr": (work / "stderr.txt").read_text(),
        }
    assert result.returncode == 0, (work / "stderr.txt").read_text()
    value = json.loads(
        next(
            line.removeprefix("PORTABLE_CHILD_RESULT=")
            for line in text.splitlines()
            if line.startswith("PORTABLE_CHILD_RESULT=")
        )
    )
    return {"command": command, "exit_code": result.returncode, "result": value}


def mutate_feature(payload, case):
    """Rehash every affected envelope: the numerical reader must still detect the change."""
    state = payload / "features/states/pretrained"
    if case == "table":
        path = state / "task_summary.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        key = "margin_helpful_mass_share"
        rows[0][key] = float(np.nextafter(float(rows[0][key]), np.inf))
        path.write_bytes(csv_bytes(rows))
    elif case == "attribution":
        path = state / "coordinate_attribution.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        value = arrays["margin_removal_gain"]
        value[0, 0] = np.nextafter(value[0, 0], np.inf)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    else:
        raise ValueError("Unknown feature mutation")
    state_manifest = read_json(state / "manifest.json")
    state_manifest["outputs"][path.name].update(file_identity(path))
    (state / "manifest.json").write_text(json.dumps(state_manifest))
    feature_manifest = payload / "features/manifest.json"
    feature_value = read_json(feature_manifest)
    feature_value["states"]["pretrained"].update(file_identity(state / "manifest.json"))
    feature_manifest.write_text(json.dumps(feature_value))
    evidence_path = payload / "inference/evidence.json"
    evidence = read_json(evidence_path)
    evidence["dimension_feature_reader"]["manifest_sha256"] = file_identity(feature_manifest)[
        "sha256"
    ]
    evidence_path.write_text(json.dumps(evidence))
    inference_path = payload / "inference/manifest.json"
    inference = read_json(inference_path)
    inference["plan"]["evidence_sha256"] = digest(evidence)
    inference_path.write_text(json.dumps(inference))
    manifest_path = payload / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["metadata"]["authoring_evidence_sha256"] = digest(evidence)
    manifest["metadata"]["authoring_plan"] = inference["plan"]
    for name in manifest["files"]:
        manifest["files"][name] = file_identity(payload / name)
    manifest["summary"]["bytes"] = sum(row["bytes"] for row in manifest["files"].values())
    manifest_path.write_text(json.dumps(manifest))
    return file_identity(manifest_path)["sha256"]


def run(args):
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError("Require CPU-only assertions and a new empty audit directory")
    before = handoff()
    sources = [binding(root / name) for name in SOURCES]
    fixture = read_json(args.fixture)
    assert file_identity(args.fixture)["sha256"] == args.fixture_sha256
    assert fixture["full_dimension"] == 768 and fixture["states"] == 61
    assert fixture["upstream_admission_simulated"] is True
    for row in fixture["sources"]:
        verify_file(row["path"], row)
    parent = None
    if args.replay:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        parent = read_json(args.replay)
        require_same(parent["sources"], sources)
        require_same(parent["fixture"], binding(args.fixture))
        for row in parent["artifacts"]:
            verify_file(row["path"], row)
        original = Path(parent["payload_root"])
    else:
        original = Path(fixture["payload"])
    anchor = fixture["manifest_sha256"]
    files.inspect(original, anchor)
    payload = work / "relocated-payload"
    shutil.copytree(original, payload)
    forbidden = [str(original), str(Path(fixture["payload"]).parent / "producer")]
    result = child(work / "cold-reconstruction", payload, anchor, forbidden)
    if parent:
        require_same(result["result"], parent["cold_child"]["result"])
    negative = []
    for case in ("table", "attribution"):
        changed = work / ("rehashed-" + case)
        shutil.copytree(payload, changed)
        changed_anchor = mutate_feature(changed, case)
        files.inspect(changed, changed_anchor)
        refusal = child(
            work / ("refusal-" + case), changed, changed_anchor, forbidden, expect_failure=True
        )
        assert "differs from fresh numerical computation" in refusal["stderr"]
        negative.append({"case": case, "new_external_test_anchor": changed_anchor, **refusal})
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    contract = FunctionalInferenceContract.load(root / INFERENCE_PROTOCOL, primary)
    from types import SimpleNamespace

    missing = SimpleNamespace(
        experiment_root=work / "missing-primary", output=work / "forbidden-authoring"
    )
    try:
        build(contract, missing)
    except ValueError as error:
        assert "ordinary retained run" in str(error)
        actual_refusal = str(error)
    else:
        raise AssertionError("Actual checkpoint authoring accepted missing primary inputs")
    assert not missing.output.exists()
    for row in sources:
        verify_file(row["path"], row)
    require_same(handoff(), before)
    receipt = {
        "scope": "engineering_full_768_v3_vector_reconstruction_audit",
        "passed": True,
        "fresh_replay": parent is not None,
        "replay_parent": None if args.replay is None else binding(args.replay),
        "fixture": binding(args.fixture),
        "sources": sources,
        "payload_root": str(payload),
        "external_manifest_sha256": anchor,
        "cold_child": result,
        "numerical_alterations_rejected": negative,
        "actual_primary_authoring_refusal": actual_refusal,
        "post_execution_dispatchers": handoff(),
        "full_width": 768,
        "raw_states_recomputed": 61,
        "upstream_admission_simulated": True,
        "scientific_completion": False,
        "final_portable_publication_verified": False,
        "artifacts": [binding(path) for path in sorted(work.rglob("*")) if path.is_file()],
    }
    write_new(work / "result.json", receipt)
    print(
        {"passed": True, "scientific_completion": False, **file_identity(work / "result.json")},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "fixture", "workdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
