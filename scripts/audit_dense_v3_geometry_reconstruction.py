"""Native-shape synthetic geometry replay with archived sources and forbidden old paths."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_reconstruction_authoring import build
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.geometry_reconstruction_controls import CASES, mutate
from scripts.geometry_reconstruction_fixture import make
from scripts.geometry_reconstruction_oracle import audit as independent_oracle

SOURCES = (
    "src/embed_optim/primary_v3_geometry_primitives.py",
    "src/embed_optim/primary_v3_geometry_reconstruction.py",
    "scripts/geometry_reconstruction_fixture.py",
    "scripts/geometry_reconstruction_controls.py",
    "scripts/geometry_reconstruction_oracle.py",
    "scripts/vector_reconstruction_fixture.py",
    "scripts/outcome_reconstruction_fixture.py",
    "tests/test_primary_v3_geometry_reconstruction.py",
    "scripts/audit_dense_v3_geometry_reconstruction.py",
)
CHILD = """
import json, os, runpy, sys
from pathlib import Path
root, anchor, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
forbidden = json.loads(sys.argv[4])
def disallowed(path):
    name = os.path.abspath(os.fsdecode(path))
    return any(name == p or name.startswith(p+os.sep) for p in forbidden)
removed_search_paths = [p for p in sys.path if disallowed(p)]
sys.path[:] = [p for p in sys.path if not disallowed(p)]
def guard(event, args):
    if event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo'}:
        raise RuntimeError('Network access forbidden in geometry reconstruction')
    if event in {'open','os.listdir','os.scandir'} and args and isinstance(args[0],(str,bytes,os.PathLike)):
        if disallowed(args[0]):
            raise RuntimeError('Original producer/source path forbidden in reconstruction: '+os.fsdecode(args[0]))
sys.addaudithook(guard)
sys.argv = ['original-geometry-reconstruction','--archive-root',str(root),
    '--expected-manifest-sha256',anchor,'--output',str(output)]
runpy.run_module('embed_optim.primary_v3_geometry_reconstruction',run_name='__main__')
result = json.loads((output/'reconstruction.json').read_text())
modules = {n:str(Path(m.__file__).relative_to(root/'source/src'))
    for n,m in sys.modules.items() if n.startswith('embed_optim') and getattr(m,'__file__',None)}
assert modules and result['checkpoints_reconstructed']==60
assert result['run_pairs_recomputed']==660 and result['upstream_admission_simulated'] is True
assert result['scientific_completion'] is False and result['primary_scientific_admission'] is False
print('PORTABLE_GEOMETRY_RESULT='+json.dumps({'receipt':result,'imported_modules':modules,
    'network_and_original_paths_blocked':True,'removed_search_paths':removed_search_paths},sort_keys=True),flush=True)
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
            command, cwd=work, env=env, stdout=stdout, stderr=stderr, timeout=300
        )
    error = (work / "stderr.txt").read_text()
    if expect_failure:
        assert result.returncode != 0, "A contradictory synthetic geometry input was accepted"
        assert not (output / "reconstruction.json").exists()
        assert "Original producer/source path forbidden" not in error
        assert "Network access forbidden" not in error
        return {
            "exit_code": result.returncode,
            "success_receipt_absent": True,
            "partial_numerical_output_retained": output.exists(),
            "stderr": error,
        }
    assert result.returncode == 0, error
    value = json.loads(
        next(
            line.removeprefix("PORTABLE_GEOMETRY_RESULT=")
            for line in (work / "stdout.txt").read_text().splitlines()
            if line.startswith("PORTABLE_GEOMETRY_RESULT=")
        )
    )
    return {"exit_code": result.returncode, "result": value}


def run(args):
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError("Require CPU-only assertions and a new empty work directory")
    before = handoff()
    sources = [binding(root / name) for name in SOURCES]
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    contract = FunctionalInferenceContract.load(root / INFERENCE_PROTOCOL, primary)
    parent = None
    if args.replay:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        parent = read_json(args.replay)
        require_same(parent["sources"], sources)
        for row in parent["artifacts"]:
            verify_file(row["path"], row)
        original, anchor = Path(parent["payload_root"]), parent["external_manifest_sha256"]
        fixture = parent["fixture"]
        verify_file(fixture["path"], fixture)
    else:
        original, anchor = make(
            work / "fixture-input", contract, progress=lambda value: print(value, flush=True)
        )
        write_new(
            work / "fixture.json",
            {
                "scope": "engineering_complete_synthetic_original_geometry_fixture",
                "payload": str(original),
                "manifest_sha256": anchor,
                "sources": sources,
                "checkpoints": 60,
                "hidden_matrices_per_checkpoint": 88,
                "native_shapes_retained": True,
                "basis_rank": 16,
                "raw_weight_spectra_and_entry_counts_simulated": True,
                "upstream_primary_admission_simulated": True,
                "scientific_completion": False,
            },
        )
        fixture = binding(work / "fixture.json")
    files.inspect(original, anchor)
    payload = work / "relocated-payload"
    shutil.copytree(original, payload)
    forbidden = [
        str(root),
        "/root/embedding-optimizer-study",
        str(original),
        str(Path(read_json(fixture["path"])["payload"]).parent),
        "/tmp/dense-partition-candidate.kHXoGW",
        "/tmp/dense-weight-entry-reference.Um8TxW",
        "/tmp/dense-natural-readiness.a4xZzR",
    ]
    oracle = independent_oracle(payload)
    print({"fixture_ready": True, "independent_set_oracle_passed": True}, flush=True)
    cold = child(work / "cold-reconstruction", payload, anchor, forbidden)
    if parent:
        require_same(cold["result"], parent["cold_child"]["result"])
        require_same(oracle, parent["independent_oracle"])
    refusals = []
    for case in CASES:
        changed = work / ("rehashed-" + case)
        shutil.copytree(payload, changed)
        new_anchor = mutate(changed, case)
        result = child(
            work / ("refusal-" + case), changed, new_anchor, forbidden, expect_failure=True
        )
        assert "SHA-256 mismatch" not in result["stderr"]
        if case in {"checkpoint_row", "geometry_table", "geometry_summary", "basis_values"}:
            assert result["partial_numerical_output_retained"] is True
            assert (work / ("refusal-" + case) / "recomputed/fresh_checkpoint_rows.json").is_file()
        refusals.append({"case": case, "new_external_test_anchor": new_anchor, **result})
        print({"semantic_refusal_passed": case}, flush=True)
    missing = SimpleNamespace(
        experiment_root=work / "missing-primary", output=work / "forbidden-authoring"
    )
    try:
        build(contract, missing)
    except ValueError as error:
        actual_refusal = str(error)
        assert "ordinary retained run" in actual_refusal
    else:
        raise AssertionError("Actual missing primary authoring was accepted")
    assert not missing.output.exists()
    for row in sources:
        verify_file(row["path"], row)
    require_same(handoff(), before)
    write_new(
        work / "result.json",
        {
            "scope": "engineering_complete_original_geometry_reconstruction_audit",
            "passed": True,
            "fresh_replay": parent is not None,
            "replay_parent": None if args.replay is None else binding(args.replay),
            "sources": sources,
            "fixture": fixture,
            "payload_root": str(payload),
            "external_manifest_sha256": anchor,
            "cold_child": cold,
            "independent_oracle": oracle,
            "semantic_alterations_rejected": refusals,
            "actual_primary_authoring_refusal": actual_refusal,
            "post_execution_dispatchers": handoff(),
            "upstream_primary_admission_simulated": True,
            "scientific_completion": False,
            "final_portable_publication_verified": False,
            "artifacts": [binding(path) for path in sorted(work.rglob("*")) if path.is_file()],
        },
    )
    print(
        {"passed": True, "scientific_completion": False, **file_identity(work / "result.json")},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "workdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
