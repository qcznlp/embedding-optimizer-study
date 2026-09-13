"""Complete synthetic joint replay in source-isolated CPU processes.

Full positive runs repeat every raw branch. Late-table negative controls reuse
only the same cold run's authenticated fresh raw outputs, at the exact affected
consumer; their narrower scope is explicit and never reported as a full replay.
"""

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
from embed_optim.primary_v3_outcome_primitives import producer_path
from embed_optim.primary_v3_reconstruction_authoring import build
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.joint_reconstruction_controls import CASES, mutate
from scripts.joint_reconstruction_fixture import make
from scripts.joint_reconstruction_oracle import audit as independent_oracle

SOURCES = (
    "src/embed_optim/primary_v3_joint_reconstruction.py",
    "scripts/joint_reconstruction_fixture.py",
    "scripts/joint_reconstruction_vectors.py",
    "scripts/joint_reconstruction_geometry.py",
    "scripts/joint_reconstruction_controls.py",
    "scripts/joint_reconstruction_oracle.py",
    "scripts/audit_dense_v3_joint_reconstruction.py",
    "scripts/outcome_reconstruction_fixture.py",
    "scripts/vector_reconstruction_fixture.py",
    "scripts/geometry_reconstruction_fixture.py",
    "scripts/audit_dense_v3_bridge.py",
    "scripts/dimension_inference_reference.py",
    "tests/test_primary_v3_joint_reconstruction.py",
)

CHILD = r"""
import json, os, runpy, sys
from pathlib import Path
root, anchor, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
forbidden, mode, fresh = json.loads(sys.argv[4]), sys.argv[5], json.loads(sys.argv[6])
def disallowed(path):
    name = os.path.abspath(os.fsdecode(path))
    return any(name == p or name.startswith(p+os.sep) for p in forbidden)
removed = [p for p in sys.path if disallowed(p)]
sys.path[:] = [p for p in sys.path if not disallowed(p)]
def guard(event, args):
    if event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo'}:
        raise RuntimeError('Network access forbidden in joint reconstruction')
    if event in {'open','os.listdir','os.scandir'} and args and isinstance(args[0],(str,bytes,os.PathLike)) and disallowed(args[0]):
        raise RuntimeError('Original producer/source path forbidden: '+os.fsdecode(args[0]))
sys.addaudithook(guard)
if mode == 'full':
    sys.argv = ['joint-reconstruction','--archive-root',str(root),'--expected-manifest-sha256',anchor,'--output',str(output)]
    runpy.run_module('embed_optim.primary_v3_joint_reconstruction',run_name='__main__')
    result = json.loads((output/'reconstruction.json').read_text())
    modules = {n:str(Path(m.__file__).relative_to(root/'source/src')) for n,m in sys.modules.items() if n.startswith('embed_optim') and getattr(m,'__file__',None)}
    assert result['all_three_raw_branches_reconstructed'] is True
    assert result['original_and_functional_inference_recomputed'] is True
    assert result['scientific_completion'] is False and result['primary_scientific_admission'] is False
    print('PORTABLE_JOINT_RESULT='+json.dumps({'receipt':result,'imported_modules':modules,'network_and_original_paths_blocked':True,'removed_search_paths':removed},sort_keys=True),flush=True)
else:
    from embed_optim import primary_v3_joint_reconstruction as joint
    from embed_optim import reconstruction_files as files
    from embed_optim.primary_contract import read_json, require_same, verify_file
    from embed_optim.primary_v3_reconstruction_inputs import ReconstructionInput
    from embed_optim.primary_v3_exact_bridge import typed_csv
    inputs = ReconstructionInput.load(root, anchor)
    prior = files.inspect(Path(fresh['archive']), fresh['anchor'])
    allowed = ('bridge/', 'inference/') if mode == 'bridge' else ('inference/',)
    require_same({k:v for k,v in inputs.manifest['files'].items() if not k.startswith(allowed)}, {k:v for k,v in prior['files'].items() if not k.startswith(allowed)})
    target = Path(fresh['output'])
    verify_file(target/'reconstruction.json', fresh['receipt'])
    receipt = read_json(target/'reconstruction.json')
    assert receipt['external_archive_sha256'] == fresh['anchor']
    assert receipt['all_three_raw_branches_reconstructed'] is True
    for name, record in receipt['outputs'].items():
        verify_file(target/name, record)
    joint.provenance(inputs, inputs.evidence['original_bridge_evidence']['geometry_reader'])
    if mode == 'bridge':
        joint.original_bridge(inputs, output, target/'outcomes', read_json(target/'outcomes/reconstruction.json'), target/'geometry')
    elif mode == 'functional':
        old = {name:typed_csv(target/'bridge'/(name+'.csv')) for name in joint.bridge.TABLE_COUNTS}
        joint.functional_inference(inputs, output, target/'vectors', receipt['original_bridge_reader'], read_json(target/'bridge/evidence.json'), old)
    else:
        raise ValueError('Unknown bounded test consumer')
    raise AssertionError('A mutated late-table control was incorrectly accepted')
"""


def binding(path):
    return {"path": str(Path(path).absolute()), **file_identity(path)}


def original_locations(payload):
    """Block recorded producer directories without opening those locations."""
    evidence = read_json(payload / "inference/evidence.json")
    names = [row["path"] for row in evidence["source_bindings"]]
    outcomes = evidence["original_bridge_evidence"]["outcome_evidence"]
    names += [
        row["manifest"]["path"] for row in outcomes["validation_selection"]["source_receipts"]
    ]
    for job in outcomes["grid"]["evaluations"]:
        names.append(str(producer_path(job["plan"]["results_root"]) / "primary_admission.json"))
        names += [row["path"] for task in job["tasks"] for row in task["files"]]
    return sorted({str(producer_path(name).parent) for name in names})


def child(work, payload, anchor, forbidden, *, mode="full", fresh=None, expect_failure=False):
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
        mode,
        json.dumps(fresh),
    ]
    with (work / "stdout.txt").open("x") as stdout, (work / "stderr.txt").open("x") as stderr:
        result = subprocess.run(
            command, cwd=work, env=env, stdout=stdout, stderr=stderr, timeout=7200
        )
    error = (work / "stderr.txt").read_text()
    if expect_failure:
        assert result.returncode != 0, "A contradictory joint input was accepted"
        assert not (output / "reconstruction.json").exists()
        assert "incorrectly accepted" not in error
        assert "Original producer/source path forbidden" not in error
        assert "Network access forbidden" not in error
        assert "SHA-256 mismatch" not in error
        assert "Traceback (most recent call last)" in error
        assert any(line.startswith(("ValueError:", "KeyError:")) for line in error.splitlines()), (
            error
        )
        return {
            "exit_code": result.returncode,
            "consumer_scope": mode,
            "full_public_entrypoint_invoked": mode == "full",
            "complete_fresh_raw_reconstruction_verified_by_refusal": False,
            "same_cold_run_fresh_branches_reused": mode != "full",
            "success_receipt_absent": True,
            "partial_output_retained": output.exists(),
            "stderr": error,
        }
    assert mode == "full" and result.returncode == 0, error
    value = json.loads(
        next(
            line.removeprefix("PORTABLE_JOINT_RESULT=")
            for line in (work / "stdout.txt").read_text().splitlines()
            if line.startswith("PORTABLE_JOINT_RESULT=")
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
    if bool(args.replay) != bool(args.replay_sha256):
        raise ValueError("Replay requires both the completed result and its external anchor")
    if bool(args.components) != bool(args.component_anchor) or (args.replay and args.components):
        raise ValueError("Use paired component arguments or a replay, not both")
    before = handoff()
    sources = [binding(root / name) for name in SOURCES]
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    contract = FunctionalInferenceContract.load(root / INFERENCE_PROTOCOL, primary)
    parent = None
    if args.replay:
        verify_file(
            args.replay, {"bytes": args.replay.stat().st_size, "sha256": args.replay_sha256}
        )
        parent = read_json(args.replay)
        assert parent["passed"] is True and parent["scientific_completion"] is False
        require_same(parent["sources"], sources)
        for row in parent["artifacts"]:
            verify_file(row["path"], row)
        original, anchor = Path(parent["payload_root"]), parent["external_manifest_sha256"]
        fixture = parent["fixture"]
        verify_file(fixture["path"], fixture)
    else:
        original, anchor = make(
            work / "fixture-input",
            contract,
            progress=lambda row: print(row, flush=True),
            components=args.components,
            component_anchor=args.component_anchor,
        )
        write_new(
            work / "fixture.json",
            {
                "scope": "engineering_complete_synthetic_joint_fixture",
                "payload": str(original),
                "manifest_sha256": anchor,
                "sources": sources,
                "one_complete_run_population": True,
                "vector_dimension": 768,
                "vector_states": 61,
                "geometry_states": 60,
                "beir_task_units": 840,
                "fixture_raw_components_reused": args.components is not None,
                "raw_components_external_anchor": args.component_anchor,
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
    forbidden += original_locations(payload)
    oracle = independent_oracle(payload)
    print(
        {"complete_fixture_ready": True, "independent_join_and_numerical_oracles_passed": True},
        flush=True,
    )
    cold = child(work / "cold-reconstruction", payload, anchor, forbidden)
    if parent:
        require_same(cold["result"], parent["cold_child"]["result"])
        require_same(oracle, parent["independent_oracle"])
    fresh = {
        "archive": str(payload),
        "anchor": anchor,
        "output": str(work / "cold-reconstruction/recomputed"),
        "receipt": file_identity(work / "cold-reconstruction/recomputed/reconstruction.json"),
    }
    refusals = []
    for case, mode in CASES.items():
        changed = work / ("rehashed-joint-" + case)
        shutil.copytree(payload, changed)
        new_anchor = mutate(changed, case)
        result = child(
            work / ("refusal-" + case),
            changed,
            new_anchor,
            forbidden,
            mode=mode,
            fresh=fresh if mode != "full" else None,
            expect_failure=True,
        )
        refusals.append({"case": case, "new_external_test_anchor": new_anchor, **result})
        print({"semantic_refusal_passed": case, "consumer_scope": mode}, flush=True)
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
            "scope": "engineering_complete_joint_reconstruction_audit",
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
    parser.add_argument("--components", type=Path)
    parser.add_argument("--component-anchor")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
