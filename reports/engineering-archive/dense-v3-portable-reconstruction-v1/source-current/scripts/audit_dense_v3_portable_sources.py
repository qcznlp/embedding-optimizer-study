"""Actual source-only relocation and explicitly synthetic authoring transport audit."""

import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_reconstruction_authoring import ACCEPTANCE, build, require_parent
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL, source_selection
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff

SOURCES = (
    "src/embed_optim/reconstruction_files.py",
    "src/embed_optim/primary_v3_reconstruction_sources.py",
    "src/embed_optim/primary_v3_reconstruction_authoring.py",
    "tests/test_reconstruction_files.py",
    "tests/test_primary_v3_reconstruction_sources.py",
    "scripts/audit_dense_v3_portable_sources.py",
)
CHILD = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_reconstruction_authoring import build, require_parent
from embed_optim.reconstruction_files import inspect
root = Path(sys.argv[1])
inspect(root, sys.argv[2])
source, training = root/'source', root/'training-source'
p = PrimaryV3Contract.load(source/'configs/dense_primary_v3_protocol.json', source, training)
c = FunctionalInferenceContract.load(source/'configs/dense_primary_v3_dimension_inference_protocol.json', p)
require_parent(c)
expected = {r['run_id']: r['expected_identity_sha256'] for r in p.inputs['runs']}
assert len(expected) == 12
mods = {n: str(Path(m.__file__).relative_to(source/'src')) for n,m in sys.modules.items() if n.startswith('embed_optim') and getattr(m,'__file__',None)}
assert len(mods)>20
args = SimpleNamespace(experiment_root=Path(sys.argv[3]), output=Path(sys.argv[4]))
try:
    build(c,args)
except ValueError as error:
    assert 'ordinary retained run' in str(error)
    refusal = str(error)
else:
    raise AssertionError('Missing primary population was admitted')
assert not args.output.exists()
inspect(root, sys.argv[2])
print(json.dumps({'inference_protocol_sha256':c.sha256,'run_identities':expected,'imported_modules':mods,'actual_primary_refusal':refusal,'output_absent':True,'scientific_completion':False},sort_keys=True))
"""


def binding(path):
    return {"path": str(Path(path).resolve()), **file_identity(path)}


def child(work, payload, anchor):
    env = {
        **os.environ,
        "PYTHONPATH": str(payload / "source/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    command = [
        sys.executable,
        "-B",
        "-c",
        CHILD,
        str(payload),
        anchor,
        str(work / "absent-primary"),
        str(work / "forbidden-output"),
    ]
    result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=60)
    (work / "child-stdout.json").write_text(result.stdout)
    (work / "child-stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    return {"command": command, "exit_code": result.returncode, "result": value}


def alterations(work, source, anchor):
    output = []
    for name in ("source", "metadata", "omitted_file", "extra_directory"):
        target = work / ("altered-" + name)
        shutil.copytree(source, target)
        manifest = read_json(target / "manifest.json")
        relative = "source/src/embed_optim/primary_v3_dimension_inference.py"
        path = target / relative
        if name == "source":
            with path.open("a") as stream:
                stream.write("\n# Changed after authenticated selection.\n")
            manifest["files"][relative] = file_identity(path)
            (target / "manifest.json").write_text(json.dumps(manifest))
        elif name == "metadata":
            manifest["metadata"]["primary_results_present"] = True
            (target / "manifest.json").write_text(json.dumps(manifest))
        elif name == "omitted_file":
            path.rename(work / "preserved-omitted-source.py")
        else:
            (target / "undeclared-empty-directory").mkdir()
        try:
            files.inspect(target, anchor)
        except ValueError as error:
            output.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise AssertionError("Changed or incomplete payload passed the trusted anchor")
    return output


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
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    contract = FunctionalInferenceContract.load(root / INFERENCE_PROTOCOL, primary)
    acceptance = require_parent(contract)
    sources = [binding(root / name) for name in SOURCES]
    parent = None
    actual = work / "actual-source-payload"
    synthetic = work / "explicit-synthetic-reconstruction"
    if args.replay:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        parent = read_json(args.replay)
        require_same(parent["sources"], sources)
        for row in parent["artifacts"]:
            verify_file(row["path"], row)
        shutil.copytree(args.replay.parent / "actual-source-payload", actual)
        shutil.copytree(args.replay.parent / "explicit-synthetic-reconstruction", synthetic)
        anchor = parent["actual_payload"]["manifest_sha256"]
        synthetic_anchor = parent["synthetic_payload"]["manifest_sha256"]
    else:
        selected = source_selection(contract, {})
        files.select(selected, "source", ACCEPTANCE[0], acceptance, file_identity(acceptance))
        receipt = files.write(
            actual,
            selected,
            {
                "scope": "engineering_actual_source_contract_closure",
                "primary_results_present": False,
            },
        )
        anchor = receipt["manifest_sha256"]
        # This separate fixture intentionally simulates upstream admission; no model is invented.
        fixture = runpy.run_path(str(root / "tests/test_primary_v3_reconstruction_sources.py"))[
            "test_full_local_builder_with_explicitly_simulated_upstream_admission"
        ]
        with pytest.MonkeyPatch.context() as patch:
            fixture(contract, work, patch)
        synthetic_anchor = file_identity(synthetic / "manifest.json")["sha256"]
    actual_manifest = files.inspect(actual, anchor)
    synthetic_manifest = files.inspect(synthetic, synthetic_anchor)
    assert (
        read_json(synthetic / "inference/evidence.json")["upstream_primary_admission_simulated"]
        is True
    )
    assert synthetic_manifest["metadata"]["scientific_completion"] is False
    result_child = child(work, actual, anchor)
    if parent:
        require_same(result_child["result"], parent["fresh_child"]["result"])
    altered = alterations(work, actual, anchor)
    missing_args = SimpleNamespace(
        experiment_root=work / "absent-primary", output=work / "forbidden-local-output"
    )
    try:
        build(contract, missing_args)
    except ValueError as error:
        assert "ordinary retained run" in str(error)
        refused = str(error)
    else:
        raise AssertionError("Actual authoring admitted missing primary data")
    assert not missing_args.output.exists()
    contract.recheck()
    for row in sources:
        verify_file(row["path"], row)
    require_same(before, handoff())
    result = {
        "scope": "engineering_v3_portable_source_and_authoring_transport_audit",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "fresh_replay": parent is not None,
        "replay_parent": None if args.replay is None else binding(args.replay),
        "sources": sources,
        "actual_payload": {"manifest_sha256": anchor, **actual_manifest["summary"]},
        "synthetic_payload": {"manifest_sha256": synthetic_anchor, **synthetic_manifest["summary"]},
        "fresh_child": result_child,
        "actual_local_primary_refusal": refused,
        "alterations": altered,
        "artifacts": [binding(path) for path in sorted(work.rglob("*")) if path.is_file()],
        "post_execution_dispatchers": handoff(),
        "authoring_upstream_admission_simulated_in_positive_fixture": True,
        "primary_runs_or_vectors_admitted": 0,
        "raw_vector_numerical_reconstruction_verified": False,
        "final_portable_publication_verified": False,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    write_new(work / "result.json", result)
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
