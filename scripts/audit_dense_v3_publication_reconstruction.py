"""Source-isolated full synthetic publication replay, never a primary experiment."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from embed_optim import primary_v3_publication as publication
from embed_optim import primary_v3_publication_archive as archive
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_publication_contract import PublicationContract
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_v3_joint_reconstruction import original_locations
from scripts.publication_reconstruction_fixture import ORIGINAL, make

SOURCES = (
    "src/embed_optim/primary_v3_publication_archive.py",
    "src/embed_optim/primary_v3_publication_reconstruction.py",
    "scripts/publication_reconstruction_fixture.py",
    "scripts/audit_dense_v3_publication_reconstruction.py",
    "tests/test_primary_v3_publication_reconstruction.py",
)
CHILD = r"""
import json, os, runpy, sys
from pathlib import Path
root, anchor, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
forbidden = json.loads(sys.argv[4])
def disallowed(path):
    name = os.path.abspath(os.fsdecode(path))
    return any(name == p or name.startswith(p + os.sep) for p in forbidden)
removed = [p for p in sys.path if disallowed(p)]
sys.path[:] = [p for p in sys.path if not disallowed(p)]
def guard(event, args):
    if event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo'}:
        raise RuntimeError('Network access forbidden in publication reconstruction')
    if event in {'open', 'os.listdir', 'os.scandir'} and args and isinstance(args[0], (str, bytes, os.PathLike)) and disallowed(args[0]):
        raise RuntimeError('Original producer/source path forbidden: ' + os.fsdecode(args[0]))
sys.addaudithook(guard)
sys.argv = ['publication-reconstruction', '--archive-root', str(root),
            '--expected-manifest-sha256', anchor, '--output', str(output)]
runpy.run_module('embed_optim.primary_v3_publication_reconstruction', run_name='__main__')
result = json.loads((output / 'reconstruction.json').read_text())
assert result['all_joint_raw_branches_and_inference_recomputed'] is True
assert result['all_seven_publication_outputs_recomputed'] is True
assert result['upstream_admission_simulated'] is True
assert result['scientific_completion'] is False and result['primary_scientific_admission'] is False
modules = {n: str(Path(m.__file__).relative_to(root / 'source/src'))
           for n, m in sys.modules.items() if n.startswith('embed_optim') and getattr(m, '__file__', None)}
print('PUBLICATION_REPLAY_RESULT=' + json.dumps({'receipt': result, 'imported_modules': modules,
      'network_and_original_paths_blocked': True, 'removed_search_paths': removed}, sort_keys=True), flush=True)
"""


def binding(path):
    return {"path": str(Path(path).absolute()), **file_identity(path)}


def cold(work, payload, anchor, forbidden):
    work.mkdir()
    command = [
        sys.executable,
        "-B",
        "-c",
        CHILD,
        str(payload),
        anchor,
        str(work / "recomputed"),
        json.dumps(forbidden),
    ]
    env = {**os.environ, "PYTHONPATH": str(payload / "source/src"), "PYTHONDONTWRITEBYTECODE": "1"}
    with (work / "stdout.txt").open("x") as out, (work / "stderr.txt").open("x") as err:
        process = subprocess.run(command, cwd=work, env=env, stdout=out, stderr=err, timeout=7200)
    error = (work / "stderr.txt").read_text()
    assert process.returncode == 0, error
    rows = (work / "stdout.txt").read_text().splitlines()
    value = json.loads(
        next(
            row.removeprefix("PUBLICATION_REPLAY_RESULT=")
            for row in rows
            if row.startswith("PUBLICATION_REPLAY_RESULT=")
        )
    )
    return {"exit_code": process.returncode, "result": value}


def altered_outputs(work, payload, fresh):
    """Exact-output controls reuse only this cold run's freshly reconstructed bytes."""
    expected = {name: (fresh / "publication" / name).read_bytes() for name in publication.OUTPUTS}
    plan = read_json(fresh / "publication/manifest.json")["plan"]
    controls = []
    for name in publication.OUTPUTS:
        target = work / ("altered-" + name.replace(".", "-"))
        shutil.copytree(payload / archive.ROLE, target)
        path = target / name
        if name.endswith(".json"):
            changed = read_json(path)
            changed["undeclared_scientific_claim"] = "Synthetic alteration: optimizer superiority"
            path.write_text(json.dumps(changed, sort_keys=True))
        else:
            path.write_bytes(
                path.read_bytes() + b"\n\\newcommand{\\UndeclaredClaim}{Synthetic alteration}\n"
            )
        manifest = read_json(target / "manifest.json")
        manifest["outputs"][name] = {"path": name, **file_identity(path)}
        (target / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        try:
            publication.inspect(target, plan, expected)
        except ValueError as error:
            assert str(error) == "Publication differs from freshly reconstructed evidence: " + name
        else:
            raise AssertionError("A rehashed altered publication output was accepted")
        controls.append(
            {
                "file": name,
                "refused": True,
                "consumer_scope": "exact_publication_output_inspection",
                "same_cold_run_expected_bytes_reused": True,
                "raw_reconstruction_repeated": False,
                "fresh_process_control": False,
            }
        )
    return controls


def run(args):
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not work.is_dir()
        or list(work.iterdir())
    ):
        raise ValueError("Require assertions, CPU-only operation and a new empty audit directory")
    if bool(args.replay) != bool(args.replay_sha256):
        raise ValueError("A replay needs an external completed-result anchor")
    before = handoff()
    sources = [binding(root / name) for name in SOURCES]
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    contract = PublicationContract.load(root / archive.PROTOCOL, primary)
    parent = None
    if args.replay:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        parent = read_json(args.replay)
        assert parent["passed"] is True and parent["scientific_completion"] is False
        require_same(parent["sources"], sources)
        for row in parent["artifacts"]:
            verify_file(row["path"], row)
        original, anchor = Path(parent["payload_root"]), parent["external_manifest_sha256"]
    else:
        original, anchor = make(work / "fixture", contract)
    files.inspect(original, anchor)
    payload = work / "relocated-payload"
    shutil.copytree(original, payload)
    forbidden = [
        str(root),
        "/root/embedding-optimizer-study",
        str(original),
        str(ORIGINAL),
        "/tmp/dense-partition-candidate.kHXoGW",
        "/tmp/dense-weight-entry-reference.Um8TxW",
        "/tmp/dense-natural-readiness.a4xZzR",
        str(args.training_root.resolve()),
    ]
    forbidden += original_locations(payload)
    print(
        {"complete_synthetic_publication_fixture_ready": True, "external_anchor": anchor},
        flush=True,
    )
    checked = cold(work / "cold", payload, anchor, forbidden)
    if parent:
        require_same(checked["result"], parent["cold_child"]["result"])
    controls = altered_outputs(work, payload, work / "cold/recomputed")
    return finish(args, sources, contract, before, payload, anchor, checked, controls)


def finish(args, sources, contract, before, payload, anchor, checked, controls):
    work = args.workdir.resolve()
    absent = SimpleNamespace(
        experiment_root=work / "missing-primary", output=work / "forbidden-authoring"
    )
    try:
        archive.build(contract, absent)
    except ValueError as error:
        refusal = str(error)
        assert "ordinary retained run" in refusal
    else:
        raise AssertionError("Actual missing primary archive authoring was accepted")
    assert not absent.output.exists()
    for row in sources:
        verify_file(row["path"], row)
    require_same(handoff(), before)
    write_new(
        work / "result.json",
        {
            "scope": "engineering_complete_synthetic_publication_reconstruction_audit",
            "passed": True,
            "fresh_replay": args.replay is not None,
            "replay_parent": None
            if args.replay is None
            else {"path": str(args.replay), "sha256": args.replay_sha256},
            "payload_root": str(payload),
            "external_manifest_sha256": anchor,
            "cold_child": checked,
            "altered_publication_output_controls": controls,
            "actual_missing_primary_authoring_refusal": refusal,
            "upstream_primary_admission_simulated": True,
            "actual_checkpoint_backed_positive_authoring_verified": False,
            "strict_manuscript_consumer_integrated": False,
            "physical_cross_host_execution_verified": False,
            "reviewed_source_runtime_release_verified": False,
            "manuscript_installed": False,
            "scientific_completion": False,
            "sources": sources,
            "post_execution_dispatchers": before,
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
