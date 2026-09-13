"""Read saved worker evidence and source preservation; no numerical/GPU replay."""

import ast
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/embedding-optimizer-story-refactor")
ARCHIVE = REPO / "reports/engineering-archive/dense-v3-factorial-worker-v1"
WORK = Path("/tmp/dense-v3-factorial-worker.lIJCnUVr")
EXPERIMENT = Path("/root/embedding-optimizer-v3-experiment")


def identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Require ordinary evidence file: {path.name}")
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read(path, sha=None):
    if sha is not None:
        assert identity(path)["sha256"] == sha, path.name
    return json.loads(path.read_text())


def check_tests(version, wanted):
    path = ARCHIVE / "actual" / f"tests-{version}.xml"
    assert identity(path) == identity(WORK / path.name)
    log = path.with_suffix(".log")
    assert identity(log) == identity(WORK / log.name)
    tree = ET.parse(path)
    suites = tree.findall("testsuite")
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in wanted}
    assert counts == wanted
    cases = tree.findall(".//testcase")
    assert len(cases) == counts["tests"]
    failed = sorted(c.get("name") for c in cases if c.find("failure") is not None)
    if version == "first":
        assert failed == sorted(
            [
                "test_fresh_factory_controls[short_context]",
                "test_fresh_reader_controls_even_after_rehash[changed_factory_recipe]",
                "test_fresh_reader_controls_even_after_rehash[changed_factory_arguments]",
            ]
        )
    else:
        assert sum(c.get("classname") == "tests.test_factorial_v3_worker" for c in cases) in (0, 57)
        assert sum("test_factorial_v3_worker" in c.get("classname", "") for c in cases) == 57
    return {**counts, "failed_cases": failed, "xml": identity(path), "log": identity(log)}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parent = REPO / "reports/engineering-archive/dense-v3-factorial-factory-v1"
    parent_proof = read(
        parent / "verification.json",
        "6b0963d64374fc5f0657faf810935da059180ca9f3cfcf94967a2669c9c25d58",
    )
    prefix = "actual/source-first/"
    parent_files = {
        k.removeprefix(prefix): v for k, v in parent_proof["files"].items() if k.startswith(prefix)
    }
    assert len(parent_files) == 96
    new_file = "src/embed_optim/factorial_v3_worker.py"
    roots = (WORK / "source-first", WORK / "source-final", ARCHIVE / "actual/source-final")
    for root in roots:
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        assert actual == set(parent_files) | {new_file}, str(root)
        for rel, expected in parent_files.items():
            assert identity(root / rel) == expected, rel
    assert identity(roots[0] / new_file) == identity(
        ARCHIVE / "source-first/factorial_v3_worker.py"
    )
    for path in (
        roots[1] / new_file,
        roots[2] / new_file,
        ARCHIVE / "source-final/factorial_v3_worker.py",
    ):
        assert identity(path) == identity(REPO / new_file)
    assert identity(ARCHIVE / "source-final/test_factorial_v3_worker.py") == identity(
        REPO / "tests/test_factorial_v3_worker.py"
    )

    accepted = read(
        REPO
        / "reports/engineering-archive/dense-v3-factorial-calibration-v1/actual/actual-inputs-third.json",
        "3640dc3a0deaa142d7e09eabfcc8450ab786db1b7302e04f470518557a63176f",
    )
    dependencies = dict(accepted["source_files"])
    assert len(dependencies) == 66
    for name in (
        "factorial_v3_run_contract.py",
        "factorial_v3_bound_trainer.py",
        "factorial_v3_factory.py",
    ):
        relative = f"src/embed_optim/{name}"
        dependencies[relative] = parent_files[relative]
    assert len(dependencies) == 69
    for relative, expected in dependencies.items():
        assert identity(roots[1] / relative) == expected, relative
    dependencies[new_file] = identity(roots[1] / new_file)

    primary_assembly = read(
        EXPERIMENT / "launch/source-snapshot/source-assembly.json",
        "e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8",
    )
    assert len(primary_assembly["files"]) == 56
    for root in (
        Path("/root/embedding-optimizer-primary-v3"),
        EXPERIMENT / "launch/source-snapshot",
    ):
        for relative, row in primary_assembly["files"].items():
            assert identity(root / relative) == row["identity"], relative
    dispatches = {
        "evaluation-handoff/dispatch.py": "5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427",
        "evaluation-handoff/authorization.json": "2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c",
        "validation-handoff/validation.py": "d2fc55c67670ea5b42cacb055f48379ef278a465ce86d95808231ab86049d913",
        "validation-handoff/authorization.json": "6a8dcdd579bd6c2f6ea82e3f4cd808f88e92e9a28b9e97f41ffa6447eb0e1f6f",
        "functional-dimensions/dispatch.py": "3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5",
        "functional-dimensions/authorization.json": "d72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336",
    }
    for relative, sha in dispatches.items():
        assert identity(EXPERIMENT / "launch" / relative)["sha256"] == sha, relative
    assert (
        identity(REPO / "paper/main.tex")["sha256"]
        == "4d59c3219589204ec396d5b9d8c6269df1dfc89f29361ddf70ada26100dbbb2b"
    )
    old_docs = read(
        REPO / "reports/engineering-archive/dense-v3-weight-artifact-backup-v1/verification.json",
        "5efae5f9eeb905f2d55248669f6a41811cbe427bc828a7498d834fd1bba15315",
    )
    assert (
        identity(ARCHIVE / "before/CURRENT_EXPERIMENT.md")
        == old_docs["live_documents"]["CURRENT_EXPERIMENT.md"]
    )
    tests = {
        "first": check_tests("first", {"tests": 53, "failures": 3, "errors": 0, "skipped": 0}),
        "final": check_tests("final", {"tests": 194, "failures": 0, "errors": 0, "skipped": 0}),
    }
    attempts = read(ARCHIVE / "attempts.json")
    assert attempts["first"]["terminal"]["exit_code"] == 1
    assert attempts["first"]["launch"]["session_id"] == 33576
    assert attempts["final"]["terminal"]["exit_code"] == 0
    assert attempts["final"]["launch"]["session_id"] == 28447
    observations = read(ARCHIVE / "observations.json")
    assert observations["primary"]["observation"]["primary_tasks"] == [24, 840]
    assert observations["primary"]["observation"]["observer"]["live"] is True
    workers = observations["primary"]["observation"]["worker_observations_from_that_snapshot"]
    assert len(workers) == 8 and all(w["live"] and w["command_matches"] for w in workers)
    assert observations["validation"]["observation"]["scored_records"] == 7
    assert observations["functional"]["observation"]["encoded_states"] == []
    assert observations["resource_priority_request_answered"] is False

    docs = [REPO / "CURRENT_EXPERIMENT.md", ARCHIVE / "README.md", ARCHIVE / "plan.md"]
    links = 0
    for path in docs:
        content = path.read_text()
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", content):
            if target.startswith(("https:", "http:", "#")):
                continue
            # The verification record is authored only after this successful read.
            if path == ARCHIVE / "README.md" and target == "verification.json":
                continue
            assert (path.parent / target.split("#", 1)[0]).exists(), target
            links += 1
    for path in (REPO / new_file, REPO / "tests/test_factorial_v3_worker.py"):
        ast.parse(path.read_text())
    files = {
        p.relative_to(ARCHIVE).as_posix(): identity(p)
        for p in sorted(ARCHIVE.rglob("*"))
        if p.is_file() and p != ARCHIVE / "verification.json"
    }
    secrets = re.compile(rb"(?:wandb_v1_[A-Za-z0-9_-]{25,}|hf_[A-Za-z0-9]{20,})")
    for rel in files:
        assert secrets.search((ARCHIVE / rel).read_bytes()) is None, rel
    for path in docs:
        assert secrets.search(path.read_bytes()) is None, str(path)
    result = {
        "scope": "factorial_worker_component_preservation_readback",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "declared_parent_dependencies_unchanged": 69,
        "inherited_physical_files_unchanged_per_copy": 96,
        "source_copies_checked": 3,
        "final_source_files": dependencies,
        "primary_source_files_matched_per_root": 56,
        "primary_source_roots_checked": 2,
        "live_dispatch_and_authority_files_unchanged": 6,
        "manuscript_unchanged": True,
        "prior_current_document_preserved": True,
        "tests": tests,
        "local_links_resolved": links,
        "archive_files": files,
        "live_files": {
            str(p.relative_to(REPO)): identity(p)
            for p in (*docs, REPO / new_file, REPO / "tests/test_factorial_v3_worker.py")
        },
        "gpu_or_model_or_numerical_replay_by_this_read": False,
        "genuine_gpu_calibrations": 0,
        "formal_branches": 0,
        "execution_authorized": False,
        "scientific_admission": False,
        "source_publication": False,
        "whole_goal_complete": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
