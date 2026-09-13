"""Independent file/source/attempt preservation check; no numerical replay."""

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
WORK = Path("/tmp/dense-v3-factorial-run-contract.BEaQJZ")


def file_id(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Not an ordinary archive file: {path}")
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require_file(path, expected):
    if file_id(path) != {k: expected[k] for k in ("bytes", "sha256")}:
        raise ValueError(f"Archive source/payload differs: {path}")


def read(path):
    return json.loads(path.read_bytes())


def main():
    output = ROOT / "verification.json"
    if output.exists():
        raise ValueError("Do not overwrite an earlier archive verification")
    parent = (
        REPOSITORY
        / "reports/engineering-archive/dense-v3-factorial-calibration-v1/actual/actual-inputs-third.json"
    )
    if (
        file_id(parent)["sha256"]
        != "3640dc3a0deaa142d7e09eabfcc8450ab786db1b7302e04f470518557a63176f"
    ):
        raise ValueError("Accepted parent changed")
    admitted = read(parent)
    copies = {}
    for path in sorted(WORK.rglob("*")):
        if path.is_file():
            relative = path.relative_to(WORK)
            expected = file_id(path)
            require_file(ROOT / "actual" / relative, expected)
            copies[str(relative)] = expected
    for version in ("first", "second", "third", "fourth", "fifth"):
        source = ROOT / "actual" / f"source-{version}"
        for relative, expected in admitted["source_files"].items():
            require_file(source / relative, expected)
    for name, relative in {
        "factorial_v3_run_contract.py": "src/embed_optim/factorial_v3_run_contract.py",
        "factorial_v3_bound_trainer.py": "src/embed_optim/factorial_v3_bound_trainer.py",
        "test_factorial_v3_run_contract.py": "tests/test_factorial_v3_run_contract.py",
    }.items():
        require_file(ROOT / "source-current" / name, file_id(REPOSITORY / relative))
    for name in ("factorial_v3_run_contract.py", "factorial_v3_bound_trainer.py"):
        require_file(
            ROOT / "actual/source-fifth/src/embed_optim" / name,
            file_id(ROOT / "source-current" / name),
        )
    tests = {}
    for name, count, failures in (
        ("tests-first.xml", 83, 30),
        ("tests-second.xml", 93, 0),
        ("tests-counterexamples.xml", 3, 3),
        ("tests-final.xml", 98, 0),
        ("tests-fifth.xml", 107, 0),
    ):
        tree = ET.parse(ROOT / "actual" / name)
        suites = list(tree.iter("testsuite"))
        if len(suites) != 1:
            raise ValueError("Unexpected test XML structure")
        observed = {k: int(suites[0].attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
        if (
            observed != {"tests": count, "failures": failures, "errors": 0, "skipped": 0}
            or len(list(tree.iter("testcase"))) != count
        ):
            raise ValueError("Original test count/verdict changed")
        tests[name] = observed
    actual_views = {}
    for suffix, source_version, sha in (
        ("first", "third", "ee1ef398cf5f2792d666d9fe7873f010a40bec07100737d9d9d485f2abf93b33"),
        ("final", "fourth", "e85676395ad3ec7e87860c728b733442ed3d82d601fa8dc988bf700c7831ab1d"),
        ("fifth", "fifth", "3e87392057fdca49460f1c40883881183dc9126d64a7940aef27c0fd62b6baf4"),
    ):
        path = ROOT / "actual" / f"actual-view-{suffix}.json"
        if file_id(path)["sha256"] != sha:
            raise ValueError("Actual input result changed")
        value = read(path)
        for relative, expected in value["sources"].items():
            require_file(ROOT / "actual" / f"source-{source_version}" / relative, expected)
        if len(value["sources"]) != 68 or value["gpu_initialized"] or value["formal_run_admitted"]:
            raise ValueError("Actual input scope differs")
        for label, record in value["actual_source_weights"].items():
            source = admitted["inputs"]["sources"][label]
            expected = next(
                r for r in source["native_checkpoint"]["files"] if r["path"] == "model.safetensors"
            )
            if (
                record["payload"] != expected
                or record["checkpoint"] != source["checkpoint"]
                or record["numeric"]
                != {"tensors": 134, "parameters": 149014272, "dtype": "torch.float32"}
            ):
                raise ValueError("Source-weight identity/count differs")
        actual_views[suffix] = value["complete_actual_branch_view"]
    if (
        actual_views["first"] != actual_views["final"]
        or actual_views["first"] != actual_views["fifth"]
    ):
        raise ValueError("Actual view changed across versions")
    writer_counts = {}
    for suffix, sha, fixture in (
        (
            "first",
            "9ad36c0be928968be8ed895d9552a9eb323946b32c88668936bc8676019698f2",
            "test_factorial_v3_run_contract-second.py",
        ),
        (
            "final",
            "ac9f6ca7cc769c93874424a75a6b222c302f557f219dd9a6a1641004f0d28983",
            "test_factorial_v3_run_contract-fourth.py",
        ),
        (
            "fifth",
            "952c0613298a56a9a94a661181aaac2fe97b58405bc20922f3177112c14d319a",
            "test_factorial_v3_run_contract-fifth.py",
        ),
    ):
        root = ROOT / "actual" / f"four-rank-{suffix}"
        complete = root / "complete.json"
        if file_id(complete)["sha256"] != sha:
            raise ValueError("Original four-rank result changed")
        value = read(complete)
        require_file(ROOT / "actual" / fixture, value["fixture_source"])
        require_file(ROOT / "audit_collective_writer.py", value["audit_source"])
        for row in value["files"]:
            require_file(root / row["path"], row)
        if [r["rank"] for r in value["ranks"]] != [0, 1, 2, 3]:
            raise ValueError("Missing rank controls")
        for rank in value["ranks"]:
            if (
                rank["backend"] != "gloo"
                or rank["world_size"] != 4
                or rank["actual_trainer_executed"]
                or rank["gpu_execution"]
                or rank["scientific_admission"]
                or not rank["all_rank_bindings_identical"]
                or not rank["partial_outputs_preserved"]
                or not rank["default_deep_reader_rejects_fixture"]
            ):
                raise ValueError("Four-rank control scope differs")
        writer_counts[suffix] = len(value["files"])
    ops = read(ROOT / "ops-block.json")
    edits = {}
    # apply_patch preserves an extra separating blank line after the inserted
    # block; test the exact insertion, not a whitespace-normalized reconstruction.
    for target, name in ops["targets"]:
        before = (ROOT / "ops-before" / name).read_bytes()
        current = Path(target).read_bytes()
        needle = (ops["block"] + "\n").encode()
        if current.count(needle) != 1 or current.replace(needle, b"", 1) != before:
            raise ValueError(f"Operational edit is not exactly reversible: {name}")
        edits[target] = {
            "before": file_id(ROOT / "ops-before" / name),
            "after": file_id(Path(target)),
        }
    if (
        file_id(REPOSITORY / "paper/main.tex")["sha256"]
        != "4d59c3219589204ec396d5b9d8c6269df1dfc89f29361ddf70ada26100dbbb2b"
    ):
        raise ValueError("Manuscript changed")
    files = {str(p.relative_to(ROOT)): file_id(p) for p in sorted(ROOT.rglob("*")) if p.is_file()}
    result = {
        "scope": "factorial-run-component-archive-preservation",
        "files": files,
        "original_work_copies": len(copies),
        "unchanged_parent_sources_per_version": 66,
        "source_versions": 5,
        "test_results": tests,
        "actual_branch_view": actual_views["fifth"],
        "writer_control_file_counts": writer_counts,
        "reversible_operational_edits": edits,
        "fresh_model_or_gradient_replay": False,
        "formal_run_or_gpu_admission": False,
        "scientific_completion": False,
    }
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                **file_id(output),
                "bound_files": len(files),
                "original_work_copies": len(copies),
                "ops_edits_reversible": len(edits),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
