"""Verify this bounded factory archive; never launch or numerically replay a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

STORY = Path("/root/embedding-optimizer-story-refactor")
PARENT = STORY / "reports/engineering-archive/dense-v3-factorial-run-contract-v1"
PARENT_SHA = "fd0485f7e87ded67e1645f49035bf847da5dd1f56ae5925ed62a8536b5b07cb4"
MODEL_RECEIPTS = {
    "first": (26, 7, 67904, "42d4e7d3597f20b64f973a786751a2b8b4dc92aa6ca7438ea14b1d7381e95c24"),
    "final": (137, 10, 68440, "10516d79cb814d44fc6fda8904a7349f194228f54e3094a857fa5aa27cb21fe8"),
}


def identity(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Require an ordinary archived file: {path}")
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def inspect(root):
    actual = root / "actual"
    require(
        identity(PARENT / "verification.json")["sha256"] == PARENT_SHA, "Parent archive changed"
    )
    last = None
    for name, (test_count, controls, size, sha) in MODEL_RECEIPTS.items():
        path = actual / f"actual-loading-{name}.json"
        require(identity(path) == {"bytes": size, "sha256": sha}, "Actual receipt bytes changed")
        last = read(path)
        for field in (
            "scientific_admission",
            "execution_authorized",
            "gpu_execution",
            "forward_backward_or_training",
            "calibration_outputs_consumed",
            "formal_trainer_constructed",
        ):
            require(last[field] is False, "A loading diagnostic claims formal execution")
        records = last["actual_cpu_source_loads"]
        require(
            [r["state"] for r in records] == ["adamw_state", "muon_state"],
            "Incomplete real source loading",
        )
        require(len(last["synthetic_rate_recipe_grid"]) == 12, "Missing recipe cells")
        cells = {(r["state"], r["operator"], r["seed"]) for r in last["synthetic_rate_recipe_grid"]}
        require(
            cells
            == {
                (s, o, d)
                for s in ("adamw_state", "muon_state")
                for o in ("adamw", "muon")
                for d in (314159, 271828, 161803)
            },
            "Recipe grid differs",
        )
        for record in records:
            require(
                record["loaded_tensor_count"] == 134
                and record["loaded_parameter_count"] == 149014272,
                "Not both complete real source models",
            )
            require(
                record["every_saved_tensor_bitwise_equal"] is True
                and record["original_source_weights_unchanged"] is True,
                "Actual source loading failed",
            )
            require(
                record["synthetic_rate_and_data_metadata_not_run_admission"] is True,
                "Synthetic identity boundary omitted",
            )
            require(
                record["configuration"]["diagnostic_cpu"] is True
                and record["configuration"]["backend"] == "sdpa",
                "CPU diagnostic boundary changed",
            )
            require(
                len(record["configuration_controls"]) == controls
                and all(c["rejected"] is True for c in record["configuration_controls"]),
                "A configuration counterexample did not fail",
            )
            require(len(record["source_files"]) == 68, "Wrong parent closure")
            for relative, expected in record["source_files"].items():
                require(
                    identity(actual / "source-first" / relative) == expected,
                    "Factory parent source changed",
                )
                require(
                    identity(PARENT / "actual/source-fifth" / relative) == expected,
                    "An accepted parent was replaced",
                )
        cases = list(ET.parse(actual / f"tests-{name}.xml").getroot().iter("testcase"))
        require(len(cases) == test_count, "Testcase count differs")
        require(
            all(not list(c.iter(tag)) for c in cases for tag in ("failure", "error", "skipped")),
            "Tests did not pass",
        )
        worker = "audit_factorial_v3_factory.py" if name == "first" else "audit_factory_final.py"
        require(identity(actual / worker) == last["worker_source"], "Executed audit source changed")
        require(
            identity(actual / "test_factorial_v3_run_contract.py") == last["test_metadata_source"],
            "Test metadata helper changed",
        )
        require(
            identity(actual / "source-first/src/embed_optim/factorial_v3_factory.py")
            == last["factory_source"],
            "Actual factory source changed",
        )
    block = read(root / "ops-block.json")
    names = ("AGENTS.md", "PROJECT_STATUS.md", "EXPERIMENT_RUNNING.md", "CONTINUATION_RUNNING.md")
    anchor = (
        "**Source-bound factorial run component checked; real GPU admission pending — 2026-09-10.**"
    )
    require(len(block["targets"]) == 4, "Wrong operational edit scope")
    operational = []
    for name, target in zip(names, block["targets"], strict=True):
        before, after = root / "ops-before" / name, root / "ops-after" / name
        require(before.read_text().count(anchor) == 1, "Operational insertion anchor is ambiguous")
        require(
            after.read_text() == before.read_text().replace(anchor, block["block"] + anchor, 1),
            "Operational edit contains another change",
        )
        operational.append({"target": target, "before": identity(before), "after": identity(after)})
    return {
        "source_files": 69,
        "final_test_cases": 137,
        "actual_source_models_per_call": 2,
        "final_configuration_counterexamples": 20,
        "gpu_or_training_admission": False,
        "reversible_operational_edits": operational,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    summary = inspect(root)
    files = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Archive contains a symlink")
        if path.is_file() and path.name != "verification.json":
            files[path.relative_to(root).as_posix()] = identity(path)
    output = root / "verification.json"
    if args.verify:
        saved = read(output)
        require(
            saved["files"] == files and saved["summary"] == summary,
            "Fresh archive verification differs",
        )
    else:
        require(not output.exists(), "Preserve previous verification")
        with output.open("x") as stream:
            json.dump(
                {
                    "scope": "dense-v3-factorial-factory-archive-v1",
                    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": summary,
                    "files": files,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
    print(
        json.dumps(
            {"verified_files": len(files), "summary": summary, "verification": identity(output)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
