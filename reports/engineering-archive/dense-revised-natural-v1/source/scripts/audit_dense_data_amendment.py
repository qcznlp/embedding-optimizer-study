"""Read-only actual amendment admission and controlled invalid-file rejections."""

import argparse
import copy
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, verify_file
from embed_optim.primary_data_revision import load_amendment
from scripts.audit_dense_natural_data import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not args.workdir.is_dir()
        or any(args.workdir.iterdir())
    ):
        raise ValueError("Require CPU-only admission checks in a fresh namespace")
    before = file_identity(args.amendment)
    if before["sha256"] != args.amendment_sha256:
        raise ValueError("Actual amendment differs from declared source")
    value, inputs = load_amendment(args.amendment, args.repository)
    cases = []

    def add(name, keys, changed):
        case = copy.deepcopy(value)
        node = case
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = changed
        cases.append((name, case))

    add("formal_authorization", ["execution_authorized"], True)
    add("released_status", ["status"], "reviewed_primary_execution_lock")
    add("old_scope", ["scope"], "dense_primary_correctness_v2")
    add("boolean_schema", ["schema_version"], True)
    add("missing_source_closure", ["source_bindings"], {})
    add("missing_parent_closure", ["parents"], {})
    add("old_input_snapshot", ["inputs"], value["parents"]["inputs"])
    add("wrong_primary_parent", ["parents", "primary"], value["parents"]["validation"])
    add(
        "old_training_root",
        ["datasets", "training", "root"],
        "/root/embedding-optimizer-study/data/denseon-sft-500k-seed42",
    )
    add(
        "old_validation_root",
        ["datasets", "validation", "root"],
        "/root/embedding-optimizer-study/data/validation-4096-seed20260826",
    )
    add("short_training_count", ["datasets", "training", "rows"], 499999)
    add("dropped_validation_group", ["datasets", "validation", "rows"], 4095)
    add(
        "altered_data_inventory",
        ["datasets", "training", "files"],
        value["datasets"]["training"]["files"][:-1],
    )
    add(
        "missing_replacement_position",
        ["datasets", "training", "changed_sample_ids"],
        value["datasets"]["training"]["changed_sample_ids"][:-1],
    )
    add("omit_mandatory_groups", ["diagnostic", "include_all_training_replacement_groups"], False)
    add("more_diagnostic_steps", ["diagnostic", "steps_per_optimizer"], 4)
    add("altered_source_quota", ["diagnostic", "source_quotas", "fiqa"], 41)
    add("wrong_gpu_pool", ["diagnostic", "gpu_pool"], ["0", "1", "2", "3"])
    add("fewer_algorithms", ["diagnostic", "algorithms"], ["adamw", "muon"])
    add(
        "altered_selection_rule",
        ["unchanged_evaluation", "validation_selection", "rule"],
        "choose using BEIR",
    )
    records = []
    for name, invalid in cases:
        path = args.workdir / f"{name}.json"
        write_new(path, invalid)
        try:
            load_amendment(path, args.repository)
        except (ValueError, KeyError) as error:
            records.append(
                {
                    "case": name,
                    "rejected": True,
                    "error": f"{type(error).__name__}: {error}",
                    "file": {"path": str(path), **file_identity(path)},
                }
            )
        else:
            records.append(
                {
                    "case": name,
                    "rejected": False,
                    "file": {"path": str(path), **file_identity(path)},
                }
            )
    path = args.workdir / "duplicate_json_key.json"
    with path.open("x") as stream:
        stream.write('{"scope":"ignored",' + json.dumps(value, sort_keys=True)[1:] + "\n")
    try:
        load_amendment(path, args.repository)
    except ValueError as error:
        records.append(
            {
                "case": "duplicate_json_key",
                "rejected": True,
                "error": str(error),
                "file": {"path": str(path), **file_identity(path)},
            }
        )
    else:
        records.append(
            {
                "case": "duplicate_json_key",
                "rejected": False,
                "file": {"path": str(path), **file_identity(path)},
            }
        )
    verify_file(args.amendment, before)
    result = {
        "scope": "engineering_dense_data_amendment_admission",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "actual_baseline_accepted": True,
        "all_changed_files_rejected": all(r["rejected"] for r in records),
        "records": records,
        "derived_primary_identities": len(inputs["runs"]),
        "amendment": {"path": str(args.amendment), **before},
        "source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
        "scientific_completion": False,
        "model_updates": 0,
        "boundary": "Exercise only the actual read-only loader against full authenticated baseline data and retained changed-file copies. No changed configuration is trained and no parent/source/data file is modified.",
    }
    write_new(args.workdir / "result.json", result)
    print(
        {
            "baseline_accepted": True,
            "rejected_cases": sum(r["rejected"] for r in records),
            "cases": len(records),
        }
    )
    if not result["all_changed_files_rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
