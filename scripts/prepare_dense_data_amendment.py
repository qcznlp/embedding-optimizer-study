"""Freeze a new explicit data amendment without changing any parent execution lock."""

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, verify_file
from embed_optim.primary_data_revision import (
    ACCEPTANCE_SHA,
    INPUT_SHA,
    QUOTAS,
    SCOPE,
    SEED,
    SOURCES,
    STATUS,
    load_amendment,
)
from scripts.audit_dense_natural_data import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only preparation and a new explicit amendment file")
    input_path = root / "reports/dense-primary-v3/input-bindings.json"
    if file_identity(input_path)["sha256"] != INPUT_SHA:
        raise ValueError("Actual revised inputs differ")
    inputs = read_json(input_path)
    acceptance_path = root / "reports/engineering-archive/dense-data-candidate-v1/validation.json"
    if file_identity(acceptance_path)["sha256"] != ACCEPTANCE_SHA:
        raise ValueError("Independent complete-data acceptance differs")
    verify_file(Path(inputs["data_producer"]["path"]), inputs["data_producer"])
    producer = read_json(Path(inputs["data_producer"]["path"]))
    parents = {
        "primary": "configs/dense_primary_v2_protocol.json",
        "validation": "configs/validation_probe.json",
        "inputs": "reports/dense-primary-v2/input-bindings.json",
    }
    original = read_json(root / parents["primary"])
    datasets = {}
    for partition, result in producer["partitions"].items():
        files = result["files"]
        for row in files:
            verify_file(Path(row["path"]), row)
        manifest = next(row for row in files if Path(row["path"]).name == "manifest.json")
        datasets[partition] = {
            "root": str(Path(manifest["path"]).parent),
            "rows": result["rows"],
            "changed_sample_ids": result["changed_sample_ids"],
            "files": files,
        }
    value = {
        "schema_version": 1,
        "scope": SCOPE,
        "status": STATUS,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "execution_authorized": False,
        "scientific_completion": False,
        "production_deployed": False,
        "parents": {
            name: {"path": path, **file_identity(root / path)} for name, path in parents.items()
        },
        "inputs": {"path": str(input_path.relative_to(root)), **file_identity(input_path)},
        "data_acceptance": {
            "path": str(acceptance_path.relative_to(root)),
            **file_identity(acceptance_path),
        },
        "datasets": datasets,
        "source_bindings": {name: file_identity(root / name) for name in SOURCES},
        "unchanged_evaluation": original["evaluation"],
        "unchanged_analysis": original["analysis"],
        "new_namespace_proposal": {
            "run_prefix": "verified-v3-",
            "output_root": "outputs/dense-correctness-v3",
            "checkpoint_prefix": "corrected-dense-correctness-v3/dense",
            "primary_contract_released": False,
        },
        "diagnostic": {
            "rows": 288,
            "source_quotas": QUOTAS,
            "seed": SEED,
            "include_all_training_replacement_groups": True,
            "steps_per_optimizer": 3,
            "global_query_groups": [128, 128, 32],
            "algorithms": ["adamw", "muon", "normuon"],
            "gpu_pool": ["4", "5", "6", "7"],
            "untrained_base_required": True,
            "no_outcome_selection": True,
        },
        "data_rule": "Implement only the independently reproduced six-training/52-validation whole-group revision. All old source populations, seeded priorities, first eligible positives, seven-of-ten draws, source quotas, totals and non-target fields are preserved. Reserve old train/validation IDs and query texts plus all pinned BEIR evaluation queries and accepted replacements.",
        "derived_subset_policy": "All three old subset contents match revised parents. Their old parent manifests remain historical; this amendment records explicit new parent compatibility, not old-protocol execution or promotion of historical results.",
        "release_requirements": [
            "new primary/validation and downstream consumers must explicitly load this revision",
            "seven-source actual-entrypoint diagnostic acceptance",
            "numerical and routed-factorial integration",
            "reviewed assembled-source/runtime handoff",
            "owner WIP/deployment direction and publication/access gates",
        ],
        "boundary": "A declared prospective input amendment and bounded diagnostic plan, not retrospective preregistration, primary execution authorization, an old hash refresh, or a scientific result. No old protocol, data, checkpoint, ledger or manuscript is modified.",
    }
    write_new(args.output, value)
    load_amendment(args.output, root)
    print(
        {
            "prepared": True,
            "execution_authorized": False,
            "rows": {p: r["rows"] for p, r in datasets.items()},
            **file_identity(args.output),
        }
    )


if __name__ == "__main__":
    main()
