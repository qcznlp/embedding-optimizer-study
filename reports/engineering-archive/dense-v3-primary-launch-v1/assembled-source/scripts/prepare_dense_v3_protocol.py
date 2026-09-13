"""Prepare one explicit v3 train/durability/BEIR contract; never release or launch it."""

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import DRAFT, file_identity, read_json
from embed_optim.primary_v3_contract import (
    ADDITIONAL_CONSUMERS,
    PARENTS,
    PATHS,
    SCOPE,
    PrimaryV3Contract,
    dataset_inventory,
    expected_evaluation,
)
from scripts.audit_dense_natural_data import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only preparation and a new explicit protocol path")
    bindings, parents = {}, {}
    for key, (name, sha) in PARENTS.items():
        binding = {"path": name, **file_identity(root / name)}
        if binding["sha256"] != sha:
            raise ValueError("A declared immutable v3 parent differs")
        bindings[key], parents[key] = binding, read_json(root / name)
    parent, amendment = parents["primary_parent"], parents["data_amendment"]
    payload = {
        "schema_version": 1,
        "scope": SCOPE,
        "status": DRAFT,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        **bindings,
        **PATHS,
        **{
            k: parent[k]
            for k in (
                "world_size",
                "checkpoint_steps",
                "beir_task_revisions",
                "gpu_pools",
                "gpu_lease_root",
                "wandb",
                "checkpoint_repository",
                "worker_python",
                "training_sources",
                "analysis",
            )
        },
        "evaluation": expected_evaluation(parent),
        "datasets": {p: dataset_inventory(amendment, p) for p in ("training", "validation")},
        "consumer_sources": {
            name: file_identity(root / name)
            for name in sorted(set(parent["consumer_sources"]) | set(ADDITIONAL_CONSUMERS))
        },
        "scientific_completion": False,
        "production_deployed": False,
        "pending_release_requirements": [
            "complete v3 validation selection/outcome/dimension/publication consumers",
            "separately reviewed routed-factorial numerical integration",
            "owner WIP/deployment direction and publication/access gates",
            "reviewed assembled committed source/runtime and controller transition",
        ],
        "boundary": "New v3 core train/backup/BEIR/whole-grid contract, not a release, full downstream integration or a scientific result. Data locations are new; values, numerical training source, hyperparameters, tasks, estimands and the primary loss/lower-LR selection rule retain their explicit parents. Old v2 protocols and consumers are unchanged.",
    }
    write_new(args.output, payload)
    contract = PrimaryV3Contract.load(args.output, root, args.training_root)
    print(
        {
            "prepared": True,
            "runs": len(contract.inputs["runs"]),
            "status": DRAFT,
            **file_identity(args.output),
        }
    )


if __name__ == "__main__":
    main()
