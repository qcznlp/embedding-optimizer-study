"""Prepare a separate exact-measurement bridge sensitivity; no scientific run is executed."""

import argparse
import copy
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.bridge_named_features import MISSING_POLICY
from embed_optim.bridge_numerics import NUMERICAL_POLICY
from embed_optim.exact_bridge_measurements import (
    CHECKPOINT_FEATURES,
    MEASUREMENT_RULES,
    PAIR_FEATURES,
)
from embed_optim.primary_contract import DRAFT, file_identity
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_exact_bridge import (
    AMENDMENT,
    PARENTS,
    SCOPE,
    SOURCES,
    TABLE_COUNTS,
    ExactBridgeContract,
)
from embed_optim.primary_v3_validation_io import write_new


def payload(root):
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": DRAFT,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "parents": {
            key: {"path": name, **file_identity(root / name)} for key, (name, _) in PARENTS.items()
        },
        "sources": {name: file_identity(root / name) for name in SOURCES},
        "numerical_policy": copy.deepcopy(NUMERICAL_POLICY),
        "undefined_policy": copy.deepcopy(MISSING_POLICY),
        "measurement_rules": copy.deepcopy(MEASUREMENT_RULES),
        "amendment": copy.deepcopy(AMENDMENT),
        "checkpoint_feature_mapping": copy.deepcopy(CHECKPOINT_FEATURES),
        "pair_feature_mapping": copy.deepcopy(PAIR_FEATURES),
        "table_counts": copy.deepcopy(TABLE_COUNTS),
        "scientific_completion": False,
        "formal_execution_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only preparation and a new output path")
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    write_new(args.output, payload(root))
    contract = ExactBridgeContract.load(args.output, primary)
    print({"prepared": True, "scope": contract.payload["scope"], **file_identity(args.output)})


if __name__ == "__main__":
    main()
