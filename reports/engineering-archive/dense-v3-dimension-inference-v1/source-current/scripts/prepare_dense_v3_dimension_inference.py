"""Prepare the source-bound functional inference draft, without reading primary scores."""

import argparse
import copy
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.bridge_named_features import MISSING_POLICY
from embed_optim.bridge_numerics import NUMERICAL_POLICY
from embed_optim.dimension_inference import RULES, TABLE_COUNTS
from embed_optim.primary_contract import DRAFT, file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import (
    PARENTS,
    SCOPE,
    SOURCES,
    FunctionalInferenceContract,
)
from embed_optim.primary_v3_validation_io import write_new


def payload(root):
    scientific = read_json(root / PARENTS["scientific"][0])
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": DRAFT,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "parents": {
            key: {"path": name, **file_identity(root / name)} for key, (name, _) in PARENTS.items()
        },
        "sources": {name: file_identity(root / name) for name in SOURCES},
        "rules": copy.deepcopy(RULES),
        "bridge_numerics": copy.deepcopy(NUMERICAL_POLICY),
        "undefined_policy": copy.deepcopy(MISSING_POLICY),
        "table_counts": copy.deepcopy(TABLE_COUNTS),
        "scientific_rules": {
            key: scientific[key]
            for key in (
                "corrected_primary_comparison",
                "rotation_control",
                "retrieval_bridge",
                "figure_contract",
                "claim_boundary",
            )
        },
        "scientific_completion": False,
        "formal_execution_authorized": False,
        "manuscript_installation_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only draft preparation and a new file")
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    write_new(args.output, payload(root))
    FunctionalInferenceContract.load(args.output, primary)
    print({"prepared": True, "manuscript_installed": False, **file_identity(args.output)})


if __name__ == "__main__":
    main()
