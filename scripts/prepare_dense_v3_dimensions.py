"""Prepare the complete fixed-probe v3 export/feature contract; execute no model."""

import argparse
import copy
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.dimension_interventions import NUMERICAL_POLICY
from embed_optim.primary_contract import DRAFT, file_identity
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_contract import (
    ADMISSION,
    ENCODING,
    INPUT_EXECUTION,
    PARENTS,
    SCOPE,
    SOURCES,
    TABLE_COUNTS,
    DimensionContract,
    planned_states,
)
from embed_optim.primary_v3_validation_io import write_new


def payload(root, primary):
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": DRAFT,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "parents": {
            key: {"path": name, **file_identity(root / name)} for key, (name, _) in PARENTS.items()
        },
        "sources": {name: file_identity(root / name) for name in SOURCES},
        "encoding": copy.deepcopy(ENCODING),
        "input_execution": copy.deepcopy(INPUT_EXECUTION),
        "numerical_policy": copy.deepcopy(NUMERICAL_POLICY),
        "admission": copy.deepcopy(ADMISSION),
        "table_counts": copy.deepcopy(TABLE_COUNTS),
        "states": planned_states(primary),
        "scientific_completion": False,
        "formal_execution_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise ValueError("Require CPU-only preparation and a new output path")
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    write_new(args.output, payload(root, primary))
    contract = DimensionContract.load(args.output, primary)
    print(
        {"prepared": True, "states": len(contract.payload["states"]), **file_identity(args.output)}
    )


if __name__ == "__main__":
    main()
