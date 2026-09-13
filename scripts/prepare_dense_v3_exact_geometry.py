"""Prepare an explicit exact-geometry layer without changing old approximate features."""

import argparse
import copy
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import DRAFT, file_identity
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_exact_geometry import (
    PARENTS,
    RULES,
    SCOPE,
    SETTINGS,
    SOURCES,
    TABLE_COUNTS,
    ExactGeometryContract,
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
        "settings": copy.deepcopy(SETTINGS),
        "measurement_rules": copy.deepcopy(RULES),
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
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    write_new(args.output, payload(root))
    ExactGeometryContract.load(args.output, primary)
    print({"exact_geometry_prepared": True, **file_identity(args.output)})


if __name__ == "__main__":
    main()
