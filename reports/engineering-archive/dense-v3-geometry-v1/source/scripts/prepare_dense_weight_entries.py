"""Prepare an explicit changed-entry measurement correction; old geometry is never rewritten."""

import argparse
import copy
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import DRAFT, file_identity
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.primary_weight_entries import DEFINITION, PARENTS, SCOPE, SOURCES, load_amendment


def payload(root, primary):
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": DRAFT,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "primary": {
            "path": primary.path.relative_to(root).as_posix(),
            **file_identity(primary.path),
        },
        "parents": {
            key: {"path": name, **file_identity(root / name)} for key, (name, _) in PARENTS.items()
        },
        "sources": {name: file_identity(root / name) for name in SOURCES},
        "measurement": copy.deepcopy(DEFINITION),
        "scientific_completion": False,
        "formal_execution_authorized": False,
        "boundary": "The original matrix-level proxy remains in old source/artifacts. This component computes the intended individual-entry statistic from actual saved values, using the existing base and checkpoint anchors. Full geometry/dimension/publication integration remains required. It creates no optimizer finding.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    write_new(args.output, payload(root, primary))
    load_amendment(args.output, primary)
    print({"prepared": True, **file_identity(args.output)})


if __name__ == "__main__":
    main()
