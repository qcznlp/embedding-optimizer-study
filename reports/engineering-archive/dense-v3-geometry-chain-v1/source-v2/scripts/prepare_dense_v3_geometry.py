"""Prepare full v3 geometry without launching any primary computation or release."""

import argparse
import copy
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import DRAFT, file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_geometry import (
    IMPLEMENTATION_REVISION,
    PARENTS,
    SCOPE,
    SETTINGS,
    SOURCES,
    TABLE_COUNTS,
    VALIDITY,
    GeometryContract,
)
from embed_optim.primary_v3_validation_io import write_new


def payload(root):
    original = read_json(root / PARENTS["analysis"][0])
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
        "table_counts": copy.deepcopy(TABLE_COUNTS),
        "weight_space_rules": copy.deepcopy(original["weight_space"]),
        "subspace_validity": copy.deepcopy(VALIDITY),
        "implementation_revision": copy.deepcopy(IMPLEMENTATION_REVISION),
        "scientific_completion": False,
        "formal_execution_authorized": False,
        "boundary": "Prepared complete geometry consumer. Original spectral kernel, ranks, seeds, stages and all-rate aggregation are retained. Global norm/cosine reductions use an explicit FP64 implementation correction without changing their definitions or tolerances. Exact entry counts use the separate accepted amendment. Unsupported nonzero subspace signal rank is refused, not imputed. This does not release training or certify retrieval mechanisms/dimension utility; draft-parent release transition and downstream consumers remain required.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require CPU-only preparation")
    root = args.repository.resolve()
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    write_new(args.output, payload(root))
    GeometryContract.load(args.output, primary)
    print({"geometry_prepared": True, **file_identity(args.output)})


if __name__ == "__main__":
    main()
