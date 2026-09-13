"""Prepare an immutable exact-publication draft without reading model results."""

import argparse
import copy
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import DRAFT, file_identity
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_exact_publication_contract import (
    OUTPUTS,
    PARENTS,
    RULES,
    SCOPE,
    SOURCES,
    ExactPublicationContract,
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
        "rules": copy.deepcopy(RULES),
        "outputs": list(OUTPUTS),
        "scientific_completion": False,
        "formal_execution_authorized": False,
        "manuscript_installation_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only preparation and a new exact publication draft")
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    write_new(args.output, payload(root))
    ExactPublicationContract.load(args.output, primary)
    print({"prepared": True, "scientific_completion": False, **file_identity(args.output)})


if __name__ == "__main__":
    main()
