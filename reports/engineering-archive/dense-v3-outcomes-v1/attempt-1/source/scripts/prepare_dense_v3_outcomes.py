"""Prepare the v3 statistical consumer draft without executing a model or selecting a recipe."""

import argparse
import copy
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import DRAFT, file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import (
    PARENTS,
    SCIENCE_KEYS,
    SCOPE,
    SOURCES,
    TABLE_COUNTS,
    OutcomeContract,
)
from embed_optim.primary_v3_validation import ValidationContract
from embed_optim.primary_v3_validation_io import write_new


def build_payload(root):
    parent = read_json(root / PARENTS["statistical_parent"][0])
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": DRAFT,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "parents": {
            key: {"path": name, **file_identity(root / name)} for key, (name, _) in PARENTS.items()
        },
        "scientific_rules": {key: copy.deepcopy(parent[key]) for key in SCIENCE_KEYS},
        "table_counts": TABLE_COUNTS,
        "sources": {name: file_identity(root / name) for name in SOURCES},
        "scientific_completion": False,
        "formal_execution_authorized": False,
        "boundary": "Preparation-only outcome consumer. The old statistical kernel and estimands are unchanged. It requires real complete v3 training, validation and BEIR evidence; fixtures and diagnostics are not primary results. Released-source/runtime handoff and downstream mechanism/publication consumers remain separate required work.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only preparation and a new output path")
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    validation = ValidationContract.load(root / PARENTS["validation"][0], primary)
    write_new(args.output, build_payload(root))
    checked = OutcomeContract.load(args.output, validation)
    print({"prepared": True, "scope": checked.payload["scope"], **file_identity(args.output)})


if __name__ == "__main__":
    main()
