"""Prepare the separate v3 validation implementation contract without modifying its parents."""

import argparse
import copy
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import DRAFT, file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation import (
    CORE_ACCEPTANCE_SHA,
    SCOPE,
    SETTINGS,
    SOURCES,
    ValidationContract,
)
from embed_optim.primary_v3_validation_io import write_new


def build_payload(root, primary):
    old = read_json(root / "configs/validation_probe.json")
    for new_key, old_key in (
        ("model_dtype", "model_dtype"),
        ("forward_dtype", "forward_dtype"),
        ("model_mode", "model_mode"),
        ("flash_attention", "flash_attention"),
        ("batch_size", "dense_batch_size"),
        ("rows", "expected_sample_records_per_job"),
    ):
        if SETTINGS[new_key] != old["evaluation"][old_key]:
            raise ValueError("A v3 validation numerical setting differs from its original scorer")
    accepted = root / "reports/engineering-archive/dense-primary-v3-chain-v1/validation.json"
    if file_identity(accepted)["sha256"] != CORE_ACCEPTANCE_SHA:
        raise ValueError("Wrong v3 core-chain acceptance")
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": DRAFT,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "primary": {
            "path": primary.path.relative_to(root).as_posix(),
            **file_identity(primary.path),
        },
        "core_acceptance": {
            "path": accepted.relative_to(root).as_posix(),
            **file_identity(accepted),
        },
        "dataset": primary.payload["datasets"]["validation"],
        "settings": copy.deepcopy(SETTINGS),
        "sources": {name: file_identity(root / name) for name in SOURCES},
        "scientific_completion": False,
        "production_deployed": False,
        "selection_parent": "The v3 primary execution/outcome loss-then-lower-LR rule overrides the older exploratory margin tie breaker. No BEIR metric is read by selection.",
        "additional_evidence": "Store all eight candidate cosine scores and all original FP32 metrics for each original-field-bound row. Replay metrics independently in scalar float64 without changing recorded values or introducing a selection tolerance.",
        "boundary": "New v3 validation scorer/reader/selector, not formal execution authorization or complete scientific inference. Existing primary, data and historical validation contracts are unchanged. Full primary runs, downstream outcome/mechanism/publication and reviewed assembled-source handoff remain required.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only preparation and a new explicit protocol path")
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    write_new(args.output, build_payload(root, primary))
    validated = ValidationContract.load(args.output, primary)
    print({"prepared": True, "status": validated.payload["status"], **file_identity(args.output)})


if __name__ == "__main__":
    main()
