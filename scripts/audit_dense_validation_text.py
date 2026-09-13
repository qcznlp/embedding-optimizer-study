"""Read-only text eligibility check of the existing frozen 4096-query holdout."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset

from embed_optim.primary_contract import file_identity, verify_file
from embed_optim.validation_data import audit_validation_data
from scripts.audit_dense_natural_data import write_new
from scripts.audit_dense_text_integrity import content_audit, manifest_linkage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require a CPU-only read and new receipt")
    root = args.experiment_root / "data/validation-4096-seed20260826"
    spec = args.repository / "configs/validation_probe.json"
    original = audit_validation_data(
        root, args.experiment_root / "data/denseon-sft-500k-seed42", spec_path=spec
    )
    files = [{"path": str(p), **file_identity(p)} for p in sorted(root.rglob("*")) if p.is_file()]
    dataset = Dataset.load_from_disk(str(root / "dataset"))
    content = content_audit(dataset.data.table)
    linkage = manifest_linkage(dataset, root / "rows.jsonl")
    for row in files:
        verify_file(Path(row["path"]), row)
    result = {
        "scope": "engineering_dense_frozen_validation_text",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scientific_completion": False,
        "model_outputs_read": False,
        "data_modified": False,
        "frozen_validation_audit": original,
        "content": content,
        "actual_materialized_id_linkage": linkage,
        "authenticated_files_unchanged": files,
        "source_bindings": [
            {"path": str(p), **file_identity(p)}
            for p in (
                Path(__file__).resolve(),
                args.repository / "scripts/audit_dense_text_integrity.py",
                args.repository / "src/embed_optim/validation_data.py",
                spec,
            )
        ],
    }
    write_new(args.output, result)
    print(
        json.dumps(
            {
                "rows": len(dataset),
                "invalid_text_rows": content["invalid_text_rows"],
                "identical_positive_negative_rows": content[
                    "positive_negative_identical_text_rows"
                ],
                "repeated_negative_text_rows": content["repeated_negative_text_rows"],
                "output": str(args.output),
                **file_identity(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
