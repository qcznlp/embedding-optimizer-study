#!/usr/bin/env python3
"""Read-only, checkpoint-deep audit of the completed primary DenseOn runs.

Only the requested report directory is written. Run with the audited checkout's
src directory on PYTHONPATH. A passing partial audit never certifies the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def identity(path: Path, repository: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": os.path.relpath(path.resolve(), repository),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def audit_materialized_rows(root: Path) -> dict[str, Any]:
    """Compare every materialized example identity to its canonical sampling row."""
    from datasets import Dataset

    columns = [
        "sample_id",
        "source",
        "query_id",
        "positive_id",
        *(f"negative_{index}_id" for index in range(7)),
    ]
    dataset = Dataset.load_from_disk(str(root / "dataset")).select_columns(columns)
    counts: Counter[str] = Counter()
    seen = set()
    mismatched = duplicate_ids = invalid_negatives = count = 0
    with (root / "rows.jsonl").open(encoding="utf-8") as handle:
        records = (json.loads(line) for line in handle if line.strip())
        for row, declared in itertools.zip_longest(dataset, records):
            count += 1
            if row is None or declared is None:
                mismatched += 1
                continue
            negative_ids = [row[f"negative_{index}_id"] for index in range(7)]
            observed = {key: row[key] for key in columns[:4]}
            observed["negative_ids"] = negative_ids
            expected = {key: declared[key] for key in observed}
            mismatched += observed != expected
            duplicate_ids += row["sample_id"] in seen
            seen.add(row["sample_id"])
            invalid_negatives += len(set(negative_ids)) != 7 or row["positive_id"] in negative_ids
            counts[row["source"]] += 1
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    complete = (
        count == 500_000
        and not mismatched
        and not duplicate_ids
        and not invalid_negatives
        and dict(counts) == manifest["quotas"]
    )
    return {
        "complete": complete,
        "rows": count,
        "unique_sample_ids": len(seen),
        "row_identity_mismatches": mismatched,
        "duplicate_sample_ids": duplicate_ids,
        "invalid_negative_groups": invalid_negatives,
        "source_counts": dict(counts),
        "boundary": "All stored row identities checked; upstream texts were not re-downloaded.",
    }


def audit_bindings(repository: Path) -> dict[str, Any]:
    protocols = [
        "dense_no_packing_execution_protocol.json",
        "dense_no_packing_analysis_protocol.json",
        "dense_no_packing_outcome_protocol.json",
        "dense_no_packing_bridge_implementation_protocol_v2.json",
        "dense_no_packing_sensitivity_implementation_protocol.json",
        "dense_no_packing_publication_protocol.json",
        "dense_no_packing_state_operator_factorial_implementation_protocol.json",
    ]
    checked = []
    problems = []

    def visit(value: Any, origin: str) -> None:
        if isinstance(value, list):
            for entry in value:
                visit(entry, origin)
        elif isinstance(value, dict):
            if isinstance(value.get("path"), str) and "sha256" in value:
                source = repository / value["path"]
                if not source.is_file():
                    problems.append(f"{origin}: missing {value['path']}")
                    return
                actual = identity(source, repository)
                valid = actual["sha256"] == value["sha256"] and (
                    "bytes" not in value or actual["bytes"] == value["bytes"]
                )
                checked.append({"protocol": origin, **actual, "matches": valid})
                if not valid:
                    problems.append(f"{origin}: changed {value['path']}")
            else:
                for entry in value.values():
                    visit(entry, origin)

    for name in protocols:
        payload = json.loads((repository / "configs" / name).read_text(encoding="utf-8"))
        # Historical and amendment identities describe past bytes, not the current lock.
        for field in ("source_bindings", "parent_bindings", "configuration_bindings"):
            visit(payload.get(field, {}), name)
    return {"complete": not problems, "checked_bindings": checked, "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output.resolve()
    os.chdir(repository)
    from embed_optim import aggregate, config, corrected_progress

    for module in (aggregate, config, corrected_progress):
        if not Path(module.__file__).resolve().is_relative_to(repository / "src"):
            raise RuntimeError("The audit imported a different checkout; set PYTHONPATH explicitly")
    matrix = repository / "configs/dense_no_packing_retrain.yaml"
    configs = config.load_matrix(matrix)
    if len(configs) != 12 or any(item.model_family != "dense" for item in configs):
        raise ValueError("Expected the declared 12-run primary Dense matrix")
    print("Checking data and exact materialized row identities", flush=True)
    dataset = aggregate.audit_dataset_artifacts(configs)
    row_linkage = audit_materialized_rows(Path(configs[0].dataset_path))
    print("Checking artifact progress and deep completed-run payloads", flush=True)
    progress = corrected_progress.build_progress(matrix)
    completed_ids = {row["run_id"] for row in progress["runs"] if row["state"] == "complete"}
    completed = [item for item in configs if item.run_id in completed_ids]
    training = aggregate.audit_training_artifacts(
        completed, deep=True, expected_dataset_fingerprint=dataset["training_view_fingerprint"]
    )
    bindings = audit_bindings(repository)
    valid = dataset["complete"] and row_linkage["complete"] and training["complete"]
    valid = valid and bindings["complete"]
    summaries = {}
    for label, relative in {
        "outcomes": "reports/dense-no-packing-outcomes/summary_manifest.json",
        "dimension_utilization": "reports/dense-primary-dimension-utilization/summary_manifest.json",
        "factorial": "reports/state-operator-factorial/summary_manifest.json",
        "main_completion": "logs/dense-no-packing-finalization/pipeline-ledger.json",
    }.items():
        path = repository / relative
        summaries[label] = {"present": path.is_file()}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            summaries[label].update(
                {
                    "identity": identity(path, repository),
                    "declared_status": payload.get("status"),
                    "declared_complete": payload.get("complete"),
                    "content_validated_by_this_audit": False,
                }
            )
    code_paths = [
        "train",
        "losses",
        "optimizers",
        "collators",
        "config",
        "data",
        "aggregate",
        "corrected_progress",
        "corrected_beir_evaluation",
        "corrected_outcome_summary",
        "decontamination",
    ]
    payload = {
        "schema_version": 1,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "audit_scope": "data linkage, completed training payloads, and current source bindings",
        "audited_portion_valid": bool(valid),
        "scientific_completion": False,
        "audit_implementation": identity(Path(__file__), repository),
        "matrix": identity(matrix, repository),
        "sources": [
            identity(repository / f"src/embed_optim/{name}.py", repository) for name in code_paths
        ],
        "dataset": dataset,
        "materialized_row_linkage": row_linkage,
        "completed_run_ids": sorted(completed_ids),
        "completed_training_audit": training,
        "protocol_bindings": bindings,
        "progress": progress,
        "downstream_presence_only": summaries,
        "boundary": (
            "This partial audit does not certify benchmark scores, final paper claims, seed "
            "robustness, or a numerical continuation from every saved checkpoint. A completed "
            "run means its saved payloads pass the deep contract; all remaining runs and "
            "scientific publication gates still have to finish."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "audited_portion_valid": bool(valid),
                "completed_runs": len(completed),
                "verified_checkpoints": training["verified_checkpoints"],
                "output": str(output),
                "protocol_problems": bindings["problems"],
            },
            indent=2,
        )
    )
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
