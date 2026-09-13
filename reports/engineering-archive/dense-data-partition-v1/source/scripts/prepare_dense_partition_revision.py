"""Derive a prospective replacement set from authenticated content/partition audits."""

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset

from embed_optim.primary_contract import digest, file_identity, read_json, verify_file
from scripts.audit_dense_natural_data import write_new

AUDITS = {
    "training_content": (
        "reports/engineering-archive/dense-natural-readiness-v1/full-data-content.json",
        "fbfc06240961e943bfa6869cec9a70b40c7ea98415fb8963e495915faa1ac66d",
    ),
    "validation_content": (
        "reports/engineering-archive/dense-natural-readiness-v1/validation-content.json",
        "a26438d638b0320f1303454e324697166c129aa512e38b632cb5f363320de5b2",
    ),
    "protection": (
        "/tmp/dense-data-revision.qaPjm8/protected-queries.json",
        "f20353ce500807a7bfe4ddf30169560bfe6ef415b082d2291060102946e37f66",
    ),
    "string_replay": (
        "/tmp/dense-data-revision.qaPjm8/string-replay.json",
        "4afa162215c5f16138d71d96d6e1444e9b25221032b295e3ff0c37dd53deec3f",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only plan derivation and a new receipt")
    audits, bindings = {}, []
    for name, (relative, sha) in AUDITS.items():
        path = args.repository / relative
        if file_identity(path)["sha256"] != sha:
            raise ValueError("The deciding audit is not the declared immutable receipt")
        audits[name] = read_json(path)
        bindings.append({"path": str(path), **file_identity(path)})
    for row in (
        audits["training_content"]["authenticated_inputs_unchanged"]
        + audits["validation_content"]["authenticated_files_unchanged"]
    ):
        verify_file(Path(row["path"]), row)
    training = Dataset.load_from_disk(
        str(args.experiment_root / "data/denseon-sft-500k-seed42/dataset")
    )
    validation = Dataset.load_from_disk(
        str(args.experiment_root / "data/validation-4096-seed20260826/dataset")
    )
    train_bad = {row["index"] for row in audits["training_content"]["content"]["affected_rows"]}
    train_overlap = {
        row["training"]["sample_id"] for row in audits["string_replay"]["training_beir_pairs"]
    }
    val_bad = {row["index"] for row in audits["validation_content"]["content"]["affected_rows"]}
    val_overlap = set(audits["string_replay"]["affected_validation_rows"])
    selected = {}
    for name, dataset, bad, overlap in (
        ("training", training, train_bad, train_overlap),
        ("validation", validation, val_bad, val_overlap),
    ):
        rows = []
        for index in sorted(bad | overlap):
            row = dataset[index]
            rows.append(
                {
                    "sample_id": index,
                    "source": row["source"],
                    "query_id": row["query_id"],
                    "original_row_sha256": digest(row),
                    "reasons": (["empty_document_text"] if index in bad else [])
                    + (
                        ["query_text_already_in_beir_eval"]
                        if name == "training" and index in overlap
                        else []
                    )
                    + (
                        ["query_text_already_in_training"]
                        if name == "validation" and index in overlap
                        else []
                    ),
                }
            )
        selected[name] = {
            "total_rows": len(dataset),
            "replace_rows": len(rows),
            "retain_rows_unchanged": len(dataset) - len(rows),
            "replacement_source_counts": dict(Counter(row["source"] for row in rows)),
            "rows": rows,
        }
    if selected["training"]["replace_rows"] != 6 or selected["validation"]["replace_rows"] != 52:
        raise ValueError("The proposed scope no longer agrees with the actual independent audits")
    impacts = []
    for name, path, parent, replacements in (
        (
            "training_representation_probe",
            "data/probes/training-1024-seed1729/dataset",
            training,
            train_bad | train_overlap,
        ),
        (
            "factorial_branch_subset",
            "data/short-branch-50k-seed20260826/dataset",
            training,
            train_bad | train_overlap,
        ),
        (
            "historical_candidate_breadth_queries",
            "data/candidate-breadth-224-seed20260901/queries",
            validation,
            val_bad | val_overlap,
        ),
    ):
        subset = Dataset.load_from_disk(str(args.experiment_root / path))
        columns = [c for c in subset.column_names if c in parent.column_names]
        ids = list(subset["sample_id"])
        wanted = parent.select(ids).select_columns(columns)
        observed = subset.select_columns(columns)
        if wanted.to_dict() != observed.to_dict():
            raise ValueError("A declared derived subset differs from its original parent")
        files = [
            {"path": str(p), **file_identity(p)}
            for p in sorted((args.experiment_root / path).rglob("*"))
            if p.is_file()
        ]
        impacts.append(
            {
                "name": name,
                "rows": len(subset),
                "parent_row_values_exact": True,
                "affected_parent_sample_ids": sorted(set(ids) & replacements),
                "files": files,
            }
        )
    proposal = {
        "scope": "prepared_dense_partition_revision_v2",
        "status": "proposal_not_materialized_not_execution_authorized",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scientific_completion": False,
        "replacement_queries_selected": 0,
        "data_modified": False,
        "training_executed": False,
        "deciding_audits": bindings,
        "partitions": selected,
        "derived_subset_impact": impacts,
        "rules": {
            "source": "lightonai/embeddings-fine-tuning@1ca463331ed637d25c1058567e932e0d3bad2983",
            "seed_by_partition": {"training": 42, "validation": 20260826},
            "priority": "Reconstruct each original population and original seeded priority order before any new exclusions. Validation's original population excludes all original training IDs. Filter that fixed order; never re-permute a newly shortened population.",
            "unit": "Replace a whole query/positive/seven-negative group at its original source-qualified sample position, retaining source quotas and 500000/4096 totals. No alternative positive or text imputation.",
            "original_mining": "First original eligible score record; threshold 0.95, pool ten, original per-query seed and sorted seven-of-ten draw. Reject the whole candidate query if the resulting group is ineligible; do not search another positive for it.",
            "preserve": "Keep every non-target original row and all its fields/IDs unchanged; verify per-row digests and exact whole-dataset difference sets.",
            "protected_ids": "Reserve all old training and validation source-qualified IDs, including removed ones. Reserve each new training ID before validation replacement, and each accepted replacement before later candidates.",
            "protected_texts": "Use the same lower/NFKD/whitespace normalized query strings for exclusions across namespaces: reserve all original train/validation query texts, all fourteen frozen BEIR evaluation query texts, and earlier accepted replacements. Recheck zero train-validation and train-BEIR overlap on full revised sets; keep validation internally unique and disjoint from BEIR.",
            "ordering": "Training replacements first, then validation replacements; within each source assign eligible unused candidates by original fixed priority to ascending affected sample positions.",
            "no_outcome_selection": True,
            "release_boundary": "Prepare separate parent-bound datasets and an explicit protocol amendment. Old source locks, identities, data and failed receipts remain immutable. No formal execution or mere hash refresh is authorized by this plan.",
        },
        "source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
        "boundary": "The earlier six-group empty-text plan is insufficient: actual raw-equal train/validation and train/BEIR overlaps require a 6-training/52-validation proposal. No replacement candidate is chosen here. Zero affected derived-subset rows establishes content preservation potential, not that old source-bound consumers already accept a new parent dataset.",
    }
    write_new(args.output, proposal)
    print(
        json.dumps(
            {
                "training_replacements": 6,
                "validation_replacements": 52,
                "derived_subset_affected_rows": {
                    r["name"]: len(r["affected_parent_sample_ids"]) for r in impacts
                },
                "replacement_queries_selected": 0,
                "output": str(args.output),
                **file_identity(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
