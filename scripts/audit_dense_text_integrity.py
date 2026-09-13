"""Read-only full-data content eligibility and manifest-to-Arrow linkage audit."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset

from embed_optim.primary_contract import digest, file_identity, verify_file
from scripts.audit_dense_natural_data import TEXT_COLUMNS, input_files, write_new


def content_audit(table):
    """Report text defects without cleaning, reordering, or changing any example."""
    empty, cells, masks = {}, {}, {}
    for name in TEXT_COLUMNS:
        values = table[name]
        if not (pa.types.is_string(values.type) or pa.types.is_large_string(values.type)):
            raise ValueError("Training text must have an Arrow string type")
        null = pc.is_null(values)
        blank = pc.fill_null(pc.equal(pc.utf8_trim_whitespace(values), ""), False)
        masks[name] = pc.or_(null, blank)
        cells[name] = {
            "null": int(pc.sum(null).as_py()),
            "blank_including_empty": int(pc.sum(blank).as_py()),
            "empty_string": int(pc.sum(pc.fill_null(pc.equal(values, ""), False)).as_py()),
        }
        empty[name] = masks[name]
    invalid = empty[TEXT_COLUMNS[0]]
    for mask in list(empty.values())[1:]:
        invalid = pc.or_(invalid, mask)
    matches = {}
    for left, right in itertools.combinations(TEXT_COLUMNS[1:], 2):
        equal = pc.fill_null(pc.equal(table[left], table[right]), False)
        valid = pc.invert(pc.or_(empty[left], empty[right]))
        matches[f"{left}=={right}"] = pc.and_(equal, valid)
    conflict = pa.chunked_array([pa.array([False] * len(table))])
    repeated_negative = conflict
    for names, mask in matches.items():
        if names.startswith("positive=="):
            conflict = pc.or_(conflict, mask)
        else:
            repeated_negative = pc.or_(repeated_negative, mask)
    affected = pc.or_(invalid, pc.or_(conflict, repeated_negative))
    indices = pc.indices_nonzero(affected).to_pylist()
    records = []
    for i in indices:
        records.append(
            {
                "index": i,
                "sample_id": table["sample_id"][i].as_py(),
                "source": table["source"][i].as_py(),
                "query_id": table["query_id"][i].as_py(),
                "positive_id": table["positive_id"][i].as_py(),
                "negative_ids": [table[f"negative_{n}_id"][i].as_py() for n in range(7)],
                "invalid_text_columns": [c for c in TEXT_COLUMNS if masks[c][i].as_py()],
                "equal_nonempty_document_pairs": [
                    names for names, mask in matches.items() if mask[i].as_py()
                ],
            }
        )
    per_source = {}
    for source in sorted(set(table["source"].to_pylist())):
        source_mask = pc.equal(table["source"], source)
        per_source[source] = {
            "rows": int(pc.sum(source_mask).as_py()),
            "invalid_text_rows": int(pc.sum(pc.and_(source_mask, invalid)).as_py()),
            "positive_negative_identical_text_rows": int(
                pc.sum(pc.and_(source_mask, conflict)).as_py()
            ),
            "repeated_negative_text_rows": int(
                pc.sum(pc.and_(source_mask, repeated_negative)).as_py()
            ),
        }
    return {
        "rows": len(table),
        "text_cells": len(table) * len(TEXT_COLUMNS),
        "by_column": cells,
        "invalid_text_rows": int(pc.sum(invalid).as_py()),
        "positive_negative_identical_text_rows": int(pc.sum(conflict).as_py()),
        "repeated_negative_text_rows": int(pc.sum(repeated_negative).as_py()),
        "affected_rows_union": len(indices),
        "by_source": per_source,
        "affected_rows": records,
        "comparison_rule": "Empty/Unicode-whitespace/null text eligibility; exact nonempty document-text equality only, no case folding, normalization or near-duplicate inference.",
    }


def manifest_linkage(dataset, path):
    columns = [
        "sample_id",
        "source",
        "query_id",
        "positive_id",
        *(f"negative_{i}_id" for i in range(7)),
    ]
    view = dataset.select_columns(columns)
    counts, errors, seen = Counter(), [], 0
    with path.open() as handle:
        for index, pair in enumerate(itertools.zip_longest(iter(view), handle)):
            actual, line = pair
            seen += 1
            if actual is None or line is None:
                errors.append({"index": index, "reason": "unequal_length"})
                continue
            expected = json.loads(line)
            wanted = {k: expected[k] for k in ("sample_id", "source", "query_id", "positive_id")}
            wanted.update(
                {f"negative_{i}_id": value for i, value in enumerate(expected["negative_ids"])}
            )
            if actual != wanted:
                errors.append({"index": index, "actual": actual, "expected": wanted})
            counts[actual["source"]] += 1
    return {
        "rows_compared": seen,
        "every_materialized_id_matches_manifest": not errors,
        "mismatch_count": len(errors),
        "mismatches": errors,
        "source_counts": dict(counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise ValueError("Require CPU-only audit and a new receipt path")
    inputs, files = input_files(args.repository.resolve())
    dataset = Dataset.load_from_disk(
        str(Path(inputs["data_linkage_audit"]["dataset_path"]) / "dataset")
    )
    content = content_audit(dataset.data.table)
    print(
        json.dumps(
            {
                k: content[k]
                for k in (
                    "rows",
                    "text_cells",
                    "invalid_text_rows",
                    "positive_negative_identical_text_rows",
                    "repeated_negative_text_rows",
                )
            }
        ),
        flush=True,
    )
    linkage = manifest_linkage(dataset, Path(inputs["row_manifest"]["path"]))
    for row in files:
        verify_file(Path(row["path"]), row)
    result = {
        "scope": "engineering_dense_full_text_integrity",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "audit_execution_complete": True,
        "scientific_completion": False,
        "training_executed": False,
        "data_modified": False,
        "content": content,
        "actual_materialized_id_linkage": linkage,
        "authenticated_inputs_unchanged": files,
        "source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
        "source_materializer": {
            "path": str(args.repository / "src/embed_optim/data.py"),
            **file_identity(args.repository / "src/embed_optim/data.py"),
        },
        "all_nine_texts_nonempty": content["invalid_text_rows"] == 0,
        "boundary": "Exhaustive nine-text eligibility, exact within-group document-text equality and actual Arrow IDs compared with every original manifest row. No row is replaced, dropped, imputed or re-ranked. A repeated negative is reported separately from a positive-negative identity collision; neither is silently redefined as seven distinct negative texts. Exact text equality is not a semantic duplicate detector. These are data checks, not optimizer/retrieval results.",
    }
    write_new(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                **file_identity(args.output),
                "data_unchanged": True,
                "content_sha256": digest(content),
                "all_nine_texts_nonempty": result["all_nine_texts_nonempty"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
