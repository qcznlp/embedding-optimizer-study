import json

import pyarrow as pa
import pytest
from datasets import Dataset

from scripts.audit_dense_natural_data import TEXT_COLUMNS
from scripts.audit_dense_text_integrity import content_audit, manifest_linkage


def example():
    return {
        "sample_id": 0,
        "source": "fixture",
        "query_id": 10,
        "positive_id": 20,
        **{f"negative_{i}_id": i + 30 for i in range(7)},
        **{c: f"unique {c}" for c in TEXT_COLUMNS},
    }


def test_distinct_nonempty_fixture():
    value = content_audit(pa.Table.from_pylist([example()]))
    assert value["text_cells"] == 9
    assert value["affected_rows_union"] == 0


@pytest.mark.parametrize("invalid", ["", "  ", "\u2003"])
def test_empty_and_unicode_whitespace_are_invalid(invalid):
    row = example()
    row["positive"] = invalid
    value = content_audit(pa.Table.from_pylist([row]))
    assert value["invalid_text_rows"] == 1
    assert value["affected_rows"][0]["invalid_text_columns"] == ["positive"]


def test_null_text_and_blank_pairs_are_not_nonempty_collisions():
    original = example()
    changed = {**original, "sample_id": 1, "positive": None, "negative_0": "", "negative_1": ""}
    value = content_audit(pa.Table.from_pylist([original, changed]))
    assert value["invalid_text_rows"] == 1
    assert value["repeated_negative_text_rows"] == 0
    assert value["by_column"]["positive"]["null"] == 1


def test_label_collision_and_duplicate_negatives_are_separate():
    row = example()
    row["negative_0"] = row["positive"]
    row["negative_2"] = row["negative_1"]
    value = content_audit(pa.Table.from_pylist([row]))
    assert value["invalid_text_rows"] == 0
    assert value["positive_negative_identical_text_rows"] == 1
    assert value["repeated_negative_text_rows"] == 1
    assert value["affected_rows_union"] == 1


def test_exact_equality_does_not_infer_semantic_collision():
    row = example()
    row["positive"] = "Text"
    row["negative_0"] = "text"
    row["negative_1"] = "Text "
    assert content_audit(pa.Table.from_pylist([row]))["affected_rows_union"] == 0


@pytest.mark.parametrize(
    "change", [None, "positive_id", "negative_6_id", "query_id", "source", "short_manifest"]
)
def test_manifest_compares_actual_materialized_ids(tmp_path, change):
    row = example()
    original = {k: row[k] for k in ("sample_id", "source", "query_id", "positive_id")}
    original["negative_ids"] = [row[f"negative_{i}_id"] for i in range(7)]
    path = tmp_path / "rows.jsonl"
    path.write_text("" if change == "short_manifest" else json.dumps(original) + "\n")
    if change and change != "short_manifest":
        row[change] = "changed" if change == "source" else 999
    result = manifest_linkage(Dataset.from_list([row]), path)
    assert result["every_materialized_id_matches_manifest"] is (change is None)
