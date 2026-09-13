"""Synthetic unit controls for the real-data diagnostic; not primary evidence."""

import json

import pytest
from datasets import Dataset

from scripts.audit_dense_natural_data import (
    TEXT_COLUMNS,
    logged_observations,
    select_prefix,
    write_new,
)


def rows():
    return [{**{name: f"{name} {i}" for name in TEXT_COLUMNS}, "length": i + 1} for i in range(300)]


def test_fixed_prefix_keeps_every_original_text_and_order():
    original = rows()
    selected, receipt = select_prefix(Dataset.from_list(original))
    assert list(selected) == original[:288]
    assert receipt["selected_row_count"] == 288
    assert receipt["expected_global_query_groups"] == [128, 128, 32]
    assert len(receipt["row_sha256"]) == 288
    changed = rows()
    changed[2]["negative_6"] = "A changed original negative"
    _, altered = select_prefix(Dataset.from_list(changed))
    assert altered["selected_rows_sha256"] != receipt["selected_rows_sha256"]


@pytest.mark.parametrize("count", [1, 287, 289, 300])
def test_rejects_unplanned_prefix_size(count):
    with pytest.raises(ValueError, match="predeclared"):
        select_prefix(Dataset.from_list(rows()), count=count)


def test_rejects_short_original():
    with pytest.raises(ValueError, match="predeclared"):
        select_prefix(Dataset.from_list(rows()[:287]))


@pytest.mark.parametrize("column", TEXT_COLUMNS)
@pytest.mark.parametrize("invalid", [None, "", " "])
def test_rejects_incomplete_original_text_group(column, invalid):
    original = rows()
    original[0][column] = invalid
    with pytest.raises(ValueError, match="nine original texts"):
        select_prefix(Dataset.from_list(original))


@pytest.mark.parametrize("length", [0, -1, 1.5, None])
def test_rejects_bad_original_length(length):
    original = rows()
    original[0]["length"] = length
    with pytest.raises(ValueError, match="length feature"):
        select_prefix(Dataset.from_list(original))


def history(tmp_path, records):
    (tmp_path / "trainer_state_final.json").write_text(json.dumps({"log_history": records}))


def test_checks_actual_finite_logged_observation(tmp_path):
    row = {"step": 1, "loss": 0.5, "grad_norm": 25.0, "learning_rate": 0.0}
    history(tmp_path, [row, {"step": 3, "train_loss": 0.4}])
    assert logged_observations(tmp_path) == row


@pytest.mark.parametrize(
    "field,value",
    [
        ("loss", float("nan")),
        ("loss", -1),
        ("grad_norm", float("inf")),
        ("grad_norm", 0),
        ("learning_rate", 1e-4),
    ],
)
def test_rejects_invalid_training_observation(tmp_path, field, value):
    row = {"step": 1, "loss": 0.5, "grad_norm": 25.0, "learning_rate": 0.0, field: value}
    history(tmp_path, [row])
    with pytest.raises(ValueError):
        logged_observations(tmp_path)


@pytest.mark.parametrize("count", [0, 2])
def test_rejects_absent_or_ambiguous_first_step(tmp_path, count):
    history(tmp_path, [{"step": 1, "loss": 0.5, "grad_norm": 25.0, "learning_rate": 0.0}] * count)
    with pytest.raises(ValueError, match="missing or duplicated"):
        logged_observations(tmp_path)


def test_receipt_never_overwrites_existing_evidence(tmp_path):
    path = tmp_path / "receipt.json"
    write_new(path, {"first": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, {"first": False})
    assert path.read_bytes() == original
