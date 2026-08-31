from __future__ import annotations

import copy

import pytest

from embed_optim import candidate_breadth_data
from embed_optim.candidate_breadth_data import (
    _validate_candidate_records,
    load_candidate_breadth_protocol,
    nested_candidate_ids,
    parse_args,
    select_validation_rows,
)
from embed_optim.data import SPLITS


def _validation_rows(per_source: int = 40) -> list[dict]:
    rows = []
    sample_id = 0
    for source in SPLITS:
        for query_id in range(per_source):
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "query_id": query_id,
                    "positive_id": 100_000 + sample_id,
                    "negative_ids": list(range(7)),
                    "negative_pool_indices": list(range(7)),
                }
            )
            sample_id += 1
    return rows


def test_protocol_is_post_hoc_and_fixes_nested_widths() -> None:
    _, protocol = load_candidate_breadth_protocol("configs/candidate_breadth_probe.json")
    assert protocol["status"] == "post_hoc_shortlist_corpus_mechanism_probe"
    assert protocol["candidate_construction"]["negative_widths"] == [
        7,
        10,
        32,
        128,
        512,
        2048,
    ]
    assert protocol["timing"]["candidate_breadth_data_or_scores_visible"] is False


def test_candidate_breadth_prepare_modes_are_mutually_exclusive() -> None:
    assert parse_args(["--resume"]).resume is True
    assert parse_args(["--audit-only"]).audit_only is True
    with pytest.raises(SystemExit):
        parse_args(["--resume", "--overwrite"])


def test_only_explicit_audit_mode_reconstructs_the_pinned_source(tmp_path, monkeypatch) -> None:
    output = tmp_path / "candidate-data"
    output.mkdir()
    calls = []

    def fake_audit(protocol, observed_output, *, verify_source=True):
        calls.append((protocol, observed_output, verify_source))
        return {"status": "complete"}

    monkeypatch.setattr(candidate_breadth_data, "audit_candidate_breadth_data", fake_audit)
    candidate_breadth_data.main(
        ["--protocol", "protocol.json", "--output", str(output), "--resume"]
    )
    candidate_breadth_data.main(
        ["--protocol", "protocol.json", "--output", str(output), "--audit-only"]
    )

    assert [call[2] for call in calls] == [False, True]


def test_balanced_selection_is_deterministic_and_source_complete() -> None:
    rows = _validation_rows()
    selected = select_validation_rows(rows, count=224, seed=20260901)
    repeated = select_validation_rows(reversed(rows), count=224, seed=20260901)
    assert selected == repeated
    assert len(selected) == 224
    assert {source: sum(row["source"] == source for row in selected) for source in SPLITS} == {
        source: 32 for source in SPLITS
    }
    assert [row["sample_id"] for row in selected] == sorted(row["sample_id"] for row in selected)


def test_nested_candidates_preserve_canonical_seven_and_extend_in_mined_order() -> None:
    raw_ids = [90_000, *range(1, 2049)]
    raw_scores = [1.0, *([0.9] * 2048)]
    pool_indices = [1, 3, 4, 5, 6, 7, 8]
    canonical = [2, 4, 5, 6, 7, 8, 9]
    row = {
        "positive_id": 90_000,
        "negative_ids": canonical,
        "negative_pool_indices": pool_indices,
    }
    candidates = nested_candidate_ids(
        raw_ids,
        raw_scores,
        row,
        threshold=0.95,
        maximum_width=2048,
    )
    assert candidates[:7] == canonical
    assert candidates[:10] == [*canonical, 1, 3, 10]
    assert len(candidates) == len(set(candidates)) == 2048
    assert 90_000 not in candidates
    for smaller, larger in zip((7, 10, 32, 128, 512), (10, 32, 128, 512, 2048)):
        assert candidates[:smaller] == candidates[:larger][:smaller]


def test_nested_candidates_reject_a_nonreproducible_canonical_seven() -> None:
    raw_ids = [90_000, *range(1, 2049)]
    raw_scores = [1.0, *([0.9] * 2048)]
    row = {
        "positive_id": 90_000,
        "negative_ids": [2, 4, 5, 6, 7, 8, 11],
        "negative_pool_indices": [1, 3, 4, 5, 6, 7, 8],
    }
    with pytest.raises(ValueError, match="canonical seven"):
        nested_candidate_ids(
            raw_ids,
            raw_scores,
            copy.deepcopy(row),
            threshold=0.95,
            maximum_width=2048,
        )


def _candidate_records() -> tuple[list[dict], list[dict]]:
    selected = _validation_rows(per_source=1)
    records = []
    for index, row in enumerate(selected):
        records.append(
            {
                "query_index": index,
                "sample_id": row["sample_id"],
                "source": row["source"],
                "query_id": row["query_id"],
                "positive_id": row["positive_id"],
                "negative_ids": [*row["negative_ids"], 7, 8, 9],
                "source_score_file": f"{row['source']}.parquet",
                "source_score_row_group": 0,
                "source_score_row_offset": index,
            }
        )
    return selected, records


def test_candidate_ledger_is_bound_to_frozen_selection_and_canonical_seven() -> None:
    selected, records = _candidate_records()
    _validate_candidate_records(records, selected, maximum_width=10)

    tampered = copy.deepcopy(records)
    tampered[0]["query_id"] += 1
    with pytest.raises(ValueError, match="ledger row 1"):
        _validate_candidate_records(tampered, selected, maximum_width=10)

    tampered = copy.deepcopy(records)
    tampered[0]["negative_ids"][0] = 99
    with pytest.raises(ValueError, match="ledger row 1"):
        _validate_candidate_records(tampered, selected, maximum_width=10)


def test_full_source_reconstruction_detects_resigned_extended_candidates() -> None:
    selected, records = _candidate_records()
    source_records = [
        {key: value for key, value in record.items() if key != "query_index"} for record in records
    ]
    tampered = copy.deepcopy(records)
    tampered[0]["negative_ids"][-1] = 99

    # The local semantic layer permits a different unique extension because it has
    # deliberately avoided rescanning upstream parquet.  The mandatory release
    # layer then compares all 2,048 positions with the pinned source reconstruction.
    _validate_candidate_records(tampered, selected, maximum_width=10)
    with pytest.raises(ValueError, match="pinned mined-score reconstruction"):
        _validate_candidate_records(
            tampered,
            selected,
            maximum_width=10,
            source_records=source_records,
        )
