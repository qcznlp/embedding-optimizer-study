from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embed_optim.confirmatory_data import (
    _receipt_path,
    _scan_negative_pools,
    _selected_pool_indices,
    audit_confirmatory_data,
    load_confirmatory_protocol,
    main,
)


def test_receipt_paths_are_relative_inside_the_repository(tmp_path: Path):
    repository = tmp_path / "checkout"
    artifact = repository / "data" / "view"
    external = tmp_path / "external" / "view"

    assert _receipt_path(artifact, repository) == "data/view"
    assert Path(_receipt_path(external, repository)).is_absolute()


def _score_fixture(tmp_path: Path, *, duplicate_pool_id: bool = False):
    score_root = tmp_path / "scores"
    score_root.mkdir()
    pool_ids = list(range(200, 210))
    if duplicate_pool_id:
        pool_ids[-1] = pool_ids[-2]
    documents = [100, 999, *pool_ids]
    scores = [1.0, 0.99, *[0.94 - index * 0.01 for index in range(10)]]
    table = pa.table(
        {
            "query_id": pa.array([7], type=pa.int64()),
            "document_ids": pa.array([documents], type=pa.list_(pa.int64())),
            "scores": pa.array([scores], type=pa.list_(pa.float64())),
        }
    )
    pq.write_table(table, score_root / "fever-00000-of-00001.parquet")
    indices = _selected_pool_indices(42, "fever", 7)
    base = {
        "sample_id": 0,
        "source": "fever",
        "query_id": 7,
        "positive_id": 100,
        "negative_pool_indices": indices,
        "negative_ids": [pool_ids[index] for index in indices],
    }
    return base, list(range(200, 210))


def _append_alternate_positive(tmp_path: Path):
    path = tmp_path / "scores" / "fever-00000-of-00001.parquet"
    original = pq.read_table(path)
    alternate = pa.table(
        {
            "query_id": pa.array([7], type=pa.int64()),
            "document_ids": pa.array([[101, *range(300, 310)]], type=pa.list_(pa.int64())),
            "scores": pa.array(
                [[1.0, *[0.9 - index * 0.01 for index in range(10)]]],
                type=pa.list_(pa.float64()),
            ),
        }
    )
    pq.write_table(pa.concat_tables([original, alternate]), path)


def test_frozen_confirmatory_protocol_has_three_new_seeds():
    path, protocol = load_confirmatory_protocol("configs/confirmatory_protocol.json")

    assert path.name == "confirmatory_protocol.json"
    assert protocol["confirmatory_data"]["seeds"] == [314159, 271828, 161803]
    assert protocol["source"]["exploratory_seed"] not in protocol["confirmatory_data"]["seeds"]
    assert protocol["training"]["expected_runs"] == 18
    assert protocol["training"]["expected_beir_units"] == 252
    assert protocol["freeze_context"]["strict_beir_valid_units"] == 150
    assert protocol["freeze_context"]["query_disjoint_validation_outputs_visible"] is False
    assert protocol["freeze_context"]["confirmatory_model_outputs_exist"] is False


def test_receipt_binds_the_audited_confirmatory_views():
    root = Path(__file__).parents[1]
    receipt = json.loads((root / "reports/confirmatory-data/receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert len(receipt["views"]) == 3
    for view in receipt["views"]:
        assert view["rows"] == 500_000
        assert view["query_positive_identity_sha256"] == receipt["query_positive_identity_sha256"]
        assert view["dataset_fingerprint"]

    expected_pairs = {
        "314159_vs_271828": "0.991580",
        "314159_vs_161803": "0.991684",
        "271828_vs_161803": "0.991772",
    }
    for pair, rendered in expected_pairs.items():
        assert receipt["changed_negative_group_fractions"][pair] == pytest.approx(float(rendered))


def test_audit_receipt_promotes_identical_pairwise_fractions(monkeypatch):
    fractions = {"314159_vs_271828": 0.99158}
    views = {
        seed: {
            "seed": seed,
            "query_positive_identity_sha256": "q" * 64,
            "changed_negative_group_fractions": fractions.copy(),
        }
        for seed in (314159, 271828, 161803)
    }
    monkeypatch.setattr(
        "embed_optim.confirmatory_data.audit_negative_pool",
        lambda *_args, **_kwargs: {"source_rescanned": False},
    )
    monkeypatch.setattr(
        "embed_optim.confirmatory_data.audit_confirmatory_view",
        lambda _protocol, seed: views[seed].copy(),
    )

    receipt = audit_confirmatory_data("configs/confirmatory_protocol.json")

    assert receipt["changed_negative_group_fractions"] == fractions
    assert all("changed_negative_group_fractions" not in view for view in receipt["views"])


def test_audit_rejects_pairwise_fraction_disagreement(monkeypatch):
    def view(_protocol, seed):
        return {
            "seed": seed,
            "query_positive_identity_sha256": "q" * 64,
            "changed_negative_group_fractions": {"pair": seed / 1_000_000},
        }

    monkeypatch.setattr(
        "embed_optim.confirmatory_data.audit_negative_pool",
        lambda *_args, **_kwargs: {"source_rescanned": False},
    )
    monkeypatch.setattr("embed_optim.confirmatory_data.audit_confirmatory_view", view)

    with pytest.raises(ValueError, match="disagree on pairwise changed-negative fractions"):
        audit_confirmatory_data("configs/confirmatory_protocol.json")


def test_protocol_rejects_reusing_exploratory_seed(tmp_path: Path):
    _, protocol = load_confirmatory_protocol("configs/confirmatory_protocol.json")
    protocol["confirmatory_data"]["seeds"] = [42, 271828, 161803]
    protocol["confirmatory_data"]["outputs"]["42"] = protocol["confirmatory_data"]["outputs"].pop(
        "314159"
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="exploratory seed"):
        load_confirmatory_protocol(path)


def test_scan_reconstructs_the_exact_seed42_pool(tmp_path: Path):
    base, expected_pool = _score_fixture(tmp_path)

    observed = _scan_negative_pools(
        tmp_path,
        "fever",
        [base],
        exploratory_seed=42,
        threshold=0.95,
        pool_size=10,
        sampled_negatives=7,
        batch_size=8,
    )

    assert observed == [
        {
            "sample_id": 0,
            "source": "fever",
            "query_id": 7,
            "positive_id": 100,
            "negative_pool_ids": expected_pool,
        }
    ]


def test_scan_matches_original_first_eligible_positive_rule(tmp_path: Path):
    base, expected_pool = _score_fixture(tmp_path)
    _append_alternate_positive(tmp_path)

    observed = _scan_negative_pools(
        tmp_path,
        "fever",
        [base],
        exploratory_seed=42,
        threshold=0.95,
        pool_size=10,
        sampled_negatives=7,
        batch_size=8,
    )

    assert observed[0]["positive_id"] == 100
    assert observed[0]["negative_pool_ids"] == expected_pool


def test_scan_rejects_duplicate_pool_documents(tmp_path: Path):
    base, _ = _score_fixture(tmp_path, duplicate_pool_id=True)

    with pytest.raises(ValueError, match="duplicate/positive pool"):
        _scan_negative_pools(
            tmp_path,
            "fever",
            [base],
            exploratory_seed=42,
            threshold=0.95,
            pool_size=10,
            sampled_negatives=7,
            batch_size=8,
        )


def test_confirmatory_selection_is_deterministic_distinct_and_seeded():
    first = _selected_pool_indices(314159, "nq", 123)
    repeated = _selected_pool_indices(314159, "nq", 123)
    alternate = _selected_pool_indices(271828, "nq", 123)

    assert first == repeated
    assert first == sorted(set(first))
    assert len(first) == 7
    assert all(0 <= index < 10 for index in first)
    assert first != alternate


def test_audit_only_does_not_rewrite_the_materialization_receipt(monkeypatch, tmp_path: Path):
    writes = []
    audits = []

    def audit(*_args, **kwargs):
        audits.append(kwargs)
        return {"schema_version": 1, "status": "complete"}

    monkeypatch.setattr(
        "embed_optim.confirmatory_data.audit_confirmatory_data",
        audit,
    )
    monkeypatch.setattr(
        "embed_optim.confirmatory_data._atomic_json",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    main(["--audit-only", "--receipt", str(tmp_path / "receipt.json")])

    assert writes == []
    assert audits == [{"verify_source": False}]


def test_materialization_rescans_the_pinned_score_source(monkeypatch, tmp_path: Path):
    writes = []
    audits = []
    monkeypatch.setattr(
        "embed_optim.confirmatory_data.prepare_negative_pool", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "embed_optim.confirmatory_data.prepare_confirmatory_views", lambda *_a, **_k: None
    )

    def audit(*_args, **kwargs):
        audits.append(kwargs)
        return {"schema_version": 1, "status": "complete"}

    monkeypatch.setattr("embed_optim.confirmatory_data.audit_confirmatory_data", audit)
    monkeypatch.setattr(
        "embed_optim.confirmatory_data._atomic_json",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    main(["--receipt", str(tmp_path / "receipt.json")])

    assert audits == [{"verify_source": True}]
    assert len(writes) == 1


def test_audit_only_rescans_source_only_when_explicitly_requested(monkeypatch, tmp_path: Path):
    audits = []

    def audit(*_args, **kwargs):
        audits.append(kwargs)
        return {"schema_version": 1, "status": "complete"}

    monkeypatch.setattr("embed_optim.confirmatory_data.audit_confirmatory_data", audit)
    monkeypatch.setattr(
        "embed_optim.confirmatory_data._atomic_json",
        lambda *_args, **_kwargs: pytest.fail("audit-only must not write"),
    )

    main(
        [
            "--audit-only",
            "--verify-source",
            "--receipt",
            str(tmp_path / "receipt.json"),
        ]
    )

    assert audits == [{"verify_source": True}]
