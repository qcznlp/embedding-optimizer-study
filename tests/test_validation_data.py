from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from embed_optim.data import SOURCE_REPO, SOURCE_REVISION, SPLITS
from embed_optim.geometry import _sha256
from embed_optim.validation_data import (
    _canonical_row,
    _validate_expected,
    ensure_selection_ledger,
    load_validation_spec,
    training_query_ids,
)


def _training_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "training"
    root.mkdir()
    rows = [
        {"sample_id": 0, "source": "fiqa", "query_id": 10},
        {"sample_id": 1, "source": "hotpotqa", "query_id": 20},
    ]
    digest = hashlib.sha256()
    with (root / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            digest.update(_canonical_row(row))
    quotas = {split: 0 for split in SPLITS}
    quotas.update({"fiqa": 1, "hotpotqa": 1})
    manifest = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "total_queries": 2,
        "sampled_negatives": 7,
        "row_manifest_sha256": digest.hexdigest(),
        "quotas": quotas,
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return root


def test_training_query_ledger_is_recomputed_and_partitioned(tmp_path: Path):
    root = _training_fixture(tmp_path)
    query_ids, manifest = training_query_ids(
        root,
        expected_manifest_sha256=_sha256(root / "manifest.json"),
        expected_ledger_sha256=_sha256(root / "rows.jsonl"),
        expected_total=2,
    )

    assert query_ids["fiqa"] == {10}
    assert query_ids["hotpotqa"] == {20}
    assert all(not query_ids[split] for split in set(SPLITS) - {"fiqa", "hotpotqa"})
    assert (
        manifest["row_manifest_sha256"]
        == hashlib.sha256(
            _canonical_row({"sample_id": 0, "source": "fiqa", "query_id": 10})
            + _canonical_row({"sample_id": 1, "source": "hotpotqa", "query_id": 20})
        ).hexdigest()
    )


def test_training_query_ledger_rejects_duplicate_query(tmp_path: Path):
    root = _training_fixture(tmp_path)
    duplicate = {"sample_id": 1, "source": "fiqa", "query_id": 10}
    with (root / "rows.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate) + "\n")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["total_queries"] = 3
    manifest["quotas"]["fiqa"] = 2
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate or invalid"):
        training_query_ids(root, expected_total=3)


def test_frozen_validation_protocol_is_disjoint_and_beir_independent():
    path, spec = load_validation_spec("configs/validation_probe.json")

    assert path.name == "validation_probe.json"
    assert spec["selection"]["count"] == 4096
    assert spec["selection"]["allocation"] == "balanced"
    assert spec["selection"]["sampled_negatives"] == 7
    assert spec["freeze_context"]["strict_beir_valid_units"] == 144
    assert spec["freeze_context"]["validation_examples_or_model_outputs_visible"] is False
    assert spec["recipe_selection"]["checkpoint"] == "final checkpoint only"
    assert len(spec["source"]["manifest_sha256"]) == 64
    assert len(spec["source"]["row_ledger_sha256"]) == 64

    # The 500K materialization is intentionally gitignored.  Keep this unit test
    # hermetic on clean clones while still checking the pinned artifact whenever
    # it is present; the validation CLI performs the mandatory full artifact
    # audit before preparation or evaluation.
    local_manifest = Path(spec["source"]["training_data"]) / "manifest.json"
    if local_manifest.is_file():
        assert _sha256(local_manifest) == spec["source"]["manifest_sha256"]


def test_validation_expectations_are_exact():
    observed = {"rows": 4096, "overlap": 0}
    _validate_expected(observed, dict(observed))
    with pytest.raises(ValueError, match="expectation differs"):
        _validate_expected(observed, {"rows": 4095})


def test_selection_ledger_alias_is_exact_and_idempotent(tmp_path: Path):
    output = tmp_path / "validation"
    output.mkdir()
    row_path = output / "rows.jsonl"
    row_path.write_text('{"sample_id":0}\n', encoding="utf-8")

    selection = ensure_selection_ledger(output)
    before = selection.stat().st_mtime_ns
    resumed = ensure_selection_ledger(output)

    assert selection.read_bytes() == row_path.read_bytes()
    assert resumed == selection
    assert selection.stat().st_mtime_ns == before
