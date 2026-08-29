from __future__ import annotations

import json
from types import SimpleNamespace

from embed_optim.geometry import _sha256
from embed_optim.supplemental_training_audit import audit_derived_training_artifacts


def _fixture(tmp_path, monkeypatch):
    dataset = tmp_path / "derived"
    output = tmp_path / "outputs" / "run"
    dataset.mkdir()
    output.mkdir(parents=True)
    manifest = {"rows": 10, "dataset_fingerprint": "serialized"}
    manifest_path = dataset / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "completed.json").write_text(
        json.dumps({"dataset_rows": 10, "dataset_fingerprint": "training-view"}),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        model_family="dense",
        run_id="muon-derived",
        dataset_path=str(dataset),
        output_dir=output,
    )
    generic = {
        "complete": False,
        "verified_runs": 0,
        "expected_runs": 1,
        "verified_checkpoints": 5,
        "expected_checkpoints": 5,
        "deep_validation": True,
        "errors": ["dense/muon-derived: completion dataset row count does not match manifest"],
    }
    monkeypatch.setattr(
        "embed_optim.supplemental_training_audit.audit_training_artifacts",
        lambda *args, **kwargs: generic,
    )
    receipt = {
        "rows": 10,
        "training_view_fingerprint": "training-view",
        "manifest_sha256": _sha256(manifest_path),
    }
    return config, output, receipt


def test_derived_audit_reconciles_only_the_proven_schema_difference(tmp_path, monkeypatch):
    config, _, receipt = _fixture(tmp_path, monkeypatch)

    result = audit_derived_training_artifacts([config], receipt)

    assert result["complete"] is True
    assert result["verified_runs"] == 1
    assert result["verified_checkpoints"] == 5
    assert result["errors"] == []


def test_derived_audit_rejects_completion_row_drift(tmp_path, monkeypatch):
    config, output, receipt = _fixture(tmp_path, monkeypatch)
    (output / "completed.json").write_text(
        json.dumps({"dataset_rows": 9, "dataset_fingerprint": "training-view"}),
        encoding="utf-8",
    )

    result = audit_derived_training_artifacts([config], receipt)

    assert result["complete"] is False
    assert any("completion row count differs" in error for error in result["errors"])
    assert any("does not match manifest" in error for error in result["errors"])


def test_derived_audit_preserves_unrelated_deep_errors(tmp_path, monkeypatch):
    config, _, receipt = _fixture(tmp_path, monkeypatch)
    schema_error = "dense/muon-derived: completion dataset row count does not match manifest"
    checkpoint_error = "dense/muon-derived/checkpoint-2: invalid optimizer state"

    def generic(*args, **kwargs):
        return {
            "complete": False,
            "verified_runs": 0,
            "expected_runs": 1,
            "verified_checkpoints": 4,
            "expected_checkpoints": 5,
            "deep_validation": True,
            "errors": [schema_error, checkpoint_error],
        }

    monkeypatch.setattr(
        "embed_optim.supplemental_training_audit.audit_training_artifacts",
        generic,
    )

    result = audit_derived_training_artifacts([config], receipt)

    assert result["complete"] is False
    assert result["errors"] == [checkpoint_error]
    assert result["verified_checkpoints"] == 4
