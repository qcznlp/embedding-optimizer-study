import json

import pytest

from scripts.audit_dense_gpu_fresh_replay import (
    compare_fingerprints,
    verify_baseline,
    verify_identity,
)
from scripts.audit_dense_gpu_replay_update import identity


def test_source_file_change_is_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{}")
    sealed = identity(path)
    verify_identity(sealed)
    path.write_text("[]")
    with pytest.raises(ValueError):
        verify_identity(sealed)


def test_symlinked_source_is_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("{}")
    sealed = identity(path)
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        verify_identity({**sealed, "path": str(link)})


def test_receipt_requires_explicit_trusted_digest(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Untrusted"):
        verify_baseline(path, "0" * 64)


@pytest.mark.parametrize("field", ["acceptance_passed", "audit_execution_complete"])
def test_incomplete_or_failed_baseline_is_rejected(tmp_path, field):
    path = tmp_path / "result.json"
    receipt = {
        "scope": "engineering_canonical_reduction_deterministic_backward_control",
        "target": "resume",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": True,
        "acceptance_passed": True,
    }
    receipt[field] = False
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="completed passing"):
        verify_baseline(path, identity(path)["sha256"])


def test_exact_comparison_does_not_ignore_any_path():
    expected = {"entry": {"weights": "abc"}, "future": [{"sha256": "def"}]}
    assert compare_fingerprints(expected, expected)["exact"]
    actual = {"entry": {"weights": "abc"}, "future": [{"sha256": "deg"}]}
    result = compare_fingerprints(expected, actual)
    assert result["exact"] is False
    assert result["first_mismatch_paths"] == ["state/future/0/sha256"]


def test_mismatch_path_limit_does_not_change_acceptance():
    result = compare_fingerprints(list(range(100)), [None] * 100, limit=3)
    assert len(result["first_mismatch_paths"]) == 3
    assert result["exact"] is False
