"""Pinned-reference loader guards; real upstream numerical results are separate."""

import hashlib

import pytest
import torch

from scripts import audit_normuon_reference_edges as audit


def test_upstream_digest_is_required_before_any_function_execution():
    with pytest.raises(ValueError, match="digest"):
        audit.reference_functions(b"raise AssertionError('must not run')")


def test_diagnostic_control_changes_only_the_expected_denominator(monkeypatch):
    # Intentionally synthetic source to test the AST guard, not an upstream numerical reference.
    raw = b"def zeropower_via_newtonschulz5(X):\n    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)\n    return X\ndef normuon_update(X):\n    return zeropower_via_newtonschulz5(X)\n"
    monkeypatch.setattr(audit, "SHA256", hashlib.sha256(raw).hexdigest())
    ordinary = audit.reference_functions(raw)
    control = audit.reference_functions(raw, clamp_control=True)
    value = torch.ones(2, 2) * 1e-8
    torch.testing.assert_close(ordinary(value), value / (value.norm() + 1e-7), rtol=0, atol=0)
    torch.testing.assert_close(control(value), value / value.norm().clamp_min(1e-7), rtol=0, atol=0)
    assert not torch.equal(ordinary(value), control(value))


@pytest.mark.parametrize(
    "source",
    [
        b"def normuon_update(X):\n    return X\n",
        b"def zeropower_via_newtonschulz5(X):\n    return X\ndef normuon_update(X):\n    return X\n",
    ],
)
def test_control_rejects_changed_source_topology(monkeypatch, source):
    monkeypatch.setattr(audit, "SHA256", hashlib.sha256(source).hexdigest())
    with pytest.raises(ValueError):
        audit.reference_functions(source, clamp_control=True)
