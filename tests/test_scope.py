import json
from pathlib import Path

import pytest

from embed_optim.scope import (
    canonical_scope_amendment,
    normalize_families,
    resolve_scope,
)


def test_family_scope_is_canonical():
    assert normalize_families(["late", "dense", "dense"]) == ("dense", "late")


def test_reduced_scope_requires_amendment():
    with pytest.raises(ValueError, match="requires --scope-amendment"):
        resolve_scope(["dense"], None)


def test_frozen_dense_scope_amendment_passes():
    families, amendment = resolve_scope(["dense"], "configs/dense_scope_amendment.json")

    assert families == ("dense",)
    assert amendment["status"] == "user_directed_post_hoc_scope_amendment"
    assert amendment["path"] == "configs/dense_scope_amendment.json"
    assert len(amendment["sha256"]) == 64


def test_changed_scope_binding_is_rejected(tmp_path, monkeypatch):
    source = "configs/dense_scope_amendment.json"
    payload = json.loads(open(source, encoding="utf-8").read())
    payload["source_bindings"][0]["sha256"] = "0" * 64
    root = tmp_path
    (root / "configs").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='test'\n")
    amendment = root / source
    amendment.write_text(json.dumps(payload))
    monkeypatch.setattr("embed_optim.scope.resolve_matrix_path", lambda _: amendment)

    with pytest.raises(ValueError, match="Scope-amendment source differs"):
        resolve_scope(["dense"], amendment)


def test_scope_identity_accepts_the_legacy_absolute_checkout_path():
    _, amendment = resolve_scope(["dense"], "configs/dense_scope_amendment.json")
    legacy = {
        **amendment,
        "path": str((Path.cwd() / amendment["path"]).resolve()),
    }

    assert canonical_scope_amendment(legacy, Path.cwd()) == amendment
