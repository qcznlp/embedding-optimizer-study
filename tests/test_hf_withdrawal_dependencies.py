import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import audit_hf_withdrawal_dependencies as audit
from scripts import hf_obsolete_cleanup as cleanup


def row(path, digest="a" * 64, size=12):
    return {"path": path, "size": size, "blob_id": "b" * 40, "lfs_sha256": digest}


def fixture():
    old = row("dense/old/scheduler.pt")
    good = row("corrected-dense-no-packing-v1/dense/new/scheduler.pt")
    snapshots = {
        kind: {
            "repo_id": repo,
            "repo_type": kind,
            "revision": "c" * 40,
            "files": [old, good] if kind == "model" else [],
        }
        for kind, repo in cleanup.REPOSITORIES.items()
    }
    plan = cleanup.build_plan({"repositories": list(snapshots.values())}, "before-sha")
    return plan, snapshots


def receipt_file(tmp_path):
    path = tmp_path / audit.RECEIPT_DIRS[0] / "new.json"
    path.parent.mkdir(parents=True)
    receipt = {
        "status": "complete",
        "repo_id": cleanup.REPOSITORIES["model"],
        "repo_type": "model",
        "remote_prefix": "corrected-dense-no-packing-v1/dense/new",
        "commit_oid": "d" * 40,
        "commit_url": "https://huggingface.co/"
        + cleanup.REPOSITORIES["model"]
        + "/commit/"
        + "d" * 40,
        "inventory": {
            "complete": True,
            "local_files": 1,
            "remote_files": 1,
            "local_bytes": 12,
            "remote_bytes": 12,
            "missing": [],
            "extra": [],
            "size_mismatch": [],
        },
    }
    path.write_text(json.dumps(receipt))
    return path, receipt


class ReadOnlyApi:
    def __init__(self, snapshots):
        self.snapshots = snapshots
        self.calls = []

    def list_repo_refs(self, repo_id, **kwargs):
        self.calls.append(("refs", repo_id, kwargs))
        return SimpleNamespace(branches=[], tags=[], converts=[], pull_requests=[])

    def list_repo_tree(self, repo_id, **kwargs):
        self.calls.append(("tree", repo_id, kwargs))
        good = self.snapshots["model"]["files"][1]
        return [
            SimpleNamespace(
                path=good["path"],
                size=good["size"],
                blob_id=good["blob_id"],
                lfs=SimpleNamespace(sha256=good["lfs_sha256"]),
            )
        ]

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected API operation: {name}")


def test_shared_object_retains_each_path_and_is_never_a_purge_approval():
    plan, snapshots = fixture()
    result = audit.repository_dependencies(plan["repositories"][0], snapshots["model"])
    assert result["shared_lfs_count"] == 1
    assert result["shared_lfs_unique_bytes"] == 12
    assert result["shared_lfs_objects"][0]["retained_paths"] == [
        snapshots["model"]["files"][1]["path"]
    ]
    assert result["safe_to_purge"] is False


@pytest.mark.parametrize(
    "mutation", ["changed", "new_invalid", "duplicate", "shared_size", "protected_plan"]
)
def test_drift_and_invalid_inventory_rejected(mutation):
    plan, snapshots = fixture()
    planned, snapshot = plan["repositories"][0], snapshots["model"]
    if mutation == "changed":
        snapshot = copy.deepcopy(snapshot)
        snapshot["files"][0]["size"] += 1
    elif mutation == "new_invalid":
        snapshot["files"].append(row("dense/new-invalid/optimizer.pt"))
    elif mutation == "duplicate":
        snapshot["files"].append(snapshot["files"][0])
    elif mutation == "shared_size":
        snapshot["files"][1]["size"] += 1
    else:
        planned["delete"] = [dict(snapshot["files"][1], reason="injected")]
    with pytest.raises(ValueError):
        audit.repository_dependencies(planned, snapshot)


def test_new_protected_upload_is_preserved():
    plan, snapshots = fixture()
    snapshots["model"]["files"].append(row("corrected-dense-no-packing-v1/dense/later/model"))
    result = audit.repository_dependencies(plan["repositories"][0], snapshots["model"])
    assert result["protected_files"] == 2
    assert len(result["shared_lfs_objects"][0]["retained_paths"]) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("commit_oid", "e" * 40),
        ("commit_url", "https://example.org/commit/" + "d" * 40),
        ("status", "failed"),
        ("remote_prefix", "dense/old"),
        ("remote_prefix", "corrected-dense-no-packing-v1/../dense/old"),
    ],
)
def test_invalid_reference_rejected(tmp_path, field, value):
    path, receipt = receipt_file(tmp_path)
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        audit.declared_reference(path, tmp_path)


def test_legacy_url_only_receipt_is_identified_without_rewriting_it(tmp_path):
    path, receipt = receipt_file(tmp_path)
    receipt.pop("commit_oid")
    path.write_text(json.dumps(receipt))
    before = path.read_bytes()
    reference = audit.declared_reference(path, tmp_path)
    assert reference["commit_oid"] == "d" * 40
    assert reference["original_receipt_includes_digest_check"] is False
    assert path.read_bytes() == before


def test_pinned_digest_drift_is_reported_not_hidden(tmp_path):
    _, snapshots = fixture()
    api = ReadOnlyApi(copy.deepcopy(snapshots))
    path, _ = receipt_file(tmp_path)
    reference = audit.declared_reference(path, tmp_path)
    snapshots["model"]["files"][1]["lfs_sha256"] = "f" * 64
    checked = audit.check_reference(api, reference, snapshots["model"])
    assert checked["immutable_prefix_readable"] is True
    assert checked["pinned_files_unchanged_at_observed_head"] is False
    assert checked["changed_or_missing_at_observed_head"]
    assert api.calls[0][2]["revision"] == "d" * 40
    assert checked["payload_downloaded_or_locally_rehashed"] is False


def test_whole_cli_uses_only_reads_and_preserves_receipts(tmp_path, monkeypatch):
    plan, snapshots = fixture()
    api = ReadOnlyApi(snapshots)
    path, _ = receipt_file(tmp_path)
    before = path.read_bytes()
    plan_path, output = tmp_path / "plan.json", tmp_path / "report.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(audit, "HfApi", lambda: api)
    monkeypatch.setattr(audit, "inventory", lambda _, kind: snapshots[kind])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--plan",
            str(plan_path),
            "--plan-sha256",
            hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "--repository",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    audit.main()
    result = json.loads(output.read_text())
    assert result["reference_receipts"] == 1
    assert result["all_pinned_files_unchanged_at_observed_heads"] is True
    assert result["hf_mutations"] is result["safe_to_purge"] is False
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        audit.main()
