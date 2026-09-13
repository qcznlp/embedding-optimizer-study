from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.hf_api import RepoFile

from embed_optim import dimension_archive as archive
from embed_optim.config import load_matrix
from embed_optim.primary_dimension_probe import planned_cells

ROOT = Path(__file__).resolve().parents[1]


class FakeHub:
    """Immutable snapshots with both Git-blob and LFS identities; never uses network."""

    def __init__(self):
        self.head = "initial"
        self.snapshots = {self.head: {}}
        self.commits = []
        self.reads = []
        self.corrupt_digest = False

    def repo_info(self, repo_id, **kwargs):
        assert repo_id == archive.REPO_ID and kwargs == {"repo_type": "dataset"}
        return SimpleNamespace(sha=self.head)

    def list_repo_tree(self, repo_id, **kwargs):
        assert repo_id == archive.REPO_ID and kwargs["repo_type"] == "dataset"
        revision = kwargs["revision"]
        self.reads.append(revision)
        snapshot = self.snapshots[revision]
        prefix = kwargs["path_in_repo"] + "/"
        found = False
        for name, value in sorted(snapshot.items()):
            if not name.startswith(prefix):
                continue
            found = True
            digest = archive._bytes_inventory(value)
            lfs = None
            if name.endswith(".npz"):
                lfs = {
                    "size": len(value),
                    "oid": "0" * 64 if self.corrupt_digest else digest["sha256"],
                    "pointerSize": 130,
                }
            yield RepoFile(path=name, size=len(value), oid=digest["git_blob_sha1"], lfs=lfs)
        if not found:
            # Exercise exceptions raised while consuming the lazy HF iterator.
            raise RemoteEntryNotFoundError(
                "No such archive prefix",
                response=httpx.Response(
                    404, request=httpx.Request("GET", "https://example.invalid")
                ),
            )

    def create_commit(self, **kwargs):
        assert kwargs["repo_id"] == archive.REPO_ID and kwargs["repo_type"] == "dataset"
        assert kwargs["parent_commit"] == self.head
        snapshot = dict(self.snapshots[self.head])
        paths = []
        for operation in kwargs["operations"]:
            value = operation.path_or_fileobj
            value = value if isinstance(value, bytes) else Path(value).read_bytes()
            assert operation.path_in_repo not in snapshot
            snapshot[operation.path_in_repo] = value
            paths.append(operation.path_in_repo)
        self.commits.append(paths)
        self.head = f"commit-{len(self.commits)}"
        self.snapshots[self.head] = snapshot
        return SimpleNamespace(oid=self.head)


@pytest.fixture
def archive_source(tmp_path, monkeypatch):
    root = tmp_path / "embedding-optimizer-study"
    shutil.copytree(ROOT / "src", root / "src")
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copy(ROOT / "pyproject.toml", root / "pyproject.toml")
    calls = []
    monkeypatch.setattr(
        archive.publication, "audit_report", lambda *a, **k: calls.append(k.get("portable", False))
    )
    portable = root / archive.release.MANIFEST
    portable.parent.mkdir(parents=True)
    portable.write_text('{"synthetic":true}\n')
    monkeypatch.setattr(archive.release, "selected_files", lambda *_: [portable])
    scientific = json.loads((root / archive.release.PROTOCOL).read_text())
    cells = planned_cells(
        load_matrix(root / "configs/dense_no_packing_retrain.yaml"),
        scientific["inputs"]["checkpoint_stages"],
    )
    outputs = []
    for index, cell in enumerate(cells):
        path = root / archive.EXPORT_ROOT / f"{cell}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-vector-{index}".encode())
        outputs.append({"cell": cell, "export": archive.publication._identity(path, root)})
    handoff = root / archive.EXPORT_ROOT / "primary_exports.json"
    handoff.write_text(
        json.dumps(
            {
                "status": "complete",
                "scope": "primary_dense_fixed_probe_exports",
                "cells": cells,
                "outputs": outputs,
            }
        )
    )
    return root, calls


def test_contract_binds_current_sources():
    assert archive.load_contract(ROOT)["destination"]["repo_type"] == "dataset"


def test_changed_archive_source_is_rejected(archive_source):
    root, _ = archive_source
    source = root / archive.SOURCES[0]
    source.write_text(source.read_text() + "# altered\n")
    with pytest.raises(ValueError, match="binding changed"):
        archive.build_bundle(root)


def test_atomic_archive_and_audit_remain_pinned_when_head_advances(archive_source):
    root, calls = archive_source
    hub = FakeHub()
    first = archive.archive(root, api=hub)
    assert calls[0] is False  # Full authoring audit happens before remote work.
    assert len(hub.commits) == 1
    assert sum(path.endswith(".npz") for path in hub.commits[0]) == 61
    assert first["commit_oid"] == "commit-1"
    assert first["scientific_completion"] is False
    assert first["inventory"]["complete"] is True
    hub.head = "new-unrelated-head"
    hub.snapshots[hub.head] = {"unrelated-file": b"not the archive"}
    second = archive.archive(root, audit_only=True, api=hub)
    assert second["commit_oid"] == first["commit_oid"]
    assert second["first_verified_at_utc"] == first["first_verified_at_utc"]
    assert second["local_inventory"] == first["local_inventory"]
    assert hub.reads[-1] == "commit-1" and len(hub.commits) == 1


def test_remote_hash_corruption_never_produces_complete_receipt(archive_source):
    root, _ = archive_source
    hub = FakeHub()
    hub.corrupt_digest = True
    with pytest.raises(ValueError, match="inventory differs"):
        archive.archive(root, api=hub)
    assert not list((root / archive.RECEIPT_ROOT).glob("*.json"))


def test_missing_receipt_audit_does_not_create_a_remote_commit(archive_source):
    root, _ = archive_source
    hub = FakeHub()
    with pytest.raises(FileNotFoundError, match="No verified"):
        archive.archive(root, audit_only=True, api=hub)
    assert not hub.commits and not hub.reads


def test_lost_receipt_recovers_a_verified_snapshot_without_reupload(archive_source):
    root, _ = archive_source
    hub = FakeHub()
    first = archive.archive(root, api=hub)
    receipt = root / archive.RECEIPT_ROOT / f"{first['archive_sha256']}.json"
    receipt.rename(receipt.with_suffix(".interrupted"))
    recovered = archive.archive(root, api=hub)
    assert len(hub.commits) == 1
    assert recovered["commit_provenance"] == "recovered_verified_snapshot"
    assert recovered["commit_oid"] == first["commit_oid"]


def test_conflicting_existing_prefix_is_preserved_not_overwritten(archive_source):
    root, _ = archive_source
    hub = FakeHub()
    payload, _ = archive.build_bundle(root)
    sha = hashlib.sha256(archive._json_bytes(payload)).hexdigest()
    name = f"{archive.REMOTE_ROOT}/{sha}/unexpected.txt"
    hub.snapshots[hub.head][name] = b"retain conflicting evidence"
    with pytest.raises(ValueError, match="inventory differs"):
        archive.archive(root, api=hub)
    assert not hub.commits and hub.snapshots[hub.head][name] == b"retain conflicting evidence"


def test_changed_source_during_staging_aborts_before_upload(archive_source, monkeypatch):
    root, _ = archive_source
    copy = archive.shutil.copyfile

    def change(source, target):
        result = copy(source, target)
        if str(source).endswith("pretrained.npz"):
            Path(target).write_bytes(b"changed during staging")
        return result

    monkeypatch.setattr(archive.shutil, "copyfile", change)
    hub = FakeHub()
    with pytest.raises(ValueError, match="changed while staging"):
        archive.archive(root, api=hub)
    assert not hub.commits


def test_raw_vector_digest_and_canonical_location_are_required(archive_source):
    root, _ = archive_source
    path = root / archive.EXPORT_ROOT / "pretrained.npz"
    retained = path.with_suffix(".original")
    path.rename(retained)
    path.symlink_to(retained)
    with pytest.raises(ValueError, match="vector path differs"):
        archive.build_bundle(root)


def test_download_verification_is_independent_and_rejects_corruption(archive_source, tmp_path):
    root, _ = archive_source
    hub = FakeHub()
    receipt = archive.archive(root, api=hub)
    downloaded = tmp_path / "downloaded"
    prefix = receipt["remote_prefix"] + "/"
    for name, content in hub.snapshots[receipt["commit_oid"]].items():
        path = downloaded / name.removeprefix(prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    checked = archive.audit_download(downloaded, receipt["archive_sha256"])
    assert checked["complete"] is True and checked["scientific_completion"] is False
    with pytest.raises(ValueError, match="trusted receipt hash"):
        archive.audit_download(downloaded, "0" * 64)
    vector = downloaded / "payload" / archive.EXPORT_ROOT / "pretrained.npz"
    vector.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload differs"):
        archive.audit_download(downloaded, receipt["archive_sha256"])


@pytest.mark.parametrize(
    "name", [".", "../escape", "/absolute", "x/../../escape", "x//y", "x\\y", ".cache/token"]
)
def test_archive_paths_cannot_escape(name):
    with pytest.raises(ValueError, match="Unsafe"):
        archive._safe_relative(name)


def test_read_only_plan_does_not_instantiate_network_client(monkeypatch, capsys):
    monkeypatch.setattr(archive, "HfApi", lambda: pytest.fail("dry run used network"))
    archive.main(["--repository", str(ROOT), "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["missing_inputs"] and result["network_used"] is False
    assert result["upload_ready"] is False
