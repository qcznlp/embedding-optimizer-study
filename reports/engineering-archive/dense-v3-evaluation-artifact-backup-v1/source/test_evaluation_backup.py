"""Bounded file/remote guards; no upload, model or numerical experiment runs here."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("backup.py")
spec = importlib.util.spec_from_file_location("evaluation_artifact_backup", SCRIPT)
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


@pytest.mark.parametrize("name", ("", ".", "..", "../file", "/file", "a//b", "a/./b", "a\\b", 17))
def test_noncanonical_or_escaping_name_is_refused(name):
    with pytest.raises(ValueError):
        backup.safe_name(name)


def test_plain_payload_name_is_retained():
    assert (
        backup.safe_name("exact/runs/verified-v3-muon-3e-4/checkpoint-2345.json")
        == "exact/runs/verified-v3-muon-3e-4/checkpoint-2345.json"
    )


def inventories():
    expected = {
        "weights": {"bytes": 4, "sha256": "a" * 64, "git_blob_sha1": "b" * 40},
        "metadata": {"bytes": 2, "sha256": "c" * 64, "git_blob_sha1": "d" * 40},
    }
    actual = {
        "weights": {"bytes": 4, "kind": "sha256", "digest": "a" * 64},
        "metadata": {"bytes": 2, "kind": "git_blob_sha1", "digest": "d" * 40},
    }
    return expected, actual


def test_lfs_and_git_digests_are_both_checked():
    backup.compare_remote(*inventories())


@pytest.mark.parametrize("change", ("missing", "extra", "size", "lfs_digest", "git_digest", "kind"))
def test_remote_inventory_changes_are_refused(change):
    expected, source = inventories()
    actual = copy.deepcopy(source)
    if change == "missing":
        actual.pop("weights")
    elif change == "extra":
        actual["extra"] = actual["weights"]
    elif change == "size":
        actual["weights"]["bytes"] += 1
    elif change == "lfs_digest":
        actual["weights"]["digest"] = "0" * 64
    elif change == "git_digest":
        actual["metadata"]["digest"] = "0" * 40
    else:
        actual["weights"]["kind"] = "untrusted"
    with pytest.raises(ValueError):
        backup.compare_remote(expected, actual)


def test_changed_payload_is_preserved(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text("{}")
    expected = backup.file_identity(path)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        backup.compare_file(path, expected)
    assert path.read_text() == '{"changed":true}'


def test_symlink_parent_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "file.json").write_text("{}")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        backup.compare_file(alias / "file.json", backup.file_identity(real / "file.json"))


def test_attempted_upload_cannot_be_blindly_repeated(tmp_path):
    (tmp_path / "upload-started.json").write_text("{}")
    with pytest.raises(ValueError, match="attempted"):
        backup.upload(tmp_path)


@pytest.mark.parametrize("value", ('import os\nprint("synthetic")', "hf_" + "a" * 24))
def test_code_or_credential_shaped_text_is_refused(tmp_path, value):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"synthetic_test": value}))
    with pytest.raises(ValueError):
        backup.scan_text(path)


@pytest.mark.parametrize(
    "value",
    (
        {"query": "synthetic text"},
        {"input_ids": [1, 2]},
        {"api_key": "synthetic-short"},
        {"authorization": "Bearer synthetic"},
    ),
)
def test_raw_examples_and_named_credentials_are_refused(tmp_path, value):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        backup.scan_text(path)


def remote_trees():
    before = {
        "README.md": {"kind": "RepoFile", "blob_id": "old-card"},
        backup.NAMESPACE: {"kind": "RepoFolder", "tree_id": "old-container"},
    }
    after = copy.deepcopy(before)
    after[backup.NAMESPACE]["tree_id"] = "new-container"
    subtree = {
        f"{backup.NAMESPACE}/weight-space": {"kind": "RepoFolder", "tree_id": "original-weights"}
    }
    new_subtree = {
        **copy.deepcopy(subtree),
        backup.ADDITION: {"kind": "RepoFolder", "tree_id": "new-training"},
    }
    return before, after, subtree, new_subtree


def test_only_new_training_subtree_is_added():
    backup.compare_preserved(*remote_trees())


@pytest.mark.parametrize(
    "change",
    [
        "root_card",
        "root_extra",
        "root_missing",
        "weight_tree",
        "subtree_extra",
        "existing_training",
    ],
)
def test_existing_remote_content_cannot_be_replaced(change):
    before, after, old, new = remote_trees()
    if change == "root_card":
        after["README.md"]["blob_id"] = "changed"
    elif change == "root_extra":
        after["extra"] = {"kind": "RepoFolder"}
    elif change == "root_missing":
        after.pop("README.md")
    elif change == "weight_tree":
        new[f"{backup.NAMESPACE}/weight-space"]["tree_id"] = "changed"
    elif change == "subtree_extra":
        new["unrelated"] = {"kind": "RepoFolder"}
    else:
        old[backup.ADDITION] = copy.deepcopy(new[backup.ADDITION])
    with pytest.raises(ValueError):
        backup.compare_preserved(before, after, old, new)


@pytest.mark.parametrize("value", (
    {"query": "synthetic text"}, {"input_ids": [1, 2]},
    {"api_key": "synthetic-short"}, {"authorization": "Bearer synthetic"},
    {"metadata": "import os\nprint('synthetic')"},
))
def test_jsonl_publication_scan_rejects_content_on_later_lines(tmp_path, value):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"row": {"position": 0}}) + "\n" + json.dumps(value) + "\n")
    with pytest.raises(ValueError):
        backup.scan_text(path)


def test_jsonl_numeric_records_are_retained(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"row": {"position": 0, "source": "synthetic", "query_id": 7},
                                "scores": [0.2] * 8, "metrics": {"contrastive_loss": 1.0}}) + "\n")
    backup.scan_text(path)


def test_existing_attributes_cannot_change(tmp_path):
    before, after, old, new = remote_trees()
    before[".gitattributes"] = {"kind": "RepoFile", "blob_id": "original-attributes"}
    after[".gitattributes"] = {"kind": "RepoFile", "blob_id": "changed-attributes"}
    with pytest.raises(ValueError, match="root entries"):
        backup.compare_preserved(before, after, old, new)
