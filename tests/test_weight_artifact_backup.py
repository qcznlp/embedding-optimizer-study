"""Bounded file/remote guards; no upload, model or numerical experiment runs here."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reports/engineering-archive/dense-v3-weight-artifact-backup-v1/backup.py"
)
spec = importlib.util.spec_from_file_location("weight_artifact_backup", SCRIPT)
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
