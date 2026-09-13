"""Transport/integrity checks only; synthetic payloads are never model evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import restore_primary_v3 as recovery

REPOSITORY = Path(__file__).resolve().parents[1]
INDEX = REPOSITORY / "docs/primary-v3-checkpoints.json"
INDEX_SHA = "76e1253373e24525044ebbb459aadcb56afa76da3db629e5b00a652ed44dbe89"


def actual_index():
    return recovery.load_index(INDEX, INDEX_SHA)


def synthetic_index():
    value = copy.deepcopy(actual_index())
    payload = b"not a model; synthetic recovery test\n"
    sha = hashlib.sha256(payload).hexdigest()
    blob = hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False
    ).hexdigest()
    for row in value["checkpoints"]:
        for entry in row["files"]:
            entry["bytes"], entry["sha256"] = len(payload), sha
            kind = entry["remote"]["kind"]
            entry["remote"] = {
                "bytes": len(payload),
                "kind": kind,
                "digest": sha if kind == "sha256" else blob,
            }
        row["total_bytes"] = 20 * len(payload)
    value["total_bytes"] = 1200 * len(payload)
    return recovery.validate_index(value), payload


def test_real_complete_index_has_all_current_cells():
    value = actual_index()
    assert len(value["checkpoints"]) == 60 and value["total_files"] == 1200
    assert value["total_bytes"] == 89889820336
    assert value["scientific_completion"] is False
    assert len({r["revision"] for r in value["checkpoints"]}) == 60


@pytest.mark.parametrize(
    "change",
    (
        "old_protocol",
        "moving_revision",
        "old_prefix",
        "duplicate_cell",
        "wrong_step",
        "unsafe_file",
        "missing_file",
        "wrong_lfs",
        "wrong_total",
        "scientific_claim",
    ),
)
def test_rehashed_invalid_index_is_rejected(tmp_path, change):
    value = actual_index()
    row = value["checkpoints"][0]
    if change == "old_protocol":
        value["protocol_sha256"] = "0" * 64
    elif change == "moving_revision":
        row["revision"] = "main"
    elif change == "old_prefix":
        row["prefix"] = "corrected-dense-no-packing-v1/dense/padded-adamw-3e-5/checkpoint-782"
    elif change == "duplicate_cell":
        value["checkpoints"][1] = copy.deepcopy(row)
    elif change == "wrong_step":
        row["step"] = 391
    elif change == "unsafe_file":
        row["files"][0]["path"] = "../../outside"
    elif change == "missing_file":
        row["files"].pop()
    elif change == "wrong_lfs":
        next(f for f in row["files"] if f["remote"]["kind"] == "sha256")["remote"]["digest"] = (
            "0" * 64
        )
    elif change == "wrong_total":
        value["total_bytes"] += 1
    else:
        value["scientific_completion"] = True
    path = tmp_path / "index.json"
    content = json.dumps(value).encode()
    path.write_bytes(content)
    with pytest.raises(ValueError):
        recovery.load_index(path, hashlib.sha256(content).hexdigest())


def test_index_digest_cannot_be_self_substituted():
    with pytest.raises(ValueError, match="authentication"):
        recovery.load_index(INDEX, "0" * 64)


def test_selection_requires_explicit_scope():
    value = actual_index()
    with pytest.raises(ValueError, match="Explicitly"):
        recovery.select(value)
    with pytest.raises(ValueError, match="not both"):
        recovery.select(value, all_checkpoints=True, run_id=recovery.RUNS[0])
    assert len(recovery.select(value, all_checkpoints=True)) == 60
    assert len(recovery.select(value, run_id=recovery.RUNS[0])) == 5
    assert len(recovery.select(value, run_id=recovery.RUNS[0], step=2345)) == 1


@pytest.mark.parametrize("relative", ("../escape", "/absolute", "a/../../escape"))
def test_path_escape_is_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        recovery.safe_target(tmp_path, relative)


def test_symlinked_destination_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        recovery.safe_target(linked, "anything")


def install_fake_download(monkeypatch, payload):
    import huggingface_hub

    calls = []

    def transfer(**kwargs):
        assert kwargs["token"] is False and kwargs["force_download"] is False
        assert kwargs["endpoint"] == "https://huggingface.co"
        assert kwargs["repo_id"] == recovery.REPO and recovery.hexadecimal(kwargs["revision"], 40)
        path = kwargs["local_dir"] / kwargs["filename"]
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", transfer)
    return calls


def test_download_and_repeat_resume_verify_without_overwrite(monkeypatch, tmp_path):
    value, payload = synthetic_index()
    rows = recovery.select(value, run_id=recovery.RUNS[0], step=2345)
    calls = install_fake_download(monkeypatch, payload)
    destination = tmp_path / "new"
    first = recovery.download(destination, rows, workers=1)
    assert first == {"downloaded_files": 20, "reused_verified_files": 0, "verified_files": 20}
    assert len(calls) == 20
    second = recovery.download(destination, rows, resume=True, workers=1)
    assert second == {"downloaded_files": 0, "reused_verified_files": 20, "verified_files": 20}
    assert len(calls) == 20


@pytest.mark.parametrize("problem", ("corrupt", "extra", "symlink"))
def test_resume_refuses_existing_damage_before_network(monkeypatch, tmp_path, problem):
    value, payload = synthetic_index()
    rows = recovery.select(value, run_id=recovery.RUNS[0], step=2345)
    calls = install_fake_download(monkeypatch, payload)
    destination = tmp_path / "new"
    root = destination / rows[0]["prefix"]
    root.mkdir(parents=True)
    if problem == "corrupt":
        target = root / "model.safetensors"
        target.write_bytes(b"preserve corrupt evidence")
    elif problem == "extra":
        target = root / "unexpected.txt"
        target.write_bytes(b"preserve extra evidence")
    else:
        target = root / "model.safetensors"
        target.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        recovery.download(destination, rows, resume=True, workers=1)
    assert calls == [] and (target.exists() or target.is_symlink())


def test_missing_payload_fails_verification(tmp_path):
    rows = recovery.select(actual_index(), run_id=recovery.RUNS[0], step=2345)
    with pytest.raises(ValueError, match="Missing"):
        recovery.existing_payloads(tmp_path, rows, require_complete=True)


def test_nonempty_destination_requires_resume_before_network(monkeypatch, tmp_path):
    value, payload = synthetic_index()
    rows = recovery.select(value, run_id=recovery.RUNS[0], step=2345)
    calls = install_fake_download(monkeypatch, payload)
    (tmp_path / "user-file").write_text("preserve")
    with pytest.raises(ValueError, match="not empty"):
        recovery.download(tmp_path, rows)
    assert calls == [] and (tmp_path / "user-file").read_text() == "preserve"
