import copy
import hashlib
import json

import pytest

from scripts.audit_checkpoint_download import (
    PREFIX,
    REPO_ID,
    REQUIRED,
    select_download,
    verify_download,
)

RUN_ID = "padded-normuon-3e-4"
STEP = 3126


def digest(payload):
    return {
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "git_blob_sha1": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
    }


@pytest.fixture
def download(tmp_path):
    root = tmp_path / "fresh-download"
    run = root / PREFIX / RUN_ID
    inventory = {}
    for name in sorted(REQUIRED):
        relative = f"checkpoint-{STEP}/{name}"
        payload = f"synthetic fixture {relative}".encode()
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inventory[relative] = digest(payload)
    (run / "run_config.json").write_bytes(b"{}")
    inventory["run_config.json"] = digest(b"{}")
    data = {
        "schema_version": 1,
        "audit_scope": "immutable remote file content versus completed local runs",
        "complete_for_audited_runs": True,
        "scientific_completion": False,
        "records": [
            {
                "run_id": RUN_ID,
                "repo_id": REPO_ID,
                "prefix": f"{PREFIX}/{RUN_ID}",
                "revision": "1" * 40,
                "local_inventory": inventory,
                "remote_inventory": {
                    k: {"size": v["size"], "digest_kind": "sha256", "digest": v["sha256"]}
                    for k, v in inventory.items()
                },
            }
        ],
    }
    audit = tmp_path / "trusted-audit.json"

    def select(changed=None, **kwargs):
        audit.write_text(json.dumps(data if changed is None else changed))
        trusted = hashlib.sha256(audit.read_bytes()).hexdigest()
        return select_download(
            audit,
            kwargs.get("sha256", trusted),
            kwargs.get("run_id", RUN_ID),
            kwargs.get("step", STEP),
        )

    return root, run, data, select


def test_exact_selected_checkpoint_and_config_verify_without_writes(download):
    root, run, _, select = download
    before = {p: p.stat().st_mtime_ns for p in run.rglob("*")}
    selected = select()
    result = verify_download(selected, root)
    assert result["complete_for_selected_download"]
    assert result["files"] == len(REQUIRED) + 1
    assert result["checkpoint_root"] == str(run / f"checkpoint-{STEP}")
    assert result["inventory"] == selected["files"]
    assert before == {p: p.stat().st_mtime_ns for p in run.rglob("*")}


@pytest.mark.parametrize(
    "key,value",
    [("scientific_completion", True), ("complete_for_audited_runs", False), ("schema_version", 2)],
)
def test_rejects_wrong_audit_scope(download, key, value):
    _, _, data, select = download
    data[key] = value
    with pytest.raises(ValueError, match="content audit"):
        select(data)


def test_requires_independently_trusted_audit_hash(download):
    with pytest.raises(ValueError, match="trusted SHA-256"):
        download[3](sha256="0" * 64)
    with pytest.raises(ValueError, match="trusted audit SHA-256"):
        download[3](sha256="main")


@pytest.mark.parametrize(
    "field,value",
    [("repo_id", "other/private-model"), ("prefix", "dense/old-run"), ("revision", "main")],
)
def test_rejects_wrong_remote_identity(download, field, value):
    _, _, data, select = download
    data["records"][0][field] = value
    with pytest.raises(ValueError, match="repository, prefix or immutable"):
        select(data)


def test_rejects_duplicate_run_identity(download):
    _, _, data, select = download
    data["records"].append(copy.deepcopy(data["records"][0]))
    with pytest.raises(ValueError, match="exactly once"):
        select(data)


@pytest.mark.parametrize("step", [True, 0, 3125])
def test_rejects_unknown_checkpoint_stage(download, step):
    with pytest.raises(ValueError, match="scheduled"):
        download[3](step=step)


def test_rejects_legacy_or_missing_run(download):
    with pytest.raises(ValueError, match="primary run ID"):
        download[3](run_id="normuon-3e-4")
    with pytest.raises(ValueError, match="exactly once"):
        download[3](run_id="padded-muon-3e-4")


def test_rejects_inconsistent_remote_content(download):
    _, _, data, select = download
    data["records"][0]["remote_inventory"]["run_config.json"]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="inconsistent inventories"):
        select(data)


def test_rejects_missing_required_payload_even_if_inventory_agrees(download):
    _, _, data, select = download
    for field in ["local_inventory", "remote_inventory"]:
        data["records"][0][field].pop(f"checkpoint-{STEP}/optimizer.pt")
    with pytest.raises(ValueError, match="required reconstruction"):
        select(data)


@pytest.mark.parametrize(
    "unsafe",
    [
        "../outside",
        "/outside",
        "checkpoint-3126/../../outside",
        "checkpoint-3126//config.json",
        "checkpoint-3126/.cache/file",
        "checkpoint-3126\\file",
    ],
)
def test_rejects_unsafe_inventory_paths(download, unsafe):
    _, _, data, select = download
    row = data["records"][0]
    for field in ["local_inventory", "remote_inventory"]:
        row[field][unsafe] = copy.deepcopy(row[field]["run_config.json"])
    with pytest.raises(ValueError, match="Unsafe inventory"):
        select(data)


def test_rejects_corrupted_download_without_repair(download):
    root, run, _, select = download
    selected = select()
    target = run / f"checkpoint-{STEP}/model.safetensors"
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="content mismatch"):
        verify_download(selected, root)
    assert target.read_bytes() == b"corrupt"


def test_rejects_unexpected_checkpoint_file(download):
    root, run, _, select = download
    selected = select()
    (run / f"checkpoint-{STEP}/unexpected").write_text("not a frozen payload")
    with pytest.raises(ValueError, match="coverage"):
        verify_download(selected, root)


def test_rejects_missing_downloaded_file(download):
    root, run, _, select = download
    selected = select()
    (run / f"checkpoint-{STEP}/optimizer.pt").rename(run / "retained-outside-checkpoint.pt")
    with pytest.raises(ValueError, match="coverage"):
        verify_download(selected, root)


def test_rejects_payload_symlink(download):
    root, run, _, select = download
    selected = select()
    target = run / f"checkpoint-{STEP}/model.safetensors"
    retained = run / "retained-model.safetensors"
    target.rename(retained)
    target.symlink_to(retained)
    with pytest.raises(ValueError, match="symlink"):
        verify_download(selected, root)


def test_rejects_download_root_symlink(download, tmp_path):
    root, _, _, select = download
    link = tmp_path / "aliased-download"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        verify_download(select(), link)
