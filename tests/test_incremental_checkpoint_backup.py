import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.incremental_checkpoint_backup import (
    REQUIRED_CHECKPOINT_FILES,
    _recover_upload_commit,
    compare_checkpoint_inventories,
    inventory_digest,
    local_checkpoint_inventory,
    stat_signature,
    validate_sealed_checkpoint,
)


def _record(size: int, sha256: str, git_blob_sha1: str) -> dict[str, object]:
    return {"size": size, "sha256": sha256, "git_blob_sha1": git_blob_sha1}


def test_checkpoint_inventory_matches_lfs_and_git_blob_digests() -> None:
    local = {
        "model.safetensors": _record(10, "sha-model", "git-model"),
        "trainer_state.json": _record(4, "sha-state", "git-state"),
    }
    remote = {
        "model.safetensors": {
            "size": 10,
            "digest_kind": "sha256",
            "digest": "sha-model",
        },
        "trainer_state.json": {
            "size": 4,
            "digest_kind": "git_blob_sha1",
            "digest": "git-state",
        },
    }

    audit = compare_checkpoint_inventories(local, remote)

    assert audit == {
        "complete": True,
        "local_files": 2,
        "local_bytes": 14,
        "remote_files": 2,
        "remote_bytes": 14,
        "missing": [],
        "extra": [],
        "size_mismatch": [],
        "digest_mismatch": [],
    }
    assert inventory_digest(local) == inventory_digest(dict(reversed(list(local.items()))))


def test_checkpoint_inventory_rejects_path_size_and_digest_drift() -> None:
    local = {
        "a": _record(1, "sha-a", "git-a"),
        "b": _record(2, "sha-b", "git-b"),
        "d": _record(4, "sha-d", "git-d"),
    }
    remote = {
        "a": {"size": 9, "digest_kind": "sha256", "digest": "sha-a"},
        "c": {"size": 3, "digest_kind": "sha256", "digest": "sha-c"},
        "d": {"size": 4, "digest_kind": "sha256", "digest": "wrong"},
    }

    audit = compare_checkpoint_inventories(local, remote)

    assert audit["complete"] is False
    assert audit["missing"] == ["b"]
    assert audit["extra"] == ["c"]
    assert audit["size_mismatch"] == ["a"]
    assert audit["digest_mismatch"] == ["d"]


def test_upload_commit_recovery_matches_exact_title() -> None:
    api = SimpleNamespace(
        list_repo_commits=lambda *_args, **_kwargs: [
            SimpleNamespace(
                title="unrelated",
                commit_id="old",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                title="target",
                commit_id="abc123",
                created_at=datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
            ),
        ]
    )

    oid, url, created_at = _recover_upload_commit(api, repo_id="owner/repo", title="target")

    assert oid == "abc123"
    assert url == "https://huggingface.co/owner/repo/commit/abc123"
    assert created_at == "2026-02-03T04:05:06Z"


def test_incremental_backup_has_public_cli_entrypoint() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    scripts = pyproject.split("[project.scripts]", 1)[1].split("\n[", 1)[0]

    assert (
        "embed-optim-backup-sealed-checkpoint = "
        '"embed_optim.incremental_checkpoint_backup:main"' in scripts
    )


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        run_id="padded-adamw-test",
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=1e-5),
        model_name="example/model",
        dataset_path="example/data",
        output_root=str(tmp_path / "outputs"),
        dense_can_flatten_inputs=False,
    )


def _write_checkpoint(config: RunConfig, step: int) -> Path:
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "checkpoint_schedule.json").write_text(
        json.dumps({"max_steps": 10, "steps": [step]}), encoding="utf-8"
    )
    checkpoint = config.output_dir / f"checkpoint-{step}"
    checkpoint.mkdir()
    for name in REQUIRED_CHECKPOINT_FILES:
        if name in {"model.safetensors", "trainer_state.json"}:
            continue
        (checkpoint / name).write_bytes(b"payload")
    save_file({"weight": np.ones((2, 2), dtype=np.float32)}, checkpoint / "model.safetensors")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    return checkpoint


def test_sealed_checkpoint_validation_and_content_inventory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = _write_checkpoint(config, 7)

    assert validate_sealed_checkpoint(config, 7) == checkpoint
    before = stat_signature(checkpoint)
    inventory = local_checkpoint_inventory(checkpoint)

    assert set(REQUIRED_CHECKPOINT_FILES) <= set(inventory)
    assert sum(item["size"] for item in inventory.values()) > 0
    assert before == stat_signature(checkpoint)


def test_sealed_checkpoint_rejects_wrong_state_or_missing_payload(tmp_path: Path) -> None:
    config = _config(tmp_path)
    checkpoint = _write_checkpoint(config, 7)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 6}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="expected 7"):
        validate_sealed_checkpoint(config, 7)

    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 7}), encoding="utf-8")
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(RuntimeError, match="optimizer.pt"):
        validate_sealed_checkpoint(config, 7)
