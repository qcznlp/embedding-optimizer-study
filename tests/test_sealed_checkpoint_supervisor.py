import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.incremental_checkpoint_backup import REQUIRED_CHECKPOINT_FILES
from embed_optim.sealed_checkpoint_supervisor import (
    _exclusive_lease,
    _inventory_is_complete,
    _new_state,
    _process_cycle,
)


def _config(tmp_path: Path, run_id: str) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=1e-5),
        model_name="example/model",
        dataset_path="example/data",
        output_root=str(tmp_path / "outputs"),
        dense_can_flatten_inputs=False,
    )


def _schedule(config: RunConfig) -> list[int]:
    steps = [10, 20, 30, 40, 50]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "checkpoint_schedule.json").write_text(
        json.dumps({"max_steps": 50, "steps": steps}), encoding="utf-8"
    )
    return steps


def _checkpoint(config: RunConfig, step: int) -> Path:
    checkpoint = config.output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in REQUIRED_CHECKPOINT_FILES:
        if name in {"model.safetensors", "trainer_state.json"}:
            continue
        (checkpoint / name).write_bytes(b"payload")
    save_file({"weight": np.ones((2, 2), dtype=np.float32)}, checkpoint / "model.safetensors")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    return checkpoint


def _args(tmp_path: Path) -> Namespace:
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("frozen matrix\n", encoding="utf-8")
    return Namespace(
        matrix=matrix,
        repo_id="owner/checkpoints",
        remote_prefix="corrected/dense",
        receipt_root=tmp_path / "incremental",
        full_receipt_root=tmp_path / "full",
        completion_ledger=tmp_path / "completion.json",
        stability_seconds=0.0,
        final_grace_seconds=180.0,
    )


def _complete_inventory() -> dict[str, object]:
    return {
        "complete": True,
        "local_files": 17,
        "remote_files": 17,
        "local_bytes": 123,
        "remote_bytes": 123,
        "missing": [],
        "extra": [],
        "size_mismatch": [],
        "digest_mismatch": [],
    }


def _incremental_receipt(args: Namespace, config: RunConfig, step: int) -> None:
    import hashlib

    args.receipt_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "scientific_completion": False,
        "run_id": config.run_id,
        "checkpoint_step": step,
        "repo_id": args.repo_id,
        "repo_type": "model",
        "remote_prefix": f"{args.remote_prefix}/{config.run_id}/checkpoint-{step}",
        "required_files": sorted(REQUIRED_CHECKPOINT_FILES),
        "matrix": {"sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest()},
        "commit_oid": "incremental-commit",
        "commit_url": "https://example.test/incremental-commit",
        "inventory": _complete_inventory(),
    }
    path = args.receipt_root / f"{config.run_id}-checkpoint-{step}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _full_receipt(args: Namespace, config: RunConfig) -> None:
    args.full_receipt_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "run_id": config.run_id,
        "repo_id": args.repo_id,
        "repo_type": "model",
        "remote_prefix": f"{args.remote_prefix}/{config.run_id}",
        "commit_oid": "full-commit",
        "commit_url": "https://example.test/full-commit",
        "inventory": _complete_inventory(),
    }
    (args.full_receipt_root / f"{config.run_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_cycle_counts_full_run_and_incremental_receipt_without_upload(tmp_path: Path) -> None:
    args = _args(tmp_path)
    full = _config(tmp_path, "padded-adamw-full")
    partial = _config(tmp_path, "padded-adamw-partial")
    _schedule(full)
    _schedule(partial)
    _full_receipt(args, full)
    _incremental_receipt(args, partial, 10)
    uploads = []

    state, events = _process_cycle(
        [full, partial],
        args,
        _new_state({"sha256": "test"}, 10),
        api=object(),
        backup=lambda *call_args, **call_kwargs: uploads.append((call_args, call_kwargs)),
    )

    assert events == []
    assert uploads == []
    assert state["covered_checkpoints"] == 6
    assert state["pending_checkpoints"] == 4
    assert state["runs"][full.run_id]["covered_checkpoints"] == 5
    assert (
        state["runs"][partial.run_id]["checkpoints"]["10"]["coverage"]["kind"]
        == "incremental_checkpoint_receipt"
    )


def test_cycle_uploads_only_a_newly_sealed_checkpoint(tmp_path: Path) -> None:
    args = _args(tmp_path)
    config = _config(tmp_path, "padded-adamw-new")
    _schedule(config)
    _checkpoint(config, 20)
    calls = []

    def backup(_api, received, step, **kwargs):
        calls.append((received.run_id, step, kwargs))
        return {
            "commit_oid": "new-commit",
            "remote_prefix": f"{args.remote_prefix}/{received.run_id}/checkpoint-{step}",
            "inventory": {"remote_files": 17, "remote_bytes": 456},
        }

    state, events = _process_cycle(
        [config],
        args,
        _new_state({"sha256": "test"}, 5),
        api=object(),
        backup=backup,
    )

    assert [(run_id, step) for run_id, step, _ in calls] == [(config.run_id, 20)]
    assert calls[0][2]["audit_only"] is False
    assert events == [
        {
            "event": "checkpoint_backed_up",
            "run_id": config.run_id,
            "step": 20,
            "commit_oid": "new-commit",
            "files": 17,
            "bytes": 456,
        }
    ]
    assert state["covered_checkpoints"] == 1
    assert state["scientific_completion"] is False


def test_partial_checkpoint_waits_for_seal_without_upload_failure(tmp_path: Path) -> None:
    args = _args(tmp_path)
    config = _config(tmp_path, "padded-adamw-writing")
    _schedule(config)
    checkpoint = config.output_dir / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 10}', encoding="utf-8")

    state, events = _process_cycle(
        [config],
        args,
        _new_state({"sha256": "test"}, 5),
        api=object(),
        backup=lambda *_args, **_kwargs: pytest.fail("partial checkpoint must not upload"),
    )

    assert events == []
    assert state["cycle_failures"] == 0
    assert (
        state["runs"][config.run_id]["checkpoints"]["10"]["status"] == "waiting_for_checkpoint_seal"
    )


def test_final_checkpoint_gives_full_run_controller_priority(tmp_path: Path) -> None:
    args = _args(tmp_path)
    config = _config(tmp_path, "padded-adamw-final")
    _schedule(config)
    checkpoint = _checkpoint(config, 50)
    newest = max(item.stat().st_mtime for item in checkpoint.rglob("*") if item.is_file())

    state, _ = _process_cycle(
        [config],
        args,
        _new_state({"sha256": "test"}, 5),
        api=object(),
        backup=lambda *_args, **_kwargs: pytest.fail("final checkpoint is still in grace"),
        now_epoch=newest + 30,
    )
    assert (
        state["runs"][config.run_id]["checkpoints"]["50"]["status"]
        == "waiting_for_full_run_backup_grace"
    )

    args.completion_ledger.write_text(
        json.dumps({"active_run": config.run_id, "backups": {}}), encoding="utf-8"
    )
    state, _ = _process_cycle(
        [config],
        args,
        state,
        api=object(),
        backup=lambda *_args, **_kwargs: pytest.fail("full-run backup is active"),
        now_epoch=newest + 300,
    )
    assert (
        state["runs"][config.run_id]["checkpoints"]["50"]["status"]
        == "waiting_for_active_full_run_backup"
    )


def test_invalid_existing_receipt_fails_closed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    config = _config(tmp_path, "padded-adamw-invalid")
    _schedule(config)
    _checkpoint(config, 10)
    args.receipt_root.mkdir()
    (args.receipt_root / f"{config.run_id}-checkpoint-10.json").write_text(
        "not-json", encoding="utf-8"
    )

    state, events = _process_cycle(
        [config],
        args,
        _new_state({"sha256": "test"}, 5),
        api=object(),
        backup=lambda *_args, **_kwargs: pytest.fail("invalid receipt must be reviewed"),
    )

    assert events == []
    assert state["cycle_failures"] == 1
    assert state["runs"][config.run_id]["checkpoints"]["10"]["status"] == "invalid_receipt"


def test_inventory_validation_rejects_malformed_counts() -> None:
    assert not _inventory_is_complete(
        {"inventory": {"complete": True, "local_files": None}}, require_digest=False
    )


def test_supervisor_lease_rejects_duplicate_holder(tmp_path: Path) -> None:
    lease = tmp_path / "supervisor.lease"
    with _exclusive_lease(lease):
        with pytest.raises(RuntimeError, match="already active"):
            with _exclusive_lease(lease):
                pytest.fail("duplicate lease unexpectedly acquired")


def test_sealed_backup_supervisor_has_public_cli_entrypoint() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    scripts = pyproject.split("[project.scripts]", 1)[1].split("\n[", 1)[0]

    assert (
        "embed-optim-supervise-sealed-checkpoint-backup = "
        '"embed_optim.sealed_checkpoint_supervisor:main"' in scripts
    )
