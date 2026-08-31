from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from embed_optim import evaluate_matrix, short_branch_evaluation, spectral_transplant_matrix
from embed_optim.gpu_lease import (
    acquire_gpu_lease,
    evaluation_gpu_tokens,
    parse_gpu_tokens,
    validate_disjoint_gpu_pools,
)


def test_gpu_tokens_are_canonical_unique_and_disjoint():
    assert parse_gpu_tokens("0,1,2,3", expected_count=4) == ("0", "1", "2", "3")
    assert validate_disjoint_gpu_pools("0,1,2,3", "4,5,6,7") == {
        "a": ("0", "1", "2", "3"),
        "b": ("4", "5", "6", "7"),
    }
    for value in ("0, 1,2,3", "0,1,1,2", "00,1,2,3", "0,1,2,"):
        with pytest.raises(ValueError):
            parse_gpu_tokens(value)
    with pytest.raises(ValueError, match="disjoint"):
        validate_disjoint_gpu_pools("0,1,2,3", "3,4,5,6")


def test_dense_lease_does_not_claim_the_unused_second_pool():
    assert evaluation_gpu_tokens(
        has_dense=True,
        has_late=False,
        gpus_a="4,5,6,7",
        gpus_b="0,1,2,3",
    ) == ("4", "5", "6", "7")
    assert evaluation_gpu_tokens(
        has_dense=True,
        has_late=True,
        gpus_a="4,5,6,7",
        gpus_b="0,1,2,3",
    ) == tuple(str(index) for index in range(8))


def test_gpu_lease_times_out_exclusively_and_records_both_ledgers(tmp_path):
    outer_ledger = tmp_path / "outer.json"
    inner_ledger = tmp_path / "inner.json"
    with acquire_gpu_lease(
        ("0",),
        lock_dir=tmp_path / "locks",
        timeout_seconds=1,
        purpose="outer",
        ledger_path=outer_ledger,
        poll_seconds=0.001,
    ):
        assert json.loads(outer_ledger.read_text())["status"] == "acquired"
        with pytest.raises(TimeoutError):
            with acquire_gpu_lease(
                ("0",),
                lock_dir=tmp_path / "locks",
                timeout_seconds=0.01,
                purpose="inner",
                ledger_path=inner_ledger,
                poll_seconds=0.001,
            ):
                raise AssertionError("exclusive token was acquired twice")
        assert json.loads(inner_ledger.read_text())["status"] == "timeout"
    assert json.loads(outer_ledger.read_text())["status"] == "released"


def test_evaluation_input_manifest_rejects_checkpoint_content_changes(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "completed.json").write_text(
        json.dumps({"model_family": "dense", "run_id": "adamw-selected"})
    )
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"first payload")
    results = tmp_path / "results"
    results.mkdir()

    evaluate_matrix._record_evaluation_inputs(results, {"dense": [checkpoint]})
    manifest = json.loads((results / "evaluation_inputs.json").read_text())
    identity = manifest["checkpoints"][str(checkpoint.resolve())]
    assert identity["run_id"] == "adamw-selected"
    assert len(identity["model_sha256"]) == 64
    audit = evaluate_matrix.audit_evaluation_inputs(results, [checkpoint])
    assert audit["checkpoints"] == 1
    assert audit["bytes"] == (results / "evaluation_inputs.json").stat().st_size
    assert len(audit["sha256"]) == 64

    manifest["checkpoints"][str(tmp_path / "stale-checkpoint")] = identity
    (results / "evaluation_inputs.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact checkpoints"):
        evaluate_matrix.audit_evaluation_inputs(results, [checkpoint])
    del manifest["checkpoints"][str(tmp_path / "stale-checkpoint")]
    (results / "evaluation_inputs.json").write_text(json.dumps(manifest))

    weights.write_bytes(b"different payload")
    with pytest.raises(ValueError, match="content differs"):
        evaluate_matrix.audit_evaluation_inputs(results, [checkpoint])
    with pytest.raises(RuntimeError, match="content changed"):
        evaluate_matrix._record_evaluation_inputs(results, {"dense": [checkpoint]})


def test_evaluation_input_manifest_will_not_adopt_unbound_cached_results(tmp_path):
    checkpoint = tmp_path / "run" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "completed.json").write_text(
        json.dumps({"model_family": "dense", "run_id": "adamw-selected"})
    )
    (checkpoint / "model.safetensors").write_bytes(b"payload")
    stale = tmp_path / "results" / "dense" / "model" / "Task.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")

    with pytest.raises(RuntimeError, match="without a pre-existing"):
        evaluate_matrix._record_evaluation_inputs(tmp_path / "results", {"dense": [checkpoint]})


def test_formal_result_file_audit_rejects_unselected_json(tmp_path):
    root = tmp_path / "results"
    selected = root / "run-a" / "SciFactDecontaminated.json"
    selected.parent.mkdir(parents=True)
    selected.write_text("{}")
    rows = [{"result_path": str(selected)}]

    audit = evaluate_matrix.audit_evaluation_result_files(root, rows)
    assert audit == {"root": str(root.resolve()), "files": 1}

    unexpected = root / "stale-run" / "NFCorpusDecontaminated.json"
    unexpected.parent.mkdir()
    unexpected.write_text("{}")
    with pytest.raises(ValueError, match="result-file coverage"):
        evaluate_matrix.audit_evaluation_result_files(root, rows)


def test_early_lease_blocks_later_short_branch_and_spectral_gpu_steps(tmp_path, monkeypatch):
    """Canonical confirmation cannot fall through into an unleased GPU phase."""

    lock_dir = tmp_path / "locks"
    monkeypatch.chdir(tmp_path)
    short_started = []
    spectral_started = []
    monkeypatch.setattr(
        short_branch_evaluation,
        "_run_gpu_tiers",
        lambda *args: short_started.append(True) or 0,
    )
    monkeypatch.setattr(
        spectral_transplant_matrix,
        "spectral_transplant_job_complete",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        spectral_transplant_matrix,
        "common_state_job_complete",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        spectral_transplant_matrix,
        "_execute_pending_jobs",
        lambda *args: spectral_started.append(True) or 0,
    )
    short_args = SimpleNamespace(
        gpus="0,1,2,3,4,5,6,7",
        gpu_lock_dir=lock_dir,
        gpu_lock_timeout_seconds=0.01,
        tier="both",
    )
    spectral_args = SimpleNamespace(
        spectral_spec=tmp_path / "spectral.json",
        common_state_spec=tmp_path / "common.json",
        verify_hashes=False,
        audit_only=False,
        dry_run=False,
        gpus="0,1,2,3,4,5,6,7",
        gpu_lock_dir=lock_dir,
        gpu_lock_timeout_seconds=0.01,
        log_dir=tmp_path / "spectral-logs",
        families=["dense"],
    )
    fake_job = SimpleNamespace(label="dense/fake", common_state=object())

    with acquire_gpu_lease(
        ("0",),
        lock_dir=lock_dir,
        timeout_seconds=1,
        purpose="early-confirmatory-tail",
    ):
        with pytest.raises(TimeoutError):
            short_branch_evaluation._run_gpu_tiers_with_lease(short_args, ("dense",), [], [])
        with pytest.raises(TimeoutError):
            spectral_transplant_matrix.run_matrix([fake_job], spectral_args)

    assert short_started == []
    assert spectral_started == []
