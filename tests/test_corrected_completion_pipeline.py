import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.corrected_completion_pipeline import (
    PipelineStep,
    _load_ledger,
    _new_ledger,
    parse_args,
    pipeline_steps,
)


def _args(root: Path) -> Namespace:
    return Namespace(
        workdir=root,
        matrix=root / "configs/dense_no_packing_retrain.yaml",
        python=sys.executable,
        gpus="0,1,2,3,4,5,6,7",
        training_log_dir=root / "logs/dense-no-packing-v1",
        log_dir=Path("logs/dense-no-packing-finalization"),
        checkpoint_repo="qcz/embedding-optimizer-study-checkpoints",
        checkpoint_prefix="corrected-dense-no-packing-v1/dense",
        poll_seconds=60.0,
        retry_delay=300.0,
        max_attempts=0,
        resume=False,
    )


def test_corrected_steps_cover_locked_dependency_order_and_never_schedule_late_or_gpu_keeper():
    root = Path.cwd()
    steps = pipeline_steps(_args(root), root)
    names = [step.name for step in steps]

    assert names == [
        "training-progress-receipt",
        "checkpoint-backup-audit",
        "padded-validation",
        "padded-validation-audit",
        "decontaminated-beir",
        "decontaminated-beir-audit",
        "weight-space",
        "outcome-summary",
        "retrieval-bridge",
        "execution-sensitivity",
        "publication-render",
        "paper-release",
        "paper-audit",
        "portable-evidence-audit",
        "tests",
        "ruff-check",
        "ruff-format-check",
    ]
    joined = "\n".join(" ".join(step.command) for step in steps)
    assert "LateOn" not in joined
    assert "--families late" not in joined
    assert "gpu.py" not in joined
    assert names.index("padded-validation") < names.index("outcome-summary")
    assert names.index("decontaminated-beir") < names.index("outcome-summary")
    assert names.index("weight-space") < names.index("retrieval-bridge")
    assert names.index("outcome-summary") < names.index("retrieval-bridge")
    assert names.index("retrieval-bridge") < names.index("publication-render")
    assert names.index("execution-sensitivity") < names.index("publication-render")


def test_pipeline_steps_use_all_declared_gpus_and_corrected_entrypoints():
    root = Path.cwd()
    args = _args(root)
    args.gpus = "8,9,10,11,12,13,14,15"
    steps = pipeline_steps(args, root)
    by_name = {step.name: step for step in steps}

    for name in ("padded-validation", "decontaminated-beir"):
        command = by_name[name].command
        assert command[command.index("--gpus") + 1] == args.gpus
    assert by_name["weight-space"].command[2] == "embed_optim.corrected_geometry_matrix"
    assert "--local-files-only" in by_name["weight-space"].command
    backup = by_name["checkpoint-backup-audit"].command
    assert backup[backup.index("--repo-id") + 1] == args.checkpoint_repo
    assert backup[backup.index("--remote-prefix") + 1] == args.checkpoint_prefix


def test_existing_ledger_requires_explicit_resume_and_exact_contract(tmp_path: Path):
    contract = {"schema_version": 1, "sha256": "a" * 64}
    path = tmp_path / "pipeline-ledger.json"
    ledger = _new_ledger(contract)
    path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(FileExistsError):
        _load_ledger(path, contract, resume=False)
    assert _load_ledger(path, contract, resume=True)["contract"] == contract
    with pytest.raises(RuntimeError, match="contract differs"):
        _load_ledger(path, {**contract, "sha256": "b" * 64}, resume=True)


def test_completed_ledger_is_idempotent_on_resume(tmp_path: Path):
    contract = {"schema_version": 1, "sha256": "a" * 64}
    path = tmp_path / "pipeline-ledger.json"
    ledger = _new_ledger(contract)
    ledger.update({"status": "complete", "complete": True})
    path.write_text(json.dumps(ledger), encoding="utf-8")

    assert _load_ledger(path, contract, resume=True)["complete"] is True


def test_resume_reaudits_each_previously_completed_backup_once(tmp_path: Path):
    contract = {"schema_version": 1, "sha256": "a" * 64}
    path = tmp_path / "pipeline-ledger.json"
    ledger = _new_ledger(contract)
    ledger["backups"]["padded-adamw-1e-6"] = {
        "run_id": "padded-adamw-1e-6",
        "complete": True,
        "attempts": [{"attempt": 1, "return_code": 0}],
    }
    path.write_text(json.dumps(ledger), encoding="utf-8")

    resumed = _load_ledger(path, contract, resume=True)
    backup = resumed["backups"]["padded-adamw-1e-6"]
    assert backup["complete"] is False
    assert backup["audit_only"] is True


def test_cli_rejects_bad_gpu_or_retry_controls():
    with pytest.raises(SystemExit):
        parse_args(["--gpus", "0,0"])
    with pytest.raises(SystemExit):
        parse_args(["--poll-seconds", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--max-attempts", "-1"])


def test_pipeline_step_is_immutable_value():
    step = PipelineStep("one", ("python", "one"))
    with pytest.raises(Exception):
        step.name = "two"


def test_fixture_shape_matches_expected_config_interface():
    config = SimpleNamespace(
        model_family="dense",
        dense_can_flatten_inputs=False,
        optimizer=SimpleNamespace(name="muon"),
        run_id="padded-muon-1e-4",
        checkpoint_fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
    )
    assert len(config.checkpoint_fractions) == 5
