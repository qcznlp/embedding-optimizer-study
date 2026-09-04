import json
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import embed_optim.matrix_handoff_guard as guard


def _protocol() -> dict:
    return {
        "matrix": "configs/dense_no_packing_retrain.yaml",
        "current_run_ids": ["current-a", "current-b"],
        "successor_run_ids": ["next-a", "next-b"],
        "runtime": {
            "python": "/usr/bin/python3",
            "gpus_a": "0,1,2,3",
            "gpus_b": "4,5,6,7",
            "port_a": 29610,
            "port_b": 29620,
            "training_log_dir": "logs/training",
            "recovery_state_file": "logs/training/recovery.json",
            "guard_poll_seconds": 10,
            "successor_grace_seconds": 30,
            "supervisor_poll_seconds": 30,
            "restart_delay_seconds": 30,
        },
    }


def _args(root: Path) -> Namespace:
    return Namespace(
        workdir=root,
        protocol=root / "protocol.json",
        log_dir=root / "guard",
    )


def _configs(root: Path) -> dict[str, SimpleNamespace]:
    return {
        run_id: SimpleNamespace(
            run_id=run_id,
            model_family="dense",
            output_dir=root / "outputs" / run_id,
        )
        for run_id in ("current-a", "current-b", "next-a", "next-b")
    }


def test_guard_yields_when_existing_matrix_creates_successor_artifact(monkeypatch, tmp_path):
    protocol = _protocol()
    configs = _configs(tmp_path)
    now = [0.0]
    monkeypatch.setattr(guard, "_load_protocol", lambda *_: protocol)
    monkeypatch.setattr(guard, "resolve_matrix_path", lambda path: path)
    monkeypatch.setattr(guard, "_selected_configs", lambda _: configs)
    monkeypatch.setattr(
        guard,
        "_run_is_complete",
        lambda config: config.run_id.startswith("current") and now[0] >= 10,
    )
    monkeypatch.setattr(
        guard,
        "_successor_artifacts",
        lambda config, _: [f"log-{config.run_id}"] if now[0] >= 20 else [],
    )

    result = guard.run_guard(
        _args(tmp_path),
        run_command=lambda *_args, **_kwargs: pytest.fail("takeover must not run"),
        sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
        clock=lambda: now[0],
    )

    assert result == 0
    payload = json.loads((tmp_path / "guard/state.json").read_text())
    assert payload["status"] == "yielded_to_existing_matrix"
    assert payload["takeover_launched"] is False


def test_guard_takes_over_only_after_full_absence_grace(monkeypatch, tmp_path):
    protocol = _protocol()
    configs = _configs(tmp_path)
    now = [0.0]
    takeover = [False]
    monkeypatch.setattr(guard, "_load_protocol", lambda *_: protocol)
    monkeypatch.setattr(guard, "resolve_matrix_path", lambda path: path)
    monkeypatch.setattr(guard, "_selected_configs", lambda _: configs)
    monkeypatch.setattr(
        guard,
        "_run_is_complete",
        lambda config: config.run_id.startswith("current") or takeover[0],
    )
    monkeypatch.setattr(guard, "_successor_artifacts", lambda *_: [])
    monkeypatch.setattr(guard, "_supervisor_command", lambda *_: ("python", "supervisor"))

    def run(command, **kwargs):
        assert command == ("python", "supervisor")
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is False
        takeover[0] = True
        return subprocess.CompletedProcess(command, 0)

    result = guard.run_guard(
        _args(tmp_path),
        run_command=run,
        sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
        clock=lambda: now[0],
    )

    assert result == 0
    assert now[0] == 30
    payload = json.loads((tmp_path / "guard/state.json").read_text())
    assert payload["status"] == "complete"
    assert payload["takeover_launched"] is True


def test_checked_in_guard_contract_is_source_bound_and_process_free():
    root = Path.cwd()
    protocol = guard._load_protocol(
        root / "configs/dense_no_packing_matrix_handoff_guard.json", root
    )
    command = guard._supervisor_command(protocol, root)

    assert protocol["status"] == "locked_before_current_pair_completion"
    assert protocol["process_inspection"] is False
    assert protocol["signals_processes"] is False
    assert protocol["current_run_ids"] == ["padded-adamw-1e-5", "padded-adamw-3e-5"]
    assert protocol["successor_run_ids"] == ["padded-muon-1e-4", "padded-muon-3e-4"]
    assert "gpu.py" not in " ".join(command)
    assert command[command.index("--max-launches") + 1] == "0"
