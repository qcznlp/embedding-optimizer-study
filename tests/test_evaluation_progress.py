from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.aggregate import DECONTAMINATED_TASK_NAMES
from embed_optim.evaluation_progress import evaluation_progress, watch_progress, write_progress


def _configs():
    return [
        SimpleNamespace(model_family="dense", run_id="adamw-lr1e-6"),
        SimpleNamespace(model_family="late", run_id="muon-lr1e-4"),
    ]


def _row(config, stage, task):
    return {
        "model_family": config.model_family,
        "run_id": config.run_id,
        "stage": stage,
        "task": task,
    }


def test_evaluation_progress_counts_only_collected_rows(monkeypatch):
    configs = _configs()
    rows = [
        _row(configs[0], 1, DECONTAMINATED_TASK_NAMES[0]),
        _row(configs[1], 5, DECONTAMINATED_TASK_NAMES[-1]),
    ]
    monkeypatch.setattr("embed_optim.evaluation_progress.load_matrix", lambda matrix: configs)
    monkeypatch.setattr(
        "embed_optim.evaluation_progress.collect_evaluations",
        lambda results_root, observed_configs: rows,
    )

    snapshot = evaluation_progress("matrix.yaml", "results")

    assert snapshot["valid_units"] == 2
    assert snapshot["expected_units"] == 2 * 5 * len(DECONTAMINATED_TASK_NAMES)
    assert snapshot["families"] == {"dense": 1, "late": 1}
    assert snapshot["runs"] == {"dense/adamw-lr1e-6": 1, "late/muon-lr1e-4": 1}
    assert not snapshot["complete"]
    assert snapshot["error"] is None


def test_write_progress_is_atomic_json(tmp_path: Path, monkeypatch):
    payload = {
        "schema_version": 1,
        "audited_at": "fixed",
        "complete": True,
        "error": None,
    }
    monkeypatch.setattr(
        "embed_optim.evaluation_progress.evaluation_progress",
        lambda matrix, results_root: payload,
    )
    output = tmp_path / "nested" / "progress.json"

    observed = write_progress("matrix.yaml", "results", output)

    assert observed == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not list(output.parent.glob(".*.tmp.*"))


def test_watcher_retries_collector_errors_and_stops_when_complete(tmp_path: Path, monkeypatch):
    attempts = iter(
        [
            RuntimeError("transient audit failure"),
            {"schema_version": 1, "complete": False, "valid_units": 3, "error": None},
            {"schema_version": 1, "complete": True, "valid_units": 4, "error": None},
        ]
    )

    def fake_write(matrix, results_root, output):
        del matrix, results_root, output
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("embed_optim.evaluation_progress.write_progress", fake_write)
    sleeps = []
    output = tmp_path / "progress.json"

    snapshot = watch_progress(
        "matrix.yaml", "results", output, interval_seconds=7, sleeper=sleeps.append
    )

    assert snapshot["complete"]
    assert sleeps == [7, 7]
    error_snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert error_snapshot["error"] == "RuntimeError: transient audit failure"


def test_watcher_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="positive"):
        watch_progress("matrix.yaml", "results", "progress.json", interval_seconds=0)
