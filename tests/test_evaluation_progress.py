from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.aggregate import DECONTAMINATED_TASK_NAMES
from embed_optim.candidate_breadth_release import (
    RELEASE_STEP_NAMES,
    UPSTREAM_FINALIZATION_STEP_NAMES,
)
from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.dense_completion_pipeline import CORE_STEP_NAMES
from embed_optim.evaluation_progress import (
    _ledger_progress,
    evaluation_progress,
    study_evaluation_progress,
    watch_progress,
    watch_study_progress,
    write_progress,
)


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

    with pytest.raises(ValueError, match="positive"):
        watch_study_progress(".", "progress.json", interval_seconds=0)


def _dense_config(run_id: str, output_root: str) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=3e-5),
        model_name="dense",
        dataset_path="data",
        output_root=output_root,
    )


def _complete_rows(configs, stages):
    return [
        {
            "model_family": config.model_family,
            "run_id": config.run_id,
            "stage": stage,
            "task": task,
        }
        for config in configs
        for stage in stages
        for task in DECONTAMINATED_TASK_NAMES
    ]


def test_study_progress_uses_exact_1750_beir_units_and_separates_short_probes(
    tmp_path: Path, monkeypatch
):
    discovery = [_dense_config(f"discovery-{index}", "outputs/dense") for index in range(12)]
    hybrid = [_dense_config(f"hybrid-{index}", "outputs/hybrid") for index in range(4)]
    confirmatory = [
        _dense_config(name, "outputs/confirmatory") for name in ("adamw", "muon", "normuon")
    ]

    def fake_load_matrix(path):
        name = Path(path).name
        if name == "experiment.yaml":
            return discovery
        if name == "hybrid_adamw.yaml":
            return hybrid
        if name.startswith("seed"):
            return confirmatory
        raise AssertionError(path)

    def fake_collect(results_root, configs):
        path = str(results_root)
        if path.endswith("results/decontaminated-beir"):
            return _complete_rows(configs, range(1, 6))
        if path.endswith("results/hybrid-adamw-beir-dynamics"):
            return _complete_rows(configs, range(1, 5))
        if path.endswith("results/hybrid-adamw-beir"):
            return _complete_rows(configs, (5,))
        if "/results/confirmatory-beir/" in path:
            return _complete_rows(configs, (5,))
        if path.endswith("seed314159"):
            return _complete_rows(configs, range(1, 5))
        if path.endswith("seed271828"):
            return _complete_rows(configs, range(1, 5))[:8]
        if path.endswith("seed161803"):
            return []
        raise AssertionError(path)

    monkeypatch.setattr("embed_optim.evaluation_progress.load_matrix", fake_load_matrix)
    monkeypatch.setattr("embed_optim.evaluation_progress.collect_evaluations", fake_collect)

    snapshot = study_evaluation_progress(tmp_path)

    assert snapshot["beir"]["valid_units"] == 1422
    assert snapshot["beir"]["expected_units"] == 1750
    assert snapshot["beir"]["completion_fraction"] == pytest.approx(1422 / 1750)
    assert snapshot["beir"]["suites"]["discovery"]["expected_units"] == 840
    assert snapshot["beir"]["suites"]["hybrid"]["expected_units"] == 280
    assert snapshot["beir"]["suites"]["confirmatory"]["expected_units"] == 630
    assert snapshot["shared_start_probes"] == {
        "included_in_beir_total": False,
        "full_corpus_beir": False,
        "query_disjoint_checkpoint_jobs": 45,
        "unseen_probe_jobs": 46,
        "explanation": (
            "The shared-start mechanism branch uses two frozen functional probes and is not "
            "an additional 126-unit full-corpus BEIR panel."
        ),
    }
    assert not snapshot["complete"]
    assert all(item["status"] == "pending" for item in snapshot["controllers"].values())


@pytest.mark.parametrize(
    "names",
    (CORE_STEP_NAMES, UPSTREAM_FINALIZATION_STEP_NAMES, RELEASE_STEP_NAMES),
)
def test_ledger_progress_requires_exact_ordered_complete_contract(tmp_path: Path, names):
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "complete": False,
                "steps": [
                    {"name": names[0], "complete": True},
                    {"name": names[1], "complete": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    partial = _ledger_progress(path, names)
    assert partial["status"] == "running"
    assert partial["completed_steps"] == 1
    assert partial["current_step"] == names[1]

    path.write_text(
        json.dumps(
            {
                "complete": True,
                "steps": [{"name": name, "complete": True} for name in names],
            }
        ),
        encoding="utf-8",
    )
    complete = _ledger_progress(path, names)
    assert complete["status"] == "complete"
    assert complete["complete"]

    path.write_text(
        json.dumps(
            {
                "complete": True,
                "steps": [{"name": name, "complete": True} for name in reversed(names)],
            }
        ),
        encoding="utf-8",
    )
    invalid = _ledger_progress(path, names)
    assert invalid["status"] == "invalid"
    assert not invalid["complete"]
