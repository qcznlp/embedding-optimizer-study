from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import dense_retrieval_dynamics_evaluation as dynamics
from embed_optim.decontamination import DECONTAMINATED_TASK_NAMES
from embed_optim.evaluate_matrix import _record_evaluation_inputs


def _configs(count: int, *, optimizer: str = "adamw"):
    return [
        SimpleNamespace(
            model_family="dense",
            run_id=f"run-{index}",
            optimizer=SimpleNamespace(name=optimizer, lr=float(index + 1)),
        )
        for index in range(count)
    ]


def _rows(configs):
    return [
        {
            "model_family": config.model_family,
            "run_id": config.run_id,
            "stage": stage,
            "task": task,
        }
        for config in configs
        for stage in dynamics.DYNAMICS_STAGES
        for task in DECONTAMINATED_TASK_NAMES
    ]


def test_extension_cardinality_is_exactly_728_units():
    assert dynamics.EXPECTED_UNITS == {"hybrid": 224, "confirmatory": 504}
    assert dynamics.TOTAL_EXPECTED_UNITS == 728


def test_dynamics_coverage_accepts_four_stages_and_rejects_stage_five():
    configs = _configs(4)
    rows = _rows(configs)

    assert dynamics.coverage_problems(rows, configs) == []

    rows[-1] = {**rows[-1], "stage": 5}
    problems = dynamics.coverage_problems(rows, configs)
    assert any("formal stage 5 leaked" in problem for problem in problems)
    assert any("missing dynamics identities" in problem for problem in problems)


def test_checked_in_contract_is_source_bound_and_isolates_formal_roots():
    contract = dynamics.load_dynamics_contract()

    assert contract.payload["evaluation"]["expected_additional_units"] == 728
    assert contract.payload["decision_timing"] == {
        "discovery_scores_visible": True,
        "hybrid_training_complete": True,
        "hybrid_beir_scores_visible": False,
        "confirmatory_terminal_runs_visible": False,
        "confirmatory_beir_scores_visible": False,
        "extension_rule_result_contingent": False,
        "reason": (
            "The extension restores the user-requested five-stage evaluation coverage; it does "
            "not select stages, tasks, runs, or claims from hybrid or confirmatory outcomes."
        ),
    }
    for suite in dynamics.SUITES:
        dynamics_root = contract.result_root(suite)
        formal_root = (
            contract.repository / contract.payload["suites"][suite]["formal_results_root"]
        ).resolve()
        assert dynamics_root != formal_root
        assert not dynamics._inside(dynamics_root, formal_root)
        assert not dynamics._inside(formal_root, dynamics_root)


def test_input_manifest_is_rehashed_before_dynamics_receipt(tmp_path):
    output = tmp_path / "outputs" / "dense" / "run"
    steps = [10, 20, 30, 40, 50]
    output.mkdir(parents=True)
    (output / "checkpoint_schedule.json").write_text(json.dumps({"steps": steps}))
    (output / "completed.json").write_text(json.dumps({"model_family": "dense", "run_id": "run"}))
    for step in steps:
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(f"weights-{step}".encode())
    config = SimpleNamespace(output_dir=output)
    results = tmp_path / "results"
    results.mkdir()
    _record_evaluation_inputs(
        results,
        {"dense": [output / f"checkpoint-{step}" for step in steps[:4]]},
    )

    audit = dynamics._audit_input_manifest(results, [config], repository=tmp_path)

    assert audit["checkpoints"] == 4
    assert len(audit["sha256"]) == 64
    (output / "checkpoint-20" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="content differs"):
        dynamics._audit_input_manifest(results, [config], repository=tmp_path)


def test_hybrid_runner_reuses_locked_evaluator_for_stages_one_to_four(tmp_path, monkeypatch):
    configs = _configs(4, optimizer="hybrid_adamw")
    audit = {"complete": True, "errors": [], "verified_runs": 4}
    contract = SimpleNamespace(
        source_path=lambda name: tmp_path / f"{name}.yaml",
        result_root=lambda suite: tmp_path / "results" / suite,
        log_root=lambda suite: tmp_path / "logs" / suite,
    )
    args = Namespace(
        gpus_a="0,1,2,3,4,5,6,7",
        gpus_b="4,5,6,7",
        worker_python="/usr/bin/python3",
        gpu_lock_dir=tmp_path / "leases",
        gpu_lock_timeout_seconds=100.0,
    )
    monkeypatch.setattr(dynamics, "_hybrid_configs", lambda _contract: configs)
    monkeypatch.setattr(dynamics, "audit_hybrid_training", lambda selected, families: audit)
    calls = []
    monkeypatch.setattr(
        dynamics,
        "run_evaluation_after_specialized_audit",
        lambda worker, training, *, label: calls.append((worker, training, label)) or 0,
    )

    dynamics._run_hybrid(args, contract)

    assert len(calls) == 1
    worker, observed_audit, label = calls[0]
    assert worker.stages == [1, 2, 3, 4]
    assert worker.tasks == list(DECONTAMINATED_TASK_NAMES)
    assert Path(worker.results_root) == (tmp_path / "results/hybrid").resolve()
    assert observed_audit is audit
    assert "stage-1--4" in label


def test_confirmatory_runner_keeps_each_seed_in_a_content_isolated_root(tmp_path, monkeypatch):
    seeds = (314159, 271828, 161803)
    matrices = {seed: tmp_path / f"seed{seed}.yaml" for seed in seeds}
    configs = _configs(3)
    contract = SimpleNamespace(
        source_path=lambda name: tmp_path / f"{name}.json",
        result_root=lambda suite: tmp_path / "results" / suite,
        log_root=lambda suite: tmp_path / "logs" / suite,
    )
    args = Namespace(
        gpus_a="0,1,2,3,4,5,6,7",
        gpus_b="4,5,6,7",
        worker_python="/usr/bin/python3",
        gpu_lock_dir=tmp_path / "leases",
        gpu_lock_timeout_seconds=100.0,
    )
    monkeypatch.setattr(
        dynamics,
        "_confirmatory_context",
        lambda _contract: (tmp_path / "protocol.json", {}, matrices, "manifest"),
    )
    monkeypatch.setattr(dynamics, "_confirmatory_configs", lambda _matrix: configs)
    monkeypatch.setattr(
        dynamics,
        "audit_confirmatory_training",
        lambda protocol, seed, selected, families: {
            "complete": True,
            "errors": [],
            "verified_runs": 3,
        },
    )
    calls = []
    monkeypatch.setattr(
        dynamics,
        "run_evaluation_after_specialized_audit",
        lambda worker, training, *, label: calls.append((worker, label)) or 0,
    )

    dynamics._run_confirmatory(args, contract)

    assert len(calls) == 3
    assert [Path(worker.results_root).name for worker, _ in calls] == [
        f"seed{seed}" for seed in seeds
    ]
    assert all(worker.stages == [1, 2, 3, 4] for worker, _ in calls)
    assert all("stage-1--4" in label for _, label in calls)


def test_full_audit_aggregates_224_plus_504_without_promoting_dynamics(tmp_path, monkeypatch):
    contract_file = tmp_path / "configs/dynamics.json"
    contract_file.parent.mkdir()
    contract_file.write_text("{}")
    contract = SimpleNamespace(
        path=contract_file,
        repository=tmp_path,
        source_path=lambda name: tmp_path / f"{name}.json",
        result_root=lambda suite: tmp_path / "results" / suite,
    )
    matrices = {seed: tmp_path / f"seed{seed}.yaml" for seed in (314159, 271828, 161803)}
    monkeypatch.setattr(dynamics, "load_dynamics_contract", lambda _path: contract)
    monkeypatch.setattr(
        dynamics,
        "resolve_scope",
        lambda families, amendment: (("dense",), {"active_scope": "dense"}),
    )
    monkeypatch.setattr(dynamics, "_hybrid_configs", lambda _contract: _configs(4))
    monkeypatch.setattr(
        dynamics,
        "_confirmatory_context",
        lambda _contract: (tmp_path / "protocol.json", {}, matrices, "matrix-sha"),
    )
    monkeypatch.setattr(dynamics, "_confirmatory_configs", lambda _matrix: _configs(3))

    def fake_root_audit(root, configs, **kwargs):
        units = len(configs) * 4 * 14
        return {
            "complete": True,
            "valid_units": units,
            "expected_units": units,
            "errors": [],
            "input_manifest": {"path": "inputs.json", "sha256": "a" * 64},
            "runtime_manifest": {"path": "runtime.json", "sha256": "b" * 64},
            "result_sources": [],
        }

    monkeypatch.setattr(dynamics, "_audit_result_root", fake_root_audit)

    receipt = dynamics.audit_dense_retrieval_dynamics(contract_file)

    assert receipt["complete"] is True
    assert receipt["valid_units"] == receipt["expected_units"] == 728
    assert receipt["suites"]["hybrid"]["valid_units"] == 224
    assert receipt["suites"]["confirmatory"]["valid_units"] == 504
    assert receipt["formal_inference_stage"] == 5
    assert receipt["formal_inference_uses_dynamics_rows"] is False


def test_cli_defaults_to_all_suites_and_does_not_offer_stage_five():
    args = dynamics.parse_args([])

    assert args.suite == "all"
    assert args.gpus_a == "0,1,2,3,4,5,6,7"
    assert dynamics.DYNAMICS_STAGES == (1, 2, 3, 4)
