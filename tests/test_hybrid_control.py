from pathlib import Path

import pytest
import torch

from embed_optim import hybrid_control
from embed_optim.aggregate import _linear_schedule_multiplier
from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.decontamination import DECONTAMINATED_TASK_NAMES
from embed_optim.hybrid_control import (
    _hybrid_optimizer_contract_problem,
    _validate_protocol,
    summarize_final_evaluations,
)


def test_frozen_hybrid_protocol_rehashes_tracked_sources_in_distribution():
    root = Path(__file__).parents[1]
    protocol = _validate_protocol(
        root / "configs" / "hybrid_adamw_control.json", require_external=False
    )

    assert protocol["selection"]["expected_runs"] == 8
    assert protocol["selection"]["expected_beir_units"] == 112
    assert protocol["observed_before_freeze"]["strict_beir"]["valid_units"] == 140


def test_hybrid_optimizer_contract_requires_three_adamw_groups():
    config = RunConfig(
        run_id="hybrid-test",
        model_family="dense",
        optimizer=OptimizerConfig(name="hybrid_adamw", lr=1e-5, aux_lr=3e-6),
        model_name="model",
        dataset_path="dataset",
    )
    step, final_step = 40, 100
    multiplier = _linear_schedule_multiplier(step, final_step, config.warmup_ratio)

    def state():
        return {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.zeros(3),
            "exp_avg_sq": torch.zeros(3),
        }

    optimizer = {
        "state": {index: state() for index in range(3)},
        "param_groups": [
            {
                "params": [0],
                "algorithm": "adamw",
                "lr": config.optimizer.lr * multiplier,
                "betas": (config.optimizer.beta1, config.optimizer.beta2),
                "eps": config.optimizer.eps,
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": [1],
                "algorithm": "adamw",
                "lr": config.optimizer.aux_lr * multiplier,
                "betas": (config.optimizer.aux_beta1, config.optimizer.aux_beta2),
                "eps": config.optimizer.aux_eps,
                "weight_decay": config.optimizer.weight_decay,
            },
            {
                "params": [2],
                "algorithm": "adamw",
                "lr": config.optimizer.aux_lr * multiplier,
                "betas": (config.optimizer.aux_beta1, config.optimizer.aux_beta2),
                "eps": config.optimizer.aux_eps,
                "weight_decay": 0.0,
            },
        ],
    }

    assert _hybrid_optimizer_contract_problem(optimizer, config, step, final_step) is None
    optimizer["param_groups"][0]["algorithm"] = "hybrid_adamw"
    assert "algorithm is not AdamW" in _hybrid_optimizer_contract_problem(
        optimizer, config, step, final_step
    )


def _evaluation_rows(optimizer: str):
    rows = []
    for family_index, family in enumerate(("dense", "late")):
        for rate_index, learning_rate in enumerate((1e-6, 3e-6, 1e-5, 3e-5)):
            for task_index, task in enumerate(DECONTAMINATED_TASK_NAMES):
                native = 0.2 + family_index * 0.01 + rate_index * 0.001 + task_index * 0.0001
                rows.append(
                    {
                        "model_family": family,
                        "optimizer": optimizer,
                        "learning_rate": learning_rate,
                        "stage": 5,
                        "task": task,
                        "ndcg_at_10": native + (0.002 if optimizer == "hybrid_adamw" else 0.0),
                        "result_path": f"/{optimizer}/{family}/{learning_rate}/{task}.json",
                    }
                )
    return rows


def test_final_hybrid_summary_is_paired_over_all_112_units():
    contrasts, summaries = summarize_final_evaluations(
        _evaluation_rows("adamw"), _evaluation_rows("hybrid_adamw")
    )

    assert len(contrasts) == 112
    assert len(summaries) == 8
    assert all(row["hybrid_minus_adamw"] == pytest.approx(0.002) for row in contrasts)
    assert all(row["tasks"] == 14 and row["hybrid_task_wins"] == 14 for row in summaries)

    with pytest.raises(ValueError, match="coverage differs"):
        summarize_final_evaluations(
            _evaluation_rows("adamw"), _evaluation_rows("hybrid_adamw")[:-1]
        )


def test_hybrid_formal_artifacts_require_exact_stage5_provenance(tmp_path, monkeypatch):
    configs = [
        RunConfig(
            run_id="hybrid-a",
            model_family="dense",
            optimizer=OptimizerConfig(name="hybrid_adamw", lr=1e-5),
            model_name="model",
            dataset_path="dataset",
        )
    ]
    checkpoint = tmp_path / "hybrid-a" / "checkpoint-5"
    rows = [{"result_path": str(tmp_path / "SciFactDecontaminated.json")}]
    calls = []
    monkeypatch.setattr(
        hybrid_control,
        "checkpoint_paths",
        lambda config, stages: [checkpoint],
    )
    monkeypatch.setattr(
        hybrid_control,
        "audit_evaluation_artifacts",
        lambda root, checkpoints, selected_rows: (
            calls.append((root, checkpoints, selected_rows))
            or {"input_manifest": {"sha256": "a" * 64}, "result_files": {"files": 1}}
        ),
    )

    audit = hybrid_control.audit_hybrid_formal_evaluation_artifacts(
        tmp_path,
        configs,
        rows,
    )

    assert audit["input_manifest"]["sha256"] == "a" * 64
    assert calls == [(tmp_path, [checkpoint], rows)]

    monkeypatch.setattr(
        hybrid_control,
        "audit_evaluation_artifacts",
        lambda *args: (_ for _ in ()).throw(ValueError("unexpected formal result file")),
    )
    with pytest.raises(ValueError, match="unexpected formal result file"):
        hybrid_control.audit_hybrid_formal_evaluation_artifacts(tmp_path, configs, rows)
