from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embed_optim import gradient_probe
from embed_optim import state_operator_factorial as factorial
from embed_optim import state_operator_factorial_evaluation as factorial_evaluation
from embed_optim import state_operator_factorial_probe as factorial_probe
from embed_optim import state_operator_factorial_summary as factorial_summary
from embed_optim.config import OptimizerConfig, RunConfig, load_matrix
from embed_optim.geometry import _sha256


def _write_calibration(root: Path, state: str, adamw: float, muon: float) -> None:
    direction_root = root / state / "directions"
    direction_root.mkdir(parents=True)
    metrics = direction_root / "metrics.jsonl"
    rows = []
    for index in range(factorial.EXPECTED_HIDDEN_TENSORS):
        rows.append(
            {
                "tensor": f"tensor.{index}",
                "parameters": (
                    factorial.EXPECTED_HIDDEN_PARAMETERS - factorial.EXPECTED_HIDDEN_TENSORS + 1
                    if index == 0
                    else 1
                ),
                "weight_frobenius_norm": 2.0,
                "algorithms": {
                    "adamw": {"frobenius_norm": adamw},
                    "muon": {"frobenius_norm": muon},
                },
            }
        )
    metrics.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "state": state,
        "tensors": factorial.EXPECTED_HIDDEN_TENSORS,
        "parameters": factorial.EXPECTED_HIDDEN_PARAMETERS,
        "analysis_config": {"weight_decay_included": False},
        "output": {
            "path": metrics.name,
            "bytes": metrics.stat().st_size,
            "sha256": _sha256(metrics),
        },
    }
    (direction_root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_factorial_jobs_expand_to_exact_crossed_design() -> None:
    jobs = factorial.factorial_jobs()

    assert len(jobs) == 12
    assert len({(job.state, job.operator, job.seed) for job in jobs}) == 12
    assert {job.state for job in jobs} == {"adamw_state", "muon_state"}
    assert {job.operator for job in jobs} == {"adamw", "muon"}
    assert {job.seed for job in jobs} == {314159, 271828, 161803}
    assert all("__" in job.run_id for job in jobs)
    assert len({(job.seed, job.run_id) for job in jobs}) == 12


def test_padded_dense_loader_records_observed_execution(tmp_path: Path, monkeypatch) -> None:
    first = SimpleNamespace(can_flatten_inputs=True)
    model = SimpleNamespace(_first_module=lambda: first)
    monkeypatch.setattr(gradient_probe, "_load_model", lambda *_args, **_kwargs: model)
    receipt_path = tmp_path / "receipt.json"
    receipt = {"schema_version": 1, "status": "in_progress", "request": {}}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with factorial._padded_dense_loader(receipt_path, receipt):
        observed = gradient_probe._load_model("dense", tmp_path)

    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert observed is model
    assert first.can_flatten_inputs is False
    assert saved["status"] == "model_verified"
    assert saved["observed_input_execution"] == {
        "mode": "independently_padded",
        "sentence_transformers_can_flatten_inputs": False,
    }


def test_scale_calibration_uses_global_hidden_frobenius_ratio(tmp_path: Path) -> None:
    _write_calibration(tmp_path, "adamw_state", adamw=1.0, muon=4.0)

    identity, ratios = factorial._load_calibration_metrics(
        "adamw_state", tmp_path, verify_provenance=False
    )

    assert identity["metrics"]["sha256"]
    assert math.isclose(ratios["adamw"], 0.5)
    assert math.isclose(ratios["muon"], 2.0)


def test_generate_and_audit_six_two_run_matrices(tmp_path: Path, monkeypatch) -> None:
    calibration = tmp_path / "calibration"
    matrices = tmp_path / "matrices"
    _write_calibration(calibration, "adamw_state", adamw=1.0, muon=2.0)
    _write_calibration(calibration, "muon_state", adamw=2.0, muon=4.0)
    original_load = factorial._load_calibration_metrics

    def load_without_provenance(state, root, **_kwargs):
        return original_load(state, root, verify_provenance=False)

    monkeypatch.setattr(factorial, "_load_calibration_metrics", load_without_provenance)

    manifest = factorial.generate_factorial_matrices(
        matrix_root=matrices,
        calibration_root=calibration,
        deep_data_audit=False,
    )
    audit = factorial.audit_factorial_matrices(
        matrix_root=matrices,
        calibration_root=calibration,
    )

    assert manifest["expected_matrices"] == audit["matrices"] == 6
    assert manifest["expected_runs"] == audit["runs"] == 12
    assert len(manifest["matrices"]) == 6
    for item in manifest["matrices"]:
        runs = load_matrix(item["path"])
        assert len(runs) == 2
        assert all(run.dense_can_flatten_inputs is False for run in runs)
        assert {run.optimizer.name for run in runs} == {"hybrid_adamw", "muon"}
        assert all(run.model_revision is None for run in runs)


def test_gradient_export_rejects_manifest_without_padded_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint-2345"
    checkpoint.mkdir()
    protocol_path = tmp_path / "scientific.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    protocol = {
        "source_states": {
            "states": [
                {
                    "label": "adamw_state",
                    "run_id": "source",
                    "checkpoint": str(checkpoint),
                }
            ]
        },
        "scale_calibration": {"gradient_history_steps": 8},
    }
    monkeypatch.setattr(
        factorial,
        "load_factorial_protocol",
        lambda _path: (protocol_path, protocol),
    )
    output = tmp_path / "calibration"
    gradient_root = output / "adamw_state" / "gradients"
    gradient_root.mkdir(parents=True)
    (gradient_root / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="untagged gradient artifact"):
        factorial.export_padded_gradient_calibration(
            "adamw_state", protocol_path, calibration_root=output
        )


def test_full_beir_cell_matrix_names_are_disjoint(tmp_path: Path) -> None:
    assert factorial_evaluation.matrix_path("adamw_state", 314159, tmp_path) == (
        tmp_path / "adamw_state-seed314159.yaml"
    )
    assert factorial_evaluation.matrix_path("muon_state", 314159, tmp_path) != (
        tmp_path / "adamw_state-seed314159.yaml"
    )


def test_probe_summary_contains_every_predeclared_metric() -> None:
    scores = torch.tensor(
        [
            [0.8, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4],
            [0.1, 0.5, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5],
        ],
        dtype=torch.float32,
    )
    reference = torch.tensor(
        [
            [0.7, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5],
            [0.6, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4],
        ],
        dtype=torch.float32,
    )

    result = factorial_probe._summary(scores, reference)

    assert set(result) == {
        "samples",
        "contrastive_loss_mean",
        "positive_margin_mean",
        "positive_margin_p05",
        "mean_reciprocal_rank",
        "top1_accuracy",
        "pretrained_top1_agreement",
    }
    assert result["samples"] == 2
    assert result["top1_accuracy"] == pytest.approx(0.5)
    assert result["mean_reciprocal_rank"] == pytest.approx(0.75)
    assert result["pretrained_top1_agreement"] == pytest.approx(0.5)
    assert all(math.isfinite(float(value)) for key, value in result.items() if key != "samples")


def test_padded_probe_loader_records_observed_execution(tmp_path: Path, monkeypatch) -> None:
    first = SimpleNamespace(can_flatten_inputs=True)
    model = SimpleNamespace(_first_module=lambda: first)
    monkeypatch.setattr(factorial_probe.probe_export, "_load_model", lambda *_a, **_k: model)
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "schema_version": 1,
        "status": "in_progress",
        "request": {
            "input_execution": {
                "mode": "independently_padded",
                "sentence_transformers_can_flatten_inputs": False,
            }
        },
    }

    with factorial_probe._padded_probe_loader(receipt_path, receipt):
        observed = factorial_probe.probe_export._load_model("dense", tmp_path)

    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert observed is model
    assert first.can_flatten_inputs is False
    assert saved["status"] == "model_verified"
    assert saved["observed_input_execution"] == receipt["request"]["input_execution"]


def test_probe_job_builder_covers_reference_and_sixty_checkpoints(tmp_path: Path) -> None:
    _, protocol = factorial.load_factorial_protocol()
    configs = []
    for seed in protocol["branch_data"]["order_seeds"]:
        for state in ("adamw_state", "muon_state"):
            for operator in ("adamw", "muon"):
                name = "hybrid_adamw" if operator == "adamw" else "muon"
                config = RunConfig(
                    run_id=f"{state.replace('_', '-')}__{operator}-reset",
                    model_family="dense",
                    optimizer=OptimizerConfig(name=name, lr=1e-4),
                    model_name=str(tmp_path / "source"),
                    dataset_path="data/short-branch-50k-seed20260826",
                    output_root=str(tmp_path / state / f"seed{seed}"),
                    seed=seed,
                    dense_can_flatten_inputs=False,
                )
                config.output_dir.mkdir(parents=True)
                steps = protocol["factorial_design"]["training"]["expected_checkpoint_steps"]
                (config.output_dir / "checkpoint_schedule.json").write_text(
                    json.dumps(
                        {
                            "steps": steps,
                            "fractions": list(config.checkpoint_fractions),
                        }
                    ),
                    encoding="utf-8",
                )
                for step in steps:
                    (config.output_dir / f"checkpoint-{step}").mkdir()
                configs.append(config)
    reference = tmp_path / "reference"
    reference.mkdir()

    jobs = factorial_probe.build_probe_jobs(
        configs,
        protocol,
        reference,
        tmp_path / "probe-results",
        ("a" * 64, "b" * 64),
    )

    assert len(jobs) == 61
    assert sum(job.kind == "reference" for job in jobs) == 1
    assert sum(job.kind == "checkpoint" for job in jobs) == 60
    assert len({job.label for job in jobs}) == 61
    assert all(job.family == "dense" for job in jobs)
    assert np.unique([job.probe_manifest_sha256 for job in jobs]).tolist() == ["a" * 64]


def test_factorial_estimands_separate_state_operator_and_interaction() -> None:
    rows = []
    values = {
        ("adamw_state", "adamw"): 0.50,
        ("adamw_state", "muon"): 0.51,
        ("muon_state", "adamw"): 0.52,
        ("muon_state", "muon"): 0.55,
    }
    for seed in (1, 2, 3):
        for task_index, task in enumerate(factorial_summary.DECONTAMINATED_TASK_NAMES):
            for (state, operator), value in values.items():
                rows.append(
                    {
                        "state": state,
                        "operator": operator,
                        "seed": seed,
                        "task": task,
                        "ndcg_at_10": value + task_index * 1e-4,
                    }
                )

    effects = factorial_summary._effect_rows(rows)
    first = {
        row["estimand"]: row["contrast_ndcg_at_10"]
        for row in effects
        if row["seed"] == 1 and row["task"] == "ArguAna"
    }

    assert first["weight_state_effect"] == pytest.approx(0.03)
    assert first["operator_effect"] == pytest.approx(0.02)
    assert first["state_operator_interaction"] == pytest.approx(0.02)
    assert len(effects) == 126


def test_two_way_bootstrap_uses_locked_decision_rule() -> None:
    positive = factorial_summary.two_way_cluster_bootstrap(np.ones((3, 14)))
    negative = factorial_summary.two_way_cluster_bootstrap(-np.ones((3, 14)))
    mixed = factorial_summary.two_way_cluster_bootstrap(
        np.tile(np.asarray([-1.0, 0.0, 1.0])[:, None], (1, 14))
    )

    assert positive["decision"] == "supported_positive"
    assert negative["decision"] == "supported_negative"
    assert mixed["decision"] == "inconclusive"
    assert positive["bootstrap_samples"] == 100_000
    assert positive["bootstrap_seed"] == 20_260_904
