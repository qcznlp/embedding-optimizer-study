import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.aggregate import (
    _accepted_timing_problems,
    _contains_run_id,
    _dataset_rows_audit,
    _linear_schedule_multiplier,
    _optimizer_contract_problem,
    _optimizer_summaries,
    _paired_comparisons,
    _plot,
    _render_results,
    _render_systems,
    _replace_marked,
    _run_settings_scope_matches,
    _system_summaries,
    _timing_adjustment_problems,
    _trajectory_auc,
    audit_dataset_artifacts,
    audit_experiment_contract,
    audit_training_artifacts,
    collect_evaluations,
    collect_system_metrics,
    render_blog,
)
from embed_optim.config import MUON_NS_IMPLEMENTATION, OptimizerConfig, RunConfig, load_matrix
from embed_optim.decontamination import DECONTAMINATED_BEIR
from embed_optim.evaluate_matrix import _evaluation_source_manifest


def test_run_id_matching_does_not_confuse_muon_and_normuon():
    muon = Path("results/dense/muon-lr1e-4__checkpoint-10/task.json")
    normuon = Path("results/dense/normuon-lr1e-4__checkpoint-10/task.json")
    assert _contains_run_id(muon, "muon-lr1e-4")
    assert not _contains_run_id(normuon, "muon-lr1e-4")
    assert _contains_run_id(normuon, "normuon-lr1e-4")


def test_optimizer_contract_validates_mixed_muon_topology():
    import torch

    config = RunConfig(
        run_id="muon-test",
        model_family="late",
        optimizer=OptimizerConfig(name="muon", lr=1e-4, aux_lr=3e-6),
        model_name="model",
        dataset_path="dataset",
    )
    step, final_step = 40, 100
    multiplier = _linear_schedule_multiplier(step, final_step, config.warmup_ratio)

    def adam_state():
        return {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.zeros(3),
            "exp_avg_sq": torch.zeros(3),
        }

    optimizer = {
        "state": {
            0: {"momentum_buffer": torch.zeros(3, 2)},
            1: adam_state(),
            2: adam_state(),
        },
        "param_groups": [
            {
                "params": [0],
                "algorithm": "muon",
                "lr": config.optimizer.lr * multiplier,
                "momentum": config.optimizer.momentum,
                "beta2": config.optimizer.normuon_beta2,
                "ns_steps": config.optimizer.ns_steps,
                "ns_implementation": MUON_NS_IMPLEMENTATION,
                "adjust_lr_fn": config.optimizer.adjust_lr_fn,
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

    assert _optimizer_contract_problem(optimizer, config, step, final_step) is None
    implementation = optimizer["param_groups"][0].pop("ns_implementation")
    assert "ns_implementation" in _optimizer_contract_problem(optimizer, config, step, final_step)
    optimizer["param_groups"][0]["ns_implementation"] = implementation
    optimizer["param_groups"][0]["lr"] *= 2
    assert "parameter group 0 lr" in _optimizer_contract_problem(
        optimizer, config, step, final_step
    )


def test_timing_adjustment_requires_nonoverlapping_timestamp_evidence(tmp_path):
    path = tmp_path / "timing_adjustment.json"
    payload = {
        "prior_training_wall_time_seconds": 180.0,
        "included_through_checkpoint_step": 4,
        "segments": [
            {
                "started_at_utc": "2026-01-01T00:00:00Z",
                "checkpoint_completed_at_utc": "2026-01-01T00:01:00Z",
                "wall_time_seconds": 60.0,
                "included_through_checkpoint_step": 2,
            },
            {
                "started_at_utc": "2026-01-01T00:02:00Z",
                "checkpoint_completed_at_utc": "2026-01-01T00:04:00Z",
                "wall_time_seconds": 120.0,
                "included_through_checkpoint_step": 4,
            },
        ],
        "evidence": "W&B start times and checkpoint mtimes",
        "reason": "Retain only useful work through durable checkpoints",
    }
    path.write_text(json.dumps(payload))
    assert _timing_adjustment_problems(path, [2, 4, 6, 8, 10]) == []

    payload["prior_training_wall_time_seconds"] = 181.0
    path.write_text(json.dumps(payload))
    assert "timing adjustment total does not match" in " ".join(
        _timing_adjustment_problems(path, [2, 4, 6, 8, 10])
    )

    payload["prior_training_wall_time_seconds"] = 180.0
    payload["segments"][1]["started_at_utc"] = "2026-01-01T00:00:30Z"
    path.write_text(json.dumps(payload))
    assert "overlaps its predecessor" in " ".join(
        _timing_adjustment_problems(path, [2, 4, 6, 8, 10])
    )


def test_run_settings_scope_schema_is_selected_by_mteb_version():
    singular = {"split": "test", "subset": "default"}
    plural = {"splits": ["test"], "subsets": ["default"]}

    assert _run_settings_scope_matches(singular, "test", "default", "2.18.16")
    assert not _run_settings_scope_matches(plural, "test", "default", "2.18.16")
    assert _run_settings_scope_matches(plural, "test", "default", "2.19.3")
    assert not _run_settings_scope_matches(singular, "test", "default", "2.19.3")
    assert not _run_settings_scope_matches(
        {"split": "test", "subset": "default", "splits": ["test"], "subsets": ["default"]},
        "test",
        "default",
        "2.19.3",
    )
    with pytest.raises(ValueError, match="Unsupported MTEB"):
        _run_settings_scope_matches(plural, "test", "default", "1.38.0")


def test_replace_marked_preserves_the_markers():
    text = "before\n<!-- A -->\nold\n<!-- B -->\nafter\n"
    result = _replace_marked(text, ("<!-- A -->", "<!-- B -->"), "new")
    assert result == "before\n<!-- A -->\n\nnew\n\n<!-- B -->\nafter\n"


def test_dataset_row_audit_proves_identity_negatives_and_checksum(tmp_path):
    row = {
        "sample_id": 0,
        "source": "fiqa",
        "query_id": 10,
        "positive_id": 20,
        "negative_ids": [21, 22, 23, 24, 25, 26, 27],
        "negative_pool_indices": [0, 1, 2, 4, 5, 7, 9],
    }
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps(row) + "\n")
    manifest = {
        "total_queries": 1,
        "quotas": {"fiqa": 1},
        "row_manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    audit = _dataset_rows_audit(rows_path, manifest)
    assert audit["errors"] == []
    assert audit["rows"] == 1
    assert audit["unique_source_queries"] == 1

    row["negative_ids"][-1] = row["negative_ids"][0]
    rows_path.write_text(json.dumps(row) + "\n")
    invalid = _dataset_rows_audit(rows_path, manifest)
    assert any("seven distinct negatives" in error for error in invalid["errors"])


def test_dataset_audit_rejects_more_than_one_training_dataset(tmp_path):
    configs = [
        SimpleNamespace(dataset_path=str(tmp_path / "one")),
        SimpleNamespace(dataset_path=str(tmp_path / "two")),
    ]
    audit = audit_dataset_artifacts(configs)
    assert not audit["complete"]
    assert "expected one shared dataset path" in audit["errors"][0]


def test_experiment_contract_accepts_matrix_and_rejects_semantic_drift():
    configs = load_matrix("configs/experiment.yaml")
    audit = audit_experiment_contract(configs)
    assert audit["complete"]
    assert audit["observed_runs"] == audit["expected_runs"] == 24

    changed = [replace(configs[0], max_length=512), *configs[1:]]
    invalid = audit_experiment_contract(changed)
    assert not invalid["complete"]
    assert any("max_length" in error for error in invalid["errors"])


def test_optimizer_summary_reports_observed_auc_and_lr_robustness():
    summary = []
    for run_id, lr, score in (("adamw-low", 1e-6, 0.2), ("adamw-high", 3e-6, 0.4)):
        for stage in range(1, 6):
            summary.append(
                {
                    "model_family": "dense",
                    "optimizer": "adamw",
                    "learning_rate": lr,
                    "run_id": run_id,
                    "stage": stage,
                    "fraction": stage / 5,
                    "mean_ndcg_at_10": score,
                }
            )

    rows, best_dynamics = _optimizer_summaries(summary)
    assert len(rows) == 1
    assert rows[0]["best_run_id"] == "adamw-high"
    assert rows[0]["final_mean_across_lrs"] == pytest.approx(0.3)
    assert rows[0]["best_config_observed_auc_ndcg_at_10"] == pytest.approx(0.4)
    assert rows[0]["observed_auc_mean_across_lrs"] == pytest.approx(0.3)
    assert _trajectory_auc(best_dynamics) == pytest.approx(0.4)


def test_optimizer_summary_does_not_mix_same_run_id_across_families():
    summary = []
    for family, score in (("dense", 0.2), ("late", 0.8)):
        for stage in range(1, 6):
            summary.append(
                {
                    "model_family": family,
                    "optimizer": "adamw",
                    "learning_rate": 1e-6,
                    "run_id": "adamw-lr1e-6",
                    "stage": stage,
                    "fraction": stage / 5,
                    "mean_ndcg_at_10": score,
                }
            )

    rows, best_dynamics = _optimizer_summaries(summary)
    by_family = {row["model_family"]: row for row in rows}
    assert by_family["dense"]["best_config_observed_auc_ndcg_at_10"] == pytest.approx(0.2)
    assert by_family["late"]["best_config_observed_auc_ndcg_at_10"] == pytest.approx(0.8)
    assert len(best_dynamics) == 10


def test_system_summary_reports_speedup_against_family_adamw():
    rows = []
    for optimizer, wall_time, throughput in (("adamw", 4.0, 10.0), ("muon", 2.0, 20.0)):
        rows.append(
            {
                "model_family": "dense",
                "optimizer": optimizer,
                "wall_time_hours": wall_time,
                "samples_per_second": throughput,
                "steps_per_second": throughput / 128,
                "peak_allocated_gib": 1.0,
                "peak_reserved_gib": 2.0,
                "checkpoint_gib": 3.0,
                "optimizer_state_gib": 4.0,
                "gpu_name": "GPU",
                "world_size": 4,
            }
        )

    by_optimizer = {row["optimizer"]: row for row in _system_summaries(rows)}
    assert by_optimizer["adamw"]["throughput_vs_adamw"] == 1.0
    assert by_optimizer["muon"]["throughput_vs_adamw"] == 2.0
    assert by_optimizer["muon"]["wall_time_speedup_vs_adamw"] == 2.0
    assert "2.00×" in _render_systems(list(by_optimizer.values()))


def test_result_render_reports_auc_paired_task_counts_and_figure_paths():
    optimizer_rows = []
    dynamics = []
    task_rows = []
    for family in ("dense", "late"):
        for optimizer_index, optimizer in enumerate(("adamw", "muon", "normuon")):
            score = 0.4 + optimizer_index * 0.01
            optimizer_rows.append(
                {
                    "model_family": family,
                    "optimizer": optimizer,
                    "best_learning_rate": 1e-4,
                    "best_final_ndcg_at_10": score,
                    "final_mean_across_lrs": score,
                    "final_population_std_across_lrs": 0.01,
                    "final_min_across_lrs": score - 0.01,
                    "final_max_across_lrs": score + 0.01,
                    "best_config_observed_auc_ndcg_at_10": score - 0.02,
                    "observed_auc_mean_across_lrs": score - 0.02,
                }
            )
            for stage in range(1, 6):
                dynamics.append(
                    {
                        "model_family": family,
                        "optimizer": optimizer,
                        "stage": stage,
                        "mean_ndcg_at_10": score - (5 - stage) * 0.01,
                    }
                )
        for task_index in range(14):
            task_rows.append(
                {
                    "model_family": family,
                    "task": f"task-{task_index}",
                    "adamw": 0.4,
                    "muon": 0.41,
                    "normuon": 0.39,
                    "muon_minus_adamw": 0.01,
                    "normuon_minus_adamw": -0.01,
                }
            )

    paired_rows = _paired_comparisons(task_rows, bootstrap_samples=1_000)
    rendered = _render_results(optimizer_rows, dynamics, task_rows, paired_rows)
    assert "4-LR trajectory AUC" in rendered
    assert "muon beats AdamW on 14/14 tasks" in rendered
    assert "normuon beats AdamW on 0/14 tasks" in rendered
    assert "Paired bootstrap 95% CI" in rendered
    assert "../reports/figures/dense-training-dynamics.png" in rendered
    assert "../reports/figures/late-training-dynamics.png" in rendered
    assert "../reports/figures/dense-training-dynamics-by-run.png" in rendered
    assert "../reports/figures/late-training-dynamics-by-run.png" in rendered
    assert "../reports/figures/dense-lr-sensitivity.png" in rendered
    assert "../reports/figures/late-lr-sensitivity.png" in rendered
    muon = next(
        row for row in paired_rows if row["model_family"] == "dense" and row["optimizer"] == "muon"
    )
    assert muon["wins"] == 14
    assert muon["ties"] == 0
    assert muon["losses"] == 0
    assert muon["mean_delta"] == pytest.approx(0.01)
    assert muon["bootstrap_ci_95_lower"] == pytest.approx(0.01)
    assert muon["bootstrap_ci_95_upper"] == pytest.approx(0.01)
    assert muon["exact_sign_test_p_value"] == pytest.approx(2 / 2**14)
    assert muon["holm_sign_test_p_value"] == pytest.approx(4 * 2 / 2**14)
    assert paired_rows == _paired_comparisons(task_rows, bootstrap_samples=1_000)


def test_render_blog_replaces_both_sections_and_completion_status(tmp_path, monkeypatch):
    blog = tmp_path / "blog.md"
    blog.write_text(
        "**Experiment status:** training matrix in progress. This document already records the frozen protocol;\n"
        "the results sections are populated only from strictly validated aggregation artifacts after coverage reaches\n"
        "1,680/1,680.\n\n"
        "<!-- RESULTS:BEGIN -->\nold results\n<!-- RESULTS:END -->\n\n"
        "<!-- SYSTEMS:BEGIN -->\nold systems\n<!-- SYSTEMS:END -->\n"
    )
    monkeypatch.setattr("embed_optim.aggregate._render_results", lambda *args: "new results")
    monkeypatch.setattr("embed_optim.aggregate._render_systems", lambda *args: "new systems")

    render_blog(blog, [], [], [], [], [])

    rendered = blog.read_text()
    assert "<!-- RESULTS:BEGIN -->\n\nnew results\n\n<!-- RESULTS:END -->" in rendered
    assert "<!-- SYSTEMS:BEGIN -->\n\nnew systems\n\n<!-- SYSTEMS:END -->" in rendered
    assert (
        "**Experiment status:** complete — 24/24 training runs and 1,680/1,680 "
        "checkpoint/task evaluations." in rendered
    )
    assert "training matrix in progress" not in rendered


def test_plot_generates_every_figure_referenced_by_the_blog(tmp_path):
    summary = []
    for family in ("dense", "late"):
        for optimizer_index, optimizer in enumerate(("adamw", "muon", "normuon")):
            for learning_rate in (1e-5, 1e-4):
                for stage in range(1, 6):
                    summary.append(
                        {
                            "model_family": family,
                            "optimizer": optimizer,
                            "learning_rate": learning_rate,
                            "run_id": f"{optimizer}-lr{learning_rate:.0e}",
                            "stage": stage,
                            "fraction": stage / 5,
                            "mean_ndcg_at_10": 0.3 + optimizer_index * 0.01 + stage * 0.005,
                        }
                    )

    _plot(summary, tmp_path)

    expected = (
        "dense-training-dynamics.png",
        "late-training-dynamics.png",
        "dense-training-dynamics-by-run.png",
        "late-training-dynamics-by-run.png",
        "dense-lr-sensitivity.png",
        "late-lr-sensitivity.png",
    )
    for name in expected:
        payload = (tmp_path / "figures" / name).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 1_000


def test_system_metrics_add_audited_prior_training_segment(tmp_path):
    output = tmp_path / "dense" / "adamw-lr1e-6"
    output.mkdir(parents=True)
    (output / "completed.json").write_text(
        json.dumps(
            {
                "global_step": 540,
                "dataset_rows": 5400,
                "system_metrics": {
                    "wall_time_seconds_max_rank": 3600,
                    "trainer": {
                        "train_samples_per_second": 9.9,
                        "train_steps_per_second": 0.99,
                    },
                },
            }
        )
    )
    (output / "timing_adjustment.json").write_text(
        json.dumps({"prior_training_wall_time_seconds": 1800})
    )
    config = SimpleNamespace(
        output_dir=output,
        model_family="dense",
        optimizer=SimpleNamespace(name="adamw", lr=1e-6),
        run_id="adamw-lr1e-6",
    )

    row = collect_system_metrics([config])[0]
    assert row["wall_time_hours"] == 1.5
    assert row["recorded_segment_wall_time_hours"] == 1.0
    assert row["prior_training_wall_time_hours"] == 0.5
    assert row["samples_per_second"] == 1.0
    assert row["steps_per_second"] == 0.1
    assert row["trainer_reported_samples_per_second"] == 9.9


def test_system_metrics_prefers_checkpoint_accepted_timing_ledger(tmp_path):
    output = tmp_path / "late" / "muon-lr1e-4"
    output.mkdir(parents=True)
    (output / "completed.json").write_text(
        json.dumps(
            {
                "global_step": 100,
                "dataset_rows": 1000,
                "system_metrics": {"wall_time_seconds_max_rank": 9999, "trainer": {}},
            }
        )
    )
    (output / "timing_adjustment.json").write_text(
        json.dumps({"prior_training_wall_time_seconds": 10})
    )
    (output / "accepted_timing.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "segments": [
                    {"wall_time_seconds_max_rank": 20},
                    {"wall_time_seconds_max_rank": 30},
                ],
                "total_wall_time_seconds_max_rank": 50,
            }
        )
    )
    config = SimpleNamespace(
        output_dir=output,
        model_family="late",
        optimizer=SimpleNamespace(name="muon", lr=1e-4),
        run_id="muon-lr1e-4",
    )

    row = collect_system_metrics([config])[0]
    assert row["wall_time_hours"] == pytest.approx(60 / 3600)
    assert row["samples_per_second"] == pytest.approx(1000 / 60)
    assert row["accepted_timing_path"] == str(output / "accepted_timing.json")


def test_accepted_timing_audit_requires_contiguous_steps_and_matching_total(tmp_path):
    path = tmp_path / "accepted_timing.json"
    payload = {
        "schema_version": 1,
        "segments": [
            {
                "start_step_exclusive": 4,
                "end_step_inclusive": 6,
                "started_at_utc": "2026-01-01T00:00:00Z",
                "checkpoint_completed_at_utc": "2026-01-01T00:01:00Z",
                "wall_time_seconds_max_rank": 60.0,
            },
            {
                "start_step_exclusive": 6,
                "end_step_inclusive": 8,
                "started_at_utc": "2026-01-01T00:01:00Z",
                "checkpoint_completed_at_utc": "2026-01-01T00:02:00Z",
                "wall_time_seconds_max_rank": 60.0,
            },
        ],
        "total_wall_time_seconds_max_rank": 120.0,
    }
    path.write_text(json.dumps(payload))
    assert _accepted_timing_problems(path, expected_start_step=4, expected_final_step=8) == []

    payload["segments"][1]["start_step_exclusive"] = 5
    payload["total_wall_time_seconds_max_rank"] = 1.0
    path.write_text(json.dumps(payload))
    problems = _accepted_timing_problems(path, expected_start_step=4, expected_final_step=8)
    assert "accepted timing segment 1 starts at 5, expected 6" in problems
    assert "accepted timing total does not match its segments" in problems


def test_evaluation_collection_requires_pinned_result_provenance(tmp_path):
    config = RunConfig(
        run_id="adamw-test",
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=1e-6),
        model_name="model",
        dataset_path="data",
        output_root=str(tmp_path / "outputs"),
    )
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "checkpoint_schedule.json").write_text(
        json.dumps({"steps": [2, 4, 6, 8, 10]})
    )
    training_versions = {
        "torch": "1",
        "transformers": "1",
        "sentence-transformers": "1",
        "pylate": "1",
        "late-interaction-kernels": "1",
    }
    (config.output_dir / "completed.json").write_text(json.dumps({"versions": training_versions}))

    result = (
        tmp_path
        / "results"
        / "dense"
        / "adamw-test__checkpoint-2"
        / "adamw-test"
        / "checkpoint-2"
        / "local"
        / "SciFactDecontaminated.json"
    )
    result.parent.mkdir(parents=True)
    payload = {
        "dataset_revision": DECONTAMINATED_BEIR["SciFact"][1],
        "task_name": "SciFactDecontaminated",
        "mteb_version": "2.19.3",
        "evaluation_time": 1.0,
        "scores": {
            "test": [
                {
                    "hf_subset": "default",
                    "main_score": 0.5,
                    "ndcg_at_10": 0.5,
                    "mteb_version": "2.19.3",
                }
            ]
        },
    }
    result.write_text(json.dumps(payload))
    (result.parent / "model_meta.json").write_text(
        json.dumps(
            {
                "name": "adamw-test/checkpoint-2",
                "revision": "local",
                "max_tokens": 8192,
                "embed_dim": 768,
                "similarity_fn_name": "cosine",
                "framework": ["Sentence Transformers", "PyTorch"],
            }
        )
    )
    evaluation_versions = {
        "mteb": "2.19.3",
        "torch": "1",
        "sentence-transformers": "1",
        "flash-attn": "1",
        "transformers": "1",
    }
    runtime_versions = {
        **evaluation_versions,
        "pylate": "1",
        "fast-plaid": "1",
        "late-interaction-kernels": "1",
    }
    source_files = _evaluation_source_manifest(Path(__file__).resolve().parents[1])
    (tmp_path / "results" / "evaluation_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "python": "/system/python",
                "versions": runtime_versions,
                "source_files": source_files,
            }
        )
    )
    (result.parent / "run_settings.jsonl").write_text(
        json.dumps(
            {
                "task": "SciFactDecontaminated",
                "splits": ["test"],
                "subsets": ["default"],
                "version": evaluation_versions,
                "encode_kwargs": {},
            }
        )
        + "\n"
    )

    rows = collect_evaluations(tmp_path / "results", [config])
    assert len(rows) == 1
    assert rows[0]["task"] == "SciFact"
    assert rows[0]["ndcg_at_10"] == 0.5

    settings_path = result.parent / "run_settings.jsonl"
    valid_settings = settings_path.read_text()
    settings_path.write_text(
        json.dumps(
            {
                "task": "SciFactDecontaminated",
                "split": "test",
                "subset": "default",
                "version": evaluation_versions,
                "encode_kwargs": {},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="Missing/ambiguous run settings"):
        collect_evaluations(tmp_path / "results", [config])
    settings_path.write_text(valid_settings)

    payload["dataset_revision"] = "wrong-revision"
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unexpected dataset revision"):
        collect_evaluations(tmp_path / "results", [config])

    payload["dataset_revision"] = DECONTAMINATED_BEIR["SciFact"][1]
    result.write_text(json.dumps(payload))
    model_meta_path = result.parent / "model_meta.json"
    model_meta = json.loads(model_meta_path.read_text())
    model_meta["max_tokens"] = 512
    model_meta_path.write_text(json.dumps(model_meta))
    with pytest.raises(ValueError, match="Unexpected model evaluation semantics"):
        collect_evaluations(tmp_path / "results", [config])
    model_meta["max_tokens"] = 8192
    model_meta_path.write_text(json.dumps(model_meta))

    runtime_versions["pylate"] = "different"
    (tmp_path / "results" / "evaluation_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "python": "/system/python",
                "versions": runtime_versions,
                "source_files": source_files,
            }
        )
    )
    with pytest.raises(ValueError, match="Training/evaluation pylate versions differ"):
        collect_evaluations(tmp_path / "results", [config])


def test_evaluation_collection_ignores_unrelated_smoke_results_without_runtime(tmp_path):
    config = RunConfig(
        run_id="adamw-test",
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=1e-6),
        model_name="model",
        dataset_path="data",
        output_root=str(tmp_path / "outputs"),
    )
    smoke_result = (
        tmp_path
        / "results"
        / "smoke-dense"
        / "results"
        / "no_model_name__available"
        / "SciFactDecontaminated.json"
    )
    smoke_result.parent.mkdir(parents=True)
    smoke_result.write_text("{}")

    assert collect_evaluations(tmp_path / "results", [config]) == []


def test_training_artifact_audit_requires_resumable_five_checkpoint_run(tmp_path):
    config = RunConfig(
        run_id="adamw-test",
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=1e-6),
        model_name="model",
        dataset_path=str(tmp_path / "shared-data"),
        output_root=str(tmp_path),
    )
    output = config.output_dir
    output.mkdir(parents=True)
    dataset = Path(config.dataset_path)
    dataset.mkdir()
    manifest = {
        "total_queries": 10,
        "dataset_fingerprint": "dataset-fingerprint",
        "row_manifest_sha256": "row-sha256",
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    (output / "dataset_manifest.json").write_text(json.dumps(manifest))
    (output / "run_config.json").write_text(json.dumps(config.as_dict()))
    steps = [2, 4, 6, 8, 10]
    (output / "checkpoint_schedule.json").write_text(json.dumps({"steps": steps}))
    (output / "completed.json").write_text(
        json.dumps(
            {
                "global_step": 10,
                "checkpoints": steps,
                "run_id": config.run_id,
                "model_family": config.model_family,
                "dataset_rows": 10,
                "dataset_fingerprint": "dataset-fingerprint",
                "optimizer_partition": {
                    "hidden": {"tensors": 1, "parameters": 4},
                    "aux_decay": {"tensors": 1, "parameters": 2},
                    "aux_no_decay": {"tensors": 1, "parameters": 1},
                },
                "system_metrics": {
                    "world_size": 4,
                    "wall_time_seconds_max_rank": 1.0,
                    "peak_allocated_bytes_max_rank": 1,
                    "peak_reserved_bytes_max_rank": 1,
                    "checkpoint_bytes": {f"checkpoint-{step}": 1 for step in steps},
                    "optimizer_state_bytes": {f"checkpoint-{step}": 1 for step in steps},
                    "trainer": {
                        "train_runtime": 1.0,
                        "train_samples_per_second": 10.0,
                        "train_steps_per_second": 1.0,
                        "train_loss": 0.5,
                        "epoch": 1.0,
                    },
                    "gpu_name": "Test GPU",
                },
                "versions": {
                    "torch": "1",
                    "transformers": "1",
                    "sentence-transformers": "1",
                    "pylate": "1",
                    "late-interaction-kernels": "1",
                },
            }
        )
    )
    (output / "trainer_state_final.json").write_text(
        json.dumps({"global_step": 10, "log_history": [{"step": 10, "loss": 0.5}]})
    )
    final = output / "final"
    final.mkdir()
    (final / "model.safetensors").write_bytes(b"model")
    for step in steps:
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": step,
                    "log_history": [
                        {
                            "step": step,
                            "loss": 0.5,
                            "grad_norm": 1.0,
                            "learning_rate": 1e-5,
                            "epoch": step / 10,
                        }
                    ],
                }
            )
        )
        (checkpoint / "training_args.bin").write_bytes(b"args")
        (checkpoint / "model.safetensors").write_bytes(b"model")
        (checkpoint / "rng_state_0.pth").write_bytes(b"rng")
        (checkpoint / "rng_state_1.pth").write_bytes(b"rng")
        (checkpoint / "rng_state_2.pth").write_bytes(b"rng")
        (checkpoint / "rng_state_3.pth").write_bytes(b"rng")

    complete = audit_training_artifacts([config])
    assert complete == {
        "complete": True,
        "verified_runs": 1,
        "expected_runs": 1,
        "verified_checkpoints": 5,
        "expected_checkpoints": 5,
        "deep_validation": False,
        "errors": [],
    }

    exact_dataset_view = audit_training_artifacts(
        [config], expected_dataset_fingerprint="dataset-fingerprint"
    )
    assert exact_dataset_view["complete"] is True
    wrong_dataset_view = audit_training_artifacts(
        [config], expected_dataset_fingerprint="different-fingerprint"
    )
    assert wrong_dataset_view["complete"] is False
    assert wrong_dataset_view["errors"] == [
        "dense/adamw-test: training dataset view fingerprint differs from audited dataset"
    ]

    derived_manifest = dict(manifest)
    derived_manifest["rows"] = derived_manifest.pop("total_queries")
    (dataset / "manifest.json").write_text(json.dumps(derived_manifest))
    (output / "dataset_manifest.json").write_text(json.dumps(derived_manifest))
    derived_dataset = audit_training_artifacts(
        [config],
        expected_dataset_fingerprint="dataset-fingerprint",
        expected_dataset_rows=10,
    )
    assert derived_dataset["complete"] is True
    wrong_dataset_rows = audit_training_artifacts([config], expected_dataset_rows=11)
    assert wrong_dataset_rows["complete"] is False
    assert wrong_dataset_rows["errors"] == [
        "dense/adamw-test: source dataset row count differs from audited dataset"
    ]
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    (output / "dataset_manifest.json").write_text(json.dumps(manifest))

    import torch
    from safetensors.torch import save_file

    save_file({"weight": torch.ones(1)}, final / "model.safetensors")

    def write_deep_payload(step):
        checkpoint = output / f"checkpoint-{step}"
        save_file({"weight": torch.tensor([float(step)])}, checkpoint / "model.safetensors")
        scheduled_lr = config.optimizer.lr * _linear_schedule_multiplier(
            step, steps[-1], config.warmup_ratio
        )
        torch.save(
            {
                "state": {
                    parameter_id: {
                        "step": torch.tensor(float(step)),
                        "exp_avg": torch.zeros(1),
                        "exp_avg_sq": torch.zeros(1),
                    }
                    for parameter_id in (0, 1)
                },
                "param_groups": [
                    {
                        "params": [0],
                        "algorithm": "adamw",
                        "lr": scheduled_lr,
                        "betas": (config.optimizer.beta1, config.optimizer.beta2),
                        "eps": config.optimizer.eps,
                        "weight_decay": config.optimizer.weight_decay,
                    },
                    {
                        "params": [1],
                        "algorithm": "adamw",
                        "lr": scheduled_lr,
                        "betas": (config.optimizer.beta1, config.optimizer.beta2),
                        "eps": config.optimizer.eps,
                        "weight_decay": 0.0,
                    },
                ],
            },
            checkpoint / "optimizer.pt",
        )
        torch.save(
            {
                "base_lrs": [config.optimizer.lr, config.optimizer.lr],
                "last_epoch": step,
                "_step_count": step + 1,
                "_last_lr": [scheduled_lr, scheduled_lr],
                "lr_lambdas": [{}, {}],
            },
            checkpoint / "scheduler.pt",
        )
        torch.save(
            SimpleNamespace(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                num_train_epochs=1.0,
                max_steps=-1,
                learning_rate=1e-6,
                max_grad_norm=1.0,
                bf16=True,
                tf32=True,
                fp16=False,
                seed=42,
                data_seed=42,
                gradient_checkpointing=True,
                dataloader_num_workers=8,
                dataloader_pin_memory=True,
                dataloader_persistent_workers=True,
                dataloader_prefetch_factor=4,
                dataloader_drop_last=True,
                remove_unused_columns=False,
                ddp_find_unused_parameters=False,
                train_sampling_strategy="group_by_length",
                logging_steps=10,
                run_name="dense-adamw-test",
                project="embedding-optimizer-study",
                lr_scheduler_type="linear",
                save_strategy="no",
                report_to=["wandb"],
                warmup_steps=0.1,
            ),
            checkpoint / "training_args.bin",
        )
        for rank in range(4):
            torch.save({"rank": rank}, checkpoint / f"rng_state_{rank}.pth")

    for step in steps:
        write_deep_payload(step)
    deep_complete = audit_training_artifacts([config], deep=True)
    assert deep_complete["complete"] is True
    assert deep_complete["deep_validation"] is True
    assert deep_complete["verified_checkpoints"] == 5

    corrupt_optimizer = output / "checkpoint-2" / "optimizer.pt"
    corrupt_optimizer.write_bytes(b"not-a-pytorch-state")
    corrupt = audit_training_artifacts([config], deep=True)
    assert corrupt["complete"] is False
    assert corrupt["verified_checkpoints"] == 4
    assert any("invalid optimizer state" in error for error in corrupt["errors"])
    write_deep_payload(2)

    wrong_optimizer = torch.load(
        output / "checkpoint-2" / "optimizer.pt", map_location="cpu", weights_only=True
    )
    wrong_optimizer["param_groups"][0]["algorithm"] = "muon"
    torch.save(wrong_optimizer, output / "checkpoint-2" / "optimizer.pt")
    corrupt = audit_training_artifacts([config], deep=True)
    assert corrupt["complete"] is False
    assert corrupt["verified_checkpoints"] == 4
    assert any("algorithm is 'muon', expected 'adamw'" in error for error in corrupt["errors"])
    write_deep_payload(2)

    wrong_scheduler = torch.load(
        output / "checkpoint-2" / "scheduler.pt", map_location="cpu", weights_only=True
    )
    wrong_scheduler["base_lrs"][0] *= 2
    torch.save(wrong_scheduler, output / "checkpoint-2" / "scheduler.pt")
    corrupt = audit_training_artifacts([config], deep=True)
    assert corrupt["complete"] is False
    assert corrupt["verified_checkpoints"] == 4
    assert any("scheduler base_lrs[0]" in error for error in corrupt["errors"])
    write_deep_payload(2)

    nonfinite_optimizer = output / "checkpoint-2" / "optimizer.pt"
    optimizer_state = torch.load(nonfinite_optimizer, map_location="cpu", weights_only=True)
    optimizer_state["state"][0]["step"] = torch.tensor(float("nan"))
    torch.save(optimizer_state, nonfinite_optimizer)
    corrupt = audit_training_artifacts([config], deep=True)
    assert corrupt["complete"] is False
    assert corrupt["verified_checkpoints"] == 4
    assert any(
        "optimizer state contains a non-finite tensor" in error for error in corrupt["errors"]
    )
    write_deep_payload(2)

    invalid_training_args = output / "checkpoint-2" / "training_args.bin"
    training_args = torch.load(invalid_training_args, map_location="cpu", weights_only=False)
    training_args.gradient_accumulation_steps = 3
    torch.save(training_args, invalid_training_args)
    corrupt = audit_training_artifacts([config], deep=True)
    assert corrupt["complete"] is False
    assert corrupt["verified_checkpoints"] == 4
    assert any(
        "gradient_accumulation_steps is 3, expected 4" in error for error in corrupt["errors"]
    )
    write_deep_payload(2)

    corrupt_model = output / "checkpoint-4" / "model.safetensors"
    corrupt_model.write_bytes(b"not-a-safetensors-payload")
    corrupt = audit_training_artifacts([config], deep=True)
    assert any("invalid safetensors payload" in error for error in corrupt["errors"])
    write_deep_payload(4)

    save_file({"weight": torch.tensor([float("nan")])}, corrupt_model)
    corrupt = audit_training_artifacts([config], deep=True)
    assert any("non-finite tensor" in error for error in corrupt["errors"])
    write_deep_payload(4)

    shutil.copy2(output / "checkpoint-2" / "model.safetensors", corrupt_model)
    corrupt = audit_training_artifacts([config], deep=True)
    assert any("model payload is unchanged" in error for error in corrupt["errors"])
    write_deep_payload(4)

    torch.save({"last_epoch": 5}, output / "checkpoint-6" / "scheduler.pt")
    corrupt = audit_training_artifacts([config], deep=True)
    assert any("scheduler state does not match" in error for error in corrupt["errors"])
    write_deep_payload(6)

    corrupt_rng = output / "checkpoint-8" / "rng_state_2.pth"
    corrupt_rng.write_bytes(b"not-a-pytorch-archive")
    corrupt = audit_training_artifacts([config], deep=True)
    assert any("RNG state is not a PyTorch archive" in error for error in corrupt["errors"])
    write_deep_payload(8)

    second_config = RunConfig(
        run_id="adamw-test-two",
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=2e-6),
        model_name="model",
        dataset_path=str(dataset),
        output_root=str(tmp_path),
    )
    shutil.copytree(output, second_config.output_dir)
    (second_config.output_dir / "run_config.json").write_text(json.dumps(second_config.as_dict()))
    second_completed_path = second_config.output_dir / "completed.json"
    second_completed = json.loads(second_completed_path.read_text())
    second_completed["run_id"] = second_config.run_id
    second_completed["dataset_fingerprint"] = "different-training-view"
    second_completed_path.write_text(json.dumps(second_completed))
    inconsistent_dataset_view = audit_training_artifacts([config, second_config])
    assert inconsistent_dataset_view["verified_runs"] == 1
    assert inconsistent_dataset_view["verified_checkpoints"] == 10
    assert inconsistent_dataset_view["errors"] == [
        "dense/adamw-test-two: training dataset view fingerprint differs across runs"
    ]

    (output / "checkpoint-6" / "scheduler.pt").unlink()
    incomplete = audit_training_artifacts([config])
    assert not incomplete["complete"]
    assert incomplete["verified_runs"] == 0
    assert incomplete["verified_checkpoints"] == 4
    assert incomplete["errors"] == ["dense/adamw-test/checkpoint-6: missing/empty scheduler.pt"]

    (output / "checkpoint-6" / "scheduler.pt").write_bytes(b"scheduler")
    (output / "checkpoint-8" / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 8,
                "log_history": [
                    {"step": 4, "loss": 0.8},
                    {"step": 4, "loss": 0.7},
                ],
            }
        )
    )
    duplicate = audit_training_artifacts([config])
    assert duplicate["verified_checkpoints"] == 4
    assert duplicate["errors"] == [
        "dense/adamw-test/checkpoint-8: loss-history steps are duplicated or non-monotonic"
    ]

    (output / "checkpoint-8" / "trainer_state.json").write_text(
        json.dumps({"global_step": 8, "log_history": [{"step": 8, "loss": float("nan")}]})
    )
    non_finite = audit_training_artifacts([config])
    assert non_finite["verified_checkpoints"] == 4
    assert non_finite["errors"] == [
        "dense/adamw-test/checkpoint-8: loss-history loss is non-finite at step 8"
    ]

    (output / "checkpoint-8" / "trainer_state.json").write_text(
        json.dumps({"global_step": 8, "log_history": [{"step": 8, "loss": 0.5}]})
    )
    completed = json.loads((output / "completed.json").read_text())
    completed["dataset_fingerprint"] = None
    (output / "completed.json").write_text(json.dumps(completed))
    invalid_dataset_view = audit_training_artifacts([config])
    assert invalid_dataset_view["verified_checkpoints"] == 5
    assert invalid_dataset_view["errors"] == [
        "dense/adamw-test: missing/invalid training dataset view fingerprint"
    ]

    completed["dataset_fingerprint"] = "training-view-fingerprint"
    completed["system_metrics"]["wall_time_seconds_max_rank"] = 0
    (output / "completed.json").write_text(json.dumps(completed))
    invalid_system_metric = audit_training_artifacts([config])
    assert invalid_system_metric["verified_checkpoints"] == 5
    assert invalid_system_metric["errors"] == [
        "dense/adamw-test: system metric wall_time_seconds_max_rank is non-finite/non-positive"
    ]

    completed["system_metrics"] = None
    (output / "completed.json").write_text(json.dumps(completed))
    missing_system_metrics = audit_training_artifacts([config])
    assert missing_system_metrics["verified_checkpoints"] == 5
    assert missing_system_metrics["errors"] == [
        "dense/adamw-test: missing/invalid completion system metrics",
        "dense/adamw-test: expected completion world_size 4, got None",
    ]
