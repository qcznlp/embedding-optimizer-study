import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.aggregate import (
    _contains_run_id,
    _dataset_rows_audit,
    _optimizer_summaries,
    _paired_comparisons,
    _render_results,
    _render_systems,
    _replace_marked,
    _system_summaries,
    _trajectory_auc,
    audit_dataset_artifacts,
    audit_experiment_contract,
    audit_training_artifacts,
    collect_evaluations,
    collect_system_metrics,
)
from embed_optim.config import OptimizerConfig, RunConfig, load_matrix
from embed_optim.decontamination import DECONTAMINATED_BEIR


def test_run_id_matching_does_not_confuse_muon_and_normuon():
    muon = Path("results/dense/muon-lr1e-4__checkpoint-10/task.json")
    normuon = Path("results/dense/normuon-lr1e-4__checkpoint-10/task.json")
    assert _contains_run_id(muon, "muon-lr1e-4")
    assert not _contains_run_id(normuon, "muon-lr1e-4")
    assert _contains_run_id(normuon, "normuon-lr1e-4")


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


def test_result_render_reports_auc_and_paired_task_counts():
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
    assert paired_rows == _paired_comparisons(task_rows, bootstrap_samples=1_000)


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
    (tmp_path / "results" / "evaluation_runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": "/system/python",
                "versions": runtime_versions,
            }
        )
    )
    (result.parent / "run_settings.jsonl").write_text(
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

    rows = collect_evaluations(tmp_path / "results", [config])
    assert len(rows) == 1
    assert rows[0]["task"] == "SciFact"
    assert rows[0]["ndcg_at_10"] == 0.5

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
                "schema_version": 1,
                "python": "/system/python",
                "versions": runtime_versions,
            }
        )
    )
    with pytest.raises(ValueError, match="Training/evaluation pylate versions differ"):
        collect_evaluations(tmp_path / "results", [config])


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
        "errors": [],
    }

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
