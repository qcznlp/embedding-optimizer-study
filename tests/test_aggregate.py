import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from embed_optim.aggregate import (
    _contains_run_id,
    _replace_marked,
    audit_training_artifacts,
    collect_system_metrics,
)
from embed_optim.config import OptimizerConfig, RunConfig


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
