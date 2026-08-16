import json
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
        dataset_path="data",
        output_root=str(tmp_path),
    )
    output = config.output_dir
    output.mkdir(parents=True)
    steps = [2, 4, 6, 8, 10]
    (output / "checkpoint_schedule.json").write_text(json.dumps({"steps": steps}))
    (output / "completed.json").write_text(
        json.dumps(
            {
                "global_step": 10,
                "checkpoints": steps,
                "system_metrics": {"world_size": 2},
            }
        )
    )
    (output / "trainer_state_final.json").write_text(json.dumps({"global_step": 10}))
    final = output / "final"
    final.mkdir()
    (final / "model.safetensors").write_bytes(b"model")
    for step in steps:
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        (checkpoint / "training_args.bin").write_bytes(b"args")
        (checkpoint / "model.safetensors").write_bytes(b"model")
        (checkpoint / "rng_state_0.pth").write_bytes(b"rng")
        (checkpoint / "rng_state_1.pth").write_bytes(b"rng")

    complete = audit_training_artifacts([config])
    assert complete == {
        "complete": True,
        "verified_runs": 1,
        "expected_runs": 1,
        "verified_checkpoints": 5,
        "expected_checkpoints": 5,
        "errors": [],
    }

    (output / "checkpoint-6" / "scheduler.pt").unlink()
    incomplete = audit_training_artifacts([config])
    assert not incomplete["complete"]
    assert incomplete["verified_runs"] == 0
    assert incomplete["verified_checkpoints"] == 4
    assert incomplete["errors"] == ["dense/adamw-test/checkpoint-6: missing/empty scheduler.pt"]
