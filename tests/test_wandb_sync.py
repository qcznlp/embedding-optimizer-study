import json
from dataclasses import replace

import pytest

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.wandb_sync import (
    CanonicalRun,
    _mark_canonical_current,
    build_canonical_run,
    canonical_history,
    history_sha256,
    main,
    parse_args,
    verify_remote_canonical_history,
    verify_remote_current_matrix,
)


class _RemoteRun:
    def __init__(self, tags, status=None, history=()):
        self.tags = tags
        self.summary = {} if status is None else {"canonical_status": status}
        self.history = tuple(history)
        self.updates = 0

    def update(self):
        self.updates += 1

    def scan_history(self, page_size):
        assert page_size == 1000
        return iter(self.history)


def test_canonical_history_is_sorted_and_merges_duplicate_steps():
    state = {
        "global_step": 20,
        "log_history": [
            {"step": 20, "loss": 0.8, "epoch": 0.2},
            {"step": 10, "loss": 1.0, "learning_rate": 1e-5},
            {"step": 20, "grad_norm": 2.0},
            {"step": 20, "loss": 0.7},
        ],
    }

    history = canonical_history(state)

    assert [row["global_step"] for row in history] == [10, 20]
    assert history[1] == {
        "global_step": 20,
        "train/loss": 0.7,
        "train/epoch": 0.2,
        "train/grad_norm": 2.0,
    }
    assert history_sha256(history) == history_sha256(history)


def test_canonical_history_rejects_steps_past_final_step():
    with pytest.raises(ValueError, match="outside"):
        canonical_history({"global_step": 10, "log_history": [{"step": 11, "loss": 1.0}]})


def test_canonical_history_always_reaches_terminal_step():
    history = canonical_history(
        {"global_step": 23, "epoch": 1.0, "log_history": [{"step": 20, "loss": 0.8}]}
    )

    assert history[-1] == {"global_step": 23, "train/epoch": 1.0}


def test_canonical_history_excludes_resume_local_trainer_summaries():
    history = canonical_history(
        {
            "global_step": 20,
            "log_history": [
                {"step": 10, "loss": 0.8},
                {
                    "step": 20,
                    "train_loss": 0.04,
                    "train_runtime": 2.0,
                    "train_samples_per_second": 500.0,
                    "train_steps_per_second": 10.0,
                },
            ],
        }
    )

    assert history == (
        {"global_step": 10, "train/loss": 0.8},
        {"global_step": 20},
    )


def test_build_canonical_run_is_content_addressed(tmp_path):
    config = RunConfig(
        run_id="adamw-test",
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=1e-6),
        model_name="model",
        dataset_path="data",
        output_root=str(tmp_path),
        wandb_entity="entity",
    )
    output = config.output_dir
    output.mkdir(parents=True)
    (output / "completed.json").write_text(
        json.dumps(
            {
                "global_step": 20,
                "dataset_rows": 100,
                "system_metrics": {"wall_time_seconds_max_rank": 99.0},
            }
        )
    )
    (output / "accepted_timing.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start_step_exclusive": 10,
                        "end_step_inclusive": 20,
                        "wall_time_seconds_max_rank": 6.0,
                    }
                ]
            }
        )
    )
    (output / "timing_adjustment.json").write_text(
        json.dumps({"prior_training_wall_time_seconds": 4.0})
    )
    (output / "trainer_state_final.json").write_text(
        json.dumps({"global_step": 20, "log_history": [{"step": 20, "loss": 0.8}]})
    )

    first = build_canonical_run(config)
    second = build_canonical_run(replace(config))

    assert first == second
    assert first.wandb_run_id.startswith("canonical-dense-adamw-test-")
    assert first.source_wandb_run_id == "study-v2-dense-adamw-test-seed42"
    assert first.history[-1] == {
        "global_step": 20,
        "train/loss": 0.8,
        "system/useful_training_wall_time_seconds": 10.0,
        "system/useful_samples_per_second": 10.0,
        "system/useful_steps_per_second": 2.0,
    }


def test_current_canonical_marker_is_non_destructive_and_idempotent():
    run = _RemoteRun(["dense", "canonical", "canonical-superseded"], "superseded")

    assert _mark_canonical_current(run) is True
    assert run.tags == ["dense", "canonical", "canonical-current"]
    assert run.summary["canonical_status"] == "current"
    assert run.updates == 1

    assert _mark_canonical_current(run) is False
    assert run.updates == 1


def _canonical_spec(history):
    config = RunConfig(
        run_id="adamw-test",
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=1e-6),
        model_name="model",
        dataset_path="data",
        wandb_entity="entity",
    )
    history = tuple(history)
    return CanonicalRun(config, history, history_sha256(history), "canonical-test", "source-test")


def test_remote_canonical_history_is_read_back_and_rehashed():
    expected = _canonical_spec(
        (
            {"global_step": 10, "train/loss": 0.8, "train/epoch": 0.5},
            {
                "global_step": 20,
                "train/epoch": 1.0,
                "system/useful_training_wall_time_seconds": 10.0,
                "system/useful_samples_per_second": 100.0,
                "system/useful_steps_per_second": 2.0,
            },
        )
    )
    remote = _RemoteRun(
        ["canonical-current"],
        history=(
            {
                "_step": 0,
                "_timestamp": 123.0,
                "global_step": 10.0,
                "train/loss": 0.8,
                "train/epoch": 0.5,
            },
            {
                "_step": 1,
                "global_step": 20,
                "train/epoch": 1,
                "system/useful_training_wall_time_seconds": 10,
                "system/useful_samples_per_second": 100,
                "system/useful_steps_per_second": 2,
            },
        ),
    )

    assert verify_remote_canonical_history(remote, expected) == 2


def test_remote_canonical_history_rejects_changed_or_duplicate_rows():
    expected = _canonical_spec(({"global_step": 10, "train/loss": 0.8},))
    changed = _RemoteRun(["canonical-current"], history=({"global_step": 10, "train/loss": 0.7},))
    duplicate = _RemoteRun(
        ["canonical-current"],
        history=(
            {"global_step": 10, "train/loss": 0.8},
            {"global_step": 10, "train/loss": 0.8},
        ),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        verify_remote_canonical_history(changed, expected)
    with pytest.raises(RuntimeError, match="duplicated"):
        verify_remote_canonical_history(duplicate, expected)


def test_remote_history_verification_is_enabled_by_default():
    defaults = parse_args([])
    assert defaults.families == ["dense"]
    assert defaults.scope_amendment is None
    assert defaults.skip_remote_history_verification is False
    assert parse_args(["--families", "dense", "late"]).families == ["dense", "late"]
    assert (
        parse_args(["--skip-remote-history-verification"]).skip_remote_history_verification is True
    )


def test_wandb_sync_validates_dense_scope_before_training_or_remote_io(monkeypatch):
    monkeypatch.setattr(
        "embed_optim.wandb_sync.load_matrix",
        lambda matrix: pytest.fail("scope must be validated before training history I/O"),
    )
    with pytest.raises(ValueError, match="requires --scope-amendment"):
        main([])


def test_remote_current_matrix_requires_one_exact_run_per_identity():
    dense = _canonical_spec(({"global_step": 10, "train/loss": 0.8},))
    late_config = replace(dense.config, model_family="late")
    late = replace(dense, config=late_config, wandb_run_id="canonical-late-test")

    class CurrentRun:
        def __init__(self, spec, *, status="current"):
            self.id = spec.wandb_run_id
            self.tags = ["canonical", "canonical-current"]
            self.config = {
                "model_family": spec.config.model_family,
                "run_id": spec.config.run_id,
            }
            self.summary = {"canonical_status": status}

    unrelated = CurrentRun(dense)
    unrelated.config = {"model_family": "dense", "run_id": "unrelated"}
    audit = verify_remote_current_matrix(
        [CurrentRun(dense), CurrentRun(late), unrelated], [dense, late]
    )

    assert audit == {
        "selected_runs": 2,
        "selected_current_runs": 2,
        "project_current_runs": 3,
    }

    with pytest.raises(RuntimeError, match="has 2 current runs"):
        verify_remote_current_matrix([CurrentRun(dense), CurrentRun(dense)], [dense])
    with pytest.raises(RuntimeError, match="identity/status differs"):
        verify_remote_current_matrix([CurrentRun(dense, status="superseded")], [dense])
