import json
from dataclasses import replace

import pytest

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.wandb_sync import build_canonical_run, canonical_history, history_sha256


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
    (output / "completed.json").write_text(json.dumps({"global_step": 20}))
    (output / "trainer_state_final.json").write_text(
        json.dumps({"global_step": 20, "log_history": [{"step": 20, "loss": 0.8}]})
    )

    first = build_canonical_run(config)
    second = build_canonical_run(replace(config))

    assert first == second
    assert first.wandb_run_id.startswith("canonical-dense-adamw-test-")
    assert first.source_wandb_run_id == "study-v2-dense-adamw-test-seed42"
