from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import train
from embed_optim.callbacks import StopAfterStepCallback
from embed_optim.config import load_matrix


def test_training_arguments_explicitly_pin_ddp_drop_last(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(train, "SentenceTransformerTrainingArguments", SimpleNamespace)
    matrix = Path(__file__).parents[1] / "configs" / "experiment.yaml"
    config = load_matrix(matrix)[0]

    arguments = train._training_arguments(config)

    assert arguments.dataloader_drop_last is True
    assert arguments.per_device_train_batch_size == 8
    assert arguments.gradient_accumulation_steps == 4


def test_stop_after_step_preserves_horizon_and_requests_durable_save():
    callback = StopAfterStepCallback(1905)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_train_begin(
        None,
        SimpleNamespace(global_step=1563, max_steps=3907),
        control,
    )
    callback.on_step_end(None, SimpleNamespace(global_step=1904), control)
    assert not control.should_save
    assert not control.should_training_stop

    callback.on_step_end(None, SimpleNamespace(global_step=1905), control)
    assert control.should_save
    assert control.should_training_stop

    with pytest.raises(ValueError, match="after resumed step"):
        callback.on_train_begin(
            None,
            SimpleNamespace(global_step=1905, max_steps=3907),
            control,
        )


def test_formal_training_verifies_declared_runtime(monkeypatch):
    config = SimpleNamespace(model_family="late", run_id="muon-test")
    observed = {}
    monkeypatch.setattr(train, "matrix_runtime_spec", lambda path: Path("runtime.json"))
    monkeypatch.setattr(train, "load_matrix", lambda path: [config])
    monkeypatch.setattr(
        "embed_optim.runtime.verify_runtime_spec",
        lambda path: {
            "python_executable": "/formal/python",
            "packages": {"torch": "2.9.1+cu129"},
            "torch_cuda": "12.9",
        },
    )
    monkeypatch.setattr(
        train,
        "run_training",
        lambda selected, resume_from_checkpoint=None: observed.update(
            config=selected, resume=resume_from_checkpoint
        ),
    )

    train.main(["--matrix", "matrix.yaml", "--model-family", "late", "--run-id", "muon-test"])

    assert observed == {"config": config, "resume": None}
