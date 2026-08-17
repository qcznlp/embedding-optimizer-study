from pathlib import Path

from embed_optim.config import load_matrix
from embed_optim.train import _training_arguments


def test_training_arguments_explicitly_pin_ddp_drop_last(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    matrix = Path(__file__).parents[1] / "configs" / "experiment.yaml"
    config = load_matrix(matrix)[0]

    arguments = _training_arguments(config)

    assert arguments.dataloader_drop_last is True
    assert arguments.per_device_train_batch_size == 8
    assert arguments.gradient_accumulation_steps == 4
