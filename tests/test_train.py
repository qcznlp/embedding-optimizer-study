from pathlib import Path
from types import SimpleNamespace

from embed_optim import train
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
