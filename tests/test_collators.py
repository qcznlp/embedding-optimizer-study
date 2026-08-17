import json
from types import SimpleNamespace

import torch
from transformers import TrainerControl, TrainerState

from embed_optim.callbacks import (
    AcceptedTimingCallback,
    FractionalCheckpointCallback,
    sanitize_pylate_checkpoint,
)
from embed_optim.collators import TEXT_COLUMNS, DenseGroupCollator, LateGroupCollator


class _FirstModule:
    can_flatten_inputs = True


class _LateModel:
    def __init__(self):
        self.first = _FirstModule()

    def _first_module(self):
        return self.first

    def tokenize(self, texts, is_query, pad):
        assert pad is False
        width = max(len(text) for text in texts)
        return {
            "input_ids": torch.ones(len(texts), width, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), width, dtype=torch.long),
        }


def test_dense_collator_applies_query_and_document_prefixes():
    seen = {}

    def preprocess(texts, prompt, task):
        del task
        seen[prompt] = seen.get(prompt, 0) + 1
        return {"input_ids": torch.ones(len(texts), 2, dtype=torch.long)}

    row = {column: f"{column} text" for column in TEXT_COLUMNS}
    batch = DenseGroupCollator(preprocess)([row])
    assert seen == {"query: ": 1, "document: ": 8}
    assert all(f"{column}_input_ids" in batch for column in TEXT_COLUMNS)


def test_late_collator_disables_flattening_and_keeps_explicit_groups():
    model = _LateModel()
    collator = LateGroupCollator(model)
    assert model.first.can_flatten_inputs is False

    row = {column: f"{column} text" for column in TEXT_COLUMNS}
    batch = collator([row])
    assert batch["return_loss"] is True
    for column in TEXT_COLUMNS:
        assert batch[f"{column}_input_ids"].shape[0] == 1
        assert batch[f"{column}_attention_mask"].shape[0] == 1


def test_pylate_checkpoint_sanitizer_removes_only_st5_io_fields(tmp_path):
    dense = tmp_path / "1_Dense"
    dense.mkdir()
    config_path = dense / "config.json"
    config_path.write_text(
        '{"in_features": 2, "module_input_name": "x", '
        '"module_output_name": "y", "use_residual": true}'
    )
    sanitize_pylate_checkpoint(tmp_path)
    assert config_path.read_text().count("module_") == 0
    assert '"use_residual": true' in config_path.read_text()


def test_fractional_checkpoint_callback_requests_each_target(tmp_path):
    callback = FractionalCheckpointCallback((0.2, 0.4, 0.6, 0.8, 1.0), tmp_path)
    state = TrainerState(max_steps=10)
    control = callback.on_train_begin(None, state, TrainerControl())
    for step in (2, 4, 6, 8, 10):
        state.global_step = step
        control.should_save = False
        control = callback.on_step_end(None, state, control)
        assert control.should_save


def test_accepted_timing_callback_records_non_overlapping_resume_segments(tmp_path, monkeypatch):
    ticks = iter([10.0, 15.0, 22.0, 30.0, 34.0])
    monkeypatch.setattr("embed_optim.callbacks.time.monotonic", lambda: next(ticks))
    args = SimpleNamespace(process_index=0)
    state = TrainerState(global_step=0)
    control = TrainerControl()

    callback = AcceptedTimingCallback(tmp_path)
    callback.on_train_begin(args, state, control)
    state.global_step = 2
    callback.on_save(args, state, control)
    state.global_step = 4
    callback.on_save(args, state, control)

    resumed = AcceptedTimingCallback(tmp_path)
    resumed.on_train_begin(args, state, control)
    state.global_step = 6
    resumed.on_save(args, state, control)

    payload = json.loads((tmp_path / "accepted_timing.json").read_text())
    assert [
        (segment["start_step_exclusive"], segment["end_step_inclusive"])
        for segment in payload["segments"]
    ] == [(0, 2), (2, 4), (4, 6)]
    assert [segment["wall_time_seconds_max_rank"] for segment in payload["segments"]] == [
        5.0,
        7.0,
        4.0,
    ]
    assert payload["total_wall_time_seconds"] == 16.0


def test_accepted_timing_callback_records_slowest_distributed_rank(tmp_path, monkeypatch):
    ticks = iter([10.0, 15.0])
    monkeypatch.setattr("embed_optim.callbacks.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def replace_with_max(duration, op):
        assert op == torch.distributed.ReduceOp.MAX
        duration.fill_(9.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", replace_with_max)
    args = SimpleNamespace(process_index=0, device=torch.device("cpu"))
    state = TrainerState(global_step=0)
    callback = AcceptedTimingCallback(tmp_path)
    callback.on_train_begin(args, state, TrainerControl())
    state.global_step = 2
    callback.on_save(args, state, TrainerControl())

    payload = json.loads((tmp_path / "accepted_timing.json").read_text())
    assert payload["segments"][0]["wall_time_seconds_max_rank"] == 9.0
