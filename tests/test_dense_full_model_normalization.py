"""Guard-unit tests, not substitutes for the separate full-model CPU execution."""

from types import SimpleNamespace

import pytest
import torch

from scripts import audit_dense_full_model_normalization as audit


class ModelShape(SimpleNamespace):
    def __getitem__(self, index):
        assert index == 0
        return self.encoder


def model_shape():
    parameters = [(f"p{i}", torch.ones(1)) for i in range(134)]
    dropout = torch.nn.Dropout(0)
    model = ModelShape(
        encoder=SimpleNamespace(
            can_flatten_inputs=False,
            auto_model=SimpleNamespace(
                config=SimpleNamespace(_attn_implementation="sdpa"), is_gradient_checkpointing=True
            ),
        ),
        get_sentence_embedding_dimension=lambda: 768,
        named_parameters=lambda: parameters,
        modules=lambda: [dropout],
        max_seq_length=8192,
        prompts={"query": "query: ", "document": "document: "},
    )
    return model, parameters, dropout


def test_declared_full_model_guard_retains_coverage_boundary(monkeypatch):
    monkeypatch.setattr(
        audit, "parameter_partition", lambda _: {"hidden": range(88), "aux_decay": [0]}
    )
    model, _, _ = model_shape()
    receipt = audit.validate_model(model)
    assert receipt["parameter_tensors"] == 134
    assert receipt["maximum_context_execution_tested"] is False
    assert receipt["device"] == "cpu" and receipt["attention"] == "sdpa"


@pytest.mark.parametrize(
    "kind",
    [
        "count",
        "dimensions",
        "context",
        "prompt",
        "execution",
        "attention",
        "checkpointing",
        "dtype",
        "dropout",
        "routing",
        "nonfinite",
    ],
)
def test_full_model_guard_rejects_configuration_or_payload_drift(monkeypatch, kind):
    monkeypatch.setattr(
        audit,
        "parameter_partition",
        lambda _: {"hidden": range(87 if kind == "routing" else 88), "aux_decay": [0]},
    )
    model, parameters, dropout = model_shape()
    if kind == "count":
        parameters.pop()
    elif kind == "dimensions":
        model.get_sentence_embedding_dimension = lambda: 32
    elif kind == "context":
        model.max_seq_length = 64
    elif kind == "prompt":
        model.prompts = {}
    elif kind == "execution":
        model.encoder.can_flatten_inputs = True
    elif kind == "attention":
        model.encoder.auto_model.config._attn_implementation = "flash_attention_2"
    elif kind == "checkpointing":
        model.encoder.auto_model.is_gradient_checkpointing = False
    elif kind == "dtype":
        parameters[0] = ("p0", torch.ones(1, dtype=torch.float64))
    elif kind == "dropout":
        dropout.p = 0.1
    elif kind == "nonfinite":
        parameters[0][1][0] = float("nan")
    with pytest.raises(ValueError):
        audit.validate_model(model)
