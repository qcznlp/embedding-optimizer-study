"""Independent forward/backward checks; no real model or network in this suite."""

import pytest
import torch
from torch import nn

from embed_optim.losses import ExplicitDenseInfoNCELoss
from scripts.audit_dense_loss_contract import (
    check_loss_graph,
    reference_scores,
    synthetic_rows,
    verify_token_columns,
)


class ProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(5, 7, dtype=torch.float64)

    def forward(self, features):
        return {"sentence_embedding": self.projection(features["values"])}


def fixture():
    torch.manual_seed(61042)
    model = ProjectionModel()
    features = [{"values": torch.randn(3, 5, dtype=torch.float64)} for _ in range(9)]
    return model, features


def test_production_loss_and_all_gradients_match_independent_row_reference():
    model, features = fixture()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    result, captured = check_loss_graph(model, features, 0.02)
    assert result["query_count"] == 3
    assert result["candidate_count_per_query"] == 8
    assert result["parameter_gradients_compared"] == 2
    assert len(captured) == 9
    assert max(r["max_absolute_error"] for r in result["gradient_errors"]) < 1e-10
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), before[name])
        assert parameter.grad is None


def test_reference_gradient_has_no_cross_query_document_dependency():
    torch.manual_seed(61201)
    columns = [torch.randn(3, 5, dtype=torch.float64, requires_grad=True) for _ in range(9)]
    scores, per_query = reference_scores(columns, 0.02)
    assert scores.shape == (3, 8)
    gradients = torch.autograd.grad(per_query[0], tuple(columns))
    for gradient in gradients:
        assert torch.equal(gradient[1:], torch.zeros_like(gradient[1:]))
        assert gradient[0].abs().sum() > 0


class WrongGradientLoss(ExplicitDenseInfoNCELoss):
    def forward(self, features, labels=None):
        value = super().forward(features, labels)
        return value.detach() + (value - value.detach()) * 1.1


def test_equal_loss_with_incorrect_backward_is_rejected():
    model, features = fixture()
    with pytest.raises(AssertionError):
        check_loss_graph(model, features, 0.02, loss_type=WrongGradientLoss)


class InBatchLoss(ExplicitDenseInfoNCELoss):
    def forward(self, features, labels=None):
        embeddings = [
            torch.nn.functional.normalize(self.model(f)["sentence_embedding"], dim=-1)
            for f in features
        ]
        scores = embeddings[0] @ torch.cat(embeddings[1:]).T / self.temperature
        return torch.nn.functional.cross_entropy(scores, torch.arange(len(scores)))


def test_accidental_in_batch_candidates_are_rejected():
    model, features = fixture()
    with pytest.raises(AssertionError):
        check_loss_graph(model, features, 0.02, loss_type=InBatchLoss)


@pytest.mark.parametrize("columns,temperature", [(8, 0.02), (10, 0.02), (9, 0.0)])
def test_reference_rejects_wrong_candidate_count_or_temperature(columns, temperature):
    with pytest.raises(ValueError):
        reference_scores([torch.ones(2, 3)] * columns, temperature)


def test_synthetic_rows_have_two_queries_and_exactly_seven_negatives_each():
    rows = synthetic_rows()
    assert len(rows) == 2
    assert all(
        list(row) == ["query", "positive", *[f"negative_{i}" for i in range(7)]] for row in rows
    )


@pytest.mark.parametrize("tensor_output", [True, False])
def test_actual_prefix_and_column_check_handles_tensor_or_list_ids(tensor_output):
    class Tokenizer:
        def preprocess(self, texts):
            values = [[len(t), ord(t[0])] for t in texts]
            return {"input_ids": torch.tensor(values) if tensor_output else values}

    rows = synthetic_rows()
    tokenizer = Tokenizer()
    features = [
        tokenizer.preprocess([("query: " if c == "query" else "document: ") + r[c] for r in rows])
        for c in rows[0]
    ]
    verify_token_columns(tokenizer, rows, features)
    if tensor_output:
        features[0]["input_ids"][0, 1] += 1
    else:
        features[0]["input_ids"][0][1] += 1
    with pytest.raises(ValueError, match="prompt differs: query"):
        verify_token_columns(tokenizer, rows, features)
