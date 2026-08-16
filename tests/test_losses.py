import torch
from torch import nn

from embed_optim.losses import ExplicitDenseInfoNCELoss


class IdentityEmbeddingModel(nn.Module):
    def forward(self, features):
        return {"sentence_embedding": features["values"]}


def test_dense_loss_uses_only_each_queries_own_group():
    model = IdentityEmbeddingModel()
    loss = ExplicitDenseInfoNCELoss(model, temperature=0.1)
    queries = {"values": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    positive = {"values": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    negatives = [{"values": torch.tensor([[0.0, 1.0], [1.0, 0.0]])} for _ in range(7)]
    baseline = loss([queries, positive, *negatives])

    # Alter every document belonging to query 1. Query 0's group scores are unchanged;
    # there is no cross-query document matrix as there would be with in-batch negatives.
    changed_positive = {"values": positive["values"].clone()}
    changed_positive["values"][1] = torch.tensor([100.0, 0.0])
    changed_negatives = []
    for negative in negatives:
        values = negative["values"].clone()
        values[1] = torch.tensor([0.0, 100.0])
        changed_negatives.append({"values": values})
    changed = loss([queries, changed_positive, *changed_negatives])
    assert torch.isfinite(baseline)
    assert torch.isfinite(changed)
