import torch
from torch import nn

from embed_optim.losses import ExplicitDenseInfoNCELoss


class IdentityEmbeddingModel(nn.Module):
    def forward(self, features):
        return {"sentence_embedding": features["values"]}


def test_dense_loss_uses_only_each_queries_own_group():
    model = IdentityEmbeddingModel()
    loss = ExplicitDenseInfoNCELoss(model, temperature=0.1)
    queries = {"values": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}
    positive = {"values": torch.tensor([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]])}
    negatives = [
        {"values": torch.tensor([[0.0, 1.0, 0.0], [0.8, 0.2, 0.565685]])} for _ in range(7)
    ]
    baseline = loss([queries, positive, *negatives])

    # Rotate every query-1 document's x/z components while preserving its normalized
    # y component. Query 1's own scores and all of query 0's explicit group stay fixed,
    # while query 0's scores against query 1's documents would change if the loss used
    # in-batch negatives.
    changed_positive = {"values": positive["values"].clone()}
    changed_positive["values"][1] = torch.tensor([0.0, 0.8, 0.6])
    changed_negatives = []
    for negative in negatives:
        values = negative["values"].clone()
        values[1] = torch.tensor([0.0, 0.2, 0.979796])
        changed_negatives.append({"values": values})
    changed = loss([queries, changed_positive, *changed_negatives])
    torch.testing.assert_close(changed, baseline, atol=1e-6, rtol=1e-6)
