import torch
from torch import nn

from embed_optim.losses import ExplicitDenseInfoNCELoss, ExplicitLateInfoNCELoss


class IdentityEmbeddingModel(nn.Module):
    def forward(self, features):
        return {"sentence_embedding": features["values"]}


class IdentityTokenModel(nn.Module):
    skiplist = set()

    def forward(self, features):
        return {"token_embeddings": features["values"]}

    def skiplist_mask(self, input_ids, skiplist):
        del skiplist
        return torch.ones_like(input_ids, dtype=torch.bool)


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


def test_late_loss_passes_only_eight_documents_per_query_to_the_scorer(monkeypatch):
    import pylate.scores

    observed = {}

    def group_scorer(
        queries_embeddings,
        documents_embeddings,
        queries_mask,
        documents_mask,
        backend,
    ):
        observed["query_shape"] = tuple(queries_embeddings.shape)
        observed["document_shape"] = tuple(documents_embeddings.shape)
        observed["query_mask_shape"] = tuple(queries_mask.shape)
        observed["document_mask_shape"] = tuple(documents_mask.shape)
        observed["backend"] = backend
        return torch.einsum("bld,bnld->bn", queries_embeddings, documents_embeddings)

    monkeypatch.setattr(pylate.scores, "colbert_kd_scores", group_scorer)
    batch, tokens, dimension = 2, 3, 4

    def feature():
        return {
            "values": torch.randn(batch, tokens, dimension),
            "input_ids": torch.ones(batch, tokens, dtype=torch.long),
            "attention_mask": torch.ones(batch, tokens, dtype=torch.long),
        }

    loss = ExplicitLateInfoNCELoss(IdentityTokenModel(), temperature=0.001)(
        [feature() for _ in range(9)]
    )

    assert torch.isfinite(loss)
    assert observed == {
        "query_shape": (2, 3, 4),
        "document_shape": (2, 8, 3, 4),
        "query_mask_shape": (2, 3),
        "document_mask_shape": (2, 8, 3),
        "backend": "lik",
    }
