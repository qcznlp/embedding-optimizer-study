from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ExplicitDenseInfoNCELoss(nn.Module):
    """InfoNCE over each query's own positive and seven negatives only."""

    def __init__(self, model: nn.Module, temperature: float = 0.02) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        del labels
        features = list(sentence_features)
        if len(features) != 9:
            raise ValueError(f"Expected query + 8 documents, got {len(features)} columns")
        query = F.normalize(self.model(features[0])["sentence_embedding"], p=2, dim=-1)
        documents = torch.stack(
            [
                F.normalize(self.model(feature)["sentence_embedding"], p=2, dim=-1)
                for feature in features[1:]
            ],
            dim=1,
        )
        scores = torch.einsum("bh,bnh->bn", query, documents) / self.temperature
        targets = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        return F.cross_entropy(scores, targets)


def _pad_sequence_dimension(tensor: Tensor, length: int) -> Tensor:
    if tensor.size(1) == length:
        return tensor
    padding = (
        (0, 0, 0, length - tensor.size(1)) if tensor.ndim == 3 else (0, length - tensor.size(1))
    )
    return F.pad(tensor, padding)


class ExplicitLateInfoNCELoss(nn.Module):
    """MeanMaxSim InfoNCE over per-query document groups, with fused LIK scoring."""

    def __init__(self, model: nn.Module, temperature: float = 0.001) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        del labels
        from pylate.scores import colbert_kd_scores

        features = list(sentence_features)
        if len(features) != 9:
            raise ValueError(f"Expected query + 8 documents, got {len(features)} columns")
        embeddings = [
            F.normalize(self.model(feature)["token_embeddings"], p=2, dim=-1)
            for feature in features
        ]
        wrapped = self.model.module if hasattr(self.model, "module") else self.model
        query_mask = features[0]["attention_mask"].bool()
        document_masks = [
            torch.logical_and(
                wrapped.skiplist_mask(feature["input_ids"], wrapped.skiplist),
                feature["attention_mask"].bool(),
            )
            for feature in features[1:]
        ]
        max_document_length = max(embedding.size(1) for embedding in embeddings[1:])
        documents = torch.stack(
            [
                _pad_sequence_dimension(embedding, max_document_length)
                for embedding in embeddings[1:]
            ],
            dim=1,
        )
        masks = torch.stack(
            [_pad_sequence_dimension(mask, max_document_length) for mask in document_masks],
            dim=1,
        )
        scores = colbert_kd_scores(
            queries_embeddings=embeddings[0],
            documents_embeddings=documents,
            queries_mask=query_mask,
            documents_mask=masks,
            backend="lik",
        )
        # MeanMaxSim, as used in the DenseOn/LateOn supervised fine-tuning recipe.
        scores = scores / query_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        scores = scores / self.temperature
        targets = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        return F.cross_entropy(scores, targets)
