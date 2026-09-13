from __future__ import annotations

from typing import Any, Callable

import torch
from sentence_transformers.sentence_transformer.data_collator import (
    SentenceTransformerDataCollator,
)

TEXT_COLUMNS = ("query", "positive", *(f"negative_{index}" for index in range(7)))


class DenseGroupCollator:
    """SentenceTransformers collator restricted to the explicit 8-way group."""

    def __init__(self, preprocess_fn: Callable) -> None:
        prompts = {
            "query": "query: ",
            **{column: "document: " for column in TEXT_COLUMNS if column != "query"},
        }
        self.inner = SentenceTransformerDataCollator(
            preprocess_fn=preprocess_fn,
            prompts=prompts,
        )
        self.valid_label_columns = self.inner.valid_label_columns

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        clean = [{column: row[column] for column in TEXT_COLUMNS} for row in features]
        return self.inner(clean)


class LateGroupCollator:
    """PyLate collator with dynamic per-column padding instead of padding to 8192."""

    def __init__(self, model: Any) -> None:
        self.model = model
        # SentenceTransformers 5 enables flattened FlashAttention inputs on its
        # Transformer module. PyLate's prefix insertion requires a 2-D attention
        # mask, so explicitly keep the dynamically padded representation here.
        first_module = model._first_module()
        if hasattr(first_module, "can_flatten_inputs"):
            first_module.can_flatten_inputs = False
        self.valid_label_columns = ["label", "labels", "score", "scores"]

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor | bool] = {"return_loss": True}
        for column in TEXT_COLUMNS:
            tokenized = self.model.tokenize(
                [row[column] for row in features],
                is_query=column == "query",
                pad=False,
            )
            for key, value in tokenized.items():
                batch[f"{column}_{key}"] = value
        return batch
