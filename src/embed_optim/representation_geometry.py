from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .geometry import _atomic_json, _sha256

SCHEMA_VERSION = 1
ModelFamily = Literal["dense", "late"]


def _export_manifest_identity(
    source: Path,
    *,
    source_sha256: str,
    family: ModelFamily,
    array_metadata: dict[str, Any],
    required: bool,
) -> dict[str, Any] | None:
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    if not manifest_path.is_file():
        if required:
            raise FileNotFoundError(f"Required probe export manifest is missing: {manifest_path}")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported probe export manifest schema in {manifest_path}")
    if manifest.get("family") != family:
        raise ValueError(
            f"Probe export manifest family is {manifest.get('family')!r}, expected {family!r}"
        )
    declared_output = manifest.get("output")
    if not isinstance(declared_output, dict):
        raise ValueError(f"Probe export manifest lacks output provenance: {manifest_path}")
    if declared_output.get("sha256") != source_sha256:
        raise ValueError("Probe archive SHA-256 disagrees with its export manifest")
    if declared_output.get("arrays") != array_metadata:
        raise ValueError("Probe archive arrays disagree with their export manifest")
    encoding = manifest.get("encoding")
    if not isinstance(encoding, dict) or encoding.get("positive_candidate_index") != 0:
        raise ValueError("Probe export manifest does not preserve the positive-first convention")
    probe = manifest.get("probe")
    if not isinstance(probe, dict):
        raise ValueError(f"Probe export manifest lacks probe provenance: {manifest_path}")
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "probe_manifest_sha256": probe.get("manifest_sha256"),
        "probe_selection_sha256": probe.get("selection_sha256"),
    }


def _as_float_tensor(array: np.ndarray, name: str) -> Tensor:
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be numeric, got dtype={array.dtype}")
    tensor = torch.from_numpy(np.asarray(array)).float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains a non-finite value")
    return tensor


def _as_mask(array: np.ndarray, name: str, shape: tuple[int, ...]) -> Tensor:
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(array.shape)}")
    if array.dtype.kind not in "biu":
        raise ValueError(f"{name} must be boolean or integer, got dtype={array.dtype}")
    mask = torch.from_numpy(np.asarray(array)).bool()
    if not mask.any(dim=-1).all():
        raise ValueError(f"{name} contains an item with no valid token")
    return mask


def _as_offsets(array: np.ndarray, name: str, *, items: int, total_tokens: int) -> Tensor:
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integer, got dtype={array.dtype}")
    offsets = torch.from_numpy(np.asarray(array)).long()
    expected_shape = (items + 1,)
    if tuple(offsets.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(offsets.shape)}")
    if offsets[0].item() != 0 or offsets[-1].item() != total_tokens:
        raise ValueError(
            f"{name} must span [0, {total_tokens}], got [{offsets[0].item()}, {offsets[-1].item()}]"
        )
    if not bool((offsets[1:] > offsets[:-1]).all()):
        raise ValueError(f"{name} must be strictly increasing")
    return offsets


def _pad_ragged_batch(
    values: Tensor,
    offsets: Tensor,
    start: int,
    stop: int,
) -> tuple[Tensor, Tensor]:
    if start < 0 or stop <= start or stop >= offsets.numel():
        raise ValueError(f"Invalid ragged slice [{start}, {stop}) for {offsets.numel() - 1} items")
    lengths = offsets[start + 1 : stop + 1] - offsets[start:stop]
    maximum = int(lengths.max().item())
    result = values.new_zeros((stop - start, maximum, values.size(1)))
    mask = torch.zeros((stop - start, maximum), dtype=torch.bool)
    for local_index, item_index in enumerate(range(start, stop)):
        begin = int(offsets[item_index].item())
        end = int(offsets[item_index + 1].item())
        length = end - begin
        result[local_index, :length] = values[begin:end]
        mask[local_index, :length] = True
    return result, mask


def _gini(values: Tensor) -> float:
    values = values.detach().double().flatten()
    if values.numel() == 0:
        return 0.0
    minimum = values.min()
    if minimum < 0:
        values = values - minimum
    total = values.sum()
    if total <= 0:
        return 0.0
    ordered = values.sort().values
    indices = torch.arange(1, ordered.numel() + 1, dtype=torch.float64)
    numerator = ((2 * indices - ordered.numel() - 1) * ordered).sum()
    return float((numerator / (ordered.numel() * total)).item())


def _quantiles(values: Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    return {
        "min": float(values.min().item()),
        "p05": float(torch.quantile(values, 0.05).item()),
        "p25": float(torch.quantile(values, 0.25).item()),
        "median": float(torch.quantile(values, 0.50).item()),
        "p75": float(torch.quantile(values, 0.75).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def _sample_rows(values: Tensor, maximum: int, seed: int) -> tuple[Tensor, bool]:
    if maximum <= 0:
        raise ValueError(f"max_representation_vectors must be positive, got {maximum}")
    if values.size(0) <= maximum:
        return values, False
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(values.size(0), generator=generator)[:maximum]
    return values[indices], True


def representation_summary(
    vectors: Tensor,
    *,
    max_vectors: int,
    seed: int,
) -> dict[str, Any]:
    if vectors.ndim != 2 or vectors.size(0) == 0 or vectors.size(1) == 0:
        raise ValueError(
            f"representation vectors must be non-empty 2-D, got {tuple(vectors.shape)}"
        )
    if not torch.isfinite(vectors).all():
        raise ValueError("representation vectors contain a non-finite value")
    original_count = vectors.size(0)
    vectors, sampled = _sample_rows(vectors.float(), max_vectors, seed)
    norms = vectors.norm(dim=-1)
    unit = F.normalize(vectors, p=2, dim=-1)
    count = unit.size(0)
    if count > 1:
        resultant_sq = unit.sum(dim=0).square().sum()
        self_similarity = unit.square().sum()
        mean_pairwise_cosine = float(
            ((resultant_sq - self_similarity) / (count * (count - 1))).item()
        )
    else:
        mean_pairwise_cosine = 0.0

    centered = vectors - vectors.mean(dim=0, keepdim=True)
    if count > 1 and bool(centered.square().sum() > 0):
        singular_values = torch.linalg.svdvals(centered)
        eigenvalues = singular_values.square() / (count - 1)
        positive = eigenvalues[eigenvalues > torch.finfo(eigenvalues.dtype).eps]
        if positive.numel() > 0:
            total = positive.sum()
            probabilities = positive / total
            entropy_effective_rank = float(
                torch.exp(-(probabilities * probabilities.log()).sum()).item()
            )
            stable_rank = float((total / positive.max()).item())
            covariance_trace = float(total.item())
            rank_limit = min(count - 1, vectors.size(1))
            normalized_effective_rank = entropy_effective_rank / rank_limit
            leading_variance_fraction = float((positive.max() / total).item())
        else:
            entropy_effective_rank = 0.0
            stable_rank = 0.0
            covariance_trace = 0.0
            normalized_effective_rank = 0.0
            leading_variance_fraction = 0.0
    else:
        entropy_effective_rank = 0.0
        stable_rank = 0.0
        covariance_trace = 0.0
        normalized_effective_rank = 0.0
        leading_variance_fraction = 0.0

    return {
        "original_vectors": original_count,
        "analyzed_vectors": count,
        "sampled": sampled,
        "dimension": vectors.size(1),
        "mean_norm": float(norms.mean().item()),
        "norm_cv": float(
            (
                norms.std(unbiased=False) / norms.mean().clamp_min(torch.finfo(norms.dtype).eps)
            ).item()
        ),
        "mean_pairwise_cosine": mean_pairwise_cosine,
        "covariance_trace": covariance_trace,
        "entropy_effective_rank": entropy_effective_rank,
        "normalized_effective_rank": normalized_effective_rank,
        "stable_rank": stable_rank,
        "leading_variance_fraction": leading_variance_fraction,
    }


def ranking_summary(
    scores: Tensor,
    *,
    top_k: int,
    reference_scores: Tensor | None = None,
    sample_groups: list[str] | None = None,
) -> dict[str, Any]:
    if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
        raise ValueError(
            f"scores must have shape [samples, candidates>=2], got {tuple(scores.shape)}"
        )
    if top_k <= 0 or top_k > scores.size(1):
        raise ValueError(f"top_k must be in [1, {scores.size(1)}], got {top_k}")
    positive = scores[:, 0]
    hardest_negative = scores[:, 1:].max(dim=1).values
    margins = positive - hardest_negative
    ranks = 1 + (scores[:, 1:] > positive.unsqueeze(1)).sum(dim=1)
    result: dict[str, Any] = {
        "samples": scores.size(0),
        "candidates_per_sample": scores.size(1),
        "positive_score": _quantiles(positive),
        "hardest_negative_score": _quantiles(hardest_negative),
        "positive_hardest_negative_margin": _quantiles(margins),
        "top1_accuracy": float((ranks == 1).float().mean().item()),
        "mean_reciprocal_rank": float(ranks.float().reciprocal().mean().item()),
        "mean_candidate_score_std": float(scores.std(dim=1, unbiased=False).mean().item()),
    }
    if reference_scores is not None:
        if tuple(reference_scores.shape) != tuple(scores.shape):
            raise ValueError(
                "reference_scores must match scores: "
                f"expected {tuple(scores.shape)}, got {tuple(reference_scores.shape)}"
            )
        selected = scores.topk(top_k, dim=1).indices
        reference_selected = reference_scores.topk(top_k, dim=1).indices
        overlaps = []
        for current, reference in zip(selected, reference_selected):
            overlap = torch.isin(current, reference).sum().item() / top_k
            overlaps.append(overlap)
        result["reference_ranking"] = {
            "top_k": top_k,
            "mean_top_k_overlap": float(sum(overlaps) / len(overlaps)),
            "top1_agreement": float(
                (scores.argmax(dim=1) == reference_scores.argmax(dim=1)).float().mean().item()
            ),
            "score_drift_rms": float((scores - reference_scores).square().mean().sqrt().item()),
        }
    if sample_groups is not None:
        if len(sample_groups) != scores.size(0):
            raise ValueError(
                f"sample_groups must have {scores.size(0)} entries, got {len(sample_groups)}"
            )
        result["by_group"] = {}
        for group in sorted(set(sample_groups)):
            indices = torch.tensor(
                [index for index, value in enumerate(sample_groups) if value == group],
                dtype=torch.long,
            )
            result["by_group"][group] = ranking_summary(
                scores[indices],
                top_k=top_k,
                reference_scores=None if reference_scores is None else reference_scores[indices],
            )
    return result


def dense_probe_metrics(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    *,
    max_representation_vectors: int,
    seed: int,
    top_k: int,
    reference_scores: Tensor | None = None,
    sample_groups: list[str] | None = None,
) -> dict[str, Any]:
    if query_embeddings.ndim != 2:
        raise ValueError(
            f"dense query_embeddings must have shape [samples, dim], got {tuple(query_embeddings.shape)}"
        )
    if document_embeddings.ndim != 3:
        raise ValueError(
            "dense document_embeddings must have shape [samples, candidates, dim], got "
            f"{tuple(document_embeddings.shape)}"
        )
    samples, candidates, dimension = document_embeddings.shape
    if tuple(query_embeddings.shape) != (samples, dimension):
        raise ValueError(
            "dense query/document shapes disagree: "
            f"queries={tuple(query_embeddings.shape)}, documents={tuple(document_embeddings.shape)}"
        )
    scores = dense_probe_scores(query_embeddings, document_embeddings)
    return {
        "scorer": "cosine",
        "score_geometry": ranking_summary(
            scores,
            top_k=min(top_k, candidates),
            reference_scores=reference_scores,
            sample_groups=sample_groups,
        ),
        "representations": {
            "queries": representation_summary(
                query_embeddings,
                max_vectors=max_representation_vectors,
                seed=seed,
            ),
            "documents": representation_summary(
                document_embeddings.reshape(-1, dimension),
                max_vectors=max_representation_vectors,
                seed=seed + 1,
            ),
        },
    }


def dense_probe_scores(query_embeddings: Tensor, document_embeddings: Tensor) -> Tensor:
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 3:
        raise ValueError("Dense score inputs require 2-D queries and 3-D documents")
    if query_embeddings.size(0) != document_embeddings.size(0) or query_embeddings.size(
        1
    ) != document_embeddings.size(2):
        raise ValueError(
            "Dense score input shapes disagree: "
            f"queries={tuple(query_embeddings.shape)}, documents={tuple(document_embeddings.shape)}"
        )
    queries = F.normalize(query_embeddings.float(), p=2, dim=-1)
    documents = F.normalize(document_embeddings.float(), p=2, dim=-1)
    return torch.einsum("bd,bcd->bc", queries, documents)


def _masked_mean(values: Tensor, mask: Tensor, dimension: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dimension) / weights.sum(dim=dimension).clamp_min(1)


def _late_batch_scores(
    queries: Tensor,
    documents: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    similarities = torch.einsum("bqd,bckd->bcqk", queries, documents)
    similarities = similarities.masked_fill(~document_mask[:, :, None, :], -torch.inf)
    contributions, selected_tokens = similarities.max(dim=-1)
    contributions = contributions.masked_fill(~query_mask[:, None, :], 0.0)
    scores = contributions.sum(dim=-1) / query_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    positive_contributions = [values[mask] for values, mask in zip(contributions[:, 0], query_mask)]
    positive_selections = [values[mask] for values, mask in zip(selected_tokens[:, 0], query_mask)]
    return scores, positive_contributions, positive_selections


def late_probe_scores(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
    *,
    batch_size: int,
) -> Tensor:
    if query_embeddings.ndim != 3 or document_embeddings.ndim != 4:
        raise ValueError("Late score inputs require 3-D queries and 4-D documents")
    if query_embeddings.size(0) != document_embeddings.size(0) or query_embeddings.size(
        2
    ) != document_embeddings.size(3):
        raise ValueError(
            "Late score input shapes disagree: "
            f"queries={tuple(query_embeddings.shape)}, documents={tuple(document_embeddings.shape)}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    queries = F.normalize(query_embeddings.float(), p=2, dim=-1)
    documents = F.normalize(document_embeddings.float(), p=2, dim=-1)
    batches = []
    for start in range(0, queries.size(0), batch_size):
        stop = min(queries.size(0), start + batch_size)
        scores, _, _ = _late_batch_scores(
            queries[start:stop],
            documents[start:stop],
            query_mask[start:stop],
            document_mask[start:stop],
        )
        batches.append(scores)
    return torch.cat(batches, dim=0)


def late_probe_metrics(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
    *,
    batch_size: int,
    max_representation_vectors: int,
    seed: int,
    top_k: int,
    reference_scores: Tensor | None = None,
    sample_groups: list[str] | None = None,
) -> dict[str, Any]:
    if query_embeddings.ndim != 3:
        raise ValueError(
            "late query_embeddings must have shape [samples, query_tokens, dim], got "
            f"{tuple(query_embeddings.shape)}"
        )
    if document_embeddings.ndim != 4:
        raise ValueError(
            "late document_embeddings must have shape [samples, candidates, document_tokens, dim], got "
            f"{tuple(document_embeddings.shape)}"
        )
    samples, candidates, document_tokens, dimension = document_embeddings.shape
    if query_embeddings.size(0) != samples or query_embeddings.size(2) != dimension:
        raise ValueError(
            "late query/document shapes disagree: "
            f"queries={tuple(query_embeddings.shape)}, documents={tuple(document_embeddings.shape)}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    queries = F.normalize(query_embeddings.float(), p=2, dim=-1)
    documents = F.normalize(document_embeddings.float(), p=2, dim=-1)
    score_batches: list[Tensor] = []
    contribution_entropies: list[float] = []
    contribution_ginis: list[float] = []
    selected_document_fractions: list[float] = []
    repeated_selection_dominance: list[float] = []

    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        batch_scores, contributions, selections = _late_batch_scores(
            queries[start:stop],
            documents[start:stop],
            query_mask[start:stop],
            document_mask[start:stop],
        )
        score_batches.append(batch_scores)
        for local_index, (evidence, selected) in enumerate(zip(contributions, selections)):
            shifted = evidence.double() - evidence.double().min() + torch.finfo(torch.float64).eps
            probabilities = shifted / shifted.sum()
            if probabilities.numel() > 1:
                entropy = -(probabilities * probabilities.log()).sum() / math.log(
                    probabilities.numel()
                )
            else:
                entropy = torch.ones((), dtype=torch.float64)
            contribution_entropies.append(float(entropy.item()))
            contribution_ginis.append(_gini(shifted))
            valid_document_tokens = int(document_mask[start + local_index, 0].sum().item())
            counts = torch.bincount(selected, minlength=document_tokens)
            selected_document_fractions.append(
                float((counts > 0).sum().item() / valid_document_tokens)
            )
            repeated_selection_dominance.append(float(counts.max().item() / selected.numel()))

    scores = torch.cat(score_batches, dim=0)
    query_tokens = query_embeddings[query_mask]
    document_tokens_flat = document_embeddings.reshape(-1, document_tokens, dimension)
    document_mask_flat = document_mask.reshape(-1, document_tokens)
    document_tokens_valid = document_tokens_flat[document_mask_flat]
    pooled_queries = _masked_mean(query_embeddings, query_mask, dimension=1)
    pooled_documents = _masked_mean(document_embeddings, document_mask, dimension=2).reshape(
        -1, dimension
    )
    return {
        "scorer": "mean_maxsim_cosine",
        "storage": "padded_masks",
        "score_geometry": ranking_summary(
            scores,
            top_k=min(top_k, candidates),
            reference_scores=reference_scores,
            sample_groups=sample_groups,
        ),
        "token_utilization": {
            "positive_query_token_evidence_normalized_entropy": _quantiles(
                torch.tensor(contribution_entropies)
            ),
            "positive_query_token_evidence_gini": _quantiles(torch.tensor(contribution_ginis)),
            "positive_document_token_coverage": _quantiles(
                torch.tensor(selected_document_fractions)
            ),
            "positive_document_token_repeated_selection_dominance": _quantiles(
                torch.tensor(repeated_selection_dominance)
            ),
        },
        "representations": {
            "query_tokens": representation_summary(
                query_tokens,
                max_vectors=max_representation_vectors,
                seed=seed,
            ),
            "document_tokens": representation_summary(
                document_tokens_valid,
                max_vectors=max_representation_vectors,
                seed=seed + 1,
            ),
            "pooled_queries": representation_summary(
                pooled_queries,
                max_vectors=max_representation_vectors,
                seed=seed + 2,
            ),
            "pooled_documents": representation_summary(
                pooled_documents,
                max_vectors=max_representation_vectors,
                seed=seed + 3,
            ),
        },
    }


def _late_ragged_layout(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    query_offsets: Tensor,
    document_offsets: Tensor,
    *,
    samples: int,
) -> tuple[int, int]:
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("Ragged Late probes require 2-D packed query/document embeddings")
    if query_embeddings.size(1) != document_embeddings.size(1):
        raise ValueError(
            "Ragged Late query/document dimensions disagree: "
            f"queries={tuple(query_embeddings.shape)}, documents={tuple(document_embeddings.shape)}"
        )
    if query_offsets.numel() != samples + 1:
        raise ValueError(
            f"query_offsets must describe {samples} samples, got {query_offsets.numel() - 1}"
        )
    document_items = document_offsets.numel() - 1
    if document_items % samples:
        raise ValueError(
            f"document_offsets describes {document_items} items, not a multiple of {samples} samples"
        )
    candidates = document_items // samples
    if candidates < 2:
        raise ValueError(f"Late probes require at least two candidates, got {candidates}")
    return candidates, query_embeddings.size(1)


def late_ragged_probe_scores(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    query_offsets: Tensor,
    document_offsets: Tensor,
    *,
    samples: int,
    batch_size: int,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    candidates, dimension = _late_ragged_layout(
        query_embeddings,
        document_embeddings,
        query_offsets,
        document_offsets,
        samples=samples,
    )
    score_batches = []
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        queries, query_mask = _pad_ragged_batch(query_embeddings, query_offsets, start, stop)
        documents, document_mask = _pad_ragged_batch(
            document_embeddings,
            document_offsets,
            start * candidates,
            stop * candidates,
        )
        documents = documents.reshape(stop - start, candidates, documents.size(1), dimension)
        document_mask = document_mask.reshape(stop - start, candidates, document_mask.size(1))
        scores, _, _ = _late_batch_scores(
            F.normalize(queries.float(), p=2, dim=-1),
            F.normalize(documents.float(), p=2, dim=-1),
            query_mask,
            document_mask,
        )
        score_batches.append(scores)
    return torch.cat(score_batches, dim=0)


def late_ragged_probe_metrics(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    query_offsets: Tensor,
    document_offsets: Tensor,
    *,
    samples: int,
    batch_size: int,
    max_representation_vectors: int,
    seed: int,
    top_k: int,
    reference_scores: Tensor | None = None,
    sample_groups: list[str] | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    candidates, dimension = _late_ragged_layout(
        query_embeddings,
        document_embeddings,
        query_offsets,
        document_offsets,
        samples=samples,
    )
    score_batches: list[Tensor] = []
    pooled_query_batches: list[Tensor] = []
    pooled_document_batches: list[Tensor] = []
    contribution_entropies: list[float] = []
    contribution_ginis: list[float] = []
    selected_document_fractions: list[float] = []
    repeated_selection_dominance: list[float] = []

    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        queries, query_mask = _pad_ragged_batch(query_embeddings, query_offsets, start, stop)
        documents, document_mask = _pad_ragged_batch(
            document_embeddings,
            document_offsets,
            start * candidates,
            stop * candidates,
        )
        document_tokens = documents.size(1)
        documents = documents.reshape(stop - start, candidates, document_tokens, dimension)
        document_mask = document_mask.reshape(stop - start, candidates, document_tokens)
        batch_scores, contributions, selections = _late_batch_scores(
            F.normalize(queries.float(), p=2, dim=-1),
            F.normalize(documents.float(), p=2, dim=-1),
            query_mask,
            document_mask,
        )
        score_batches.append(batch_scores)
        pooled_query_batches.append(_masked_mean(queries, query_mask, dimension=1))
        pooled_document_batches.append(
            _masked_mean(documents, document_mask, dimension=2).reshape(-1, dimension)
        )
        for local_index, (evidence, selected) in enumerate(zip(contributions, selections)):
            shifted = evidence.double() - evidence.double().min() + torch.finfo(torch.float64).eps
            probabilities = shifted / shifted.sum()
            if probabilities.numel() > 1:
                entropy = -(probabilities * probabilities.log()).sum() / math.log(
                    probabilities.numel()
                )
            else:
                entropy = torch.ones((), dtype=torch.float64)
            contribution_entropies.append(float(entropy.item()))
            contribution_ginis.append(_gini(shifted))
            valid_document_tokens = int(document_mask[local_index, 0].sum().item())
            counts = torch.bincount(selected, minlength=document_tokens)
            selected_document_fractions.append(
                float((counts > 0).sum().item() / valid_document_tokens)
            )
            repeated_selection_dominance.append(float(counts.max().item() / selected.numel()))

    scores = torch.cat(score_batches, dim=0)
    return {
        "scorer": "mean_maxsim_cosine",
        "storage": "ragged_offsets",
        "score_geometry": ranking_summary(
            scores,
            top_k=min(top_k, candidates),
            reference_scores=reference_scores,
            sample_groups=sample_groups,
        ),
        "token_utilization": {
            "positive_query_token_evidence_normalized_entropy": _quantiles(
                torch.tensor(contribution_entropies)
            ),
            "positive_query_token_evidence_gini": _quantiles(torch.tensor(contribution_ginis)),
            "positive_document_token_coverage": _quantiles(
                torch.tensor(selected_document_fractions)
            ),
            "positive_document_token_repeated_selection_dominance": _quantiles(
                torch.tensor(repeated_selection_dominance)
            ),
        },
        "representations": {
            "query_tokens": representation_summary(
                query_embeddings,
                max_vectors=max_representation_vectors,
                seed=seed,
            ),
            "document_tokens": representation_summary(
                document_embeddings,
                max_vectors=max_representation_vectors,
                seed=seed + 1,
            ),
            "pooled_queries": representation_summary(
                torch.cat(pooled_query_batches, dim=0),
                max_vectors=max_representation_vectors,
                seed=seed + 2,
            ),
            "pooled_documents": representation_summary(
                torch.cat(pooled_document_batches, dim=0),
                max_vectors=max_representation_vectors,
                seed=seed + 3,
            ),
        },
    }


def _late_archive_layout(files: set[str], label: str) -> Literal["padded_masks", "ragged_offsets"]:
    padded = {"query_mask", "document_mask"}.issubset(files)
    ragged = {"query_offsets", "document_offsets"}.issubset(files)
    if padded == ragged:
        raise ValueError(
            f"{label} must contain exactly one Late storage layout: "
            "query_mask/document_mask or query_offsets/document_offsets"
        )
    return "padded_masks" if padded else "ragged_offsets"


def analyze_probe(
    source: Path,
    output: Path,
    *,
    family: ModelFamily,
    label: str | None = None,
    batch_size: int = 8,
    max_representation_vectors: int = 10_000,
    seed: int = 42,
    top_k: int = 3,
    require_export_manifest: bool = False,
    reference_source: Path | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = _sha256(source)
    with np.load(source, allow_pickle=False) as archive:
        required = {"sample_ids", "query_embeddings", "document_embeddings"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Probe archive is missing arrays: {sorted(missing)}")
        late_layout = (
            _late_archive_layout(set(archive.files), "Late probe archive")
            if family == "late"
            else None
        )
        sample_ids = np.asarray(archive["sample_ids"])
        if sample_ids.ndim != 1 or sample_ids.size == 0:
            raise ValueError(f"sample_ids must be non-empty 1-D, got {sample_ids.shape}")
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        queries = _as_float_tensor(archive["query_embeddings"], "query_embeddings")
        documents = _as_float_tensor(archive["document_embeddings"], "document_embeddings")
        if family == "dense" or late_layout == "padded_masks":
            counts_match = (
                queries.size(0) == sample_ids.size and documents.size(0) == sample_ids.size
            )
        else:
            counts_match = True
        if not counts_match:
            raise ValueError(
                "sample_ids/query/document counts disagree: "
                f"{sample_ids.size}, {queries.size(0)}, {documents.size(0)}"
            )
        reference_scores = None
        if "reference_scores" in archive.files and reference_source is not None:
            raise ValueError("Use either embedded reference_scores or --reference-input, not both")
        if "reference_scores" in archive.files:
            reference_scores = _as_float_tensor(archive["reference_scores"], "reference_scores")
        sample_groups = None
        if "sample_groups" in archive.files:
            raw_groups = np.asarray(archive["sample_groups"])
            if raw_groups.ndim != 1 or raw_groups.size != sample_ids.size:
                raise ValueError(
                    "sample_groups must match sample_ids: "
                    f"expected {(sample_ids.size,)}, got {raw_groups.shape}"
                )
            if raw_groups.dtype.kind not in "SUiu":
                raise ValueError(
                    "sample_groups must use a pickle-free string or integer dtype, "
                    f"got {raw_groups.dtype}"
                )
            sample_groups = [str(value) for value in raw_groups.tolist()]
        array_metadata = {
            name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
            for name in sorted(archive.files)
        }
        reference_identity = None
        if reference_source is not None:
            reference_scores, reference_identity = _reference_probe_scores(
                reference_source,
                family=family,
                sample_ids=sample_ids,
                batch_size=batch_size,
                require_export_manifest=require_export_manifest,
            )
        if family == "dense":
            metrics = dense_probe_metrics(
                queries,
                documents,
                max_representation_vectors=max_representation_vectors,
                seed=seed,
                top_k=top_k,
                reference_scores=reference_scores,
                sample_groups=sample_groups,
            )
        elif family == "late" and late_layout == "padded_masks":
            if queries.ndim != 3 or documents.ndim != 4:
                raise ValueError("late probes require 3-D queries and 4-D documents")
            query_mask = _as_mask(
                archive["query_mask"],
                "query_mask",
                (queries.size(0), queries.size(1)),
            )
            document_mask = _as_mask(
                archive["document_mask"],
                "document_mask",
                (documents.size(0), documents.size(1), documents.size(2)),
            )
            metrics = late_probe_metrics(
                queries,
                documents,
                query_mask,
                document_mask,
                batch_size=batch_size,
                max_representation_vectors=max_representation_vectors,
                seed=seed,
                top_k=top_k,
                reference_scores=reference_scores,
                sample_groups=sample_groups,
            )
        elif family == "late" and late_layout == "ragged_offsets":
            query_offsets = _as_offsets(
                archive["query_offsets"],
                "query_offsets",
                items=sample_ids.size,
                total_tokens=queries.size(0),
            )
            document_offset_array = np.asarray(archive["document_offsets"])
            if (
                document_offset_array.ndim != 1
                or document_offset_array.size < sample_ids.size * 2 + 1
            ):
                raise ValueError(
                    "document_offsets does not describe at least two candidates per sample"
                )
            document_offsets = _as_offsets(
                document_offset_array,
                "document_offsets",
                items=document_offset_array.size - 1,
                total_tokens=documents.size(0),
            )
            metrics = late_ragged_probe_metrics(
                queries,
                documents,
                query_offsets,
                document_offsets,
                samples=sample_ids.size,
                batch_size=batch_size,
                max_representation_vectors=max_representation_vectors,
                seed=seed,
                top_k=top_k,
                reference_scores=reference_scores,
                sample_groups=sample_groups,
            )
        else:
            raise ValueError(f"Unsupported family {family!r}")

    export_manifest = _export_manifest_identity(
        source,
        source_sha256=source_sha256,
        family=family,
        array_metadata=array_metadata,
        required=require_export_manifest,
    )
    if export_manifest is not None and reference_identity is not None:
        reference_manifest = reference_identity.get("export_manifest")
        if reference_manifest is not None:
            for key in ("probe_manifest_sha256", "probe_selection_sha256"):
                if export_manifest.get(key) != reference_manifest.get(key):
                    raise ValueError(f"Current and reference exports disagree on {key}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "label": label,
        "input": {
            "path": str(source),
            "sha256": source_sha256,
            "arrays": array_metadata,
            "export_manifest": export_manifest,
            "reference": reference_identity,
        },
        "parameters": {
            "batch_size": batch_size,
            "max_representation_vectors": max_representation_vectors,
            "seed": seed,
            "top_k": top_k,
            "positive_candidate_index": 0,
            "token_evidence_distribution": "minimum-shifted-positive-normalized",
            "require_export_manifest": require_export_manifest,
        },
        "metrics": metrics,
    }
    _atomic_json(output.resolve(), payload)
    return payload


def _reference_probe_scores(
    source: Path,
    *,
    family: ModelFamily,
    sample_ids: np.ndarray,
    batch_size: int,
    require_export_manifest: bool,
) -> tuple[Tensor, dict[str, Any]]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = _sha256(source)
    with np.load(source, allow_pickle=False) as archive:
        required = {"sample_ids", "query_embeddings", "document_embeddings"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Reference probe archive is missing arrays: {sorted(missing)}")
        late_layout = (
            _late_archive_layout(set(archive.files), "Late reference probe archive")
            if family == "late"
            else None
        )
        reference_ids = np.asarray(archive["sample_ids"])
        if not np.array_equal(reference_ids, sample_ids):
            raise ValueError("Current and reference probe sample_ids differ or are reordered")
        queries = _as_float_tensor(archive["query_embeddings"], "reference query_embeddings")
        documents = _as_float_tensor(
            archive["document_embeddings"], "reference document_embeddings"
        )
        array_metadata = {
            name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
            for name in sorted(archive.files)
        }
        if family == "dense":
            scores = dense_probe_scores(queries, documents)
        elif late_layout == "padded_masks":
            if queries.ndim != 3 or documents.ndim != 4:
                raise ValueError("Late reference probes require 3-D queries and 4-D documents")
            query_mask = _as_mask(
                archive["query_mask"],
                "reference query_mask",
                (queries.size(0), queries.size(1)),
            )
            document_mask = _as_mask(
                archive["document_mask"],
                "reference document_mask",
                (documents.size(0), documents.size(1), documents.size(2)),
            )
            scores = late_probe_scores(
                queries,
                documents,
                query_mask,
                document_mask,
                batch_size=batch_size,
            )
        else:
            query_offsets = _as_offsets(
                archive["query_offsets"],
                "reference query_offsets",
                items=sample_ids.size,
                total_tokens=queries.size(0),
            )
            document_offset_array = np.asarray(archive["document_offsets"])
            if (
                document_offset_array.ndim != 1
                or document_offset_array.size < sample_ids.size * 2 + 1
            ):
                raise ValueError(
                    "reference document_offsets does not describe at least two candidates per sample"
                )
            document_offsets = _as_offsets(
                document_offset_array,
                "reference document_offsets",
                items=document_offset_array.size - 1,
                total_tokens=documents.size(0),
            )
            scores = late_ragged_probe_scores(
                queries,
                documents,
                query_offsets,
                document_offsets,
                samples=sample_ids.size,
                batch_size=batch_size,
            )
    export_manifest = _export_manifest_identity(
        source,
        source_sha256=source_sha256,
        family=family,
        array_metadata=array_metadata,
        required=require_export_manifest,
    )
    return scores, {
        "path": str(source),
        "sha256": source_sha256,
        "arrays": array_metadata,
        "export_manifest": export_manifest,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze fixed dense or late-interaction probe embeddings"
    )
    parser.add_argument("--input", type=Path, required=True, help="Versioned .npz probe export")
    parser.add_argument("--output", type=Path, required=True, help="Atomic JSON result")
    parser.add_argument("--family", choices=("dense", "late"), required=True)
    parser.add_argument("--label")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-representation-vectors", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--require-export-manifest", action="store_true")
    parser.add_argument("--reference-input", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = analyze_probe(
        args.input,
        args.output,
        family=args.family,
        label=args.label,
        batch_size=args.batch_size,
        max_representation_vectors=args.max_representation_vectors,
        seed=args.seed,
        top_k=args.top_k,
        require_export_manifest=args.require_export_manifest,
        reference_source=args.reference_input,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
