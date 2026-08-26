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
) -> dict[str, str] | None:
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
    return {"path": str(manifest_path), "sha256": _sha256(manifest_path)}


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
    queries = F.normalize(query_embeddings.float(), p=2, dim=-1)
    documents = F.normalize(document_embeddings.float(), p=2, dim=-1)
    scores = torch.einsum("bd,bcd->bc", queries, documents)
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
) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = _sha256(source)
    with np.load(source, allow_pickle=False) as archive:
        required = {"sample_ids", "query_embeddings", "document_embeddings"}
        if family == "late":
            required.update({"query_mask", "document_mask"})
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Probe archive is missing arrays: {sorted(missing)}")
        sample_ids = np.asarray(archive["sample_ids"])
        if sample_ids.ndim != 1 or sample_ids.size == 0:
            raise ValueError(f"sample_ids must be non-empty 1-D, got {sample_ids.shape}")
        if len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids must be unique")
        queries = _as_float_tensor(archive["query_embeddings"], "query_embeddings")
        documents = _as_float_tensor(archive["document_embeddings"], "document_embeddings")
        if queries.size(0) != sample_ids.size or documents.size(0) != sample_ids.size:
            raise ValueError(
                "sample_ids/query/document counts disagree: "
                f"{sample_ids.size}, {queries.size(0)}, {documents.size(0)}"
            )
        reference_scores = None
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
        elif family == "late":
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
        else:
            raise ValueError(f"Unsupported family {family!r}")

    export_manifest = _export_manifest_identity(
        source,
        source_sha256=source_sha256,
        family=family,
        array_metadata=array_metadata,
        required=require_export_manifest,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "label": label,
        "input": {
            "path": str(source),
            "sha256": source_sha256,
            "arrays": array_metadata,
            "export_manifest": export_manifest,
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
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
