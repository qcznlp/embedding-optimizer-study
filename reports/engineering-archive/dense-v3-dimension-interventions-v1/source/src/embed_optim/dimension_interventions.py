"""Literal-coordinate interventions, independent of run names or publication admission.

The frozen estimands and shared masks/rotations are retained. Coordinate deletion
re-normalizes the remaining vectors directly; a total-minus-component subtraction
and a positive denominator floor are not valid substitutes near concentrated axes.
Callers must independently authenticate the vectors and complete state population.
"""

from __future__ import annotations

import math

import numpy as np

from .dimension_utilization import (
    _aggregate_checkpoint,
    _cosine_scores,
    _covariance_summary,
    _orthogonal_matrix,
    _prefix,
    _query_metrics,
    _require_rotation_invariance,
    _shared_masks,
    _summary_rows,
)

NUMERICAL_POLICY = {
    "version": 1,
    "cosine": "Explicit retained-coordinate FP64 normalization and cosine; no subtraction of a dominant coordinate from a total reduction",
    "leave_one_out": "Every coordinate is literally deleted; use the same cosine and strict-greater-than positive-rank kernel as random removal",
    "zero_remaining_norm": "Reject the complete state computation; no denominator floor, imputation, row exclusion or partial pooling",
    "array_storage": "Retain task-mean coordinate attributions in float64 without float32 archive rounding",
    "relative_margin": "Undefined (null) when the absolute task-mean baseline margin is at most the unchanged 1e-12 boundary; all absolute outcomes remain",
    "rotation": "Original three seeds and full-score 1e-6 plus exact positive-rank equality guard; no new tolerance",
    "scientific_estimands_changed": False,
    "old_sources_or_outputs_rewritten": False,
}


def validate_vectors(queries, documents):
    q, d = np.asarray(queries), np.asarray(documents)
    if (
        q.ndim != 2
        or d.ndim != 3
        or not q.shape[0]
        or q.shape[1] < 2
        or d.shape != (q.shape[0], 8, q.shape[1])
        or q.dtype.kind != "f"
        or d.dtype.kind != "f"
        or not np.isfinite(q).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError(
            "Require finite floating query/eight-document arrays of matching dimension"
        )
    return q.astype(np.float64, copy=False), d.astype(np.float64, copy=False)


def leave_one_out_metrics(queries, documents):
    """Same literal operation for every column, including very concentrated vectors."""
    queries, documents = validate_vectors(queries, documents)
    dimension = queries.shape[1]
    ndcg, margin = (np.empty(queries.shape, dtype=np.float64) for _ in range(2))
    for coordinate in range(dimension):
        keep = np.arange(dimension) != coordinate
        try:
            scores = _cosine_scores(queries, documents, keep)
        except ValueError as exc:
            raise ValueError(f"Coordinate {coordinate}: {exc}") from exc
        if not np.isfinite(scores).all():
            raise ValueError(f"Coordinate {coordinate}: non-finite literal-deletion cosine")
        values = _query_metrics(scores)
        ndcg[:, coordinate], margin[:, coordinate] = values[:2]
    return ndcg, margin


def _metadata(meta):
    if (
        set(meta) != {"run_id", "optimizer", "learning_rate", "step"}
        or not isinstance(meta["run_id"], str)
        or not meta["run_id"]
        or type(meta["step"]) is not int
        or meta["step"] < 0
    ):
        raise ValueError("State metadata must be explicit, complete and unambiguous")
    if meta["optimizer"] == "pretrained":
        if meta != {
            "run_id": "pretrained",
            "optimizer": "pretrained",
            "learning_rate": None,
            "step": 0,
        }:
            raise ValueError("Invalid pretrained state identity")
        return {**meta, "learning_rate": math.nan}
    if (
        meta["optimizer"] not in {"adamw", "muon", "normuon"}
        or isinstance(meta["learning_rate"], bool)
        or not isinstance(meta["learning_rate"], (float, int))
        or not math.isfinite(meta["learning_rate"])
        or meta["learning_rate"] <= 0
        or meta["step"] <= 0
    ):
        raise ValueError("Invalid declared optimizer state")
    return dict(meta)


def task_attributions(groups, ndcg_gain, margin_gain):
    names = sorted(set(groups.tolist()))
    return {
        "task_groups": np.asarray(names, dtype=str),
        "ndcg_removal_gain": np.stack([np.mean(ndcg_gain[groups == g], axis=0) for g in names]),
        "margin_removal_gain": np.stack([np.mean(margin_gain[groups == g], axis=0) for g in names]),
    }


def compute_state(meta, arrays, protocol):
    """Pure complete-state numerical helper; this does not establish a primary identity."""
    meta = _metadata(meta)
    q, d = validate_vectors(arrays["query_embeddings"], arrays["document_embeddings"])
    groups = np.asarray(arrays["sample_groups"])
    sample_ids = np.asarray(arrays["sample_ids"])
    if (
        groups.shape != (len(q),)
        or groups.dtype.kind != "U"
        or any(not group for group in groups.tolist())
        or sample_ids.shape != (len(q),)
        or sample_ids.dtype.kind not in "iu"
        or len(np.unique(sample_ids)) != len(q)
        or np.any(sample_ids < 0)
        or q.shape[1] != protocol["inputs"]["embedding_dimension"]
    ):
        raise ValueError("State samples, groups or declared dimensions differ")
    names = sorted(set(groups.tolist()))
    scores = _cosine_scores(q, d)
    full_ndcg, full_margin, _ = _query_metrics(scores)
    ndcg, margin = leave_one_out_metrics(q, d)
    ndcg_gain, margin_gain = ndcg - full_ndcg[:, None], margin - full_margin[:, None]
    tolerance = protocol["coordinate_ablation"]["zero_tolerance"]
    task_rows = _summary_rows(
        meta, groups, full_ndcg, full_margin, ndcg_gain, margin_gain, tolerance
    )
    checkpoint = _aggregate_checkpoint(task_rows)
    for prefix, vectors in (("query", q), ("document", d.reshape(-1, q.shape[1]))):
        checkpoint.update({f"{prefix}_{k}": v for k, v in _covariance_summary(vectors).items()})
    retention = []
    for mask in _shared_masks(q.shape[1], protocol):
        current = _query_metrics(_cosine_scores(q, d, mask["keep"]))
        for name in names:
            selected = groups == name
            baseline_ndcg, baseline_margin = (
                float(np.mean(values[selected])) for values in (full_ndcg, full_margin)
            )
            current_ndcg, current_margin = (
                float(np.mean(values[selected])) for values in current[:2]
            )
            retention.append(
                {
                    **_prefix(meta),
                    "task": name,
                    "removed_fraction": mask["removed_fraction"],
                    "draw": mask["draw"],
                    "shortlist_ndcg": current_ndcg,
                    "relative_shortlist_ndcg": current_ndcg / baseline_ndcg,
                    "margin": current_margin,
                    "relative_margin": current_margin / baseline_margin
                    if abs(baseline_margin) > 1e-12
                    else None,
                }
            )
    rotations, rotation_checks, rotated_attributions = [], [], []
    if meta["step"] in (0, protocol["inputs"]["checkpoint_stages"][-1]):
        for seed in protocol["rotation_control"]["seeds"]:
            rotation = _orthogonal_matrix(q.shape[1], seed)
            rotated_q, rotated_d = q @ rotation, d @ rotation
            rotated_scores = _cosine_scores(rotated_q, rotated_d)
            _require_rotation_invariance(scores, rotated_scores, meta["run_id"])
            rotated_full = _query_metrics(rotated_scores)
            rotated_loo = leave_one_out_metrics(rotated_q, rotated_d)
            gains = [a - b[:, None] for a, b in zip(rotated_loo, rotated_full[:2], strict=True)]
            rotations.extend(
                _summary_rows(
                    meta, groups, *rotated_full[:2], *gains, tolerance, rotation_seed=seed
                )
            )
            rotation_checks.append(
                {
                    "seed": seed,
                    "maximum_score_absolute_error": float(np.max(np.abs(scores - rotated_scores))),
                    "positive_ranks_identical": True,
                }
            )
            rotated_attributions.append({"seed": seed, **task_attributions(groups, *gains)})
    return {
        "tables": {
            "checkpoint_summary": [checkpoint],
            "task_summary": task_rows,
            "random_removal": retention,
            "rotation_summary": rotations,
        },
        "attributions": task_attributions(groups, ndcg_gain, margin_gain),
        "rotated_attributions": rotated_attributions,
        "rotation_checks": rotation_checks,
        "numerical_policy": NUMERICAL_POLICY,
        "scientific_completion": False,
    }
