"""Audit how optimizer trajectories redistribute utility across embedding coordinates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = 1
RUN_PATTERN = re.compile(r"^(?:padded-)?(adamw|muon|normuon)-(?:lr)?(.+)$")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)\.npz$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = resolved
    if root is not None:
        try:
            display = resolved.relative_to(root.resolve())
        except ValueError:
            pass
    return {
        "path": display.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _parse_archive(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root)
    if relative.as_posix() == "pretrained.npz":
        return {
            "run_id": "pretrained",
            "optimizer": "pretrained",
            "learning_rate": math.nan,
            "step": 0,
        }
    if len(relative.parts) != 2:
        raise ValueError(f"Unexpected export path: {relative}")
    run_match = RUN_PATTERN.fullmatch(relative.parts[0])
    checkpoint_match = CHECKPOINT_PATTERN.fullmatch(relative.parts[1])
    if run_match is None or checkpoint_match is None:
        raise ValueError(f"Unexpected export path: {relative}")
    return {
        "run_id": relative.parts[0],
        "optimizer": run_match.group(1),
        "learning_rate": float(run_match.group(2)),
        "step": int(checkpoint_match.group(1)),
    }


def _validate_export(
    path: Path, dimension: int, *, probe_spec: Path | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest.get("output", {})
    if manifest.get("schema_version") != 1 or manifest.get("family") != "dense":
        raise ValueError(f"Unexpected export manifest identity: {manifest_path}")
    if output.get("bytes") != path.stat().st_size or output.get("sha256") != _sha256(path):
        raise ValueError(f"Export bytes or digest disagree with manifest: {path}")
    if manifest.get("encoding", {}).get("positive_candidate_index") != 0:
        raise ValueError(f"Export is not positive-first: {manifest_path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    expected = {
        "sample_ids": (224,),
        "sample_groups": (224,),
        "query_embeddings": (224, dimension),
        "document_embeddings": (224, 8, dimension),
    }
    if set(arrays) != set(expected):
        raise ValueError(f"Unexpected arrays in {path}: {sorted(arrays)}")
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{path}:{name} has shape {arrays[name].shape}, expected {shape}")
    for name in ("query_embeddings", "document_embeddings"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Non-finite embeddings in {path}:{name}")
    ids = arrays["sample_ids"]
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Sample IDs are not unique integers in {path}")
    sample_digest = hashlib.sha256()
    for sample_id in ids:
        sample_digest.update(f"{int(sample_id)}\n".encode())
    probe = manifest.get("probe", {})
    if sample_digest.hexdigest() != probe.get("selected_sample_ids_sha256"):
        raise ValueError(f"Sample IDs differ from export provenance in {path}")
    if probe_spec is not None:
        spec = json.loads(probe_spec.read_text(encoding="utf-8"))
        frozen = spec["expected"]
        if any(
            probe.get(key) != frozen[key]
            for key in ("manifest_sha256", "selection_sha256", "selected_sample_ids_sha256")
        ) or probe.get("frozen_spec", {}).get("sha256") != _sha256(probe_spec):
            raise ValueError(f"Export uses a different frozen probe: {path}")
        if dict(Counter(arrays["sample_groups"].astype(str))) != frozen["task_counts"]:
            raise ValueError(f"Export task counts differ from the frozen probe: {path}")
    return arrays, manifest


def _cosine_scores(
    queries: np.ndarray,
    documents: np.ndarray,
    keep: np.ndarray | None = None,
) -> np.ndarray:
    queries = queries.astype(np.float64, copy=False)
    documents = documents.astype(np.float64, copy=False)
    if keep is not None:
        queries = queries[:, keep]
        documents = documents[:, :, keep]
    query_norms = np.linalg.norm(queries, axis=-1, keepdims=True)
    document_norms = np.linalg.norm(documents, axis=-1, keepdims=True)
    if np.any(query_norms == 0) or np.any(document_norms == 0):
        raise ValueError("Coordinate removal produced a zero-norm embedding")
    queries = queries / query_norms
    documents = documents / document_norms
    return np.einsum("nd,ncd->nc", queries, documents, optimize=True)


def _query_metrics(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = scores[:, 0]
    ranks = 1 + np.sum(scores[:, 1:] > positive[:, None], axis=1)
    ndcg = 1.0 / np.log2(ranks + 1.0)
    margin = positive - np.max(scores[:, 1:], axis=1)
    return ndcg, margin, ranks


def _leave_one_out_metrics(
    queries: np.ndarray,
    documents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    queries = queries.astype(np.float64, copy=False)
    documents = documents.astype(np.float64, copy=False)
    dot = np.einsum("nd,ncd->nc", queries, documents, optimize=True)
    query_sq = np.square(queries)
    document_sq = np.square(documents)
    query_total = np.sum(query_sq, axis=1)
    document_total = np.sum(document_sq, axis=2)
    numerator = dot[:, :, None] - queries[:, None, :] * documents
    query_denominator = np.sqrt(np.maximum(query_total[:, None] - query_sq, 1e-30))
    document_denominator = np.sqrt(np.maximum(document_total[:, :, None] - document_sq, 1e-30))
    scores = numerator / (query_denominator[:, None, :] * document_denominator)
    positive = scores[:, 0, :]
    ranks = 1 + np.sum(scores[:, 1:, :] > positive[:, None, :], axis=1)
    ndcg = 1.0 / np.log2(ranks + 1.0)
    margin = positive - np.max(scores[:, 1:, :], axis=1)
    return ndcg, margin


def _participation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    denominator = values.size * np.sum(np.square(values))
    if denominator <= 0:
        return 0.0
    return float(np.square(np.sum(values)) / denominator)


def _attribution_summary(gain: np.ndarray, tolerance: float) -> dict[str, float]:
    degrading = np.maximum(gain, 0.0)
    helpful = np.maximum(-gain, 0.0)
    absolute = np.abs(gain)
    total = float(np.sum(absolute))
    return {
        "degrading_coordinate_fraction": float(np.mean(gain > tolerance)),
        "degrading_attribution_mass": float(np.sum(degrading)),
        "helpful_attribution_mass": float(np.sum(helpful)),
        "helpful_mass_share": float(np.sum(helpful) / total) if total > 0 else 0.0,
        "absolute_attribution_participation_ratio": _participation(absolute),
        "helpful_attribution_participation_ratio": _participation(helpful),
    }


def _covariance_summary(vectors: np.ndarray) -> dict[str, float]:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(f"Covariance summary requires non-trivial 2-D vectors, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Covariance summary received a non-finite embedding")
    centered = values - np.mean(values, axis=0, keepdims=True)
    denominator = values.shape[0] - 1
    if centered.shape[0] <= centered.shape[1]:
        spectrum_matrix = centered @ centered.T / denominator
    else:
        spectrum_matrix = centered.T @ centered / denominator
    eigenvalues = np.linalg.eigvalsh(spectrum_matrix)
    positive = eigenvalues[eigenvalues > np.finfo(eigenvalues.dtype).eps]
    if positive.size == 0:
        return {
            "entropy_effective_rank": 0.0,
            "normalized_effective_rank": 0.0,
            "stable_rank": 0.0,
            "leading_variance_fraction": 0.0,
        }
    total = float(np.sum(positive))
    probabilities = positive / total
    effective = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    rank_limit = min(values.shape[0] - 1, values.shape[1])
    return {
        "entropy_effective_rank": effective,
        "normalized_effective_rank": effective / rank_limit,
        "stable_rank": total / float(np.max(positive)),
        "leading_variance_fraction": float(np.max(positive)) / total,
    }


def _prefix(meta: dict[str, Any]) -> dict[str, Any]:
    rate = meta["learning_rate"]
    return {
        "run_id": meta["run_id"],
        "optimizer": meta["optimizer"],
        "learning_rate": "" if not math.isfinite(rate) else f"{rate:.12g}",
        "step": meta["step"],
    }


def _shared_masks(dimension: int, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    spec = protocol["random_removal"]
    generator = np.random.default_rng(int(spec["master_seed"]))
    rows = []
    for fraction in spec["removed_fractions"]:
        removed = int(round(float(fraction) * dimension))
        for draw in range(int(spec["mask_draws_per_fraction"])):
            indices = generator.choice(dimension, size=removed, replace=False)
            keep = np.ones(dimension, dtype=bool)
            keep[indices] = False
            rows.append({"removed_fraction": float(fraction), "draw": draw, "keep": keep})
    return rows


def _orthogonal_matrix(dimension: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def _summary_rows(
    meta: dict[str, Any],
    groups: np.ndarray,
    baseline_ndcg: np.ndarray,
    baseline_margin: np.ndarray,
    ndcg_gain: np.ndarray,
    margin_gain: np.ndarray,
    tolerance: float,
    *,
    rotation_seed: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        ndcg_by_dimension = np.mean(ndcg_gain[selected], axis=0)
        margin_by_dimension = np.mean(margin_gain[selected], axis=0)
        row: dict[str, Any] = {
            **_prefix(meta),
            "task": group,
            "rotation_seed": "" if rotation_seed is None else rotation_seed,
            "baseline_shortlist_ndcg": float(np.mean(baseline_ndcg[selected])),
            "baseline_margin": float(np.mean(baseline_margin[selected])),
        }
        row.update(
            {
                f"ndcg_{key}": value
                for key, value in _attribution_summary(ndcg_by_dimension, tolerance).items()
            }
        )
        row.update(
            {
                f"margin_{key}": value
                for key, value in _attribution_summary(margin_by_dimension, tolerance).items()
            }
        )
        rows.append(row)
    return rows


def _aggregate_checkpoint(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("Cannot aggregate an empty checkpoint")
    ignored = {"run_id", "optimizer", "learning_rate", "step", "task", "rotation_seed"}
    result = {key: values[0][key] for key in ("run_id", "optimizer", "learning_rate", "step")}
    result["tasks"] = len(values)
    for key in values[0]:
        if key in ignored:
            continue
        result[key] = float(np.mean([float(row[key]) for row in values]))
    return result


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "prospective_corrected_dimension_utilization_lock":
        raise ValueError("Dimension-utilization protocol status differs")
    return protocol


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol(args.protocol)
    if args.analysis_scope == "corrected" and args.skip_rotation:
        raise ValueError("The prospective corrected analysis requires every rotation control")
    dimension = int(protocol["inputs"]["embedding_dimension"])
    expected_checkpoints = int(protocol["inputs"][f"{args.analysis_scope}_expected_checkpoints"])
    paths = sorted(args.input_root.glob("*/checkpoint-*.npz"))
    pretrained = args.input_root / "pretrained.npz"
    if len(paths) != expected_checkpoints or not pretrained.is_file():
        raise ValueError(
            f"Expected {expected_checkpoints} checkpoint exports plus pretrained, got "
            f"{len(paths)} checkpoints and pretrained={pretrained.is_file()}"
        )
    paths = [pretrained, *paths]
    probe_spec = args.repository / protocol["inputs"]["probe_spec"]
    primary_handoff = None
    if args.analysis_scope == "corrected":
        from .config import load_matrix
        from .primary_dimension_probe import audit_exports

        configs = load_matrix(args.repository / "configs/dense_no_packing_retrain.yaml")
        expected_paths = {
            f"{config.run_id}/checkpoint-{step}.npz"
            for config in configs
            for step in protocol["inputs"]["checkpoint_stages"]
        }
        if {str(path.relative_to(args.input_root)) for path in paths[1:]} != expected_paths:
            raise ValueError("Corrected dimension exports do not cover the exact primary matrix")
        audit_exports(args.input_root, args.repository)
        primary_handoff = _identity(args.input_root / "primary_exports.json", args.repository)
    masks = _shared_masks(dimension, protocol)
    tolerance = float(protocol["coordinate_ablation"]["zero_tolerance"])

    task_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    input_records = []
    identities = []
    ndcg_arrays = []
    margin_arrays = []
    group_names: list[str] | None = None
    sample_reference: tuple[np.ndarray, np.ndarray] | None = None

    final_payloads: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    for path in paths:
        meta = _parse_archive(path, args.input_root)
        arrays, _manifest = _validate_export(path, dimension, probe_spec=probe_spec)
        groups = arrays["sample_groups"].astype(str)
        if sample_reference is None:
            sample_reference = (arrays["sample_ids"].copy(), groups.copy())
        elif not (
            np.array_equal(sample_reference[0], arrays["sample_ids"])
            and np.array_equal(sample_reference[1], groups)
        ):
            raise ValueError(f"Paired sample IDs or task assignments differ in {path}")
        if group_names is None:
            group_names = sorted(set(groups.tolist()))
        elif group_names != sorted(set(groups.tolist())):
            raise ValueError(f"Task groups differ in {path}")
        queries = arrays["query_embeddings"]
        documents = arrays["document_embeddings"]
        full_scores = _cosine_scores(queries, documents)
        baseline_ndcg, baseline_margin, _ranks = _query_metrics(full_scores)
        ablated_ndcg, ablated_margin = _leave_one_out_metrics(queries, documents)
        ndcg_gain = ablated_ndcg - baseline_ndcg[:, None]
        margin_gain = ablated_margin - baseline_margin[:, None]
        rows = _summary_rows(
            meta,
            groups,
            baseline_ndcg,
            baseline_margin,
            ndcg_gain,
            margin_gain,
            tolerance,
        )
        task_rows.extend(rows)
        checkpoint_row = _aggregate_checkpoint(rows)
        for prefix, vectors in (
            ("query", queries),
            ("document", documents.reshape(-1, dimension)),
        ):
            checkpoint_row.update(
                {f"{prefix}_{key}": value for key, value in _covariance_summary(vectors).items()}
            )
        checkpoint_rows.append(checkpoint_row)
        ndcg_arrays.append(
            np.stack([np.mean(ndcg_gain[groups == group], axis=0) for group in group_names])
        )
        margin_arrays.append(
            np.stack([np.mean(margin_gain[groups == group], axis=0) for group in group_names])
        )
        for mask in masks:
            scores = _cosine_scores(queries, documents, mask["keep"])
            ndcg, margin, _ = _query_metrics(scores)
            for group in group_names:
                selected = groups == group
                base_ndcg = float(np.mean(baseline_ndcg[selected]))
                base_margin = float(np.mean(baseline_margin[selected]))
                current_ndcg = float(np.mean(ndcg[selected]))
                current_margin = float(np.mean(margin[selected]))
                retention_rows.append(
                    {
                        **_prefix(meta),
                        "task": group,
                        "removed_fraction": mask["removed_fraction"],
                        "draw": mask["draw"],
                        "shortlist_ndcg": current_ndcg,
                        "relative_shortlist_ndcg": current_ndcg / base_ndcg,
                        "margin": current_margin,
                        "relative_margin": current_margin / base_margin
                        if abs(base_margin) > 1e-12
                        else math.nan,
                    }
                )
        if meta["step"] in (0, 3907):
            final_payloads.append((meta, groups, queries, documents))
        input_records.append(
            {
                "archive": _identity(path),
                "manifest": _identity(path.with_suffix(path.suffix + ".manifest.json")),
            }
        )
        identities.append([meta["run_id"], meta["optimizer"], meta["learning_rate"], meta["step"]])

    rotation_rows: list[dict[str, Any]] = []
    if not args.skip_rotation:
        for seed in protocol["rotation_control"]["seeds"]:
            rotation = _orthogonal_matrix(dimension, int(seed))
            for meta, groups, queries, documents in final_payloads:
                original_scores = _cosine_scores(queries, documents)
                rotated_queries = queries.astype(np.float64) @ rotation
                rotated_documents = documents.astype(np.float64) @ rotation
                rotated_scores = _cosine_scores(rotated_queries, rotated_documents)
                maximum_difference = float(np.max(np.abs(original_scores - rotated_scores)))
                if maximum_difference > 1e-6:
                    raise ValueError(
                        f"Rotation changed full scores by {maximum_difference} for {meta['run_id']}"
                    )
                baseline_ndcg, baseline_margin, _ = _query_metrics(rotated_scores)
                ablated_ndcg, ablated_margin = _leave_one_out_metrics(
                    rotated_queries, rotated_documents
                )
                rotation_rows.extend(
                    _summary_rows(
                        meta,
                        groups,
                        baseline_ndcg,
                        baseline_margin,
                        ablated_ndcg - baseline_ndcg[:, None],
                        ablated_margin - baseline_margin[:, None],
                        tolerance,
                        rotation_seed=int(seed),
                    )
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "checkpoint_summary": args.output_dir / "checkpoint_summary.csv",
        "task_summary": args.output_dir / "task_summary.csv",
        "random_removal": args.output_dir / "random_removal.csv",
        "rotation_summary": args.output_dir / "rotation_summary.csv",
        "coordinate_attribution": args.output_dir / "coordinate_attribution.npz",
    }
    _atomic_csv(outputs["checkpoint_summary"], checkpoint_rows)
    _atomic_csv(outputs["task_summary"], task_rows)
    _atomic_csv(outputs["random_removal"], retention_rows)
    if rotation_rows:
        _atomic_csv(outputs["rotation_summary"], rotation_rows)
    _atomic_npz(
        outputs["coordinate_attribution"],
        identities=np.asarray(identities, dtype=str),
        task_groups=np.asarray(group_names, dtype=str),
        ndcg_removal_gain=np.asarray(ndcg_arrays, dtype=np.float32),
        margin_removal_gain=np.asarray(margin_arrays, dtype=np.float32),
    )
    output_records = {
        name: _identity(path, args.repository) for name, path in outputs.items() if path.is_file()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": f"{args.analysis_scope}_dense_dimension_utilization",
        "analysis_role": (
            "exploratory_method_development"
            if args.analysis_scope == "historical"
            else "prospective_primary_dimension_analysis"
        ),
        "implementation": _identity(Path(__file__), args.repository),
        "protocol": _identity(args.protocol, args.repository),
        "primary_export_handoff": primary_handoff,
        "inputs": input_records,
        "coverage": {
            "checkpoint_exports": expected_checkpoints,
            "pretrained_exports": 1,
            "tasks": len(group_names or []),
            "dimensions": dimension,
            "task_checkpoint_rows": len(task_rows),
            "random_removal_rows": len(retention_rows),
            "rotation_rows": len(rotation_rows),
        },
        "outputs": output_records,
        "claim_boundary": protocol["claim_boundary"],
    }
    _atomic_json(args.output_dir / "summary_manifest.json", manifest)
    return manifest


def _audit_identity(
    record: dict[str, Any], repository: Path, *, allow_archived_implementation: bool = False
) -> None:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = repository / path
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or _sha256(path) != record.get("sha256")
    ):
        if allow_archived_implementation:
            archived = (
                repository
                / "reports/engineering-archive/dimension-sources"
                / f"{record.get('sha256')}.py"
            )
            if (
                archived.is_file()
                and archived.stat().st_size == int(record.get("bytes", -1))
                and _sha256(archived) == record.get("sha256")
            ):
                return
        raise ValueError(f"Dimension-utilization identity mismatch: {path}")


def audit_outputs(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_protocol(args.protocol)
    manifest_path = args.output_dir / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_checkpoints = int(protocol["inputs"][f"{args.analysis_scope}_expected_checkpoints"])
    expected_scope = f"{args.analysis_scope}_dense_dimension_utilization"
    coverage = manifest.get("coverage", {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("scope") != expected_scope
        or coverage.get("checkpoint_exports") != expected_checkpoints
        or coverage.get("pretrained_exports") != 1
        or coverage.get("tasks") != int(protocol["inputs"]["task_groups"])
        or coverage.get("dimensions") != int(protocol["inputs"]["embedding_dimension"])
        or coverage.get("task_checkpoint_rows")
        != (expected_checkpoints + 1) * int(protocol["inputs"]["task_groups"])
        or manifest.get("claim_boundary") != protocol["claim_boundary"]
    ):
        raise ValueError("Dimension-utilization manifest coverage or claim boundary differs")
    _audit_identity(manifest["protocol"], args.repository)
    _audit_identity(
        manifest["implementation"],
        args.repository,
        allow_archived_implementation=args.analysis_scope == "historical",
    )
    input_records = manifest.get("inputs", [])
    if len(input_records) != expected_checkpoints + 1:
        raise ValueError("Dimension-utilization input identity coverage differs")
    for pair in input_records:
        _audit_identity(pair["archive"], args.repository)
        _audit_identity(pair["manifest"], args.repository)
    if args.analysis_scope == "corrected":
        from .primary_dimension_probe import audit_exports

        handoff = manifest.get("primary_export_handoff")
        if not isinstance(handoff, dict):
            raise ValueError("Primary dimension output lacks its export handoff")
        _audit_identity(handoff, args.repository)
        export_root = (args.repository / handoff["path"]).parent
        export_manifest = audit_exports(export_root, args.repository)
        expected_inputs = {
            (row["export"]["sha256"], row["export_manifest"]["sha256"])
            for row in export_manifest["outputs"]
        }
        actual_inputs = {
            (row["archive"]["sha256"], row["manifest"]["sha256"]) for row in input_records
        }
        if actual_inputs != expected_inputs or len(actual_inputs) != expected_checkpoints + 1:
            raise ValueError("Primary dimension inputs differ from the audited export handoff")
    expected_outputs = {
        "checkpoint_summary",
        "task_summary",
        "random_removal",
        "coordinate_attribution",
    }
    if not args.skip_rotation:
        expected_outputs.add("rotation_summary")
    if set(manifest.get("outputs", {})) != expected_outputs:
        raise ValueError("Dimension-utilization output inventory differs")
    for identity in manifest["outputs"].values():
        _audit_identity(identity, args.repository)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/dense_dimension_utilization_protocol.json"),
    )
    parser.add_argument(
        "--analysis-scope", choices=("historical", "corrected"), default="historical"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/representation-space/decontaminated-beir/exports/dense"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dimension-utilization-historical"),
    )
    parser.add_argument("--skip-rotation", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    args.repository = args.repository.resolve()
    args.protocol = (
        (args.repository / args.protocol).resolve()
        if not args.protocol.is_absolute()
        else args.protocol.resolve()
    )
    args.input_root = (
        (args.repository / args.input_root).resolve()
        if not args.input_root.is_absolute()
        else args.input_root.resolve()
    )
    args.output_dir = (
        (args.repository / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit_only:
        manifest = audit_outputs(args)
    else:
        manifest = analyze(args)
    print(json.dumps(manifest["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
