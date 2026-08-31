from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from .candidate_breadth_data import (
    audit_candidate_breadth_data,
    load_candidate_breadth_protocol,
)
from .geometry import SCHEMA_VERSION, _atomic_json, _atomic_jsonl, _sha256
from .gradient_probe import _temperature_from_checkpoint
from .probe_export import _checkpoint_inputs, _load_model, _validate_checkpoint_family

METRICS = (
    "contrastive_loss",
    "positive_score",
    "hardest_negative_score",
    "positive_margin",
    "reciprocal_rank",
    "top1_accuracy",
)


def candidate_width_metrics(
    scores: np.ndarray,
    widths: list[int],
    *,
    temperature: float,
) -> dict[int, dict[str, np.ndarray]]:
    scores = np.asarray(scores, dtype=np.float32)
    if (
        scores.ndim != 2
        or scores.shape[1] <= max(widths)
        or not np.isfinite(scores).all()
        or temperature <= 0
    ):
        raise ValueError("Candidate scores or temperature are invalid")
    if widths != sorted(set(widths)) or min(widths) <= 0:
        raise ValueError("Candidate widths must be positive, unique, and increasing")
    positive = scores[:, 0]
    result: dict[int, dict[str, np.ndarray]] = {}
    for width in widths:
        subset = scores[:, : width + 1]
        negatives = subset[:, 1:]
        maximum = np.max(subset / temperature, axis=1, keepdims=True)
        logsumexp = maximum[:, 0] + np.log(np.exp(subset / temperature - maximum).sum(axis=1))
        hardest = np.max(negatives, axis=1)
        rank = 1 + np.sum(negatives >= positive[:, None], axis=1)
        result[width] = {
            "contrastive_loss": logsumexp - positive / temperature,
            "positive_score": positive.copy(),
            "hardest_negative_score": hardest,
            "positive_margin": positive - hardest,
            "reciprocal_rank": 1.0 / rank.astype(np.float32),
            "top1_accuracy": (rank == 1).astype(np.float32),
        }
    return result


def _encode_dense(
    model: Any,
    texts: list[str],
    *,
    prompt: str,
    batch_size: int,
    device: str,
    forward_dtype: str,
) -> np.ndarray:
    if batch_size <= 0 or forward_dtype not in {"float32", "bfloat16"}:
        raise ValueError("Invalid dense candidate-breadth encoding settings")
    device_type = torch.device(device).type
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = [prompt + str(text) for text in texts[start : start + batch_size]]
            features = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in model.tokenize(batch).items()
            }
            with torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=forward_dtype == "bfloat16",
            ):
                embeddings = F.normalize(model(features)["sentence_embedding"], p=2, dim=-1)
            chunks.append(embeddings.float().cpu().numpy())
    if not chunks:
        raise ValueError("Cannot encode an empty text collection")
    result = np.concatenate(chunks, axis=0)
    if result.shape[0] != len(texts) or not np.isfinite(result).all():
        raise ValueError("Dense candidate-breadth embeddings are incomplete or non-finite")
    return result


def _group_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        width = int(record["negative_width"])
        grouped[(width, "__all__")].append(record)
        grouped[(width, str(record["source"]))].append(record)
    summaries = []
    for (width, source), rows in sorted(grouped.items()):
        summaries.append(
            {
                "schema_version": SCHEMA_VERSION,
                "negative_width": width,
                "source": source,
                "samples": len(rows),
                **{
                    metric: sum(float(row[metric]) for row in rows) / len(rows)
                    for metric in METRICS
                },
            }
        )
    return summaries


def _identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _verify_file(root: Path, item: dict[str, Any]) -> None:
    path = root / item["path"]
    if (
        not path.is_file()
        or path.stat().st_size != item.get("bytes")
        or _sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"Candidate-breadth evaluation file changed: {path}")


def _baseline_check(
    records: list[dict[str, Any]],
    baseline_path: Path,
    *,
    tolerance: float,
) -> dict[str, Any]:
    expected = {}
    with baseline_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                expected[int(row["sample_id"])] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid baseline row {baseline_path}:{line_number}") from error
    observed = {int(row["sample_id"]): row for row in records if int(row["negative_width"]) == 7}
    if not observed or not set(observed).issubset(expected):
        raise ValueError("Width-7 baseline does not cover selected candidate-breadth samples")
    maximum_error = 0.0
    for sample_id, row in observed.items():
        for metric in METRICS:
            maximum_error = max(
                maximum_error, abs(float(row[metric]) - float(expected[sample_id][metric]))
            )
    if maximum_error > tolerance:
        raise ValueError(
            f"Width-7 candidate scores do not reproduce validation: {maximum_error} > {tolerance}"
        )
    return {
        "path": str(baseline_path.resolve()),
        "bytes": baseline_path.stat().st_size,
        "sha256": _sha256(baseline_path),
        "samples": len(observed),
        "absolute_tolerance": tolerance,
        "maximum_absolute_error": maximum_error,
    }


def run_candidate_breadth_evaluation(
    checkpoint: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    protocol_path: str | Path = "configs/candidate_breadth_probe.json",
    device: str = "cuda",
    baseline_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    protocol_path, protocol = load_candidate_breadth_protocol(protocol_path)
    # The release controller performs the expensive pinned-source reconstruction once
    # before launching the matrix.  Each checkpoint still re-audits every local file and
    # semantic row, without rescanning the upstream score/document parquet files 12 times.
    data_audit = audit_candidate_breadth_data(protocol_path, data_root, verify_source=False)
    evaluation = protocol["evaluation"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA candidate-breadth evaluation requested without CUDA")
    if not device.startswith("cuda") and (
        evaluation["flash_attention"] or evaluation["forward_dtype"] == "bfloat16"
    ):
        raise ValueError("CPU candidate-breadth evaluation requires float32 and no FlashAttention")
    checkpoint_inputs = _checkpoint_inputs(checkpoint)
    checkpoint_config = _validate_checkpoint_family(checkpoint, "dense")
    temperature = _temperature_from_checkpoint(checkpoint, "dense", None)
    if abs(temperature - float(evaluation["temperature"])) > 1e-12:
        raise ValueError("Candidate-breadth checkpoint temperature changed")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "checkpoint": {
            "path": str(checkpoint),
            "inputs": checkpoint_inputs,
            "run_config": checkpoint_config,
        },
        "protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": _sha256(protocol_path),
        },
        "data": {
            "path": str(data_root),
            "audit": data_audit,
        },
        "negative_widths": protocol["candidate_construction"]["negative_widths"],
        "temperature": temperature,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {key: manifest.get(key) for key in identity} != identity:
            raise ValueError("Existing candidate-breadth evaluation has different inputs")
        for item in manifest.get("outputs", {}).values():
            _verify_file(output_dir, item)
        return manifest
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Partial candidate-breadth evaluation exists: {output_dir}")
    if overwrite and output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()
            else:
                raise FileExistsError(f"Refusing to delete nested output directory: {path}")

    queries = Dataset.load_from_disk(str(data_root / "queries"))
    if len(queries) != data_audit["queries"]:
        raise ValueError("Candidate-breadth query Dataset changed")
    model = _load_model(
        "dense",
        checkpoint,
        dtype=torch.float32,
        device=device,
        flash_attention=bool(evaluation["flash_attention"]),
    )
    try:
        query_embeddings = _encode_dense(
            model,
            [str(value) for value in queries["query"]],
            prompt="query: ",
            batch_size=int(evaluation["query_batch_size"]),
            device=device,
            forward_dtype=evaluation["forward_dtype"],
        )
        maximum_width = max(identity["negative_widths"])
        scores = np.full((len(queries), maximum_width + 1), np.nan, dtype=np.float32)
        for source in sorted(set(str(value) for value in queries["source"])):
            candidates = Dataset.load_from_disk(str(data_root / "candidates" / source))
            for start in range(0, len(candidates), int(evaluation["document_batch_size"])):
                batch = candidates[start : start + int(evaluation["document_batch_size"])]
                embeddings = _encode_dense(
                    model,
                    [str(value) for value in batch["document"]],
                    prompt="document: ",
                    batch_size=int(evaluation["document_batch_size"]),
                    device=device,
                    forward_dtype=evaluation["forward_dtype"],
                )
                query_indices = np.asarray(batch["query_index"], dtype=np.int64)
                candidate_indices = np.asarray(batch["candidate_index"], dtype=np.int64)
                values = np.einsum(
                    "bd,bd->b", query_embeddings[query_indices], embeddings, optimize=True
                )
                scores[query_indices, candidate_indices] = values.astype(np.float32)
                if start % (int(evaluation["document_batch_size"]) * 100) == 0:
                    print(f"{source}: {start:,}/{len(candidates):,} candidates", flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metrics_by_width = candidate_width_metrics(
        scores,
        identity["negative_widths"],
        temperature=temperature,
    )
    sample_records: list[dict[str, Any]] = []
    for width in identity["negative_widths"]:
        metrics = metrics_by_width[width]
        for index in range(len(queries)):
            sample_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "negative_width": width,
                    "query_index": index,
                    "sample_id": int(queries[index]["sample_id"]),
                    "source": str(queries[index]["source"]),
                    **{metric: float(metrics[metric][index]) for metric in METRICS},
                }
            )
    group_records = _group_summaries(sample_records)
    baseline = None
    if baseline_root is not None:
        run_id = checkpoint.parent.name
        baseline_path = Path(baseline_root).resolve() / run_id / "sample_metrics.jsonl"
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        baseline = _baseline_check(sample_records, baseline_path, tolerance=1e-5)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample_metrics.jsonl"
    group_path = output_dir / "group_metrics.jsonl"
    score_path = output_dir / "scores.npz"
    _atomic_jsonl(sample_path, sample_records)
    _atomic_jsonl(group_path, group_records)
    np.savez_compressed(
        score_path,
        scores=scores,
        sample_ids=np.asarray(queries["sample_id"], dtype=np.int64),
        negative_widths=np.asarray(identity["negative_widths"], dtype=np.int64),
    )
    manifest = {
        **identity,
        "sample_records": len(sample_records),
        "group_records": len(group_records),
        "baseline_reproduction": baseline,
        "outputs": {
            "sample_metrics": _identity(sample_path, output_dir),
            "group_metrics": _identity(group_path, output_dir),
            "scores": _identity(score_path, output_dir),
        },
        "runtime": {
            "torch": torch.__version__,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "cuda": torch.version.cuda,
            "device": device,
            "gpu_name": (
                torch.cuda.get_device_name(torch.device(device))
                if device.startswith("cuda")
                else None
            ),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DenseOn checkpoint over nested candidate widths"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/candidate-breadth-224-seed20260901")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/candidate_breadth_probe.json")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = run_candidate_breadth_evaluation(
        args.checkpoint,
        args.data_root,
        args.output_dir,
        protocol_path=args.protocol,
        device=args.device,
        baseline_root=args.baseline_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
