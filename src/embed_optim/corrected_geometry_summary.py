"""Strict weight-space summary for the corrected Dense no-packing matrix."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig
from .geometry import TensorStore, _atomic_json, _sha256
from .optimizers import parameter_partition_name

SCHEMA_VERSION = 1
OPTIMIZER_ORDER = {"adamw": 0, "muon": 1, "normuon": 2}
GEOMETRY_SUFFIX = "-rank64"


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_records(path: Path, *, expected_step: int) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not records or any(int(record.get("step", -1)) != expected_step for record in records):
        raise ValueError(f"Invalid geometry records for checkpoint {expected_step}: {path}")
    tensors = [str(record.get("tensor")) for record in records]
    if len(tensors) != len(set(tensors)):
        raise ValueError(f"Duplicate tensor records in {path}")
    return records


def _nested(payload: dict[str, Any], *keys: str) -> float:
    value: Any = payload
    for key in keys:
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite geometry metric at {keys}")
    return result


def _rss(records: list[dict[str, Any]], metric: str) -> float:
    return math.sqrt(sum(_nested(record, metric, "frobenius_norm") ** 2 for record in records))


def _weighted_metric(
    records: list[dict[str, Any]], metric: str, *keys: str, normalize_rank: bool = False
) -> float:
    numerator = 0.0
    denominator = 0
    for record in records:
        weight = int(record["parameters"])
        value = _nested(record, metric, *keys)
        if normalize_rank:
            value /= min(int(dimension) for dimension in record["shape"])
        numerator += weight * value
        denominator += weight
    if denominator <= 0:
        raise ValueError("Geometry records have no parameters")
    return numerator / denominator


def _nonzero_parameter_fraction(records: list[dict[str, Any]], metric: str) -> float:
    total = sum(int(record["parameters"]) for record in records)
    nonzero = sum(
        int(record["parameters"])
        for record in records
        if _nested(record, metric, "frobenius_norm") > 0
    )
    if total <= 0:
        raise ValueError("Geometry records have no parameters")
    return nonzero / total


def _checkpoint_rows(
    geometry_root: Path,
    configs: list[RunConfig],
    *,
    sketch_rank: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for config in configs:
        analysis_dir = geometry_root / "dense" / f"{config.run_id}{GEOMETRY_SUFFIX}"
        manifest_path = analysis_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        analysis = manifest.get("analysis_config", {})
        run = manifest.get("run", {})
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or run.get("run_id") != config.run_id
            or run.get("model_family") != "dense"
            or run.get("optimizer", {}).get("name") != config.optimizer.name
            or float(run.get("optimizer", {}).get("lr")) != config.optimizer.lr
            or analysis.get("partitions") != ["hidden"]
            or analysis.get("sketch_rank") != sketch_rank
        ):
            raise ValueError(f"Corrected geometry identity mismatch: {manifest_path}")
        record_map = manifest.get("records", {})
        if len(record_map) != len(config.checkpoint_fractions):
            raise ValueError(f"Incomplete geometry checkpoint set: {manifest_path}")
        sources.append(
            {
                "run_id": config.run_id,
                "path": str(manifest_path),
                "bytes": manifest_path.stat().st_size,
                "sha256": _sha256(manifest_path),
            }
        )
        for stage, step in enumerate(sorted(int(value) for value in record_map), start=1):
            identity = record_map[str(step)]
            record_path = analysis_dir / identity["path"]
            if (
                not record_path.is_file()
                or record_path.stat().st_size != int(identity["bytes"])
                or _sha256(record_path) != identity["sha256"]
            ):
                raise ValueError(f"Geometry record provenance mismatch: {record_path}")
            records = _read_records(record_path, expected_step=step)
            segment = "delta_from_reference" if stage == 1 else "delta_from_previous"
            if any(
                segment not in record or "delta_from_reference" not in record for record in records
            ):
                raise ValueError(f"Missing displacement metric in {record_path}")
            weight_norm = _rss(records, "weight")
            segment_norm = _rss(records, segment)
            cumulative_norm = _rss(records, "delta_from_reference")
            row = {
                "run_id": config.run_id,
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "stage": stage,
                "progress_fraction": stage / len(config.checkpoint_fractions),
                "step": step,
                "hidden_tensors": len(records),
                "hidden_parameters": sum(int(record["parameters"]) for record in records),
                "weight_frobenius_norm": weight_norm,
                "saved_segment_frobenius_norm": segment_norm,
                "saved_segment_to_weight_ratio": segment_norm / weight_norm,
                "cumulative_displacement_frobenius_norm": cumulative_norm,
                "cumulative_displacement_to_weight_ratio": cumulative_norm / weight_norm,
                "saved_segment_nonzero_parameter_fraction": _nonzero_parameter_fraction(
                    records, segment
                ),
                "saved_segment_stable_rank_parameter_weighted": _weighted_metric(
                    records, segment, "approx_stable_rank"
                ),
                "saved_segment_stable_rank_fraction_parameter_weighted": _weighted_metric(
                    records, segment, "approx_stable_rank", normalize_rank=True
                ),
                "saved_segment_sketch_effective_rank_parameter_weighted": _weighted_metric(
                    records, segment, "sketched_entropy_effective_rank"
                ),
                "saved_segment_sketch_effective_rank_fraction_parameter_weighted": (
                    _weighted_metric(
                        records,
                        segment,
                        "sketched_entropy_effective_rank",
                        normalize_rank=True,
                    )
                ),
                "saved_segment_sketch_captured_energy_parameter_weighted": _weighted_metric(
                    records, segment, "captured_frobenius_energy"
                ),
                "saved_segment_row_cv_parameter_weighted": _weighted_metric(
                    records, segment, "row_norms", "cv"
                ),
                "saved_segment_top_1pct_row_energy_parameter_weighted": _weighted_metric(
                    records, segment, "top_1pct_row_energy"
                ),
                "cumulative_stable_rank_parameter_weighted": _weighted_metric(
                    records, "delta_from_reference", "approx_stable_rank"
                ),
                "cumulative_stable_rank_fraction_parameter_weighted": _weighted_metric(
                    records,
                    "delta_from_reference",
                    "approx_stable_rank",
                    normalize_rank=True,
                ),
                "cumulative_sketch_effective_rank_parameter_weighted": _weighted_metric(
                    records, "delta_from_reference", "sketched_entropy_effective_rank"
                ),
                "cumulative_sketch_effective_rank_fraction_parameter_weighted": (
                    _weighted_metric(
                        records,
                        "delta_from_reference",
                        "sketched_entropy_effective_rank",
                        normalize_rank=True,
                    )
                ),
                "cumulative_sketch_captured_energy_parameter_weighted": _weighted_metric(
                    records, "delta_from_reference", "captured_frobenius_energy"
                ),
            }
            rows.append(row)
    return rows, sources


def _basis_seed(kind: str, stage: int, tensor: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{kind}:{stage}:{tensor}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _top_bases(
    matrix: torch.Tensor,
    *,
    rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    matrix = matrix.detach().to(device="cpu", dtype=torch.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Subspace input is not a matrix: {tuple(matrix.shape)}")
    if float(matrix.square().sum()) == 0.0:
        return None
    limit = min(matrix.shape)
    retained = min(rank, limit)
    if retained == limit:
        left, _, right_h = torch.linalg.svd(matrix, full_matrices=False)
        return left[:, :retained], right_h[:retained].T
    width = min(limit, retained + oversample)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    omega = torch.randn(matrix.shape[1], width, generator=generator, dtype=matrix.dtype)
    sample = matrix @ omega
    for _ in range(power_iterations):
        basis = torch.linalg.qr(sample, mode="reduced").Q
        sample = matrix @ (matrix.T @ basis)
    basis = torch.linalg.qr(sample, mode="reduced").Q
    projected = basis.T @ matrix
    projected_left, _, right_h = torch.linalg.svd(projected, full_matrices=False)
    left = basis @ projected_left[:, :retained]
    return left, right_h[:retained].T


def _run_stage_bases(
    config: RunConfig,
    *,
    stage: int,
    reference: TensorStore,
    rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor] | None]]:
    completed = json.loads((config.output_dir / "completed.json").read_text(encoding="utf-8"))
    steps = sorted(int(value) for value in completed["checkpoints"])
    if len(steps) != len(config.checkpoint_fractions):
        raise ValueError(f"Unexpected checkpoint count for {config.run_id}")
    step = steps[stage - 1]
    current_path = config.output_dir / f"checkpoint-{step}"
    previous_path = (
        reference.source if stage == 1 else config.output_dir / f"checkpoint-{steps[stage - 2]}"
    )
    result: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor] | None]] = {
        "saved_segment": {},
        "cumulative": {},
    }
    with ExitStack() as stack:
        current = stack.enter_context(TensorStore(current_path))
        previous = reference if stage == 1 else stack.enter_context(TensorStore(previous_path))
        names = [
            name
            for name in current.keys()
            if len(current.shape(name)) == 2 and parameter_partition_name(name, 2) == "hidden"
        ]
        if not names:
            raise ValueError(f"No hidden matrices in {current_path}")
        for name in names:
            shape = current.shape(name)
            if previous.shape(name) != shape or reference.shape(name) != shape:
                raise ValueError(f"Incompatible subspace tensor {name!r} for {config.run_id}")
            weight = current.tensor(name).float()
            segment = weight - previous.tensor(name).float()
            cumulative = weight - reference.tensor(name).float()
            for kind, delta in (("saved_segment", segment), ("cumulative", cumulative)):
                result[kind][name] = _top_bases(
                    delta,
                    rank=rank,
                    oversample=oversample,
                    power_iterations=power_iterations,
                    seed=_basis_seed(kind, stage, name, seed),
                )
    return result


def _basis_overlap(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> tuple[float, float, float]:
    left_rank = min(left[0].shape[1], right[0].shape[1])
    right_rank = min(left[1].shape[1], right[1].shape[1])
    if left_rank <= 0 or right_rank <= 0:
        raise ValueError("Subspace basis has zero rank")
    left_overlap = float((left[0].T @ right[0]).square().sum() / left_rank)
    right_overlap = float((left[1].T @ right[1]).square().sum() / right_rank)
    mean = (left_overlap + right_overlap) / 2
    for value in (left_overlap, right_overlap, mean):
        if not -1e-6 <= value <= 1 + 1e-5:
            raise ValueError(f"Subspace overlap outside [0, 1]: {value}")
    return left_overlap, right_overlap, mean


def _pair_subspace_rows(
    configs: list[RunConfig],
    reference_path: Path,
    *,
    rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(
        configs,
        key=lambda item: (
            OPTIMIZER_ORDER[item.optimizer.name],
            item.optimizer.lr,
            item.run_id,
        ),
    )
    with TensorStore(reference_path) as reference:
        for stage in range(1, len(ordered[0].checkpoint_fractions) + 1):
            bases = {
                config.run_id: _run_stage_bases(
                    config,
                    stage=stage,
                    reference=reference,
                    rank=rank,
                    oversample=oversample,
                    power_iterations=power_iterations,
                    seed=seed,
                )
                for config in ordered
            }
            for first, second in itertools.combinations(ordered, 2):
                for kind in ("saved_segment", "cumulative"):
                    first_bases = bases[first.run_id][kind]
                    second_bases = bases[second.run_id][kind]
                    if set(first_bases) != set(second_bases):
                        raise ValueError(
                            f"Subspace tensor sets differ: {first.run_id}, {second.run_id}"
                        )
                    numerator_left = 0.0
                    numerator_right = 0.0
                    numerator_mean = 0.0
                    defined_parameters = 0
                    undefined_parameters = 0
                    defined_tensors = 0
                    for name in sorted(first_bases):
                        parameters = math.prod(reference.shape(name))
                        left = first_bases[name]
                        right = second_bases[name]
                        if left is None or right is None:
                            undefined_parameters += parameters
                            continue
                        overlap_left, overlap_right, overlap_mean = _basis_overlap(left, right)
                        numerator_left += parameters * overlap_left
                        numerator_right += parameters * overlap_right
                        numerator_mean += parameters * overlap_mean
                        defined_parameters += parameters
                        defined_tensors += 1
                    total_parameters = defined_parameters + undefined_parameters
                    rows.append(
                        {
                            "stage": stage,
                            "progress_fraction": stage / len(first.checkpoint_fractions),
                            "displacement_kind": kind,
                            "first_run_id": first.run_id,
                            "first_optimizer": first.optimizer.name,
                            "first_learning_rate": first.optimizer.lr,
                            "second_run_id": second.run_id,
                            "second_optimizer": second.optimizer.name,
                            "second_learning_rate": second.optimizer.lr,
                            "defined_tensors": defined_tensors,
                            "defined_parameters": defined_parameters,
                            "undefined_zero_parameters": undefined_parameters,
                            "defined_parameter_fraction": (
                                defined_parameters / total_parameters if total_parameters else 0.0
                            ),
                            "left_subspace_overlap": (
                                numerator_left / defined_parameters if defined_parameters else None
                            ),
                            "right_subspace_overlap": (
                                numerator_right / defined_parameters if defined_parameters else None
                            ),
                            "mean_subspace_overlap": (
                                numerator_mean / defined_parameters if defined_parameters else None
                            ),
                        }
                    )
            del bases
    return rows


def _optimizer_pair_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        key = (
            row["stage"],
            row["progress_fraction"],
            row["displacement_kind"],
            row["first_optimizer"],
            row["second_optimizer"],
        )
        groups[key].append(row)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: item[0]):
        defined = [row for row in members if row["mean_subspace_overlap"] is not None]
        output.append(
            {
                "stage": key[0],
                "progress_fraction": key[1],
                "displacement_kind": key[2],
                "first_optimizer": key[3],
                "second_optimizer": key[4],
                "rate_pairs": len(members),
                "defined_rate_pairs": len(defined),
                "mean_defined_parameter_fraction": sum(
                    float(row["defined_parameter_fraction"]) for row in members
                )
                / len(members),
                "mean_subspace_overlap_across_rate_pairs": (
                    sum(float(row["mean_subspace_overlap"]) for row in defined) / len(defined)
                    if defined
                    else None
                ),
            }
        )
    return output


def summarize_corrected_geometry(
    geometry_root: Path,
    output_dir: Path,
    configs: list[RunConfig],
    reference_path: Path,
    *,
    protocol_path: Path,
    sketch_rank: int = 64,
    subspace_rank: int = 16,
    oversample: int = 8,
    power_iterations: int = 2,
    seed: int = 20260903,
) -> dict[str, Any]:
    """Aggregate all runs and compare saved-displacement subspaces without selection."""

    if len(configs) != 12 or any(config.model_family != "dense" for config in configs):
        raise ValueError("Corrected geometry summary requires the complete 12-run Dense matrix")
    geometry_root = geometry_root.resolve()
    output_dir = output_dir.resolve()
    reference_path = reference_path.resolve()
    checkpoint_rows, sources = _checkpoint_rows(geometry_root, configs, sketch_rank=sketch_rank)
    expected_checkpoint_rows = len(configs) * len(configs[0].checkpoint_fractions)
    if len(checkpoint_rows) != expected_checkpoint_rows:
        raise ValueError(
            f"Expected {expected_checkpoint_rows} checkpoint rows, found {len(checkpoint_rows)}"
        )
    pair_rows = _pair_subspace_rows(
        configs,
        reference_path,
        rank=subspace_rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
    )
    optimizer_rows = _optimizer_pair_rows(pair_rows)
    checkpoint_path = output_dir / "checkpoint_geometry.csv"
    pair_path = output_dir / "run_pair_subspace_overlap.csv"
    optimizer_path = output_dir / "optimizer_pair_subspace_summary.csv"
    _atomic_csv(checkpoint_path, checkpoint_rows, list(checkpoint_rows[0]))
    _atomic_csv(pair_path, pair_rows, list(pair_rows[0]))
    _atomic_csv(optimizer_path, optimizer_rows, list(optimizer_rows[0]))
    expected_pair_rows = math.comb(len(configs), 2) * len(configs[0].checkpoint_fractions) * 2
    if len(pair_rows) != expected_pair_rows:
        raise ValueError(f"Expected {expected_pair_rows} subspace rows, found {len(pair_rows)}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "claim_boundary": (
            "Saved-segment displacement is the parameter difference between adjacent retained "
            "checkpoints (stage 1 uses initialization), not a per-step optimizer update or path "
            "length. Subspace results are descriptive one-seed mechanism evidence."
        ),
        "runs": len(configs),
        "checkpoint_rows": len(checkpoint_rows),
        "run_pair_subspace_rows": len(pair_rows),
        "optimizer_pair_rows": len(optimizer_rows),
        "analysis": {
            "sketch_rank": sketch_rank,
            "subspace_rank": subspace_rank,
            "oversample": oversample,
            "power_iterations": power_iterations,
            "seed": seed,
            "subspace_overlap": (
                "parameter-weighted mean of left/right squared canonical overlaps, each "
                "normalized by retained rank; zero-displacement tensors are undefined and "
                "reported through defined_parameter_fraction"
            ),
            "rate_pair_policy": (
                "all unordered run pairs at the same stage; optimizer summaries average rate "
                "pairs equally and do not select a learning rate"
            ),
        },
        "protocol": {
            "path": str(protocol_path.resolve()),
            "bytes": protocol_path.stat().st_size,
            "sha256": _sha256(protocol_path),
        },
        "reference": {
            "path": str(reference_path),
            "files": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in TensorStore(reference_path).files
            ],
        },
        "sources": sources,
        "outputs": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "rows": rows,
            }
            for path, rows in (
                (checkpoint_path, len(checkpoint_rows)),
                (pair_path, len(pair_rows)),
                (optimizer_path, len(optimizer_rows)),
            )
        },
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest
