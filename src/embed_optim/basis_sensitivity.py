from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .config import resolve_matrix_path
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .update_geometry import ALGORITHMS, UpdateOperatorConfig, replay_update_directions

RECORD_METRICS = (
    "mapped_direction_cosine",
    "mapped_relative_frobenius_error",
    "mapped_norm_ratio",
    "predicted_descent_relative_error",
)
HEAD_METRICS = (
    "mapped_direction_cosine",
    "mapped_relative_frobenius_error",
    "mapped_norm_ratio",
    "singular_spectrum_relative_l2_error",
)
SUMMARY_FIELDS = (
    "family",
    "optimizer",
    "records",
    "median_mapped_direction_cosine",
    "median_mapped_relative_frobenius_error",
    "maximum_mapped_relative_frobenius_error",
    "median_absolute_norm_ratio_error",
    "median_predicted_descent_relative_error",
    "median_head_spectrum_relative_l2_error",
    "maximum_functional_invariance_error",
)


@dataclass(frozen=True)
class AnchorSource:
    family: str
    anchor: str
    checkpoint: Path
    config_path: Path
    manifest_path: Path
    shards: tuple[tuple[Path, dict[str, Any]], ...]


def _identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    rendered = resolved.relative_to(relative_to.resolve()) if relative_to is not None else resolved
    return {
        "path": str(rendered),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Refusing to write an empty basis-sensitivity table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"Basis-sensitivity rows have inconsistent fields: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {**_identity(path, relative_to=path.parent), "rows": len(rows)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _has_exact_keys(value: Any, expected: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == expected


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    protocol_path = resolve_matrix_path(path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Basis-sensitivity protocol must be a JSON object")
    common = protocol.get("common_state")
    architecture = protocol.get("architecture")
    selection = protocol.get("selection")
    transformation = protocol.get("transformation")
    calibration = protocol.get("functional_calibration")
    metrics = protocol.get("metrics")
    freeze = protocol.get("freeze_context")
    if (
        set(protocol)
        != {
            "schema_version",
            "status",
            "common_state",
            "architecture",
            "selection",
            "transformation",
            "functional_calibration",
            "metrics",
            "freeze_context",
        }
        or protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("status") != "prospective_appendix_diagnostic"
        or not all(
            isinstance(value, dict)
            for value in (
                common,
                architecture,
                selection,
                transformation,
                calibration,
                metrics,
                freeze,
            )
        )
    ):
        raise ValueError("Basis-sensitivity protocol is incomplete or not prospectively frozen")

    if not _has_exact_keys(
        common,
        {
            "root",
            "spec",
            "spec_sha256",
            "expected_anchors",
            "expected_anchors_per_family",
            "gradient_steps",
        },
    ) or not all(
        isinstance(common.get(field), str) and common[field] for field in ("root", "spec")
    ):
        raise ValueError("Basis protocol common-state binding is malformed")
    common_spec = resolve_matrix_path(common["spec"]).resolve()
    common_payload = (
        json.loads(common_spec.read_text(encoding="utf-8")) if common_spec.is_file() else {}
    )
    if (
        not common_spec.is_file()
        or not _valid_sha256(common.get("spec_sha256"))
        or common["spec_sha256"] != _sha256(common_spec)
        or common.get("expected_anchors") != 2 * common.get("expected_anchors_per_family", -1)
        or common.get("expected_anchors_per_family", 0) <= 0
        or not isinstance(common.get("gradient_steps"), int)
        or common["gradient_steps"] <= 0
        or common_payload.get("schema_version") != SCHEMA_VERSION
        or common_payload.get("selection", {}).get("gradient_steps") != common["gradient_steps"]
        or not _has_exact_keys(
            common_payload.get("operator_protocol"),
            {
                "adam_beta1",
                "adam_beta2",
                "adam_eps",
                "muon_momentum",
                "normuon_beta2",
                "ns_steps",
                "adjust_lr_fn",
            },
        )
    ):
        raise ValueError("Basis protocol common-state binding is invalid")

    if not _has_exact_keys(
        architecture,
        {
            "model_type",
            "hidden_size",
            "num_attention_heads",
            "head_dim",
            "attention_bias",
            "max_position_embeddings",
            "qkv_layout",
            "qkv_tensor_template",
            "rotary_pairing",
            "rope_bases",
        },
    ):
        raise ValueError("Basis protocol architecture fields differ from the frozen schema")
    hidden_size = architecture.get("hidden_size")
    heads = architecture.get("num_attention_heads")
    head_dim = architecture.get("head_dim")
    if (
        architecture.get("model_type") != "modernbert"
        or not all(isinstance(value, int) and value > 0 for value in (hidden_size, heads, head_dim))
        or hidden_size != heads * head_dim
        or head_dim % 2
        or architecture.get("attention_bias") is not False
        or not isinstance(architecture.get("max_position_embeddings"), int)
        or architecture["max_position_embeddings"] <= 0
        or architecture.get("qkv_layout") != "contiguous_q_k_v"
        or architecture.get("rotary_pairing") != "split_half"
        or not isinstance(architecture.get("qkv_tensor_template"), str)
        or architecture["qkv_tensor_template"].count("{layer}") != 1
        or not isinstance(architecture.get("rope_bases"), list)
        or len(architecture["rope_bases"]) != 2
        or len(set(architecture["rope_bases"])) != 2
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
            for value in architecture["rope_bases"]
        )
    ):
        raise ValueError("Basis protocol architecture is not the frozen ModernBERT QKV layout")

    if not _has_exact_keys(
        selection,
        {
            "families",
            "layers",
            "heads",
            "rotation_seeds",
            "optimizers",
            "expected_tensor_sequences",
            "expected_records",
            "expected_head_records",
            "expected_summary_rows",
        },
    ):
        raise ValueError("Basis protocol selection fields differ from the frozen schema")
    families = selection.get("families")
    layers = selection.get("layers")
    selected_heads = selection.get("heads")
    rotation_seeds = selection.get("rotation_seeds")
    optimizers = selection.get("optimizers")
    if (
        families != ["dense", "late"]
        or not all(
            isinstance(values, list) and values and len(values) == len(set(values))
            for values in (layers, selected_heads, rotation_seeds, optimizers)
        )
        or optimizers != list(ALGORITHMS)
        or not all(isinstance(value, int) and value >= 0 for value in layers)
        or max(layers) >= 10_000
        or not all(isinstance(value, int) and 0 <= value < heads for value in selected_heads)
        or not all(isinstance(value, int) and value > 0 for value in rotation_seeds)
    ):
        raise ValueError("Basis protocol selection is invalid")

    anchors = common["expected_anchors"]
    tensors = anchors * len(layers)
    records = tensors * len(rotation_seeds) * len(optimizers)
    head_records = records * 2 * len(selected_heads)
    position_pairs = calibration.get("position_pairs")
    if (
        not _has_exact_keys(
            transformation,
            {
                "group",
                "angle_distribution",
                "query_key_share_angles",
                "value_rows_unchanged",
                "map_rotated_updates_back_before_comparison",
                "function_preserving_reason",
            },
        )
        or not all(
            isinstance(transformation.get(field), str) and transformation[field]
            for field in ("group", "angle_distribution", "function_preserving_reason")
        )
        or not _has_exact_keys(
            calibration,
            {
                "dtype",
                "vector_seed",
                "position_pairs",
                "maximum_absolute_logit_error",
            },
        )
        or not isinstance(calibration.get("vector_seed"), int)
        or calibration["vector_seed"] <= 0
        or not isinstance(position_pairs, list)
        or not position_pairs
        or len(position_pairs) != len({tuple(pair) for pair in position_pairs})
        or not all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(
                isinstance(position, int)
                and 0 <= position < architecture["max_position_embeddings"]
                for position in pair
            )
            for pair in position_pairs
        )
        or not _has_exact_keys(metrics, {"full_tensor", "selected_qk_heads", "causal_boundary"})
        or metrics.get("full_tensor") != list(RECORD_METRICS)
        or metrics.get("selected_qk_heads") != list(HEAD_METRICS)
        or not isinstance(metrics.get("causal_boundary"), str)
        or not metrics["causal_boundary"]
        or not _has_exact_keys(
            freeze,
            {
                "frozen_at_utc",
                "strict_beir_valid_units",
                "strict_beir_expected_units",
                "complete_retrieval_matrix_visible",
                "common_state_output_visible",
                "formal_basis_output_visible",
                "completed_weight_trajectories_visible",
                "note",
            },
        )
        or selection.get("expected_tensor_sequences") != tensors
        or selection.get("expected_records") != records
        or selection.get("expected_head_records") != head_records
        or selection.get("expected_summary_rows") != len(families) * len(optimizers)
        or transformation.get("query_key_share_angles") is not True
        or transformation.get("value_rows_unchanged") is not True
        or transformation.get("map_rotated_updates_back_before_comparison") is not True
        or calibration.get("dtype") != "float64"
        or not isinstance(calibration.get("maximum_absolute_logit_error"), (int, float))
        or isinstance(calibration["maximum_absolute_logit_error"], bool)
        or calibration["maximum_absolute_logit_error"] <= 0
        or not isinstance(freeze.get("frozen_at_utc"), str)
        or not freeze["frozen_at_utc"].endswith("Z")
        or not isinstance(freeze.get("strict_beir_valid_units"), int)
        or isinstance(freeze["strict_beir_valid_units"], bool)
        or not isinstance(freeze.get("strict_beir_expected_units"), int)
        or isinstance(freeze["strict_beir_expected_units"], bool)
        or not 0 <= freeze["strict_beir_valid_units"] < freeze["strict_beir_expected_units"]
        or freeze.get("complete_retrieval_matrix_visible") is not False
        or freeze.get("common_state_output_visible") is not False
        or freeze.get("formal_basis_output_visible") is not False
        or freeze.get("completed_weight_trajectories_visible") is not True
        or not isinstance(freeze.get("note"), str)
        or not freeze["note"]
    ):
        raise ValueError("Basis protocol cardinality, transform, or freeze context is invalid")
    return protocol_path, protocol


def rope_commuting_angles(
    num_heads: int,
    head_dim: int,
    seed: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_heads <= 0 or head_dim <= 0 or head_dim % 2 or seed <= 0:
        raise ValueError("RoPE-commuting angles require positive heads/even head_dim/seed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    angles = torch.rand((num_heads, head_dim // 2), generator=generator, dtype=torch.float64)
    return (angles * (2 * math.pi)).to(device=device, dtype=dtype)


def _rotate_split_half(
    values: torch.Tensor, angles: torch.Tensor, *, inverse: bool
) -> torch.Tensor:
    if values.ndim < 2 or values.shape[-2] != 2 * angles.shape[-1]:
        raise ValueError("Split-half rotation shape differs from its angle grid")
    half = values.shape[-2] // 2
    first, second = values[..., :half, :], values[..., half:, :]
    cosine = angles.cos().unsqueeze(-1)
    sine = angles.sin().unsqueeze(-1)
    if inverse:
        sine = -sine
    return torch.cat((cosine * first - sine * second, sine * first + cosine * second), dim=-2)


def rotate_qk_rows(
    matrix: torch.Tensor,
    angles: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate Q/K output rows inside a contiguous fused QKV matrix; leave V unchanged."""

    if matrix.ndim != 2:
        raise ValueError(f"Expected a fused QKV matrix, got {tuple(matrix.shape)}")
    heads, half = angles.shape
    head_dim = 2 * half
    hidden_size = heads * head_dim
    if matrix.shape[0] != 3 * hidden_size:
        raise ValueError("Fused QKV rows disagree with the head/angle layout")
    values = matrix.reshape(3, heads, head_dim, matrix.shape[1])
    qk = _rotate_split_half(values[:2], angles.to(matrix), inverse=inverse)
    return torch.cat((qk, values[2:]), dim=0).reshape_as(matrix)


def _rotate_head_vectors(
    vectors: torch.Tensor,
    angles: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape != (angles.shape[0], 2 * angles.shape[1]):
        raise ValueError("Head vectors disagree with the RoPE angle layout")
    return _rotate_split_half(vectors.unsqueeze(-1), angles, inverse=inverse).squeeze(-1)


def _apply_split_half_rope(vectors: torch.Tensor, position: int, base: float) -> torch.Tensor:
    head_dim = vectors.shape[-1]
    if vectors.ndim != 2 or head_dim % 2 or position < 0 or base <= 0:
        raise ValueError("Invalid vector, position, or base for split-half RoPE")
    frequencies = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=vectors.dtype, device=vectors.device) / head_dim)
    )
    phase = frequencies * position
    cosine = torch.cat((phase.cos(), phase.cos()))
    sine = torch.cat((phase.sin(), phase.sin()))
    first, second = vectors[..., : head_dim // 2], vectors[..., head_dim // 2 :]
    rotated_half = torch.cat((-second, first), dim=-1)
    return vectors * cosine + rotated_half * sine


def functional_invariance_error(
    angles: torch.Tensor,
    *,
    vector_seed: int,
    rope_bases: list[float],
    position_pairs: list[list[int]],
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(vector_seed)
    shape = (angles.shape[0], 2 * angles.shape[1])
    query = torch.randn(shape, generator=generator, dtype=torch.float64)
    key = torch.randn(shape, generator=generator, dtype=torch.float64)
    precise_angles = angles.to(dtype=torch.float64, device="cpu")
    rotated_query = _rotate_head_vectors(query, precise_angles)
    rotated_key = _rotate_head_vectors(key, precise_angles)
    maximum = 0.0
    for base in rope_bases:
        for query_position, key_position in position_pairs:
            original = (
                _apply_split_half_rope(query, int(query_position), float(base))
                * _apply_split_half_rope(key, int(key_position), float(base))
            ).sum(dim=-1)
            transformed = (
                _apply_split_half_rope(rotated_query, int(query_position), float(base))
                * _apply_split_half_rope(rotated_key, int(key_position), float(base))
            ).sum(dim=-1)
            maximum = max(maximum, float((transformed - original).abs().max().item()))
    return maximum


def _gradient_shards(
    manifest_path: Path,
    *,
    verify_hashes: bool,
) -> tuple[dict[str, Any], tuple[tuple[Path, dict[str, Any]], ...]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("gradient_shards")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(shards, list)
        or not shards
    ):
        raise ValueError(f"Incomplete basis gradient source: {manifest_path}")
    resolved = []
    root = manifest_path.parent.resolve()
    for index, item in enumerate(shards):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or Path(item["path"]).is_absolute()
            or item.get("step_index") != index
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
            or not _valid_sha256(item.get("sha256"))
        ):
            raise ValueError(f"Invalid gradient shard declaration: {manifest_path}")
        path = (root / item["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Gradient shard escapes its source directory: {path}") from error
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or (verify_hashes and _sha256(path) != item.get("sha256"))
        ):
            raise ValueError(f"Gradient shard differs from its basis source: {path}")
        resolved.append((path, item))
    return manifest, tuple(resolved)


def _validate_model_config(path: Path, architecture: dict[str, Any]) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    observed_bases = sorted(
        float(item["rope_theta"]) for item in config.get("rope_parameters", {}).values()
    )
    if (
        config.get("model_type") != architecture["model_type"]
        or config.get("hidden_size") != architecture["hidden_size"]
        or config.get("num_attention_heads") != architecture["num_attention_heads"]
        or config.get("attention_bias") is not architecture["attention_bias"]
        or config.get("max_position_embeddings") != architecture["max_position_embeddings"]
        or observed_bases != sorted(float(value) for value in architecture["rope_bases"])
    ):
        raise ValueError(f"Checkpoint architecture differs from basis protocol: {path}")


def _load_sources(
    protocol: dict[str, Any],
    *,
    verify_hashes: bool,
) -> list[AnchorSource]:
    common = protocol["common_state"]
    architecture = protocol["architecture"]
    selection = protocol["selection"]
    root = Path(common["root"]).resolve()
    manifests = sorted(root.glob("**/gradients/manifest.json"))
    if len(manifests) != common["expected_anchors"]:
        raise ValueError(
            f"Expected {common['expected_anchors']} common-state anchors, found {len(manifests)}"
        )
    sources = []
    counts = {family: 0 for family in selection["families"]}
    tensor_names = [
        architecture["qkv_tensor_template"].format(layer=layer) for layer in selection["layers"]
    ]
    for manifest_path in manifests:
        relative_anchor = manifest_path.parent.parent.relative_to(root)
        family = relative_anchor.parts[0]
        if family not in counts:
            raise ValueError(f"Unexpected common-state family: {relative_anchor}")
        manifest, shards = _gradient_shards(manifest_path, verify_hashes=verify_hashes)
        if (
            len(shards) != common["gradient_steps"]
            or manifest.get("common_state_spec", {}).get("sha256") != common["spec_sha256"]
            or manifest.get("config", {}).get("gradient_steps") != common["gradient_steps"]
        ):
            raise ValueError(f"Gradient sequence differs from basis protocol: {manifest_path}")
        checkpoint = Path(manifest.get("checkpoint", {}).get("path", "")).resolve()
        config_path = checkpoint / "config.json"
        if not checkpoint.is_dir() or not config_path.is_file():
            raise FileNotFoundError(config_path)
        _validate_model_config(config_path, architecture)
        with safe_open(str(shards[0][0]), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in tensor_names:
                expected_shape = (3 * architecture["hidden_size"], architecture["hidden_size"])
                if (
                    name not in available
                    or tuple(handle.get_slice(name).get_shape()) != expected_shape
                ):
                    raise ValueError(f"Selected QKV tensor is absent or malformed: {name}")
        counts[family] += 1
        sources.append(
            AnchorSource(
                family=family,
                anchor=relative_anchor.as_posix(),
                checkpoint=checkpoint,
                config_path=config_path,
                manifest_path=manifest_path.resolve(),
                shards=shards,
            )
        )
    if any(count != common["expected_anchors_per_family"] for count in counts.values()):
        raise ValueError(f"Common-state family anchor counts differ: {counts}")
    return sources


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if denominator <= 0:
        raise ValueError("Cannot compare a zero optimizer direction")
    return float(torch.clamp(torch.sum(left * right) / denominator, -1, 1).item())


def _direction_metrics(original: torch.Tensor, mapped: torch.Tensor) -> dict[str, float]:
    original_norm = torch.linalg.vector_norm(original)
    mapped_norm = torch.linalg.vector_norm(mapped)
    if original_norm <= 0 or mapped_norm <= 0:
        raise ValueError("Basis diagnostic received a zero update direction")
    return {
        "mapped_direction_cosine": _cosine(original, mapped),
        "mapped_relative_frobenius_error": float(
            (torch.linalg.vector_norm(mapped - original) / original_norm).item()
        ),
        "mapped_norm_ratio": float((mapped_norm / original_norm).item()),
    }


def _head_blocks(
    matrix: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    selected_heads: list[int],
) -> tuple[list[tuple[str, int]], torch.Tensor]:
    values = matrix.reshape(3, num_heads, head_dim, matrix.shape[1])
    labels = [(role, head) for role in ("query", "key") for head in selected_heads]
    blocks = torch.stack([values[0 if role == "query" else 1, head] for role, head in labels])
    return labels, blocks


def _head_metric_rows(
    original: torch.Tensor,
    mapped: torch.Tensor,
    *,
    base: dict[str, Any],
    architecture: dict[str, Any],
    selected_heads: list[int],
) -> list[dict[str, Any]]:
    labels, original_blocks = _head_blocks(
        original,
        num_heads=architecture["num_attention_heads"],
        head_dim=architecture["head_dim"],
        selected_heads=selected_heads,
    )
    _, mapped_blocks = _head_blocks(
        mapped,
        num_heads=architecture["num_attention_heads"],
        head_dim=architecture["head_dim"],
        selected_heads=selected_heads,
    )
    original_spectra = torch.linalg.svdvals(original_blocks)
    mapped_spectra = torch.linalg.svdvals(mapped_blocks)
    rows = []
    for index, (role, head) in enumerate(labels):
        metrics = _direction_metrics(original_blocks[index], mapped_blocks[index])
        spectrum_norm = torch.linalg.vector_norm(original_spectra[index])
        if spectrum_norm <= 0:
            raise ValueError("Selected Q/K head has a zero singular spectrum")
        rows.append(
            {
                **base,
                "qk_role": role,
                "head": head,
                **metrics,
                "singular_spectrum_relative_l2_error": float(
                    (
                        torch.linalg.vector_norm(mapped_spectra[index] - original_spectra[index])
                        / spectrum_norm
                    ).item()
                ),
            }
        )
    return rows


def _source_record(source: AnchorSource, root: Path) -> dict[str, Any]:
    return {
        "family": source.family,
        "anchor": source.anchor,
        "checkpoint": str(source.checkpoint),
        "config": _identity(source.config_path),
        "gradient_manifest": _identity(source.manifest_path),
        "gradient_shards": [
            {
                "path": str(path.relative_to(root)),
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for path, item in source.shards
        ],
    }


def _summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((str(row["family"]), str(row["optimizer"])), []).append(row)
    output = []
    for (family, optimizer), rows in sorted(groups.items()):
        output.append(
            {
                "family": family,
                "optimizer": optimizer,
                "records": len(rows),
                "median_mapped_direction_cosine": statistics.median(
                    float(row["mapped_direction_cosine"]) for row in rows
                ),
                "median_mapped_relative_frobenius_error": statistics.median(
                    float(row["mapped_relative_frobenius_error"]) for row in rows
                ),
                "maximum_mapped_relative_frobenius_error": max(
                    float(row["mapped_relative_frobenius_error"]) for row in rows
                ),
                "median_absolute_norm_ratio_error": statistics.median(
                    abs(float(row["mapped_norm_ratio"]) - 1) for row in rows
                ),
                "median_predicted_descent_relative_error": statistics.median(
                    float(row["predicted_descent_relative_error"]) for row in rows
                ),
                "median_head_spectrum_relative_l2_error": statistics.median(
                    float(row["mean_head_spectrum_relative_l2_error"]) for row in rows
                ),
                "maximum_functional_invariance_error": max(
                    float(row["functional_invariance_max_abs_logit_error"]) for row in rows
                ),
            }
        )
    return output


def analyze_basis_sensitivity(
    protocol_path: str | Path = "configs/basis_sensitivity.json",
    *,
    output_dir: str | Path = "reports/basis-sensitivity",
    device: str = "cuda",
    verify_inputs: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    resolved_protocol, protocol = _load_protocol(protocol_path)
    output = Path(output_dir).resolve()
    manifest_path = output / "summary_manifest.json"
    if manifest_path.is_file() and not overwrite:
        return audit_basis_sensitivity(
            resolved_protocol,
            output_dir=output,
            verify_inputs=verify_inputs,
        )
    known = [output / "records.csv", output / "head_records.csv", output / "summary.csv"]
    if not overwrite and any(path.exists() for path in known):
        raise FileExistsError(f"Partial basis-sensitivity output exists under {output}")
    if overwrite:
        for path in [manifest_path, *known]:
            if path.is_file():
                path.unlink()

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA basis analysis was requested but CUDA is unavailable")
    architecture = protocol["architecture"]
    selection = protocol["selection"]
    calibration = protocol["functional_calibration"]
    common_root = Path(protocol["common_state"]["root"]).resolve()
    sources = _load_sources(protocol, verify_hashes=verify_inputs)
    operator = UpdateOperatorConfig(
        **json.loads(
            resolve_matrix_path(protocol["common_state"]["spec"]).read_text(encoding="utf-8")
        )["operator_protocol"]
    )
    records: list[dict[str, Any]] = []
    head_records: list[dict[str, Any]] = []
    tensor_names = [
        (
            layer,
            architecture["qkv_tensor_template"].format(layer=layer),
        )
        for layer in selection["layers"]
    ]
    calibration_errors = {}
    for rotation_seed in selection["rotation_seeds"]:
        angles = rope_commuting_angles(
            architecture["num_attention_heads"],
            architecture["head_dim"],
            rotation_seed,
            dtype=torch.float64,
        )
        error = functional_invariance_error(
            angles,
            vector_seed=calibration["vector_seed"],
            rope_bases=architecture["rope_bases"],
            position_pairs=calibration["position_pairs"],
        )
        if error > calibration["maximum_absolute_logit_error"]:
            raise ValueError(
                f"RoPE-commuting transform failed functional calibration: seed={rotation_seed}, "
                f"error={error}"
            )
        calibration_errors[rotation_seed] = error

    for source_index, source in enumerate(sources, start=1):
        print(f"Basis diagnostic anchor {source_index}/{len(sources)}: {source.anchor}", flush=True)
        with ExitStack() as stack:
            handles = [
                stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
                for path, _ in source.shards
            ]
            expected_names = set(handles[0].keys())
            if any(set(handle.keys()) != expected_names for handle in handles[1:]):
                raise ValueError(f"Gradient tensor sets differ under {source.anchor}")
            for layer, tensor_name in tensor_names:
                gradients = [handle.get_tensor(tensor_name) for handle in handles]
                original_directions = replay_update_directions(
                    gradients,
                    operator,
                    device=device,
                )
                final_gradient = gradients[-1].to(device=target, dtype=torch.float32)
                for rotation_seed in selection["rotation_seeds"]:
                    angles = rope_commuting_angles(
                        architecture["num_attention_heads"],
                        architecture["head_dim"],
                        rotation_seed,
                        device=target,
                    )
                    rotated_gradients = [
                        rotate_qk_rows(
                            gradient.to(device=target, dtype=torch.float32),
                            angles,
                        )
                        for gradient in gradients
                    ]
                    rotated_directions = replay_update_directions(
                        rotated_gradients,
                        operator,
                        device=device,
                    )
                    for optimizer in ALGORITHMS:
                        original = original_directions[optimizer]
                        rotated = rotated_directions[optimizer]
                        mapped = rotate_qk_rows(rotated, angles, inverse=True)
                        metrics = _direction_metrics(original, mapped)
                        original_descent = torch.sum(final_gradient * original)
                        rotated_descent = torch.sum(rotated_gradients[-1] * rotated)
                        descent_denominator = max(abs(float(original_descent.item())), 1e-12)
                        base = {
                            "family": source.family,
                            "anchor": source.anchor,
                            "layer": layer,
                            "tensor": tensor_name,
                            "rotation_seed": rotation_seed,
                            "optimizer": optimizer,
                        }
                        current_head_rows = _head_metric_rows(
                            original,
                            mapped,
                            base=base,
                            architecture=architecture,
                            selected_heads=selection["heads"],
                        )
                        head_records.extend(current_head_rows)
                        records.append(
                            {
                                **base,
                                **metrics,
                                "original_predicted_descent_per_unit_lr": float(
                                    original_descent.item()
                                ),
                                "rotated_predicted_descent_per_unit_lr": float(
                                    rotated_descent.item()
                                ),
                                "predicted_descent_relative_error": abs(
                                    float(rotated_descent.item() - original_descent.item())
                                )
                                / descent_denominator,
                                "mean_head_spectrum_relative_l2_error": statistics.mean(
                                    float(row["singular_spectrum_relative_l2_error"])
                                    for row in current_head_rows
                                ),
                                "maximum_head_spectrum_relative_l2_error": max(
                                    float(row["singular_spectrum_relative_l2_error"])
                                    for row in current_head_rows
                                ),
                                "functional_invariance_max_abs_logit_error": calibration_errors[
                                    rotation_seed
                                ],
                            }
                        )

    summaries = _summaries(records)
    outputs = {
        "records": _atomic_csv(output / "records.csv", records),
        "head_records": _atomic_csv(output / "head_records.csv", head_records),
        "summary": _atomic_csv(output / "summary.csv", summaries),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "protocol": _identity(resolved_protocol),
        "common_state_spec": _identity(resolve_matrix_path(protocol["common_state"]["spec"])),
        "analysis": {
            "device": str(target),
            "operator_protocol": json.loads(
                resolve_matrix_path(protocol["common_state"]["spec"]).read_text(encoding="utf-8")
            )["operator_protocol"],
            "weight_decay_included": False,
            "parameters_advanced": False,
            "causal_boundary": protocol["metrics"]["causal_boundary"],
        },
        "coverage": {
            "anchors": len(sources),
            "tensor_sequences": len(sources) * len(selection["layers"]),
            "records": len(records),
            "head_records": len(head_records),
            "summary_rows": len(summaries),
        },
        "functional_calibration": {
            "maximum_observed_absolute_logit_error": max(calibration_errors.values()),
            "maximum_allowed_absolute_logit_error": calibration["maximum_absolute_logit_error"],
            "by_rotation_seed": {
                str(seed): calibration_errors[seed] for seed in selection["rotation_seeds"]
            },
        },
        "sources": [_source_record(source, common_root) for source in sources],
        "outputs": outputs,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(target),
            "gpu_name": (torch.cuda.get_device_name(target) if target.type == "cuda" else None),
        },
    }
    _atomic_json(manifest_path, manifest)
    return audit_basis_sensitivity(
        resolved_protocol,
        output_dir=output,
        verify_inputs=verify_inputs,
    )


def _validate_finite_rows(rows: list[dict[str, str]], identity_fields: set[str]) -> None:
    for row in rows:
        for field, value in row.items():
            if field in identity_fields:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Non-numeric basis result field {field}={value!r}") from error
            if not math.isfinite(parsed):
                raise ValueError(f"Non-finite basis result field {field}={value!r}")


def _validate_metric_ranges(
    rows: list[dict[str, str]],
    *,
    functional_limit: float | None = None,
) -> None:
    for row in rows:
        cosine = float(row["mapped_direction_cosine"])
        relative_error = float(row["mapped_relative_frobenius_error"])
        norm_ratio = float(row["mapped_norm_ratio"])
        if not -1.000001 <= cosine <= 1.000001 or relative_error < 0 or norm_ratio <= 0:
            raise ValueError(f"Basis metric lies outside its mathematical range: {row}")
        for field in (
            "predicted_descent_relative_error",
            "singular_spectrum_relative_l2_error",
            "mean_head_spectrum_relative_l2_error",
            "maximum_head_spectrum_relative_l2_error",
        ):
            if field in row and float(row[field]) < 0:
                raise ValueError(f"Basis error metric is negative: {field}")
        if (
            "mean_head_spectrum_relative_l2_error" in row
            and float(row["mean_head_spectrum_relative_l2_error"])
            > float(row["maximum_head_spectrum_relative_l2_error"]) + 1e-12
        ):
            raise ValueError("Mean head-spectrum error exceeds the declared maximum")
        if functional_limit is not None:
            observed = float(row["functional_invariance_max_abs_logit_error"])
            if observed < 0 or observed > functional_limit:
                raise ValueError("Basis functional-invariance error exceeds the frozen limit")


def _integer_field(row: dict[str, str], field: str) -> int:
    value = row[field]
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError(f"Basis identity field is not a canonical integer: {field}={value!r}")
    return parsed


def _assert_unique_exact(
    observed: list[tuple[Any, ...]], expected: set[tuple[Any, ...]], label: str
) -> None:
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"Basis {label} identities differ from the frozen Cartesian product")


def _validate_summary_derivation(
    records: list[dict[str, str]], summaries: list[dict[str, str]]
) -> None:
    expected = {(str(row["family"]), str(row["optimizer"])): row for row in _summaries(records)}
    for observed in summaries:
        identity = (observed["family"], observed["optimizer"])
        derived = expected.get(identity)
        if derived is None or set(observed) != set(derived):
            raise ValueError("Basis summary schema or identity is not derivable from records")
        for field in SUMMARY_FIELDS[2:]:
            if field == "records":
                if _integer_field(observed, field) != int(derived[field]):
                    raise ValueError("Basis summary record count is not derivable")
            elif not math.isclose(
                float(observed[field]), float(derived[field]), rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"Basis summary field is not derivable: {identity}/{field}")


def audit_basis_sensitivity(
    protocol_path: str | Path = "configs/basis_sensitivity.json",
    *,
    output_dir: str | Path = "reports/basis-sensitivity",
    verify_inputs: bool = False,
) -> dict[str, Any]:
    resolved_protocol, protocol = _load_protocol(protocol_path)
    output = Path(output_dir).resolve()
    manifest_path = output / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = protocol["selection"]
    expected_coverage = {
        "anchors": protocol["common_state"]["expected_anchors"],
        "tensor_sequences": selection["expected_tensor_sequences"],
        "records": selection["expected_records"],
        "head_records": selection["expected_head_records"],
        "summary_rows": selection["expected_summary_rows"],
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("protocol", {}).get("sha256") != _sha256(resolved_protocol)
        or manifest.get("common_state_spec", {}).get("sha256")
        != protocol["common_state"]["spec_sha256"]
        or manifest.get("coverage") != expected_coverage
    ):
        raise ValueError("Basis-sensitivity summary manifest differs from the frozen protocol")
    common_payload = json.loads(
        resolve_matrix_path(protocol["common_state"]["spec"]).read_text(encoding="utf-8")
    )
    analysis = manifest.get("analysis", {})
    if (
        analysis.get("operator_protocol") != common_payload["operator_protocol"]
        or analysis.get("weight_decay_included") is not False
        or analysis.get("parameters_advanced") is not False
        or analysis.get("causal_boundary") != protocol["metrics"]["causal_boundary"]
        or not isinstance(analysis.get("device"), str)
        or not analysis["device"]
    ):
        raise ValueError("Basis analysis settings differ from the frozen replay contract")
    calibration = manifest.get("functional_calibration", {})
    calibration_by_seed = calibration.get("by_rotation_seed")
    expected_calibration_seeds = {str(seed) for seed in selection["rotation_seeds"]}
    if (
        not isinstance(calibration.get("maximum_observed_absolute_logit_error"), (int, float))
        or isinstance(calibration["maximum_observed_absolute_logit_error"], bool)
        or calibration["maximum_observed_absolute_logit_error"]
        > protocol["functional_calibration"]["maximum_absolute_logit_error"]
        or calibration.get("maximum_allowed_absolute_logit_error")
        != protocol["functional_calibration"]["maximum_absolute_logit_error"]
        or not isinstance(calibration_by_seed, dict)
        or set(calibration_by_seed) != expected_calibration_seeds
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0 <= value <= protocol["functional_calibration"]["maximum_absolute_logit_error"]
            for value in calibration_by_seed.values()
        )
        or not math.isclose(
            calibration["maximum_observed_absolute_logit_error"],
            max(calibration_by_seed.values()),
            rel_tol=0,
            abs_tol=1e-18,
        )
    ):
        raise ValueError("Basis functional-invariance calibration failed")

    common_root = Path(protocol["common_state"]["root"]).resolve()
    actual_sources = _load_sources(protocol, verify_hashes=verify_inputs)
    expected_source_records = [_source_record(source, common_root) for source in actual_sources]
    if manifest.get("sources") != expected_source_records:
        raise ValueError("Basis source ledger differs from the frozen common-state inputs")

    declared_outputs = manifest.get("outputs", {})
    tables = {}
    output_contracts = (
        (
            "records",
            selection["expected_records"],
            "records.csv",
            {
                "family",
                "anchor",
                "layer",
                "tensor",
                "rotation_seed",
                "optimizer",
                *RECORD_METRICS,
                "original_predicted_descent_per_unit_lr",
                "rotated_predicted_descent_per_unit_lr",
                "mean_head_spectrum_relative_l2_error",
                "maximum_head_spectrum_relative_l2_error",
                "functional_invariance_max_abs_logit_error",
            },
        ),
        (
            "head_records",
            selection["expected_head_records"],
            "head_records.csv",
            {
                "family",
                "anchor",
                "layer",
                "tensor",
                "rotation_seed",
                "optimizer",
                "qk_role",
                "head",
                *HEAD_METRICS,
            },
        ),
        ("summary", selection["expected_summary_rows"], "summary.csv", set(SUMMARY_FIELDS)),
    )
    for name, expected_rows, expected_path, expected_fields in output_contracts:
        item = declared_outputs.get(name, {})
        if item.get("path") != expected_path or item.get("rows") != expected_rows:
            raise ValueError(f"Basis output path differs from its fixed contract: {name}")
        path = output / expected_path
        if (
            not path.is_file()
            or path.stat().st_size != item.get("bytes")
            or _sha256(path) != item.get("sha256")
        ):
            raise ValueError(f"Basis output differs from its manifest: {name}")
        rows = _read_csv(path)
        if len(rows) != expected_rows or any(set(row) != expected_fields for row in rows):
            raise ValueError(f"Basis output schema or row count differs: {name}")
        tables[name] = rows

    _validate_finite_rows(
        tables["records"],
        {"family", "anchor", "tensor", "optimizer"},
    )
    _validate_finite_rows(
        tables["head_records"],
        {"family", "anchor", "tensor", "optimizer", "qk_role"},
    )
    _validate_finite_rows(tables["summary"], {"family", "optimizer"})
    functional_limit = float(protocol["functional_calibration"]["maximum_absolute_logit_error"])
    _validate_metric_ranges(tables["records"], functional_limit=functional_limit)
    _validate_metric_ranges(tables["head_records"])

    tensor_for_layer = {
        layer: protocol["architecture"]["qkv_tensor_template"].format(layer=layer)
        for layer in selection["layers"]
    }
    record_expected = {
        (
            source.family,
            source.anchor,
            layer,
            tensor_for_layer[layer],
            seed,
            optimizer,
        )
        for source in actual_sources
        for layer in selection["layers"]
        for seed in selection["rotation_seeds"]
        for optimizer in selection["optimizers"]
    }
    record_identities = [
        (
            row["family"],
            row["anchor"],
            _integer_field(row, "layer"),
            row["tensor"],
            _integer_field(row, "rotation_seed"),
            row["optimizer"],
        )
        for row in tables["records"]
    ]
    _assert_unique_exact(record_identities, record_expected, "record")
    head_expected = {
        (*identity, role, head)
        for identity in record_expected
        for role in ("query", "key")
        for head in selection["heads"]
    }
    head_identities = [
        (
            row["family"],
            row["anchor"],
            _integer_field(row, "layer"),
            row["tensor"],
            _integer_field(row, "rotation_seed"),
            row["optimizer"],
            row["qk_role"],
            _integer_field(row, "head"),
        )
        for row in tables["head_records"]
    ]
    _assert_unique_exact(head_identities, head_expected, "head-record")

    head_groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for identity, row in zip(head_identities, tables["head_records"], strict=True):
        head_groups.setdefault(identity[:6], []).append(row)
    for identity, record in zip(record_identities, tables["records"], strict=True):
        group = head_groups[identity]
        spectra = [float(row["singular_spectrum_relative_l2_error"]) for row in group]
        if not math.isclose(
            float(record["mean_head_spectrum_relative_l2_error"]),
            statistics.mean(spectra),
            rel_tol=0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(record["maximum_head_spectrum_relative_l2_error"]),
            max(spectra),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Basis full-tensor head aggregates are not derivable")
        seed = str(identity[4])
        if not math.isclose(
            float(record["functional_invariance_max_abs_logit_error"]),
            float(calibration_by_seed[seed]),
            rel_tol=0,
            abs_tol=1e-18,
        ):
            raise ValueError("Basis record calibration differs from its rotation seed")

    summary_expected = {
        (family, optimizer)
        for family in selection["families"]
        for optimizer in selection["optimizers"]
    }
    summary_identities = [(row["family"], row["optimizer"]) for row in tables["summary"]]
    _assert_unique_exact(summary_identities, summary_expected, "summary")
    _validate_summary_derivation(tables["records"], tables["summary"])
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure AdamW/Muon/NorMuon equivariance under RoPE-preserving Q/K bases"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/basis_sensitivity.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/basis-sensitivity"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.audit_only:
        result = audit_basis_sensitivity(
            args.protocol,
            output_dir=args.output_dir,
            verify_inputs=args.verify_inputs,
        )
    else:
        result = analyze_basis_sensitivity(
            args.protocol,
            output_dir=args.output_dir,
            device=args.device,
            verify_inputs=args.verify_inputs,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
