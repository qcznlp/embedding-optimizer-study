from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common_state_matrix import (
    CommonStateJob,
    _load_protocol,
    _resolve_reference,
    build_common_state_jobs,
    common_state_job_complete,
    resolve_common_state_spec,
)
from .config import RunConfig, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .geometry_summary import _atomic_csv
from .scope import ALL_FAMILIES, resolve_scope
from .update_geometry import ALGORITHMS

MATRIX_FIELDS = (
    "frobenius_norm",
    "top_1pct_row_energy",
    "top_10pct_row_energy",
    "spectral_norm",
    "approx_stable_rank",
    "sketched_nuclear_norm",
    "sketched_entropy_effective_rank",
    "sketched_condition_number",
    "captured_frobenius_energy",
)
DISTRIBUTION_FIELDS = ("mean", "cv", "gini", "max_to_median")
MATRIX_KEYS = {
    *MATRIX_FIELDS,
    "row_norms",
    "column_norms",
    "algorithm",
    "rank",
}
PAIRWISE = tuple(
    f"{left}__{right}"
    for left_index, left in enumerate(ALGORITHMS)
    for right in ALGORITHMS[left_index + 1 :]
)
IDENTITY_FIELDS = (
    "family",
    "anchor_kind",
    "source_optimizer",
    "learning_rate",
    "run_id",
    "stage",
    "fraction",
    "step",
    "label",
)


@dataclass(frozen=True)
class ExpectedCommonStateMetric:
    job: CommonStateJob
    anchor_kind: str
    source_optimizer: str
    learning_rate: float | str
    run_id: str
    stage: int
    fraction: float
    step: int


def expected_common_state_metrics(
    jobs: list[CommonStateJob], configs: list[RunConfig]
) -> list[ExpectedCommonStateMetric]:
    by_run = {(config.model_family, config.run_id): config for config in configs}
    expected: list[ExpectedCommonStateMetric] = []
    for job in jobs:
        parts = job.label.split("/")
        if parts == [job.family, "pretrained"]:
            expected.append(
                ExpectedCommonStateMetric(
                    job=job,
                    anchor_kind="pretrained",
                    source_optimizer="",
                    learning_rate="",
                    run_id="pretrained",
                    stage=0,
                    fraction=0.0,
                    step=0,
                )
            )
            continue
        if len(parts) != 3 or parts[0] != job.family or not parts[2].startswith("checkpoint-"):
            raise ValueError(f"Invalid common-state anchor label: {job.label}")
        config = by_run.get((job.family, parts[1]))
        if config is None:
            raise ValueError(f"Common-state anchor is absent from the matrix: {job.label}")
        try:
            step = int(parts[2].removeprefix("checkpoint-"))
            schedule = json.loads(
                (config.output_dir / "checkpoint_schedule.json").read_text(encoding="utf-8")
            )
            steps = [int(value) for value in schedule["steps"]]
            fractions = [float(value) for value in schedule["fractions"]]
            stage = steps.index(step) + 1
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot resolve anchor stage for {job.label}: {error}") from error
        if len(steps) != len(fractions) or len(steps) != len(config.checkpoint_fractions):
            raise ValueError(f"Invalid checkpoint schedule for {job.label}")
        if job.checkpoint != (config.output_dir / parts[2]).resolve():
            raise ValueError(f"Anchor checkpoint path differs from its matrix run: {job.label}")
        expected.append(
            ExpectedCommonStateMetric(
                job=job,
                anchor_kind="checkpoint",
                source_optimizer=config.optimizer.name,
                learning_rate=config.optimizer.lr,
                run_id=config.run_id,
                stage=stage,
                fraction=fractions[stage - 1],
                step=step,
            )
        )
    return expected


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _validate_matrix(payload: Any, *, context: str) -> None:
    if not isinstance(payload, dict) or set(payload) != MATRIX_KEYS or not _all_finite(payload):
        raise ValueError(f"Invalid matrix metrics in {context}")
    for name in ("row_norms", "column_norms"):
        distribution = payload[name]
        if not isinstance(distribution, dict) or set(distribution) != set(DISTRIBUTION_FIELDS):
            raise ValueError(f"Invalid {name} metrics in {context}")
    if not isinstance(payload["algorithm"], str) or not isinstance(payload["rank"], int):
        raise ValueError(f"Invalid spectral metadata in {context}")
    numeric = [payload[name] for name in MATRIX_FIELDS if payload[name] is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in numeric):
        raise ValueError(f"Non-numeric matrix metrics in {context}")


def _flatten_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    row = {name: payload[name] if payload[name] is not None else "" for name in MATRIX_FIELDS}
    row["sketch_algorithm"] = payload["algorithm"]
    row["sketch_rank"] = payload["rank"]
    for axis in ("row", "column"):
        for statistic in DISTRIBUTION_FIELDS:
            row[f"{axis}_norm_{statistic}"] = payload[f"{axis}_norms"][statistic]
    return row


def _matrix_output_fields() -> list[str]:
    return [
        *MATRIX_FIELDS,
        "sketch_algorithm",
        "sketch_rank",
        *(
            f"{axis}_norm_{statistic}"
            for axis in ("row", "column")
            for statistic in DISTRIBUTION_FIELDS
        ),
    ]


def _identity(expected: ExpectedCommonStateMetric) -> dict[str, Any]:
    return {
        "family": expected.job.family,
        "anchor_kind": expected.anchor_kind,
        "source_optimizer": expected.source_optimizer,
        "learning_rate": expected.learning_rate,
        "run_id": expected.run_id,
        "stage": expected.stage,
        "fraction": expected.fraction,
        "step": expected.step,
        "label": expected.job.label,
    }


def _read_records(
    expected: ExpectedCommonStateMetric, common_state_spec: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    job = expected.job
    if not common_state_job_complete(job, common_state_spec, verify_hashes=True):
        raise ValueError(f"Common-state anchor is missing or invalid: {job.label}")
    manifest_path = job.update_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["outputs"]["metrics"]
    metrics_path = job.update_dir / metadata["path"]
    raw = metrics_path.read_bytes()
    if len(raw) != metadata["bytes"] or _sha256(metrics_path) != metadata["sha256"]:
        raise ValueError(f"Common-state metrics differ from their manifest: {metrics_path}")
    records = [json.loads(line) for line in raw.splitlines() if line]
    if len(records) != job.hidden_tensors:
        raise ValueError(
            f"Expected {job.hidden_tensors} tensor records for {job.label}, found {len(records)}"
        )
    expected_keys = {
        "schema_version",
        "tensor",
        "shape",
        "parameters",
        "gradient_steps",
        "weight_frobenius_norm",
        "final_gradient",
        "algorithms",
        "pairwise_cosine",
    }
    names: set[str] = set()
    parameters = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid common-state tensor record for {job.label}")
        tensor = record.get("tensor")
        shape = record.get("shape")
        context = f"{job.label}/{tensor}"
        if (
            set(record) != expected_keys
            or record.get("schema_version") != SCHEMA_VERSION
            or not isinstance(tensor, str)
            or tensor in names
            or not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
            or record.get("parameters") != math.prod(shape)
            or record.get("gradient_steps") != job.gradient_steps
            or set(record.get("algorithms") or {}) != set(ALGORITHMS)
            or set(record.get("pairwise_cosine") or {}) != set(PAIRWISE)
            or not _all_finite(record)
        ):
            raise ValueError(f"Invalid common-state tensor record: {context}")
        _validate_matrix(record["final_gradient"], context=f"{context}/final_gradient")
        weight_norm = record["weight_frobenius_norm"]
        if not isinstance(weight_norm, (int, float)) or weight_norm <= 0:
            raise ValueError(f"Invalid weight norm in {context}")
        for algorithm, values in record["algorithms"].items():
            if not isinstance(values, dict):
                raise ValueError(f"Invalid {algorithm} update metrics in {context}")
            extras = {
                "cosine_with_final_gradient",
                "cosine_with_weight",
                "per_unit_lr_update_to_weight",
                "matched_frobenius_norm",
            }
            if set(values) != MATRIX_KEYS | extras:
                raise ValueError(f"Invalid {algorithm} update fields in {context}")
            _validate_matrix(
                {key: value for key, value in values.items() if key not in extras},
                context=f"{context}/{algorithm}",
            )
            if (
                any(not isinstance(values[name], (int, float)) for name in extras)
                or any(isinstance(values[name], bool) for name in extras)
                or values["per_unit_lr_update_to_weight"] <= 0
                or not math.isclose(
                    values["matched_frobenius_norm"], weight_norm, rel_tol=1e-6, abs_tol=1e-6
                )
            ):
                raise ValueError(f"Invalid {algorithm} update scalars in {context}")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in record["pairwise_cosine"].values()
        ):
            raise ValueError(f"Invalid pairwise cosine in {context}")
        names.add(tensor)
        parameters += record["parameters"]
    if parameters != job.hidden_parameters:
        raise ValueError(
            f"Expected {job.hidden_parameters} parameters for {job.label}, found {parameters}"
        )
    return manifest, sorted(records, key=lambda value: value["tensor"])


def _weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total = sum(row["parameters"] for row in rows)
    return sum(row["parameters"] * float(row[field]) for row in rows) / total


def _rss(rows: list[dict[str, Any]], field: str) -> float:
    return math.sqrt(sum(float(row[field]) ** 2 for row in rows))


def _aggregate_update_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weight_norm = _rss(rows, "weight_frobenius_norm")
    direction_norm = _rss(rows, "frobenius_norm")
    weighted = (
        "row_norm_cv",
        "row_norm_gini",
        "row_norm_max_to_median",
        "column_norm_cv",
        "column_norm_gini",
        "column_norm_max_to_median",
        "top_1pct_row_energy",
        "top_10pct_row_energy",
        "spectral_norm",
        "approx_stable_rank",
        "sketched_nuclear_norm",
        "sketched_entropy_effective_rank",
        "captured_frobenius_energy",
        "cosine_with_final_gradient",
        "cosine_with_weight",
        "per_unit_lr_update_to_weight",
    )
    return {
        "tensors": len(rows),
        "parameters": sum(row["parameters"] for row in rows),
        "weight_frobenius_norm": weight_norm,
        "direction_frobenius_norm": direction_norm,
        "global_per_unit_lr_update_to_weight": direction_norm / weight_norm,
        **{f"{field}_parameter_weighted": _weighted_mean(rows, field) for field in weighted},
    }


def _aggregate_gradient_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weight_norm = _rss(rows, "weight_frobenius_norm")
    gradient_norm = _rss(rows, "frobenius_norm")
    weighted = (
        "row_norm_cv",
        "row_norm_gini",
        "row_norm_max_to_median",
        "column_norm_cv",
        "column_norm_gini",
        "column_norm_max_to_median",
        "top_1pct_row_energy",
        "top_10pct_row_energy",
        "spectral_norm",
        "approx_stable_rank",
        "sketched_nuclear_norm",
        "sketched_entropy_effective_rank",
        "captured_frobenius_energy",
    )
    return {
        "tensors": len(rows),
        "parameters": sum(row["parameters"] for row in rows),
        "weight_frobenius_norm": weight_norm,
        "gradient_frobenius_norm": gradient_norm,
        "global_gradient_to_weight": gradient_norm / weight_norm,
        **{f"{field}_parameter_weighted": _weighted_mean(rows, field) for field in weighted},
    }


def _positive_ratio(numerator: float, denominator: float, *, context: str) -> float:
    if denominator <= 0:
        raise ValueError(f"Non-positive AdamW contrast denominator for {context}")
    return numerator / denominator


def summarize_common_state(
    expected: list[ExpectedCommonStateMetric],
    result_root: Path,
    output_dir: Path,
    *,
    common_state_spec: Path,
    allow_partial: bool = False,
    families: tuple[str, ...] = ALL_FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    result_root = result_root.resolve()
    output_dir = output_dir.resolve()
    common_state_spec = common_state_spec.resolve()
    spec, anchor_protocol = _load_protocol(common_state_spec)
    families, scope = resolve_scope(families, scope_amendment)
    expected_anchors = int(anchor_protocol["expected_anchors_per_family"]) * len(families)
    if len(expected) != expected_anchors:
        raise ValueError(
            f"Expected {expected_anchors} anchors for the requested family scope, "
            f"received {len(expected)}"
        )
    if {item.job.family for item in expected} != set(families):
        raise ValueError("Common-state identities do not match the requested family scope")
    labels = [item.job.label for item in expected]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate common-state anchor labels")
    expected_manifests = {(item.job.update_dir / "manifest.json").resolve() for item in expected}
    observed_manifests = set()
    for path in result_root.rglob("manifest.json"):
        if path.parent.name != "updates":
            continue
        relative = path.relative_to(result_root)
        if relative.parts and relative.parts[0] in set(ALL_FAMILIES) - set(families):
            continue
        observed_manifests.add(path.resolve())
    unexpected = observed_manifests.difference(expected_manifests)
    if unexpected:
        raise ValueError(f"Unexpected common-state update manifests: {sorted(unexpected)[:5]}")

    gradient_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    gradient_anchor_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    pairwise_anchor_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    missing: list[str] = []
    tensor_signature: tuple[tuple[str, tuple[int, ...], int], ...] | None = None
    for item in expected:
        if not common_state_job_complete(item.job, common_state_spec, verify_hashes=True):
            missing.append(item.job.label)
            continue
        manifest, records = _read_records(item, common_state_spec)
        signature = tuple(
            (record["tensor"], tuple(record["shape"]), record["parameters"]) for record in records
        )
        if tensor_signature is None:
            tensor_signature = signature
        elif signature != tensor_signature:
            raise ValueError(f"Hidden tensor set differs for {item.job.label}")
        identity = _identity(item)
        per_algorithm: dict[str, list[dict[str, Any]]] = {name: [] for name in ALGORITHMS}
        per_pair: dict[str, list[dict[str, Any]]] = {name: [] for name in PAIRWISE}
        for record in records:
            base = {
                **identity,
                "tensor": record["tensor"],
                "rows": record["shape"][0],
                "columns": record["shape"][1],
                "parameters": record["parameters"],
                "gradient_steps": record["gradient_steps"],
                "weight_frobenius_norm": record["weight_frobenius_norm"],
            }
            gradient_rows.append({**base, **_flatten_matrix(record["final_gradient"])})
            for algorithm in ALGORITHMS:
                values = record["algorithms"][algorithm]
                row = {
                    **base,
                    "update_operator": algorithm,
                    **_flatten_matrix(values),
                    "cosine_with_final_gradient": values["cosine_with_final_gradient"],
                    "cosine_with_weight": values["cosine_with_weight"],
                    "per_unit_lr_update_to_weight": values["per_unit_lr_update_to_weight"],
                    "matched_frobenius_norm": values["matched_frobenius_norm"],
                }
                update_rows.append(row)
                per_algorithm[algorithm].append(row)
            for pair, cosine in sorted(record["pairwise_cosine"].items()):
                row = {**base, "operator_pair": pair, "cosine": cosine}
                pairwise_rows.append(row)
                per_pair[pair].append(row)
        for algorithm in ALGORITHMS:
            anchor_rows.append(
                {
                    **identity,
                    "update_operator": algorithm,
                    **_aggregate_update_rows(per_algorithm[algorithm]),
                }
            )
        gradient_anchor_rows.append(
            {**identity, **_aggregate_gradient_rows(gradient_rows[-len(records) :])}
        )
        for pair in PAIRWISE:
            pairwise_anchor_rows.append(
                {
                    **identity,
                    "operator_pair": pair,
                    "tensors": len(per_pair[pair]),
                    "parameters": sum(row["parameters"] for row in per_pair[pair]),
                    "cosine_parameter_weighted": _weighted_mean(per_pair[pair], "cosine"),
                }
            )
        manifest_path = item.job.update_dir / "manifest.json"
        inputs.append(
            {
                "label": item.job.label,
                "gradient_manifest_path": str(item.job.gradient_dir / "manifest.json"),
                "gradient_manifest_sha256": manifest["gradient_manifest"]["sha256"],
                "update_manifest_path": str(manifest_path),
                "update_manifest_sha256": _sha256(manifest_path),
                "metrics_sha256": manifest["outputs"]["metrics"]["sha256"],
            }
        )
    if missing and not allow_partial:
        raise ValueError(
            f"Common-state matrix is incomplete: {len(missing)}/{len(expected)} missing or invalid; "
            f"first={missing[:5]}"
        )
    if not anchor_rows:
        raise ValueError("No valid common-state anchors were found")

    indexed = {(row["label"], row["update_operator"]): row for row in anchor_rows}
    pair_indexed = {(row["label"], row["operator_pair"]): row for row in pairwise_anchor_rows}
    gradient_indexed = {row["label"]: row for row in gradient_anchor_rows}
    contrast_rows: list[dict[str, Any]] = []
    update_gradient_rows: list[dict[str, Any]] = []
    ratio_fields = (
        "direction_frobenius_norm",
        "global_per_unit_lr_update_to_weight",
        "row_norm_cv_parameter_weighted",
        "row_norm_gini_parameter_weighted",
        "top_1pct_row_energy_parameter_weighted",
        "approx_stable_rank_parameter_weighted",
        "sketched_entropy_effective_rank_parameter_weighted",
        "spectral_norm_parameter_weighted",
    )
    for item in expected:
        adamw = indexed.get((item.job.label, "adamw"))
        if adamw is None:
            continue
        gradient = gradient_indexed[item.job.label]
        for algorithm in ALGORITHMS:
            candidate = indexed[(item.job.label, algorithm)]
            update_gradient_rows.append(
                {
                    **_identity(item),
                    "update_operator": algorithm,
                    "direction_frobenius_norm_to_gradient_ratio": _positive_ratio(
                        candidate["direction_frobenius_norm"],
                        gradient["gradient_frobenius_norm"],
                        context=f"{item.job.label}/gradient_frobenius_norm",
                    ),
                    **{
                        f"{field}_to_gradient_ratio": _positive_ratio(
                            candidate[f"{field}_parameter_weighted"],
                            gradient[f"{field}_parameter_weighted"],
                            context=f"{item.job.label}/{field}",
                        )
                        for field in (
                            "row_norm_cv",
                            "row_norm_gini",
                            "top_1pct_row_energy",
                            "approx_stable_rank",
                            "sketched_entropy_effective_rank",
                            "spectral_norm",
                        )
                    },
                    "cosine_with_final_gradient_parameter_weighted": candidate[
                        "cosine_with_final_gradient_parameter_weighted"
                    ],
                }
            )
        for algorithm in ("muon", "normuon"):
            candidate = indexed[(item.job.label, algorithm)]
            pair = f"adamw__{algorithm}"
            contrast_rows.append(
                {
                    **_identity(item),
                    "update_operator": algorithm,
                    **{
                        f"{field}_to_adamw_ratio": _positive_ratio(
                            candidate[field], adamw[field], context=f"{item.job.label}/{field}"
                        )
                        for field in ratio_fields
                    },
                    "cosine_with_final_gradient_minus_adamw": candidate[
                        "cosine_with_final_gradient_parameter_weighted"
                    ]
                    - adamw["cosine_with_final_gradient_parameter_weighted"],
                    "cosine_with_weight_minus_adamw": candidate[
                        "cosine_with_weight_parameter_weighted"
                    ]
                    - adamw["cosine_with_weight_parameter_weighted"],
                    "cosine_with_adamw_parameter_weighted": pair_indexed[(item.job.label, pair)][
                        "cosine_parameter_weighted"
                    ],
                }
            )

    matrix_fields = _matrix_output_fields()
    base_tensor_fields = [
        *IDENTITY_FIELDS,
        "tensor",
        "rows",
        "columns",
        "parameters",
        "gradient_steps",
        "weight_frobenius_norm",
    ]
    output_paths = {
        "gradient_tensor_metrics": output_dir / "gradient_tensor_metrics.csv",
        "update_tensor_metrics": output_dir / "update_tensor_metrics.csv",
        "pairwise_tensor_cosines": output_dir / "pairwise_tensor_cosines.csv",
        "gradient_anchor_metrics": output_dir / "gradient_anchor_metrics.csv",
        "anchor_metrics": output_dir / "anchor_metrics.csv",
        "pairwise_anchor_cosines": output_dir / "pairwise_anchor_cosines.csv",
        "update_gradient_contrasts": output_dir / "update_gradient_contrasts.csv",
        "anchor_contrasts": output_dir / "anchor_contrasts.csv",
    }
    _atomic_csv(
        output_paths["gradient_tensor_metrics"],
        gradient_rows,
        [*base_tensor_fields, *matrix_fields],
    )
    _atomic_csv(
        output_paths["update_tensor_metrics"],
        update_rows,
        [
            *base_tensor_fields,
            "update_operator",
            *matrix_fields,
            "cosine_with_final_gradient",
            "cosine_with_weight",
            "per_unit_lr_update_to_weight",
            "matched_frobenius_norm",
        ],
    )
    _atomic_csv(
        output_paths["pairwise_tensor_cosines"],
        pairwise_rows,
        [*base_tensor_fields, "operator_pair", "cosine"],
    )
    _atomic_csv(
        output_paths["gradient_anchor_metrics"],
        gradient_anchor_rows,
        list(gradient_anchor_rows[0]),
    )
    _atomic_csv(output_paths["anchor_metrics"], anchor_rows, list(anchor_rows[0]))
    _atomic_csv(
        output_paths["pairwise_anchor_cosines"],
        pairwise_anchor_rows,
        list(pairwise_anchor_rows[0]),
    )
    _atomic_csv(
        output_paths["update_gradient_contrasts"],
        update_gradient_rows,
        list(update_gradient_rows[0]),
    )
    _atomic_csv(output_paths["anchor_contrasts"], contrast_rows, list(contrast_rows[0]))
    row_counts = {
        "gradient_tensor_metrics": len(gradient_rows),
        "update_tensor_metrics": len(update_rows),
        "pairwise_tensor_cosines": len(pairwise_rows),
        "gradient_anchor_metrics": len(gradient_anchor_rows),
        "anchor_metrics": len(anchor_rows),
        "pairwise_anchor_cosines": len(pairwise_anchor_rows),
        "update_gradient_contrasts": len(update_gradient_rows),
        "anchor_contrasts": len(contrast_rows),
    }
    complete = not missing and len(inputs) == len(expected)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "allow_partial": allow_partial,
        "families": list(families),
        "scope_amendment": scope,
        "expected_anchors": len(expected),
        "valid_anchors": len(inputs),
        "missing_labels": missing,
        "common_state_spec": {
            "path": str(common_state_spec),
            "sha256": _sha256(common_state_spec),
            "freeze_context": spec["anchor_protocol"]["freeze_context"],
        },
        "inputs": inputs,
        "outputs": {
            name: {
                "path": str(path),
                "rows": row_counts[name],
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in output_paths.items()
        },
        "interpretation": (
            "All update operators replay the same ordered gradients at the same frozen anchor; "
            "weights are never advanced, per-tensor matched files equalize update and weight "
            "Frobenius norms, and weight decay is excluded."
        ),
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly aggregate common-state update geometry")
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--result-root", type=Path, default=Path("results/common-state"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/common-state"))
    parser.add_argument(
        "--common-state-spec", type=Path, default=Path("configs/common_state_probe.json")
    )
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--late-reference-checkpoint", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    all_configs = load_matrix(matrix_path)
    common_state_spec = resolve_common_state_spec(args.common_state_spec).resolve()
    spec, anchor = _load_protocol(common_state_spec)
    families, _ = resolve_scope(args.families, args.scope_amendment)
    configs = [config for config in all_configs if config.model_family in families]
    if {config.model_family for config in configs} != set(families):
        raise ValueError("Training matrix does not cover every requested model family")
    by_family = {config.model_family: config for config in configs}
    references = {}
    for family in families:
        explicit = (
            args.dense_reference_checkpoint if family == "dense" else args.late_reference_checkpoint
        )
        references[family] = _resolve_reference(by_family[family], explicit)
    jobs = build_common_state_jobs(configs, references, spec, args.result_root)
    expected = expected_common_state_metrics(jobs, configs)
    manifest = summarize_common_state(
        expected,
        args.result_root,
        args.output_dir,
        common_state_spec=common_state_spec,
        allow_partial=args.allow_partial,
        families=families,
        scope_amendment=args.scope_amendment,
    )
    print(
        f"Aggregated {manifest['valid_anchors']}/{manifest['expected_anchors']} common-state "
        f"anchors into {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
