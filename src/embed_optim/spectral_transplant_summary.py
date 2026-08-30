from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common_state_matrix import (
    _load_protocol,
    _resolve_reference,
    build_common_state_jobs,
    resolve_common_state_spec,
)
from .config import ModelFamily, load_matrix, resolve_matrix_path
from .functional_intervention_summary import METRICS, _read_jsonl, _write_csv
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .scope import ALL_FAMILIES, resolve_scope
from .spectral_transplant import (
    NATIVE_CONDITIONS,
    load_spectral_transplant_protocol,
    spectral_conditions,
)
from .spectral_transplant_matrix import (
    SpectralTransplantJob,
    build_spectral_transplant_jobs,
    spectral_transplant_job_complete,
)


def _condition_metadata(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata = {
        "adamw-native": {
            "category": "native",
            "basis_source": "adamw",
            "spectrum_operation": "native",
            "interpolation_lambda": 0.0,
            "band": "",
        },
        "muon-native": {
            "category": "native",
            "basis_source": "muon",
            "spectrum_operation": "native",
            "interpolation_lambda": 1.0,
            "band": "",
        },
    }
    for condition in spectral_conditions(spec):
        metadata[condition.name] = {
            "category": "transformed",
            "basis_source": condition.basis_source,
            "spectrum_operation": condition.spectrum_operation,
            "interpolation_lambda": (
                "" if condition.interpolation_lambda is None else condition.interpolation_lambda
            ),
            "band": condition.band or "",
        }
    return metadata


def _anchor_effects(
    label: str,
    family: str,
    records: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata = _condition_metadata(spec)
    expected_conditions = ["baseline", *NATIVE_CONDITIONS, *metadata.keys()]
    # Native keys occur in metadata already, so preserve exactly one copy of each condition.
    expected_conditions = list(dict.fromkeys(expected_conditions))
    by_condition: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        condition = record.get("condition")
        sample_id = int(record.get("sample_id"))
        if condition not in expected_conditions or sample_id in by_condition[condition]:
            raise ValueError(f"Duplicate or unexpected condition/sample in {label}")
        for metric in METRICS:
            value = float(record.get(metric))
            if not __import__("math").isfinite(value):
                raise ValueError(f"Non-finite {metric} in {label}/{condition}/{sample_id}")
        by_condition[condition][sample_id] = record
    if set(by_condition) != set(expected_conditions):
        raise ValueError(f"Condition coverage differs from the spectral protocol in {label}")
    baseline = by_condition["baseline"]
    expected_samples = spec["evaluation"]["examples"]
    if len(baseline) != expected_samples:
        raise ValueError(f"Baseline sample count differs in {label}")
    sample_ids = sorted(baseline)
    if any(set(rows) != set(sample_ids) for rows in by_condition.values()):
        raise ValueError(f"Conditions do not contain the same paired samples in {label}")

    effects = []
    for condition in expected_conditions[1:]:
        rows = by_condition[condition]
        effect = {
            "family": family,
            "anchor": label,
            "condition": condition,
            **metadata[condition],
            "samples": len(rows),
        }
        for metric in METRICS:
            values = [float(rows[sample_id][metric]) for sample_id in sample_ids]
            bases = [float(baseline[sample_id][metric]) for sample_id in sample_ids]
            effect[metric] = statistics.fmean(values)
            effect[f"delta_{metric}"] = statistics.fmean(
                value - base for value, base in zip(values, bases, strict=True)
            )
        effects.append(effect)
    return effects


def _linear_quantile(values: list[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("Quantiles require finite non-empty values and q in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Quantiles require finite values")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _worst_loss_tail(values: dict[int, float], count: int) -> set[int]:
    if count < 1 or count > len(values):
        raise ValueError("Worst-tail count is outside the sample range")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Worst-tail values must be finite")
    # The sample id is the deterministic tie-breaker; larger loss changes are worse.
    return set(sorted(values, key=lambda sample_id: (-values[sample_id], sample_id))[:count])


def _anchor_tail_effects(
    label: str,
    family: str,
    records: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata = _condition_metadata(spec)
    expected_conditions = ["baseline", *metadata]
    by_condition: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        condition = record.get("condition")
        sample_id = int(record.get("sample_id"))
        if condition not in expected_conditions or sample_id in by_condition[condition]:
            raise ValueError(f"Duplicate or unexpected tail condition/sample in {label}")
        for metric in ("contrastive_loss", "positive_margin"):
            value = float(record.get(metric))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {metric} in {label}/{condition}/{sample_id}")
        by_condition[condition][sample_id] = record
    if set(by_condition) != set(expected_conditions):
        raise ValueError(f"Tail condition coverage differs from the spectral protocol in {label}")
    sample_ids = sorted(by_condition["baseline"])
    if len(sample_ids) != spec["evaluation"]["examples"] or any(
        set(rows) != set(sample_ids) for rows in by_condition.values()
    ):
        raise ValueError(f"Tail conditions do not contain the frozen paired samples in {label}")

    tail_spec = spec["evaluation"]["tail_protocol"]
    tail_count = int(tail_spec["tail_count"])
    if tail_count != math.ceil(float(tail_spec["tail_fraction"]) * len(sample_ids)):
        raise ValueError("Frozen spectral tail count disagrees with the sample count")
    baseline = by_condition["baseline"]
    adamw = by_condition["adamw-native"]
    adam_loss_delta = {
        sample_id: float(adamw[sample_id]["contrastive_loss"])
        - float(baseline[sample_id]["contrastive_loss"])
        for sample_id in sample_ids
    }
    adam_margin_delta = {
        sample_id: float(adamw[sample_id]["positive_margin"])
        - float(baseline[sample_id]["positive_margin"])
        for sample_id in sample_ids
    }
    adam_tail = _worst_loss_tail(adam_loss_delta, tail_count)

    output = []
    for condition in expected_conditions[1:]:
        if condition == "adamw-native":
            continue
        rows = by_condition[condition]
        condition_loss_delta = {
            sample_id: float(rows[sample_id]["contrastive_loss"])
            - float(baseline[sample_id]["contrastive_loss"])
            for sample_id in sample_ids
        }
        condition_margin_delta = {
            sample_id: float(rows[sample_id]["positive_margin"])
            - float(baseline[sample_id]["positive_margin"])
            for sample_id in sample_ids
        }
        loss_contrast = {
            sample_id: condition_loss_delta[sample_id] - adam_loss_delta[sample_id]
            for sample_id in sample_ids
        }
        margin_contrast = {
            sample_id: condition_margin_delta[sample_id] - adam_margin_delta[sample_id]
            for sample_id in sample_ids
        }
        condition_tail = _worst_loss_tail(condition_loss_delta, tail_count)
        union = adam_tail | condition_tail
        output.append(
            {
                "family": family,
                "anchor": label,
                "condition": condition,
                **metadata[condition],
                "samples": len(sample_ids),
                "tail_count": tail_count,
                "mean_pairwise_loss_contrast": statistics.fmean(loss_contrast.values()),
                "p95_pairwise_loss_contrast": _linear_quantile(list(loss_contrast.values()), 0.95),
                "p99_pairwise_loss_contrast": _linear_quantile(list(loss_contrast.values()), 0.99),
                "mean_pairwise_margin_contrast": statistics.fmean(margin_contrast.values()),
                "p01_pairwise_margin_contrast": _linear_quantile(
                    list(margin_contrast.values()), 0.01
                ),
                "p05_pairwise_margin_contrast": _linear_quantile(
                    list(margin_contrast.values()), 0.05
                ),
                "mean_loss_contrast_on_adam_tail": statistics.fmean(
                    loss_contrast[sample_id] for sample_id in adam_tail
                ),
                "mean_loss_contrast_on_condition_tail": statistics.fmean(
                    loss_contrast[sample_id] for sample_id in condition_tail
                ),
                "worst_loss_tail_jaccard": len(adam_tail & condition_tail) / len(union),
                "adam_tail_baseline_margin_mean": statistics.fmean(
                    float(baseline[sample_id]["positive_margin"]) for sample_id in adam_tail
                ),
                "condition_tail_baseline_margin_mean": statistics.fmean(
                    float(baseline[sample_id]["positive_margin"]) for sample_id in condition_tail
                ),
            }
        )
    return output


def _factorial_effects(anchor_effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in anchor_effects:
        grouped[(row["family"], row["anchor"])][row["condition"]] = row
    cells = {
        "adam_adam": "adamw-native",
        "adam_muon": "adam-basis__muon-spectrum",
        "muon_adam": "muon-basis__adam-spectrum",
        "muon_muon": "muon-native",
    }
    output = []
    for (family, anchor), indexed in sorted(grouped.items()):
        if not set(cells.values()).issubset(indexed):
            raise ValueError(f"Factorial cells are incomplete in {anchor}")
        for metric in METRICS:
            values = {
                cell: float(indexed[condition][f"delta_{metric}"])
                for cell, condition in cells.items()
            }
            aa, am = values["adam_adam"], values["adam_muon"]
            ma, mm = values["muon_adam"], values["muon_muon"]
            output.append(
                {
                    "family": family,
                    "anchor": anchor,
                    "metric": metric,
                    **values,
                    "spectrum_main_effect": 0.5 * ((am - aa) + (mm - ma)),
                    "basis_main_effect": 0.5 * ((ma - aa) + (mm - am)),
                    "spectrum_basis_interaction": mm - ma - am + aa,
                }
            )
    return output


def _spectral_path_effects(anchor_effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in anchor_effects:
        grouped[(row["family"], row["anchor"])][row["condition"]] = row
    path = [
        (0.0, "adamw-native"),
        (0.25, "adam-basis__spectrum-lambda-0.25"),
        (0.5, "adam-basis__spectrum-lambda-0.50"),
        (0.75, "adam-basis__spectrum-lambda-0.75"),
        (1.0, "adam-basis__muon-spectrum"),
    ]
    output = []
    for (family, anchor), indexed in sorted(grouped.items()):
        if not {condition for _, condition in path}.issubset(indexed):
            raise ValueError(f"Spectral interpolation path is incomplete in {anchor}")
        for metric in METRICS:
            reference = float(indexed["adamw-native"][f"delta_{metric}"])
            for interpolation_lambda, condition in path:
                effect = float(indexed[condition][f"delta_{metric}"])
                output.append(
                    {
                        "family": family,
                        "anchor": anchor,
                        "metric": metric,
                        "condition": condition,
                        "interpolation_lambda": interpolation_lambda,
                        "effect_vs_baseline": effect,
                        "contrast_vs_adamw_native": effect - reference,
                    }
                )
    return output


def _band_effects(anchor_effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in anchor_effects:
        grouped[(row["family"], row["anchor"])][row["condition"]] = row
    bands = {
        "head": "adam-basis__muon-head-spectrum",
        "middle": "adam-basis__muon-middle-spectrum",
        "tail": "adam-basis__muon-tail-spectrum",
    }
    output = []
    for (family, anchor), indexed in sorted(grouped.items()):
        if not {"adamw-native", *bands.values()}.issubset(indexed):
            raise ValueError(f"Spectral band conditions are incomplete in {anchor}")
        for metric in METRICS:
            reference = float(indexed["adamw-native"][f"delta_{metric}"])
            for band, condition in bands.items():
                effect = float(indexed[condition][f"delta_{metric}"])
                output.append(
                    {
                        "family": family,
                        "anchor": anchor,
                        "metric": metric,
                        "band": band,
                        "condition": condition,
                        "effect_vs_baseline": effect,
                        "contrast_vs_adamw_native": effect - reference,
                    }
                )
    return output


def _group_summary(
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    values: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, members in sorted(grouped.items()):
        summary = {key: value for key, value in zip(keys, identity, strict=True)}
        summary["anchors"] = len(members)
        for name in values:
            observed = [float(row[name]) for row in members]
            summary[f"mean_{name}"] = statistics.fmean(observed)
            summary[f"median_{name}"] = statistics.median(observed)
            summary[f"positive_anchor_fraction_{name}"] = statistics.fmean(
                value > 0 for value in observed
            )
        output.append(summary)
    return output


def _tail_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "family",
        "condition",
        "category",
        "basis_source",
        "spectrum_operation",
        "interpolation_lambda",
        "band",
    )
    favorable_sign = {
        "mean_pairwise_loss_contrast": "negative",
        "p95_pairwise_loss_contrast": "negative",
        "p99_pairwise_loss_contrast": "negative",
        "mean_pairwise_margin_contrast": "positive",
        "p01_pairwise_margin_contrast": "positive",
        "p05_pairwise_margin_contrast": "positive",
        "mean_loss_contrast_on_adam_tail": "negative",
        "mean_loss_contrast_on_condition_tail": "negative",
    }
    descriptive = (
        "worst_loss_tail_jaccard",
        "adam_tail_baseline_margin_mean",
        "condition_tail_baseline_margin_mean",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, members in sorted(grouped.items()):
        summary = {key: value for key, value in zip(keys, identity, strict=True)}
        summary["anchors"] = len(members)
        for name, sign in favorable_sign.items():
            observed = [float(row[name]) for row in members]
            summary[f"mean_{name}"] = statistics.fmean(observed)
            summary[f"median_{name}"] = statistics.median(observed)
            summary[f"favorable_anchor_fraction_{name}"] = statistics.fmean(
                value < 0 if sign == "negative" else value > 0 for value in observed
            )
        for name in descriptive:
            observed = [float(row[name]) for row in members]
            summary[f"mean_{name}"] = statistics.fmean(observed)
            summary[f"median_{name}"] = statistics.median(observed)
        output.append(summary)
    return output


def summarize_spectral_transplants(
    jobs: list[SpectralTransplantJob],
    output_dir: str | Path,
    *,
    spectral_spec: str | Path,
    common_state_spec: str | Path,
    families: tuple[str, ...] = ALL_FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    spec_path, spec = load_spectral_transplant_protocol(spectral_spec)
    common_state_spec = Path(common_state_spec).resolve()
    families, scope = resolve_scope(families, scope_amendment)
    expected = spec["anchor_scope"]["expected_anchors_per_family"] * len(families)
    if len(jobs) != expected:
        raise ValueError(f"Spectral summary requires {expected} anchors")
    if {job.common_state.family for job in jobs} != set(families):
        raise ValueError("Spectral jobs do not match the requested family scope")
    incomplete = [
        job.label
        for job in jobs
        if not spectral_transplant_job_complete(
            job,
            spec_path,
            common_state_spec,
            verify_hashes=True,
        )
    ]
    if incomplete:
        raise ValueError("Incomplete spectral-transplant jobs: " + ", ".join(incomplete))

    sources = []
    anchor_effects = []
    anchor_tail_effects = []
    for job in jobs:
        manifest_path = job.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sample_path = job.output_dir / manifest["outputs"]["sample_metrics"]["path"]
        records = _read_jsonl(sample_path)
        anchor_effects.extend(_anchor_effects(job.label, job.common_state.family, records, spec))
        anchor_tail_effects.extend(
            _anchor_tail_effects(job.label, job.common_state.family, records, spec)
        )
        sources.append(
            {
                "label": job.label,
                "manifest": {
                    "path": str(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": _sha256(manifest_path),
                },
                "sample_metrics": {
                    "path": str(sample_path),
                    "bytes": sample_path.stat().st_size,
                    "sha256": _sha256(sample_path),
                },
            }
        )
    expected_effects = expected * (spec["intervention"]["expected_conditions_per_anchor"] - 1)
    if len(anchor_effects) != expected_effects:
        raise AssertionError("Spectral anchor-effect cardinality changed")
    expected_tail_effects = expected * (spec["intervention"]["expected_conditions_per_anchor"] - 2)
    if len(anchor_tail_effects) != expected_tail_effects:
        raise AssertionError("Spectral anchor tail-effect cardinality changed")

    factorial = _factorial_effects(anchor_effects)
    path = _spectral_path_effects(anchor_effects)
    bands = _band_effects(anchor_effects)
    condition_summary = _group_summary(
        [
            {
                "family": row["family"],
                "condition": row["condition"],
                "metric": metric,
                "effect_vs_baseline": row[f"delta_{metric}"],
            }
            for row in anchor_effects
            for metric in METRICS
        ],
        keys=("family", "condition", "metric"),
        values=("effect_vs_baseline",),
    )
    factorial_summary = _group_summary(
        factorial,
        keys=("family", "metric"),
        values=("spectrum_main_effect", "basis_main_effect", "spectrum_basis_interaction"),
    )
    path_summary = _group_summary(
        path,
        keys=("family", "metric", "condition", "interpolation_lambda"),
        values=("effect_vs_baseline", "contrast_vs_adamw_native"),
    )
    band_summary = _group_summary(
        bands,
        keys=("family", "metric", "band", "condition"),
        values=("effect_vs_baseline", "contrast_vs_adamw_native"),
    )
    tail_summary = _tail_group_summary(anchor_tail_effects)

    tables = {
        "anchor_condition_effects": anchor_effects,
        "family_condition_summary": condition_summary,
        "anchor_factorial_effects": factorial,
        "family_factorial_summary": factorial_summary,
        "anchor_spectral_path": path,
        "family_spectral_path": path_summary,
        "anchor_band_effects": bands,
        "family_band_summary": band_summary,
        "anchor_query_tail_effects": anchor_tail_effects,
        "family_query_tail_summary": tail_summary,
    }
    outputs = {}
    for name, rows in tables.items():
        path_out = output_dir / f"{name}.csv"
        _write_csv(path_out, rows)
        outputs[name] = {
            "path": str(path_out),
            "bytes": path_out.stat().st_size,
            "sha256": _sha256(path_out),
            "rows": len(rows),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "complete": True,
        "analysis_status": spec["analysis_status"],
        "families": list(families),
        "scope_amendment": scope,
        "spectral_transplant_spec": {
            "path": str(spec_path),
            "bytes": spec_path.stat().st_size,
            "sha256": _sha256(spec_path),
        },
        "common_state_spec": {
            "path": str(common_state_spec),
            "bytes": common_state_spec.stat().st_size,
            "sha256": _sha256(common_state_spec),
        },
        "anchors": len(jobs),
        "anchor_effect_records": len(anchor_effects),
        "anchor_tail_effect_records": len(anchor_tail_effects),
        "tail_protocol": spec["evaluation"]["tail_protocol"],
        "sources": sources,
        "outputs": outputs,
        "claim_boundary": spec["claim_boundary"],
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the spectral-transplant intervention")
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--dense-reference-checkpoint", type=Path)
    parser.add_argument("--late-reference-checkpoint", type=Path)
    parser.add_argument("--common-state-root", type=Path, default=Path("results/common-state"))
    parser.add_argument("--result-root", type=Path, default=Path("results/spectral-transplant"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/spectral-transplant"))
    parser.add_argument(
        "--common-state-spec", type=Path, default=Path("configs/common_state_probe.json")
    )
    parser.add_argument(
        "--spectral-spec",
        type=Path,
        default=Path("configs/spectral_transplant_intervention.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    families, _ = resolve_scope(args.families, args.scope_amendment)
    spec_path, protocol = load_spectral_transplant_protocol(args.spectral_spec)
    common_path = resolve_common_state_spec(args.common_state_spec).resolve()
    if _sha256(common_path) != protocol["source_inputs"]["common_state_spec_sha256"]:
        raise ValueError("Common-state spec differs from the spectral-transplant lock")
    common_spec, _ = _load_protocol(common_path)
    configs = [
        config
        for config in load_matrix(resolve_matrix_path(args.matrix).resolve())
        if config.model_family in families
    ]
    by_family = {config.model_family: config for config in configs}
    references: dict[ModelFamily, Path] = {}
    for family, config in by_family.items():
        explicit = (
            args.dense_reference_checkpoint if family == "dense" else args.late_reference_checkpoint
        )
        references[family] = _resolve_reference(config, explicit)
    common_jobs = build_common_state_jobs(
        configs,
        references,
        common_spec,
        args.common_state_root,
    )
    jobs = build_spectral_transplant_jobs(common_jobs, args.result_root)
    manifest = summarize_spectral_transplants(
        jobs,
        args.output_dir,
        spectral_spec=spec_path,
        common_state_spec=common_path,
        families=families,
        scope_amendment=args.scope_amendment,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
