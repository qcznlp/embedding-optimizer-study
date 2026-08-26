from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RunConfig, load_matrix, resolve_matrix_path
from .geometry import _atomic_json, _sha256
from .geometry_summary import _atomic_csv
from .probe_matrix import (
    ProbeJob,
    _declared_checkpoint_steps,
    _requested_probe_identity,
    probe_job_complete,
)
from .probes import resolve_probe_spec_path

SCHEMA_VERSION = 1
REPRESENTATION_FIELDS = [
    "original_vectors",
    "analyzed_vectors",
    "sampled",
    "dimension",
    "mean_norm",
    "norm_cv",
    "mean_pairwise_cosine",
    "covariance_trace",
    "entropy_effective_rank",
    "normalized_effective_rank",
    "stable_rank",
    "leading_variance_fraction",
]
SCORE_FIELDS = [
    "samples",
    "candidates_per_sample",
    "positive_score_mean",
    "hardest_negative_score_mean",
    "margin_mean",
    "margin_median",
    "margin_std",
    "top1_accuracy",
    "mean_reciprocal_rank",
    "mean_candidate_score_std",
    "reference_top_k",
    "reference_mean_top_k_overlap",
    "reference_top1_agreement",
    "reference_score_drift_rms",
]
TOKEN_METRICS = {
    "positive_query_token_evidence_normalized_entropy": "token_evidence_entropy",
    "positive_query_token_evidence_gini": "token_evidence_gini",
    "positive_document_token_coverage": "document_token_coverage",
    "positive_document_token_repeated_selection_dominance": "repeated_token_dominance",
}
IDENTITY_FIELDS = [
    "family",
    "kind",
    "optimizer",
    "learning_rate",
    "run_id",
    "stage",
    "fraction",
    "step",
    "label",
    "scorer",
]


@dataclass(frozen=True)
class ExpectedMetric:
    job: ProbeJob
    optimizer: str
    learning_rate: float | str
    run_id: str
    stage: int
    fraction: float
    step: int


def expected_probe_metrics(
    configs: list[RunConfig],
    result_root: Path,
    probe_identity: tuple[str, str],
) -> list[ExpectedMetric]:
    result_root = result_root.resolve()
    probe_manifest_sha256, probe_spec_sha256 = probe_identity
    expected: list[ExpectedMetric] = []
    for family in sorted({config.model_family for config in configs}):
        expected.append(
            ExpectedMetric(
                job=ProbeJob(
                    kind="reference",
                    family=family,
                    label=f"{family}/pretrained",
                    checkpoint=Path("."),
                    export=result_root / "exports" / family / "pretrained.npz",
                    metrics=result_root / "metrics" / family / "pretrained.json",
                    reference_export=None,
                    probe_manifest_sha256=probe_manifest_sha256,
                    probe_spec_sha256=probe_spec_sha256,
                ),
                optimizer="",
                learning_rate="",
                run_id="pretrained",
                stage=0,
                fraction=0.0,
                step=0,
            )
        )
    for config in sorted(configs, key=lambda item: (item.model_family, item.run_id)):
        reference = result_root / "exports" / config.model_family / "pretrained.npz"
        steps = _declared_checkpoint_steps(config)
        if len(steps) != len(config.checkpoint_fractions):
            raise ValueError(
                f"Checkpoint fraction mismatch for {config.model_family}/{config.run_id}"
            )
        for stage, (step, fraction) in enumerate(
            zip(steps, config.checkpoint_fractions, strict=True), start=1
        ):
            relative = Path(config.model_family) / config.run_id / f"checkpoint-{step}"
            expected.append(
                ExpectedMetric(
                    job=ProbeJob(
                        kind="checkpoint",
                        family=config.model_family,
                        label=f"{config.model_family}/{config.run_id}/checkpoint-{step}",
                        checkpoint=config.output_dir / f"checkpoint-{step}",
                        export=result_root / "exports" / relative.with_suffix(".npz"),
                        metrics=result_root / "metrics" / relative.with_suffix(".json"),
                        reference_export=reference,
                        probe_manifest_sha256=probe_manifest_sha256,
                        probe_spec_sha256=probe_spec_sha256,
                    ),
                    optimizer=config.optimizer.name,
                    learning_rate=config.optimizer.lr,
                    run_id=config.run_id,
                    stage=stage,
                    fraction=float(fraction),
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


def _identity_row(expected: ExpectedMetric, scorer: str) -> dict[str, Any]:
    return {
        "family": expected.job.family,
        "kind": expected.job.kind,
        "optimizer": expected.optimizer,
        "learning_rate": expected.learning_rate,
        "run_id": expected.run_id,
        "stage": expected.stage,
        "fraction": expected.fraction,
        "step": expected.step,
        "label": expected.job.label,
        "scorer": scorer,
    }


def _score_row(score: dict[str, Any]) -> dict[str, Any]:
    reference = score.get("reference_ranking") or {}
    return {
        "samples": score["samples"],
        "candidates_per_sample": score["candidates_per_sample"],
        "positive_score_mean": score["positive_score"]["mean"],
        "hardest_negative_score_mean": score["hardest_negative_score"]["mean"],
        "margin_mean": score["positive_hardest_negative_margin"]["mean"],
        "margin_median": score["positive_hardest_negative_margin"]["median"],
        "margin_std": score["positive_hardest_negative_margin"]["std"],
        "top1_accuracy": score["top1_accuracy"],
        "mean_reciprocal_rank": score["mean_reciprocal_rank"],
        "mean_candidate_score_std": score["mean_candidate_score_std"],
        "reference_top_k": reference.get("top_k", ""),
        "reference_mean_top_k_overlap": reference.get("mean_top_k_overlap", ""),
        "reference_top1_agreement": reference.get("top1_agreement", ""),
        "reference_score_drift_rms": reference.get("score_drift_rms", ""),
    }


def _token_row(metrics: dict[str, Any]) -> dict[str, Any]:
    token = metrics.get("token_utilization") or {}
    row: dict[str, Any] = {}
    for source, prefix in TOKEN_METRICS.items():
        summary = token.get(source) or {}
        row[f"{prefix}_mean"] = summary.get("mean", "")
        row[f"{prefix}_median"] = summary.get("median", "")
    return row


def _validate_payload(
    payload: dict[str, Any],
    expected: ExpectedMetric,
    *,
    expected_samples: int,
    expected_groups: set[str],
) -> None:
    metrics = payload.get("metrics")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("family") != expected.job.family
        or payload.get("label") != expected.job.label
        or not isinstance(metrics, dict)
        or not _all_finite(payload)
    ):
        raise ValueError(f"Invalid representation metric identity: {expected.job.metrics}")
    parameters = payload.get("parameters") or {}
    if (
        parameters.get("require_export_manifest") is not True
        or parameters.get("positive_candidate_index") != 0
    ):
        raise ValueError(f"Noncanonical representation parameters: {expected.job.metrics}")
    expected_scorer = "cosine" if expected.job.family == "dense" else "mean_maxsim_cosine"
    score = metrics.get("score_geometry") or {}
    if (
        metrics.get("scorer") != expected_scorer
        or score.get("samples") != expected_samples
        or score.get("candidates_per_sample") != 8
        or set((score.get("by_group") or {})) != expected_groups
    ):
        raise ValueError(f"Representation score contract changed: {expected.job.metrics}")
    reference = score.get("reference_ranking")
    if (expected.job.kind == "reference") != (reference is None):
        raise ValueError(f"Unexpected reference-ranking coverage: {expected.job.metrics}")
    expected_roles = (
        {"queries", "documents"}
        if expected.job.family == "dense"
        else {"query_tokens", "document_tokens", "pooled_queries", "pooled_documents"}
    )
    representations = metrics.get("representations") or {}
    if set(representations) != expected_roles or any(
        set(values) != set(REPRESENTATION_FIELDS) for values in representations.values()
    ):
        raise ValueError(f"Representation roles changed: {expected.job.metrics}")
    if expected.job.family == "late" and set(metrics.get("token_utilization") or {}) != set(
        TOKEN_METRICS
    ):
        raise ValueError(f"Late token-utilization contract changed: {expected.job.metrics}")


def summarize_probe_metrics(
    expected: list[ExpectedMetric],
    output_dir: Path,
    *,
    probe_manifest_path: Path,
    probe_spec_path: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
    expected_samples = int(probe_manifest["count"])
    group_counts = probe_manifest.get("task_counts") or probe_manifest.get("quotas")
    if (
        not isinstance(group_counts, dict)
        or sum(int(value) for value in group_counts.values()) != expected_samples
    ):
        raise ValueError("Probe manifest lacks complete group counts")
    expected_groups = set(group_counts)
    manifest_sha256 = _sha256(probe_manifest_path)
    spec_sha256 = _sha256(probe_spec_path)
    if any(
        item.job.probe_manifest_sha256 not in (None, manifest_sha256)
        or item.job.probe_spec_sha256 not in (None, spec_sha256)
        for item in expected
    ):
        raise ValueError("Expected jobs are bound to a different probe or specification")

    expected_metric_paths = {item.job.metrics.resolve() for item in expected}
    if expected:
        metrics_root = expected[0].job.metrics.parents[1]
        unexpected = {path.resolve() for path in metrics_root.rglob("*.json")}.difference(
            expected_metric_paths
        )
        if unexpected:
            raise ValueError(f"Unexpected representation metric files: {sorted(unexpected)[:5]}")

    main_rows = []
    representation_rows = []
    group_rows = []
    inputs = []
    missing = []
    for item in expected:
        if not probe_job_complete(item.job):
            missing.append(item.job.label)
            continue
        payload = json.loads(item.job.metrics.read_text(encoding="utf-8"))
        _validate_payload(
            payload,
            item,
            expected_samples=expected_samples,
            expected_groups=expected_groups,
        )
        metrics = payload["metrics"]
        for group, count in group_counts.items():
            group_score = metrics["score_geometry"]["by_group"][group]
            if (
                group_score.get("samples") != int(count)
                or group_score.get("candidates_per_sample") != 8
                or (item.job.kind == "reference") != (group_score.get("reference_ranking") is None)
            ):
                raise ValueError(f"Group score contract changed: {item.job.metrics}/{group}")
        identity = _identity_row(item, metrics["scorer"])
        main_rows.append(
            {**identity, **_score_row(metrics["score_geometry"]), **_token_row(metrics)}
        )
        for role, values in sorted(metrics["representations"].items()):
            representation_rows.append(
                {
                    **identity,
                    "representation_role": role,
                    **{key: values[key] for key in REPRESENTATION_FIELDS},
                }
            )
        for group, values in sorted(metrics["score_geometry"]["by_group"].items()):
            group_rows.append({**identity, "group": group, **_score_row(values)})
        inputs.append(
            {
                "label": item.job.label,
                "metrics_path": str(item.job.metrics),
                "metrics_sha256": _sha256(item.job.metrics),
                "export_path": str(item.job.export),
                "export_sha256": _sha256(item.job.export),
            }
        )
    if missing and not allow_partial:
        raise ValueError(
            f"Representation matrix is incomplete: {len(missing)}/{len(expected)} missing or invalid; "
            f"first={missing[:5]}"
        )
    if not main_rows:
        raise ValueError("No valid representation metrics were found")

    token_fields = [
        f"{prefix}_{stat}" for prefix in TOKEN_METRICS.values() for stat in ("mean", "median")
    ]
    main_fields = [*IDENTITY_FIELDS, *SCORE_FIELDS, *token_fields]
    representation_fields = [*IDENTITY_FIELDS, "representation_role", *REPRESENTATION_FIELDS]
    group_fields = [*IDENTITY_FIELDS, "group", *SCORE_FIELDS]
    outputs = {
        "checkpoint_metrics": output_dir / "checkpoint_metrics.csv",
        "representation_metrics": output_dir / "representation_metrics.csv",
        "group_metrics": output_dir / "group_metrics.csv",
    }
    _atomic_csv(outputs["checkpoint_metrics"], main_rows, main_fields)
    _atomic_csv(outputs["representation_metrics"], representation_rows, representation_fields)
    _atomic_csv(outputs["group_metrics"], group_rows, group_fields)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": not missing and len(main_rows) == len(expected),
        "allow_partial": allow_partial,
        "expected_jobs": len(expected),
        "valid_jobs": len(main_rows),
        "missing_labels": missing,
        "probe": {
            "manifest_path": str(probe_manifest_path.resolve()),
            "manifest_sha256": _sha256(probe_manifest_path),
            "spec_path": str(probe_spec_path.resolve()),
            "spec_sha256": _sha256(probe_spec_path),
            "samples": expected_samples,
            "groups": dict(sorted(group_counts.items())),
        },
        "inputs": inputs,
        "outputs": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(
                    main_rows
                    if name == "checkpoint_metrics"
                    else representation_rows
                    if name == "representation_metrics"
                    else group_rows
                ),
            }
            for name, path in outputs.items()
        },
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly aggregate a fixed representation-probe matrix"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    configs = load_matrix(matrix_path)
    resolved_probe = args.probe.resolve()
    resolved_probe_spec = resolve_probe_spec_path(args.probe_spec).resolve()
    identity = _requested_probe_identity(resolved_probe, resolved_probe_spec)
    expected = expected_probe_metrics(configs, args.result_root, identity)
    output_dir = args.output_dir or (args.result_root / "summary")
    manifest = summarize_probe_metrics(
        expected,
        output_dir,
        probe_manifest_path=resolved_probe / "manifest.json",
        probe_spec_path=resolved_probe_spec,
        allow_partial=args.allow_partial,
    )
    print(
        f"Aggregated {manifest['valid_jobs']}/{manifest['expected_jobs']} representation jobs "
        f"into {Path(output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
