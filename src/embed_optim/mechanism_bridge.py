from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .aggregate import audit_experiment_contract
from .config import RunConfig, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .geometry_summary import _atomic_csv
from .probe_matrix import _declared_checkpoint_steps
from .representation_plot import load_representation_summary

WEIGHT_FIELDS = [
    "model_family",
    "optimizer",
    "learning_rate",
    "run_id",
    "stage",
    "step",
    "hidden_tensors",
    "hidden_parameters",
    "weight_frobenius_norm",
    "reference_displacement_frobenius_norm",
    "previous_checkpoint_displacement_frobenius_norm",
    "reference_displacement_to_weight_ratio",
    "weight_row_cv_parameter_weighted",
    "reference_delta_row_cv_parameter_weighted",
    "weight_top_1pct_row_energy_parameter_weighted",
    "reference_delta_top_1pct_row_energy_parameter_weighted",
]
EVALUATION_FIELDS = [
    "model_family",
    "optimizer",
    "learning_rate",
    "run_id",
    "stage",
    "fraction",
    "checkpoint_step",
    "mean_ndcg_at_10",
    "tasks_completed",
]
TRAINING_FIELDS = [
    "model_family",
    "optimizer",
    "learning_rate",
    "run_id",
    "stage",
    "fraction",
    "checkpoint_step",
    "observed_step",
    "window_start_step",
    "window_observations",
    "mean_loss",
    "loss_standard_deviation",
    "median_grad_norm",
    "end_learning_rate",
    "end_epoch",
]
WEIGHT_METRICS = WEIGHT_FIELDS[8:]
SCORE_METRICS = [
    "margin_mean",
    "margin_median",
    "top1_accuracy",
    "mean_reciprocal_rank",
    "reference_mean_top_k_overlap",
    "reference_top1_agreement",
    "reference_score_drift_rms",
]
TOKEN_METRICS = [
    "token_evidence_entropy_mean",
    "token_evidence_gini_mean",
    "document_token_coverage_mean",
    "repeated_token_dominance_mean",
]
TIERS = ("training", "unseen")
BRIDGE_IDENTITY_FIELDS = [
    "model_family",
    "optimizer",
    "learning_rate",
    "run_id",
    "stage",
    "fraction",
    "step",
]
BRIDGE_METRIC_FIELDS = [
    "mean_training_loss",
    *WEIGHT_METRICS,
    *[
        f"{tier}_{metric}"
        for tier in TIERS
        for metric in [
            *SCORE_METRICS,
            "query_normalized_effective_rank",
            "document_normalized_effective_rank",
            *TOKEN_METRICS,
        ]
    ],
    "mean_beir_ndcg_at_10",
]
BRIDGE_FIELDS = [*BRIDGE_IDENTITY_FIELDS, *BRIDGE_METRIC_FIELDS]


def _expected_checkpoints(configs: list[RunConfig]) -> dict[tuple[str, str, int], dict[str, Any]]:
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    for config in configs:
        steps = _declared_checkpoint_steps(config)
        if len(steps) != 5:
            raise ValueError(
                f"Expected five checkpoint steps for {config.model_family}/{config.run_id}"
            )
        for stage, (fraction, step) in enumerate(
            zip(config.checkpoint_fractions, steps, strict=True), start=1
        ):
            identity = (config.model_family, config.run_id, stage)
            output[identity] = {
                "model_family": config.model_family,
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "run_id": config.run_id,
                "stage": stage,
                "fraction": float(fraction),
                "step": step,
            }
    if len(output) != 120:
        raise ValueError(f"Expected 120 checkpoint identities, found {len(output)}")
    return output


def _read_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"Unexpected CSV schema: {path}")
        return list(reader)


def _finite(value: Any, label: str, *, allow_empty: bool = False) -> float | None:
    if allow_empty and value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {label}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite numeric value for {label}")
    return parsed


def _row_identity(
    row: dict[str, str], *, family_field: str = "model_family"
) -> tuple[str, str, int]:
    try:
        return row[family_field], row["run_id"], int(row["stage"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid checkpoint identity: {row}") from error


def _validate_metadata(
    row: dict[str, str],
    expected: dict[str, Any],
    *,
    family_field: str = "model_family",
    step_field: str = "step",
) -> None:
    if (
        row[family_field] != expected["model_family"]
        or row["optimizer"] != expected["optimizer"]
        or row["run_id"] != expected["run_id"]
        or int(row["stage"]) != expected["stage"]
        or int(row[step_field]) != expected["step"]
        or not math.isclose(
            float(row["learning_rate"]), expected["learning_rate"], rel_tol=0, abs_tol=1e-15
        )
    ):
        raise ValueError(f"Checkpoint metadata differs across mechanism sources: {row}")
    if "fraction" in row and not math.isclose(
        float(row["fraction"]), expected["fraction"], rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"Checkpoint fraction differs across mechanism sources: {row}")


def _load_weight_rows(
    summary_dir: Path,
    expected: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[dict[tuple[str, str, int], dict[str, str]], dict[str, Any]]:
    summary_dir = summary_dir.resolve()
    manifest_path = summary_dir / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("outputs", {}).get("checkpoint_trajectory.csv")
    path = summary_dir / "checkpoint_trajectory.csv"
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("verify_inputs") is not True
        or manifest.get("expected_runs") != 24
        or manifest.get("observed_runs") != 24
        or manifest.get("checkpoint_rows") != 120
        or not isinstance(declared, dict)
        or declared.get("rows") != 120
        or not path.is_file()
        or path.stat().st_size != declared.get("bytes")
        or _sha256(path) != declared.get("sha256")
    ):
        raise ValueError("Weight-space summary is incomplete or differs from its strict manifest")
    rows = _read_csv(path, WEIGHT_FIELDS)
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        identity = _row_identity(row)
        if identity in indexed or identity not in expected:
            raise ValueError("Weight-space checkpoint identities are duplicated or unexpected")
        _validate_metadata(row, expected[identity])
        if int(row["hidden_tensors"]) != 88 or int(row["hidden_parameters"]) != 110_297_088:
            raise ValueError("Weight-space optimizer partition differs from the frozen contract")
        for field in WEIGHT_METRICS:
            _finite(
                row[field],
                f"weight/{identity}/{field}",
                allow_empty=field == "previous_checkpoint_displacement_frobenius_norm"
                and identity[2] == 1,
            )
        indexed[identity] = row
    if set(indexed) != set(expected):
        raise ValueError("Weight-space summary does not cover all 120 checkpoints")
    provenance = {
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "table": {"path": str(path), "sha256": _sha256(path)},
    }
    return indexed, provenance


def _load_evaluation_rows(
    reports_dir: Path,
    expected: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[dict[tuple[str, str, int], dict[str, str]], dict[str, Any]]:
    reports_dir = reports_dir.resolve()
    coverage_path = reports_dir / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if (
        coverage.get("complete") is not True
        or coverage.get("evaluation_complete") is not True
        or coverage.get("expected_results") != 1680
        or coverage.get("observed_results") != 1680
        or coverage.get("expected_checkpoint_summaries") != 120
        or coverage.get("observed_checkpoint_summaries") != 120
        or coverage.get("missing") != []
        or coverage.get("unexpected") != []
    ):
        raise ValueError("BEIR evaluation coverage is not strictly complete")
    path = reports_dir / "checkpoint_summary.csv"
    rows = _read_csv(path, EVALUATION_FIELDS)
    if len(rows) != 120:
        raise ValueError("Evaluation checkpoint summary does not contain 120 rows")
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        identity = _row_identity(row)
        if identity in indexed or identity not in expected:
            raise ValueError("Evaluation checkpoint identities are duplicated or unexpected")
        _validate_metadata(row, expected[identity], step_field="checkpoint_step")
        score = _finite(row["mean_ndcg_at_10"], f"evaluation/{identity}/mean_ndcg_at_10")
        if score is None or not 0 <= score <= 1 or int(row["tasks_completed"]) != 14:
            raise ValueError("Evaluation checkpoint row violates the 14-task score contract")
        indexed[identity] = row
    if set(indexed) != set(expected):
        raise ValueError("Evaluation summary does not cover all 120 checkpoints")
    provenance = {
        "coverage": {"path": str(coverage_path), "sha256": _sha256(coverage_path)},
        "table": {"path": str(path), "sha256": _sha256(path)},
    }
    return indexed, provenance


def _declared_output_path(summary_dir: Path, declared: dict[str, Any]) -> Path:
    raw = declared.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError("Training-dynamics manifest has no declared stage table path")
    path = Path(raw)
    if not path.is_absolute():
        path = summary_dir.parents[1] / path
    return path.resolve()


def _load_training_rows(
    summary_dir: Path,
    expected: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[dict[tuple[str, str, int], dict[str, str]], dict[str, Any]]:
    summary_dir = summary_dir.resolve()
    manifest_path = summary_dir / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = manifest.get("coverage", {})
    declared = manifest.get("outputs", {}).get("stages")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or coverage.get("runs") != 24
        or coverage.get("checkpoints") != 120
        or coverage.get("history_rows") != 9_384
        or coverage.get("trailing_observations") != 10
        or not isinstance(declared, dict)
        or declared.get("rows") != 120
    ):
        raise ValueError("Training-dynamics summary is incomplete")
    path = _declared_output_path(summary_dir, declared)
    if (
        path != summary_dir / "stage_dynamics.csv"
        or not path.is_file()
        or path.stat().st_size != declared.get("bytes")
        or _sha256(path) != declared.get("sha256")
    ):
        raise ValueError("Training-dynamics stage table differs from its strict manifest")
    rows = _read_csv(path, TRAINING_FIELDS)
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        identity = _row_identity(row)
        if identity in indexed or identity not in expected:
            raise ValueError("Training-dynamics checkpoint identities are duplicated or unexpected")
        _validate_metadata(row, expected[identity], step_field="checkpoint_step")
        mean_loss = _finite(row["mean_loss"], f"training/{identity}/mean_loss")
        loss_std = _finite(
            row["loss_standard_deviation"], f"training/{identity}/loss_standard_deviation"
        )
        median_grad = _finite(row["median_grad_norm"], f"training/{identity}/median_grad_norm")
        end_lr = _finite(row["end_learning_rate"], f"training/{identity}/end_learning_rate")
        end_epoch = _finite(row["end_epoch"], f"training/{identity}/end_epoch")
        try:
            checkpoint_step = int(row["checkpoint_step"])
            observed_step = int(row["observed_step"])
            window_start = int(row["window_start_step"])
            observations = int(row["window_observations"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid training window metadata: {row}") from error
        if (
            mean_loss is None
            or mean_loss < 0
            or loss_std is None
            or loss_std < 0
            or median_grad is None
            or median_grad < 0
            or end_lr is None
            or end_lr < 0
            or end_epoch is None
            or end_epoch < 0
            or observations != 10
            or not window_start <= observed_step <= checkpoint_step
        ):
            raise ValueError(f"Training-dynamics row violates the frozen window contract: {row}")
        indexed[identity] = row
    if set(indexed) != set(expected):
        raise ValueError("Training-dynamics summary does not cover all 120 checkpoints")
    provenance = {
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "table": {"path": str(path), "sha256": _sha256(path)},
    }
    return indexed, provenance


def _load_loss_retrieval_protocol(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    freeze = payload.get("freeze_context", {})
    analysis = payload.get("analysis", {})
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "posthoc_exploratory_diagnostic"
        or freeze.get("strict_beir_valid_units") != 1_456
        or freeze.get("strict_beir_expected_units") != 1_680
        or freeze.get("complete_retrieval_matrix_visible") is not False
        or freeze.get("training_dynamics_visible") is not True
        or freeze.get("dense_retrieval_matrix_visible") is not True
        or freeze.get("late_adamw_retrieval_matrix_visible") is not True
        or freeze.get("late_muon_retrieval_matrix_visible") is not True
        or freeze.get("partial_late_normuon_results_visible") is not True
        or not isinstance(freeze.get("strict_progress_snapshot_sha256"), str)
        or len(freeze["strict_progress_snapshot_sha256"]) != 64
        or not isinstance(analysis.get("views"), list)
        or len(analysis["views"]) != 2
        or "post-hoc" not in str(payload.get("claim_boundary", ""))
    ):
        raise ValueError("Loss-to-retrieval diagnostic does not disclose its post-hoc context")
    return {"path": str(path), "sha256": _sha256(path), "status": payload["status"]}


def _representation_lookups(
    checkpoint_rows: list[dict[str, str]],
    representation_rows: list[dict[str, str]],
    expected: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, int], dict[str, str]],
    dict[tuple[tuple[str, str, int], str], dict[str, str]],
]:
    checkpoints: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in checkpoint_rows:
        if row["kind"] == "reference":
            continue
        identity = _row_identity(row, family_field="family")
        if identity in checkpoints or identity not in expected:
            raise ValueError("Representation checkpoint identities are duplicated or unexpected")
        _validate_metadata(row, expected[identity], family_field="family")
        checkpoints[identity] = row
    representations: dict[tuple[tuple[str, str, int], str], dict[str, str]] = {}
    for row in representation_rows:
        if row["kind"] == "reference":
            continue
        identity = _row_identity(row, family_field="family")
        key = (identity, row["representation_role"])
        if key in representations or identity not in expected:
            raise ValueError("Representation-role identities are duplicated or unexpected")
        _validate_metadata(row, expected[identity], family_field="family")
        representations[key] = row
    if set(checkpoints) != set(expected):
        raise ValueError("Representation summary does not cover all 120 checkpoints")
    return checkpoints, representations


def _representation_values(
    identity: tuple[str, str, int],
    checkpoint: dict[str, str],
    representations: dict[tuple[tuple[str, str, int], str], dict[str, str]],
) -> dict[str, float | str]:
    family = identity[0]
    query_role = "queries" if family == "dense" else "query_tokens"
    document_role = "documents" if family == "dense" else "document_tokens"
    values: dict[str, float | str] = {metric: float(checkpoint[metric]) for metric in SCORE_METRICS}
    values["query_normalized_effective_rank"] = float(
        representations[(identity, query_role)]["normalized_effective_rank"]
    )
    values["document_normalized_effective_rank"] = float(
        representations[(identity, document_role)]["normalized_effective_rank"]
    )
    values.update(
        {metric: float(checkpoint[metric]) if family == "late" else "" for metric in TOKEN_METRICS}
    )
    return values


def _first_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run = {(row["model_family"], row["run_id"], row["stage"]): row for row in rows}
    output = []
    for row in rows:
        if row["stage"] == 1:
            continue
        previous = by_run[(row["model_family"], row["run_id"], row["stage"] - 1)]
        result = {field: row[field] for field in BRIDGE_IDENTITY_FIELDS}
        result["previous_stage"] = row["stage"] - 1
        result["previous_fraction"] = previous["fraction"]
        result["previous_step"] = previous["step"]
        for field in BRIDGE_METRIC_FIELDS:
            result[f"delta_{field}"] = (
                ""
                if row[field] == "" or previous[field] == ""
                else float(row[field]) - float(previous[field])
            )
        output.append(result)
    if len(output) != 96:
        raise ValueError(f"Expected 96 within-run transitions, found {len(output)}")
    return output


def _rankdata(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for index in ordered[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left_ranks)
    right_scale = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def _correlation_pairs(family: str) -> list[tuple[str, str, str]]:
    weight_metrics = [
        "reference_displacement_to_weight_ratio",
        "reference_delta_row_cv_parameter_weighted",
        "reference_delta_top_1pct_row_energy_parameter_weighted",
    ]
    pairs = [
        (metric, outcome, "weight-to-representation")
        for metric in weight_metrics
        for outcome in ("unseen_margin_mean", "unseen_query_normalized_effective_rank")
    ]
    pairs.extend(
        [
            ("mean_training_loss", "mean_beir_ndcg_at_10", "objective-to-retrieval"),
            ("training_margin_mean", "mean_beir_ndcg_at_10", "representation-to-retrieval"),
            (
                "training_query_normalized_effective_rank",
                "mean_beir_ndcg_at_10",
                "representation-to-retrieval",
            ),
            ("unseen_margin_mean", "mean_beir_ndcg_at_10", "representation-to-retrieval"),
            (
                "unseen_query_normalized_effective_rank",
                "mean_beir_ndcg_at_10",
                "representation-to-retrieval",
            ),
            (
                "unseen_reference_score_drift_rms",
                "mean_beir_ndcg_at_10",
                "representation-to-retrieval",
            ),
        ]
    )
    if family == "late":
        pairs.extend(
            [
                (
                    "unseen_token_evidence_entropy_mean",
                    "mean_beir_ndcg_at_10",
                    "token-utilization-to-retrieval",
                ),
                (
                    "unseen_document_token_coverage_mean",
                    "mean_beir_ndcg_at_10",
                    "token-utilization-to-retrieval",
                ),
                (
                    "unseen_repeated_token_dominance_mean",
                    "mean_beir_ndcg_at_10",
                    "token-utilization-to-retrieval",
                ),
            ]
        )
    return pairs


def _correlations(
    checkpoint_rows: list[dict[str, Any]],
    change_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for family in ("dense", "late"):
        scopes = [
            ("all_optimizers", None),
            *(("optimizer", value) for value in ("adamw", "muon", "normuon")),
        ]
        for scope, optimizer in scopes:
            for analysis, rows, prefix in (
                ("checkpoint_levels", checkpoint_rows, ""),
                ("within_run_first_differences", change_rows, "delta_"),
            ):
                selected = [
                    row
                    for row in rows
                    if row["model_family"] == family
                    and (optimizer is None or row["optimizer"] == optimizer)
                ]
                for predictor, outcome, bridge in _correlation_pairs(family):
                    left = [float(row[f"{prefix}{predictor}"]) for row in selected]
                    right = [float(row[f"{prefix}{outcome}"]) for row in selected]
                    rho = _spearman(left, right)
                    output.append(
                        {
                            "model_family": family,
                            "scope": scope,
                            "optimizer": optimizer or "all",
                            "analysis": analysis,
                            "bridge": bridge,
                            "predictor": predictor,
                            "outcome": outcome,
                            "observations": len(selected),
                            "spearman_rho": "" if rho is None else rho,
                        }
                    )
    return output


def build_mechanism_bridge(
    matrix_path: Path,
    weight_summary: Path,
    training_summary: Path,
    unseen_summary: Path,
    evaluation_reports: Path,
    output_dir: Path,
    *,
    training_dynamics: Path = Path("reports/training-dynamics"),
    loss_retrieval_protocol: Path = Path("configs/loss_retrieval_diagnostic.json"),
) -> dict[str, Any]:
    matrix_path = resolve_matrix_path(matrix_path).resolve()
    configs = load_matrix(matrix_path)
    audit = audit_experiment_contract(configs)
    if audit["complete"] is not True:
        raise ValueError("Experiment matrix differs from the frozen 24-run contract")
    expected = _expected_checkpoints(configs)
    weights, weight_provenance = _load_weight_rows(weight_summary, expected)
    training, training_provenance = _load_training_rows(training_dynamics, expected)
    evaluations, evaluation_provenance = _load_evaluation_rows(evaluation_reports, expected)
    loss_retrieval_provenance = _load_loss_retrieval_protocol(loss_retrieval_protocol)
    representation_sources = {}
    representation_values = {}
    for tier, summary in (("training", training_summary), ("unseen", unseen_summary)):
        checkpoint_rows, representation_rows, provenance = load_representation_summary(
            summary, configs
        )
        checkpoints, representations = _representation_lookups(
            checkpoint_rows, representation_rows, expected
        )
        representation_sources[tier] = provenance
        representation_values[tier] = (checkpoints, representations)
    if (
        representation_sources["training"]["probe"]["spec_sha256"]
        == representation_sources["unseen"]["probe"]["spec_sha256"]
    ):
        raise ValueError("Training and unseen mechanism tiers use the same probe specification")

    rows = []
    for identity, metadata in sorted(expected.items()):
        row: dict[str, Any] = dict(metadata)
        row["mean_training_loss"] = float(training[identity]["mean_loss"])
        row.update({field: weights[identity][field] for field in WEIGHT_METRICS})
        for tier in TIERS:
            checkpoints, representations = representation_values[tier]
            values = _representation_values(identity, checkpoints[identity], representations)
            row.update({f"{tier}_{field}": value for field, value in values.items()})
        row["mean_beir_ndcg_at_10"] = float(evaluations[identity]["mean_ndcg_at_10"])
        rows.append(row)
    if len(rows) != 120:
        raise ValueError("Mechanism bridge does not cover all 120 checkpoints")
    changes = _first_differences(rows)
    correlations = _correlations(rows, changes)

    output_dir = output_dir.resolve()
    outputs = {
        "checkpoint_bridge": output_dir / "checkpoint_bridge.csv",
        "within_run_changes": output_dir / "within_run_changes.csv",
        "descriptive_correlations": output_dir / "descriptive_correlations.csv",
    }
    _atomic_csv(outputs["checkpoint_bridge"], rows, BRIDGE_FIELDS)
    change_fields = [
        *BRIDGE_IDENTITY_FIELDS,
        "previous_stage",
        "previous_fraction",
        "previous_step",
        *[f"delta_{field}" for field in BRIDGE_METRIC_FIELDS],
    ]
    _atomic_csv(outputs["within_run_changes"], changes, change_fields)
    correlation_fields = [
        "model_family",
        "scope",
        "optimizer",
        "analysis",
        "bridge",
        "predictor",
        "outcome",
        "observations",
        "spearman_rho",
    ]
    _atomic_csv(outputs["descriptive_correlations"], correlations, correlation_fields)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
        "sources": {
            "weight_space": weight_provenance,
            "training_dynamics": training_provenance,
            "representation": representation_sources,
            "evaluation": evaluation_provenance,
            "loss_retrieval_protocol": loss_retrieval_provenance,
        },
        "checkpoints": len(rows),
        "within_run_transitions": len(changes),
        "correlations": len(correlations),
        "outputs": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(
                    rows
                    if name == "checkpoint_bridge"
                    else changes
                    if name == "within_run_changes"
                    else correlations
                ),
            }
            for name, path in outputs.items()
        },
        "interpretation": (
            "Spearman correlations are descriptive one-seed checkpoint associations. The "
            "within-run table removes run-level offsets but remains observational; causal claims "
            "require the common-state and short-branch interventions. The training-loss bridge is "
            "an explicitly post-hoc diagnostic added after 1,456/1,680 discovery units were visible."
        ),
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly join weight, representation, and BEIR checkpoint geometry"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--weight-summary", type=Path, default=Path("reports/weight-space"))
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("results/representation-space/training/summary"),
    )
    parser.add_argument(
        "--unseen-summary",
        type=Path,
        default=Path("results/representation-space/decontaminated-beir/summary"),
    )
    parser.add_argument("--evaluation-reports", type=Path, default=Path("reports"))
    parser.add_argument("--training-dynamics", type=Path, default=Path("reports/training-dynamics"))
    parser.add_argument(
        "--loss-retrieval-protocol",
        type=Path,
        default=Path("configs/loss_retrieval_diagnostic.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mechanism-bridge"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            build_mechanism_bridge(
                args.matrix,
                args.weight_summary,
                args.training_summary,
                args.unseen_summary,
                args.evaluation_reports,
                args.output_dir,
                training_dynamics=args.training_dynamics,
                loss_retrieval_protocol=args.loss_retrieval_protocol,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
