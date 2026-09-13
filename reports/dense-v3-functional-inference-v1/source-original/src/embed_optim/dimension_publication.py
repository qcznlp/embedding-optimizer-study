"""Infer and render the prospective DenseOn dimension-utilization results."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_matrix
from .dimension_utilization import _sha256, audit_outputs

SCHEMA_VERSION = 1
OPTIMIZERS = ("adamw", "muon", "normuon")
OPTIMIZER_LABELS = {
    "pretrained": "Pretrained",
    "adamw": "AdamW",
    "muon": "Muon",
    "normuon": "NorMuon",
}
CONTRASTS = (("muon", "adamw"), ("normuon", "adamw"), ("normuon", "muon"))
PRIMARY_FEATURES = (
    "margin_helpful_mass_share",
    "margin_helpful_attribution_participation_ratio",
    "margin_degrading_attribution_mass",
)
BRIDGE_FEATURES = (*PRIMARY_FEATURES, "random_50pct_relative_shortlist_ndcg")
FEATURE_LABELS = {
    "margin_helpful_mass_share": "helpful mass share",
    "margin_helpful_attribution_participation_ratio": "helpful participation",
    "margin_degrading_attribution_mass": "degrading mass",
    "random_50pct_relative_shortlist_ndcg": "50% random-removal retention",
}
DESIRED_SIGN = {
    "margin_helpful_mass_share": 1,
    "margin_helpful_attribution_participation_ratio": 1,
    "margin_degrading_attribution_mass": -1,
}


def _identity(path: Path, repository: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(repository.resolve()).as_posix()
    except ValueError:
        display = resolved.as_posix()
    return {"path": display, "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _resolve_identity(record: dict[str, Any], repository: Path, base: Path | None = None) -> Path:
    path = Path(str(record.get("path", "")))
    candidates = [path] if path.is_absolute() else [repository / path]
    if path.is_absolute():
        anchors = [
            i
            for i, part in enumerate(path.parts)
            if part in {"embedding-optimizer-study", "embedding-optimizer-story-refactor"}
        ]
        if anchors:
            relative = Path(*path.parts[anchors[-1] + 1 :])
            rebased = (repository / relative).resolve()
            if not rebased.is_relative_to(repository.resolve()):
                raise ValueError("Rebased dimension evidence escapes the repository")
            # Never fall back to a still-present producer tree when auditing a clone.
            candidates = [rebased]
    if not path.is_absolute() and base is not None:
        candidates.append(base / path)
    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.stat().st_size == int(record.get("bytes", -1))
            and _sha256(candidate) == record.get("sha256")
        ):
            return candidate.resolve()
    raise ValueError(f"Source-bound file identity mismatch: {record.get('path')}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_text(path: Path, text: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _finite(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {context}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value for {context}: {result}")
    return result


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "prospective_corrected_dimension_utilization_lock":
        raise ValueError("Unexpected dimension-utilization protocol")
    return payload


def _validate_panel_rows(
    rows: list[dict[str, str]], name: str, repository: Path, protocol: dict[str, Any]
) -> None:
    """Prove exact run/stage/task/draw coverage rather than just matching row counts."""
    configs = load_matrix(repository / "configs/dense_no_packing_retrain.yaml")
    run_metadata = {
        config.run_id: (config.optimizer.name, config.optimizer.lr) for config in configs
    }
    steps = [int(step) for step in protocol["inputs"]["checkpoint_stages"]]
    cells = {(run_id, step) for run_id in run_metadata for step in steps} | {("pretrained", 0)}
    spec = json.loads((repository / protocol["inputs"]["probe_spec"]).read_text(encoding="utf-8"))
    tasks = set(spec["expected"]["task_counts"])
    if name == "checkpoint_summary":
        expected = cells
    elif name == "task_summary":
        expected = {(*cell, task) for cell in cells for task in tasks}
    elif name == "random_removal":
        expected = {
            (*cell, task, float(fraction), draw)
            for cell in cells
            for task in tasks
            for fraction in protocol["random_removal"]["removed_fractions"]
            for draw in range(int(protocol["random_removal"]["mask_draws_per_fraction"]))
        }
    elif name == "rotation_summary":
        expected = {
            (*cell, task, int(seed))
            for cell in cells
            if cell[1] in (0, steps[-1])
            for task in tasks
            for seed in protocol["rotation_control"]["seeds"]
        }
    else:
        raise ValueError(f"Unknown dimension panel: {name}")
    observed = set()
    for row in rows:
        run_id = row["run_id"]
        if run_id == "pretrained":
            if row["optimizer"] != "pretrained" or row["learning_rate"] not in ("", None):
                raise ValueError("Pretrained dimension panel metadata differs")
        elif (
            run_id not in run_metadata
            or (row["optimizer"], _finite(row["learning_rate"], "learning rate"))
            != run_metadata[run_id]
        ):
            raise ValueError(f"Dimension panel run metadata differs: {run_id}")
        key = (run_id, int(row["step"]))
        if name != "checkpoint_summary":
            key += (row["task"],)
        if name == "random_removal":
            key += (float(row["removed_fraction"]), int(row["draw"]))
        elif name == "rotation_summary":
            key += (int(row["rotation_seed"]),)
        if key in observed:
            raise ValueError(f"Duplicate dimension panel identity: {name}:{key}")
        observed.add(key)
    if observed != expected:
        raise ValueError(f"Dimension panel coverage differs: {name}")


def _load_dimension_tables(
    directory: Path,
    repository: Path,
    protocol: dict[str, Any],
    *,
    verify_exports: bool = True,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    manifest_path = directory / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = manifest.get("coverage", {})
    if (
        manifest.get("status") != "complete"
        or manifest.get("scope") != "corrected_dense_dimension_utilization"
        or coverage.get("checkpoint_exports") != 60
        or coverage.get("tasks") != 14
        or coverage.get("dimensions") != 768
        or manifest.get("claim_boundary") != protocol["claim_boundary"]
    ):
        raise ValueError("Corrected dimension source manifest is incomplete or incompatible")
    protocol_path = _resolve_identity(manifest["protocol"], repository)
    if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Dimension manifest uses a different analysis protocol")
    if verify_exports:
        audit_outputs(
            argparse.Namespace(
                protocol=protocol_path,
                repository=repository,
                output_dir=directory,
                analysis_scope="corrected",
                skip_rotation=False,
            )
        )
    _resolve_identity(manifest["implementation"], repository)
    expected = {
        "checkpoint_summary": 61,
        "task_summary": 854,
        "random_removal": 68_320,
        "rotation_summary": 546,
    }
    tables = {}
    for name, rows in expected.items():
        path = _resolve_identity(manifest["outputs"][name], repository, directory)
        values = _read_csv(path)
        if len(values) != rows:
            raise ValueError(f"Expected {rows} rows in {path}, found {len(values)}")
        _validate_panel_rows(values, name, repository, protocol)
        tables[name] = values
    return tables, {
        "manifest": _identity(manifest_path, repository),
        "outputs": {name: manifest["outputs"][name] for name in expected},
    }


def _load_outcomes(
    directory: Path, repository: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest_path = directory / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("scope") != "corrected_dense_no_packing"
        or manifest.get("coverage")
        != {"runs": 12, "checkpoints": 60, "tasks": 14, "task_units": 840}
    ):
        raise ValueError("Primary retrieval outcome manifest is incomplete")
    record = manifest.get("outputs", {}).get("run_stage_scores")
    if not isinstance(record, dict):
        raise ValueError("Retrieval manifest does not bind run-stage scores")
    path = _resolve_identity(record, repository, directory)
    rows = _read_csv(path)
    if len(rows) != 60:
        raise ValueError(f"Expected 60 run-stage retrieval rows, found {len(rows)}")
    configs = load_matrix(repository / "configs/dense_no_packing_retrain.yaml")
    metadata = {config.run_id: (config.optimizer.name, config.optimizer.lr) for config in configs}
    expected = {(run_id, stage) for run_id in metadata for stage in range(1, 6)}
    observed = set()
    for row in rows:
        key = (row["run_id"], int(row["stage"]))
        if (
            key not in expected
            or key in observed
            or int(row["tasks"]) != 14
            or (row["optimizer"], _finite(row["learning_rate"], "outcome rate"))
            != metadata[row["run_id"]]
        ):
            raise ValueError(f"Outcome panel identity differs: {key}")
        _finite(row["mean_ndcg_at_10"], "outcome score")
        observed.add(key)
    if observed != expected:
        raise ValueError("Outcome panel does not cover every primary run and stage")
    return rows, {
        "manifest": _identity(manifest_path, repository),
        "run_stage_scores": record,
    }


def _group_task_values(
    rows: list[dict[str, str]], *, step: int, feature: str, rotation_seed: int | None = None
) -> tuple[list[str], dict[tuple[str, str], float]]:
    selected = []
    for row in rows:
        if row["optimizer"] not in OPTIMIZERS or int(row["step"]) != step:
            continue
        row_seed = row.get("rotation_seed", "")
        if rotation_seed is None and row_seed not in (None, ""):
            continue
        if rotation_seed is not None and int(row_seed) != rotation_seed:
            continue
        selected.append(row)
    tasks = sorted({row["task"] for row in selected})
    if len(tasks) != 14 or len(selected) != 12 * 14:
        raise ValueError(
            f"Dimension task panel differs for {feature}, rotation={rotation_seed}: "
            f"tasks={len(tasks)}, rows={len(selected)}"
        )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in selected:
        grouped[(row["task"], row["optimizer"])].append(_finite(row[feature], feature))
    means = {}
    for key, values in grouped.items():
        if len(values) != 4:
            raise ValueError(f"Expected four rates for {key}, found {len(values)}")
        means[key] = float(np.mean(values))
    return tasks, means


def _primary_intervals(
    task_rows: list[dict[str, str]], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    step = int(protocol["inputs"]["checkpoint_stages"][-1])
    columns: list[tuple[str, str, str]] = []
    vectors = []
    task_order: list[str] | None = None
    for feature in PRIMARY_FEATURES:
        tasks, means = _group_task_values(task_rows, step=step, feature=feature)
        if task_order is None:
            task_order = tasks
        elif task_order != tasks:
            raise ValueError("Primary dimension features use different task orders")
        for treatment, baseline in CONTRASTS:
            columns.append((feature, treatment, baseline))
            vectors.append([means[(task, treatment)] - means[(task, baseline)] for task in tasks])
    matrix = np.asarray(vectors, dtype=np.float64).T
    if matrix.shape != (14, 9) or not np.isfinite(matrix).all():
        raise ValueError(f"Primary dimension contrast matrix differs: {matrix.shape}")
    points = np.mean(matrix, axis=0)
    standard_errors = np.std(matrix, axis=0, ddof=1) / math.sqrt(matrix.shape[0])
    if np.any(standard_errors <= 0):
        raise ValueError("Primary dimension max-T requires positive standard errors")
    inference = protocol["corrected_primary_comparison"]
    samples = int(inference["bootstrap_samples"])
    seed = int(inference["bootstrap_seed"])
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, matrix.shape[0], size=(samples, matrix.shape[0]))
    bootstraps = np.mean(matrix[indices], axis=1)
    t_values = np.abs((bootstraps - points) / standard_errors)
    critical = float(np.quantile(np.max(t_values, axis=1), 0.95, method="linear"))
    rows = []
    for index, (feature, treatment, baseline) in enumerate(columns):
        lower = float(points[index] - critical * standard_errors[index])
        upper = float(points[index] + critical * standard_errors[index])
        decision = "positive" if lower > 0 else "negative" if upper < 0 else "inconclusive"
        rows.append(
            {
                "feature": feature,
                "treatment": treatment,
                "baseline": baseline,
                "mean_difference": float(points[index]),
                "simultaneous_ci_95_lower": lower,
                "simultaneous_ci_95_upper": upper,
                "across_task_standard_error": float(standard_errors[index]),
                "decision": decision,
                "max_t_critical_value": critical,
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
            }
        )
    indexed = {(row["feature"], row["treatment"], row["baseline"]): row for row in rows}
    constructive = {}
    for optimizer in ("muon", "normuon"):
        constructive[optimizer] = all(
            (
                indexed[(feature, optimizer, "adamw")]["simultaneous_ci_95_lower"] > 0
                if DESIRED_SIGN[feature] > 0
                else indexed[(feature, optimizer, "adamw")]["simultaneous_ci_95_upper"] < 0
            )
            for feature in PRIMARY_FEATURES
        )
    return rows, constructive


def _rotation_contrasts(
    rows: list[dict[str, str]], protocol: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    step = int(protocol["inputs"]["checkpoint_stages"][-1])
    output = []
    for seed in protocol["rotation_control"]["seeds"]:
        for feature in PRIMARY_FEATURES:
            tasks, means = _group_task_values(
                rows, step=step, feature=feature, rotation_seed=int(seed)
            )
            for treatment, baseline in CONTRASTS:
                effects = np.asarray(
                    [means[(task, treatment)] - means[(task, baseline)] for task in tasks],
                    dtype=np.float64,
                )
                output.append(
                    {
                        "rotation_seed": int(seed),
                        "feature": feature,
                        "treatment": treatment,
                        "baseline": baseline,
                        "mean_difference": float(np.mean(effects)),
                        "across_task_standard_error": float(
                            np.std(effects, ddof=1) / math.sqrt(len(effects))
                        ),
                    }
                )
    indexed = {
        (row["rotation_seed"], row["feature"], row["treatment"], row["baseline"]): row
        for row in output
    }
    stable = {}
    for optimizer in ("muon", "normuon"):
        stable[optimizer] = all(
            DESIRED_SIGN[feature]
            * indexed[(int(seed), feature, optimizer, "adamw")]["mean_difference"]
            > 0
            for seed in protocol["rotation_control"]["seeds"]
            for feature in PRIMARY_FEATURES
        )
    return output, stable


def _random50(rows: list[dict[str, str]], final_step: int) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["optimizer"] in OPTIMIZERS
            and int(row["step"]) == final_step
            and math.isclose(float(row["removed_fraction"]), 0.5)
        ):
            grouped[row["optimizer"]].append(
                _finite(row["relative_shortlist_ndcg"], "50% removal retention")
            )
    expected = 4 * 14 * 20
    if set(grouped) != set(OPTIMIZERS) or any(
        len(values) != expected for values in grouped.values()
    ):
        raise ValueError("Random-removal final panel differs from four rates x tasks x draws")
    return {optimizer: float(np.mean(grouped[optimizer])) for optimizer in OPTIMIZERS}


def _bridge_panel(
    checkpoint_rows: list[dict[str, str]],
    random_rows: list[dict[str, str]],
    outcome_rows: list[dict[str, str]],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    stages = [int(step) for step in protocol["inputs"]["checkpoint_stages"]]
    stage_by_step = {step: index for index, step in enumerate(stages, start=1)}
    checkpoints = [row for row in checkpoint_rows if row["optimizer"] in OPTIMIZERS]
    if len(checkpoints) != 60:
        raise ValueError(f"Expected 60 dimension checkpoint rows, found {len(checkpoints)}")
    random_grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in random_rows:
        if row["optimizer"] in OPTIMIZERS and math.isclose(float(row["removed_fraction"]), 0.5):
            random_grouped[(row["run_id"], int(row["step"]))].append(
                _finite(row["relative_shortlist_ndcg"], "bridge random retention")
            )
    outcomes = {(row["run_id"], int(row["stage"])): row for row in outcome_rows}
    rates: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in checkpoints:
        if int(row["step"]) == stages[0]:
            rates[row["optimizer"]].append((float(row["learning_rate"]), row["run_id"]))
    dose = {}
    centered = {}
    for optimizer in OPTIMIZERS:
        members = sorted(rates[optimizer])
        if len(members) != 4:
            raise ValueError(f"Expected four rates for {optimizer}, found {len(members)}")
        logs = np.log10([rate for rate, _run in members])
        for index, ((_, run_id), log_rate) in enumerate(zip(members, logs, strict=True), start=1):
            dose[run_id] = index
            centered[run_id] = float(log_rate - np.mean(logs))
    output = []
    for row in checkpoints:
        run_id = row["run_id"]
        step = int(row["step"])
        stage = stage_by_step.get(step)
        key = (run_id, stage or -1)
        retention = random_grouped[(run_id, step)]
        if stage is None or len(retention) != 14 * 20 or key not in outcomes:
            raise ValueError(f"Incomplete dimension bridge row: {run_id}, step {step}")
        output.append(
            {
                "run_id": run_id,
                "optimizer": row["optimizer"],
                "dose_index": dose[run_id],
                "centered_log10_learning_rate": centered[run_id],
                "stage": stage,
                "mean_ndcg_at_10": _finite(
                    outcomes[key]["mean_ndcg_at_10"], "full-corpus retrieval"
                ),
                **{feature: _finite(row[feature], feature) for feature in PRIMARY_FEATURES},
                "random_50pct_relative_shortlist_ndcg": float(np.mean(retention)),
            }
        )
    return sorted(output, key=lambda item: (item["run_id"], item["stage"]))


def _baseline_design(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append(
            [
                1.0,
                float(row["optimizer"] == "muon"),
                float(row["optimizer"] == "normuon"),
                *(float(int(row["stage"]) == stage) for stage in range(2, 6)),
                float(row["centered_log10_learning_rate"]),
            ]
        )
    design = np.asarray(matrix, dtype=np.float64)
    if design.shape != (60, 8) or np.linalg.matrix_rank(design) != 8:
        raise ValueError(f"Dimension bridge baseline design differs: {design.shape}")
    return design


def _evaluate_bridge(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = _baseline_design(rows)
    outcome = np.asarray([row["mean_ndcg_at_10"] for row in rows], dtype=np.float64)
    doses = np.asarray([row["dose_index"] for row in rows], dtype=np.int64)
    fold_rows = []
    summary_rows = []
    for feature in BRIDGE_FEATURES:
        values = np.asarray([row[feature] for row in rows], dtype=np.float64)
        pooled_observed = []
        pooled_baseline = []
        pooled_feature = []
        improved = 0
        for fold in range(1, 5):
            test = doses == fold
            train = ~test
            mean = float(np.mean(values[train]))
            scale = float(np.std(values[train]))
            if int(train.sum()) != 45 or int(test.sum()) != 15 or scale <= 0:
                raise ValueError(f"Invalid dimension bridge fold {fold} for {feature}")
            augmented = np.column_stack((baseline, (values - mean) / scale))
            baseline_prediction = (
                baseline[test] @ np.linalg.lstsq(baseline[train], outcome[train], rcond=None)[0]
            )
            feature_prediction = (
                augmented[test] @ np.linalg.lstsq(augmented[train], outcome[train], rcond=None)[0]
            )
            baseline_rmse = float(np.sqrt(np.mean((outcome[test] - baseline_prediction) ** 2)))
            feature_rmse = float(np.sqrt(np.mean((outcome[test] - feature_prediction) ** 2)))
            reduction = baseline_rmse - feature_rmse
            improved += int(reduction > 0)
            fold_rows.append(
                {
                    "feature": feature,
                    "held_out_dose_index": fold,
                    "baseline_rmse": baseline_rmse,
                    "feature_rmse": feature_rmse,
                    "rmse_reduction": reduction,
                    "feature_improves": reduction > 0,
                }
            )
            pooled_observed.extend(outcome[test])
            pooled_baseline.extend(baseline_prediction)
            pooled_feature.extend(feature_prediction)
        observed = np.asarray(pooled_observed)
        baseline_rmse = float(np.sqrt(np.mean((observed - np.asarray(pooled_baseline)) ** 2)))
        feature_rmse = float(np.sqrt(np.mean((observed - np.asarray(pooled_feature)) ** 2)))
        reduction = baseline_rmse - feature_rmse
        summary_rows.append(
            {
                "feature": feature,
                "pooled_baseline_rmse": baseline_rmse,
                "pooled_feature_rmse": feature_rmse,
                "pooled_rmse_reduction": reduction,
                "folds_improved": improved,
                "folds_total": 4,
                "predictively_useful": reduction > 0 and improved >= 3,
            }
        )
    return fold_rows, summary_rows


def _effect_text(row: dict[str, Any]) -> str:
    return (
        f"{row['mean_difference']:+.4f} "
        f"[{row['simultaneous_ci_95_lower']:+.4f}, {row['simultaneous_ci_95_upper']:+.4f}]"
    )


def _figure_points(
    tables: dict[str, list[dict[str, Any]]], repository: Path, protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    """Descriptive all-rate readouts; no new selection, intervals, or estimands."""
    for name in ("task_summary", "random_removal"):
        _validate_panel_rows(tables[name], name, repository, protocol)
    steps = [int(step) for step in protocol["inputs"]["checkpoint_stages"]]
    task_count = int(protocol["inputs"]["task_groups"])
    draws = int(protocol["random_removal"]["mask_draws_per_fraction"])
    fractions = [float(value) for value in protocol["random_removal"]["removed_fractions"]]
    points = []

    def point(feature, optimizer, step, x, values, *, mask_draws=0, identity=False):
        expected = (1 if optimizer == "pretrained" else 4) * task_count
        if mask_draws:
            expected *= mask_draws
        if not identity and len(values) != expected:
            raise ValueError(f"Incomplete figure mean: {feature}/{optimizer}/{step}/{x}")
        finite = [_finite(value, feature) for value in values]
        points.append(
            {
                "feature": feature,
                "optimizer": optimizer,
                "step": step,
                "x": x,
                "mean": 1.0 if identity else math.fsum(finite) / len(finite),
                "run_count": 1 if optimizer == "pretrained" else 4,
                "task_count": task_count,
                "mask_draws": mask_draws,
                "role": "identity_reference" if identity else "descriptive_mean",
            }
        )

    # All task/run/draw cells have equal weight in this complete balanced panel.
    for optimizer in ("pretrained", *OPTIMIZERS):
        step = 0 if optimizer == "pretrained" else steps[-1]
        point("relative_shortlist_ndcg", optimizer, step, 0.0, [], identity=True)
        for fraction in fractions:
            values = [
                row["relative_shortlist_ndcg"]
                for row in tables["random_removal"]
                if row["optimizer"] == optimizer
                and int(row["step"]) == step
                and float(row["removed_fraction"]) == fraction
            ]
            point(
                "relative_shortlist_ndcg", optimizer, step, 100 * fraction, values, mask_draws=draws
            )
    for feature in PRIMARY_FEATURES:
        for optimizer in ("pretrained", *OPTIMIZERS):
            for step in [0] if optimizer == "pretrained" else steps:
                values = [
                    row[feature]
                    for row in tables["task_summary"]
                    if row["optimizer"] == optimizer and int(row["step"]) == step
                ]
                progress = 0.0 if step == 0 else 100 * (steps.index(step) + 1) / len(steps)
                point(feature, optimizer, step, progress, values)
    return points


def _render_figure(points: list[dict[str, Any]]) -> str:
    """Native vector plot covered by exact coordinate and LaTeX recomputation."""
    panels = (
        (
            "relative_shortlist_ndcg",
            "(a) Final-state random removal",
            "Coordinates removed (\\%)",
            "Relative shortlist nDCG",
        ),
        (PRIMARY_FEATURES[0], "(b) Helpful mass share", "Training progress (\\%)", "Share"),
        (
            PRIMARY_FEATURES[1],
            "(c) Helpful participation",
            "Training progress (\\%)",
            "Normalized participation",
        ),
        (
            PRIMARY_FEATURES[2],
            "(d) Degrading mass",
            "Training progress (\\%)",
            "Margin attribution mass",
        ),
    )
    styles = {
        "pretrained": "black!55,densely dashed,no marks",
        "adamw": "dimensionAdamW,mark=*,mark options={solid,fill=dimensionAdamW}",
        "muon": "dimensionMuon,mark=square*,mark options={solid,fill=dimensionMuon}",
        "normuon": "dimensionNorMuon,mark=triangle*,mark options={solid,fill=dimensionNorMuon}",
    }
    parts = [
        r"\newcommand{\DimensionUtilizationFigure}{%",
        r"\begin{figure*}[t]",
        r"\centering",
        r"\definecolor{dimensionAdamW}{HTML}{3569A8}",
        r"\definecolor{dimensionMuon}{HTML}{D97721}",
        r"\definecolor{dimensionNorMuon}{HTML}{2E7D61}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[group style={group size=2 by 2,horizontal sep=0.15\textwidth,vertical sep=1.6cm},",
        r"width=0.37\textwidth,height=0.25\textwidth,scale only axis,",
        r"tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\footnotesize},",
        r"grid=major,grid style={black!8},major tick length=2pt,",
        r"legend style={font=\scriptsize,draw=none,fill=none,cells={anchor=west}},",
        r"every axis plot/.append style={line width=0.9pt,mark size=1.8pt}]",
    ]
    for feature, title, xlabel, ylabel in panels:
        random = feature == "relative_shortlist_ndcg"
        limits = (
            "xmin=0,xmax=75,xtick={0,10,25,50,75}"
            if random
            else "xmin=0,xmax=100,xtick={0,20,40,60,80,100},ymin=0"
        )
        parts.append(
            f"\\nextgroupplot[title={{{title}}},xlabel={{{xlabel}}},ylabel={{{ylabel}}},{limits}]"
        )
        for optimizer in ("pretrained", *OPTIMIZERS):
            selected = [
                p for p in points if p["feature"] == feature and p["optimizer"] == optimizer
            ]
            coords = [(float(p["x"]), float(p["mean"])) for p in selected]
            if optimizer == "pretrained" and not random:
                coords = [(0.0, coords[0][1]), (100.0, coords[0][1])]
            formatted = " ".join(f"({x:.12g},{y:.12g})" for x, y in coords)
            parts.append(f"\\addplot+[{styles[optimizer]}] coordinates {{{formatted}}};")
            if random:
                label = OPTIMIZER_LABELS[optimizer]
                parts.append(f"\\addlegendentry{{{label}}}")
        if random:
            parts.append(r"\draw[black!30,dotted] (axis cs:0,1) -- (axis cs:75,1);")
    parts.extend(
        [
            r"\end{groupplot}",
            r"\end{tikzpicture}",
            r"\caption{Dimension utility over the complete rate grid. (a) Final-state retention after shared random coordinate removal; the zero-removal value is one by definition. (b--d) Native-coordinate margin attribution at all five stages. Each optimizer curve averages all four rates and 14 tasks; (a) additionally averages 20 shared masks per nonzero fraction. The gray curve or horizontal line is the pretrained reference. Lines join measured points, not fitted trajectories. These are descriptive means, not confidence intervals or full-corpus retrieval scores; formal contrasts and rotation controls are reported separately.}",
            r"\label{fig:dimension-utility}",
            r"\end{figure*}%",
            "}",
        ]
    )
    return "\n".join(parts) + "\n"


def _find(rows: list[dict[str, Any]], feature: str, treatment: str) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["feature"] == feature
        and row["treatment"] == treatment
        and row["baseline"] == "adamw"
    )


def _render_latex(
    primary: list[dict[str, Any]],
    constructive: dict[str, bool],
    rotation_stable: dict[str, bool],
    retention: dict[str, float],
    bridge: list[dict[str, Any]],
    figure_points: list[dict[str, Any]],
) -> str:
    muon = {feature: _find(primary, feature, "muon") for feature in PRIMARY_FEATURES}
    main_finding = (
        "At the final stage, Muon minus AdamW changes margin helpful-mass share by "
        f"{_effect_text(muon['margin_helpful_mass_share'])}, helpful participation by "
        f"{_effect_text(muon['margin_helpful_attribution_participation_ratio'])}, and degrading "
        f"mass by {_effect_text(muon['margin_degrading_attribution_mass'])}. The joint native-basis "
        f"criterion is {'supported' if constructive['muon'] else 'not supported'}, and its required "
        f"direction is {'stable' if rotation_stable['muon'] else 'not stable'} across the three tested shared "
        f"rotations. With 50\\% of coordinates removed, relative shortlist nDCG is "
        f"{retention['adamw']:.4f}/{retention['muon']:.4f}/{retention['normuon']:.4f} for "
        "AdamW/Muon/NorMuon. Figure~\\ref{fig:dimension-utility} shows the complete all-rate readout."
    )
    supported = [
        FEATURE_LABELS[row["feature"]].replace("%", r"\%")
        for row in bridge
        if row["predictively_useful"]
    ]
    bridge_finding = (
        "The leave-dose-index-out bridge supports "
        + (", ".join(supported) if supported else "none of the four dimension-use features")
        + " beyond optimizer, stage, and dose."
    )
    if constructive["muon"] and rotation_stable["muon"]:
        conclusion = (
            "Muon distributes coordinate-level retrieval utility more constructively than AdamW, "
            "and the direction is stable under the shared-rotation controls."
        )
    elif constructive["muon"]:
        conclusion = (
            "Muon changes native-coordinate utility more constructively than AdamW, but the effect "
            "is basis dependent and does not establish broader dimensional capacity."
        )
    else:
        conclusion = (
            "The predeclared analysis does not support broader or more constructive dimension use "
            "as the explanation for Muon's retrieval behavior."
        )
    rows = []
    for treatment in ("muon", "normuon"):
        for feature in PRIMARY_FEATURES:
            row = _find(primary, feature, treatment)
            label = FEATURE_LABELS[feature].replace("%", r"\%")
            rows.append(
                f"{label} & {OPTIMIZER_LABELS[treatment]}--AdamW & {row['mean_difference']:+.4f} & "
                f"[{row['simultaneous_ci_95_lower']:+.4f}, "
                f"{row['simultaneous_ci_95_upper']:+.4f}] & {row['decision']} \\\\"
            )
    return (
        _render_figure(figure_points)
        + "% Generated by embed-optim-render-dimension-utilization; do not edit manually.\n"
        f"\\newcommand{{\\DimensionUtilizationFinding}}{{{main_finding}}}\n"
        f"\\newcommand{{\\DimensionRetrievalBridgeFinding}}{{{bridge_finding}}}\n"
        f"\\newcommand{{\\DimensionConclusionFinding}}{{{conclusion}}}\n"
        "\\newcommand{\\DimensionUtilizationAppendixTable}{%\n"
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{llrrl}\n\\toprule\n"
        "Feature & Contrast & Effect & Simultaneous 95\\% CI & Decision \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Final-stage all-rate dimension-utilization contrasts against AdamW. The common "
        "max-$T$ family also includes the corresponding NorMuon--Muon contrasts.}\n"
        "\\label{tab:dimension-utilization}\n\\end{table*}%\n}\n"
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    protocol = _load_protocol(args.protocol.resolve())
    tables, dimension_source = _load_dimension_tables(
        args.dimension_dir.resolve(), repository, protocol
    )
    outcomes, outcome_source = _load_outcomes(args.outcomes_dir.resolve(), repository)
    primary, constructive = _primary_intervals(tables["task_summary"], protocol)
    rotation, stable = _rotation_contrasts(tables["rotation_summary"], protocol)
    final_step = int(protocol["inputs"]["checkpoint_stages"][-1])
    retention = _random50(tables["random_removal"], final_step)
    bridge_panel = _bridge_panel(
        tables["checkpoint_summary"], tables["random_removal"], outcomes, protocol
    )
    folds, bridge = _evaluate_bridge(bridge_panel)
    figure_points = _figure_points(tables, repository, protocol)
    output_dir = args.output_dir.resolve()
    outputs = {
        "primary_contrasts": _atomic_csv(output_dir / "primary_contrasts.csv", primary),
        "rotation_contrasts": _atomic_csv(output_dir / "rotation_contrasts.csv", rotation),
        "bridge_panel": _atomic_csv(output_dir / "bridge_panel.csv", bridge_panel),
        "bridge_folds": _atomic_csv(output_dir / "bridge_folds.csv", folds),
        "bridge_summary": _atomic_csv(output_dir / "bridge_summary.csv", bridge),
        "figure_points": _atomic_csv(output_dir / "figure_points.csv", figure_points),
    }
    latex = _render_latex(primary, constructive, stable, retention, bridge, figure_points)
    outputs["paper_latex"] = _atomic_text(args.paper_output.resolve(), latex)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "dense_primary_dimension_utilization_publication",
        "protocol": _identity(args.protocol.resolve(), repository),
        "implementation": _identity(Path(__file__), repository),
        "sources": {"dimension": dimension_source, "outcome": outcome_source},
        "coverage": {
            "runs": 12,
            "checkpoints": 60,
            "tasks": 14,
            "primary_contrasts": len(primary),
            "rotation_contrasts": len(rotation),
            "bridge_rows": len(bridge_panel),
            "bridge_features": len(bridge),
            "figure_points": len(figure_points),
        },
        "decisions": {
            "constructive_native_coordinate_use": constructive,
            "direction_stable_across_rotations": stable,
            "predictively_useful_features": [
                row["feature"] for row in bridge if row["predictively_useful"]
            ],
        },
        "random_50pct_relative_shortlist_ndcg": retention,
        "outputs": outputs,
        "claim_boundary": protocol["claim_boundary"],
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def audit_report(args: argparse.Namespace, *, portable: bool = False) -> dict[str, Any]:
    repository = args.repository.resolve()
    if portable:
        from .dimension_release import audit_closure

        audit_closure(repository)
    protocol = _load_protocol(args.protocol.resolve())
    manifest_path = args.output_dir.resolve() / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = manifest.get("coverage", {})
    if (
        manifest.get("status") != "complete"
        or manifest.get("scope") != "dense_primary_dimension_utilization_publication"
        or coverage.get("runs") != 12
        or coverage.get("checkpoints") != 60
        or coverage.get("primary_contrasts") != 9
        or coverage.get("rotation_contrasts") != 27
        or coverage.get("bridge_rows") != 60
        or coverage.get("bridge_features") != 4
        or coverage.get("figure_points") != 68
        or manifest.get("claim_boundary") != protocol["claim_boundary"]
    ):
        raise ValueError("Dimension publication coverage or claim boundary differs")
    if (
        _resolve_identity(manifest["protocol"], repository) != args.protocol.resolve()
        or _resolve_identity(manifest["implementation"], repository) != Path(__file__).resolve()
    ):
        raise ValueError("Dimension publication implementation or protocol identity differs")
    if portable:
        tables, dimension_source = _load_dimension_tables(
            args.dimension_dir.resolve(), repository, protocol, verify_exports=False
        )
    else:
        tables, dimension_source = _load_dimension_tables(
            args.dimension_dir.resolve(), repository, protocol
        )
    outcomes, outcome_source = _load_outcomes(args.outcomes_dir.resolve(), repository)
    if manifest.get("sources") != {"dimension": dimension_source, "outcome": outcome_source}:
        raise ValueError("Dimension publication upstream sources changed")
    primary, constructive = _primary_intervals(tables["task_summary"], protocol)
    rotation, stable = _rotation_contrasts(tables["rotation_summary"], protocol)
    retention = _random50(
        tables["random_removal"], int(protocol["inputs"]["checkpoint_stages"][-1])
    )
    panel = _bridge_panel(
        tables["checkpoint_summary"], tables["random_removal"], outcomes, protocol
    )
    folds, bridge = _evaluate_bridge(panel)
    figure_points = _figure_points(tables, repository, protocol)
    expected_tables = {
        "primary_contrasts": primary,
        "rotation_contrasts": rotation,
        "bridge_panel": panel,
        "bridge_folds": folds,
        "bridge_summary": bridge,
        "figure_points": figure_points,
    }
    if set(manifest.get("outputs", {})) != {*expected_tables, "paper_latex"}:
        raise ValueError("Dimension publication output inventory differs")
    for record in manifest["outputs"].values():
        _resolve_identity(record, repository, args.output_dir.resolve())
    for name, rows in expected_tables.items():
        record = manifest["outputs"][name]
        path = _resolve_identity(record, repository, args.output_dir.resolve())
        expected = io.StringIO(newline="")
        writer = csv.DictWriter(
            expected,
            fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        if path.read_text(encoding="utf-8") != expected.getvalue() or record.get("rows") != len(
            rows
        ):
            raise ValueError(f"Dimension publication table differs from recomputation: {name}")
    paper = _resolve_identity(manifest["outputs"]["paper_latex"], repository)
    if paper != args.paper_output.resolve() or paper.read_text(encoding="utf-8") != _render_latex(
        primary, constructive, stable, retention, bridge, figure_points
    ):
        raise ValueError("Dimension publication manuscript differs from exact rendering")
    decisions = {
        "constructive_native_coordinate_use": constructive,
        "direction_stable_across_rotations": stable,
        "predictively_useful_features": [
            row["feature"] for row in bridge if row["predictively_useful"]
        ],
    }
    if (
        manifest.get("decisions") != decisions
        or manifest.get("random_50pct_relative_shortlist_ndcg") != retention
    ):
        raise ValueError("Dimension publication decisions differ from recomputation")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/dense_dimension_utilization_protocol.json")
    )
    parser.add_argument(
        "--dimension-dir", type=Path, default=Path("reports/dimension-utilization-primary")
    )
    parser.add_argument(
        "--outcomes-dir", type=Path, default=Path("reports/dense-no-packing-outcomes")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/dimension-utilization-publication")
    )
    parser.add_argument(
        "--paper-output", type=Path, default=Path("paper/generated/dimension-utilization.tex")
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--portable-audit",
        action="store_true",
        help="Recompute publication from its closed evidence bundle, without model payloads",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = (
        audit_report(args, portable=args.portable_audit)
        if args.audit_only or args.portable_audit
        else build_report(args)
    )
    print(json.dumps(manifest["coverage"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
