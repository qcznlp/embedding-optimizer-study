"""Complete functional panels and the genuine four-feature exact retrieval bridge.

Pure numerical helpers do not establish a primary experiment identity. The v3
caller must first reconstruct source features and the original geometry/outcome
bridge. No old feature is replaced with a dimension value to satisfy its schema.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .bridge_named_features import evaluate_named
from .bridge_numerics import validate_panel
from .dimension_publication import (
    BRIDGE_FEATURES,
    OPTIMIZERS,
    PRIMARY_FEATURES,
    _primary_intervals,
    _random50,
    _rotation_contrasts,
)
from .primary_contract import require_same
from .primary_v3_dimension_contract import TABLE_COUNTS as INPUT_COUNTS
from .primary_v3_dimension_contract import planned_states

TABLE_COUNTS = {
    "primary_contrasts": 9,
    "rotation_contrasts": 27,
    "figure_points": 68,
    "bridge_rows": 60,
    "leave_dose_fold_metrics": 16,
    "feature_prediction_summary": 4,
    "residual_associations": 4,
    "held_out_predictions": 240,
    "optimizer_stage_metrics": 15,
}
RULES = {
    "task_family": "Original 14 paired task units, all four rates, one nine-contrast final-stage fixed-SE max-T family; no seed-variability claim",
    "zero_standard_error": "Preserve the original refusal for any nonpositive standard error; no reduced family or invented zero-width interval",
    "primary_features": list(PRIMARY_FEATURES),
    "bridge_features": list(BRIDGE_FEATURES),
    "original_geometry_features": "Authenticate and retain every original geometry/outcome bridge column; add four distinct functional names without substituting legacy slots",
    "missing_prediction": "Report an undefined usefulness flag distinctly from a false flag; no available-fold pooling",
    "constructive": "All three original desired-direction simultaneous intervals must pass; failure of the joint rule is not equivalence",
    "rotation": "All three tested rotations must preserve all three required mean directions; descriptive sign stability is not arbitrary-basis invariance",
    "figures": "68 original all-rate task-first descriptive points; no confidence bands, selected stage/rate or fitted trajectory",
    "scope": "The eight-candidate probe is not full-corpus retrieval; dose-held-out prediction is not task-held-out generalization, mediation or causal explanation",
}


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Functional inference requires finite non-boolean typed numbers")
    return float(value)


def validate_tables(primary, tables, protocol, tasks):
    """Exact identities and every task/draw/rotation, not merely the expected counts."""
    if tasks != sorted(set(tasks)) or len(tasks) != 14:
        raise ValueError("Require the complete ordered fourteen-task population")
    require_same({key: len(rows) for key, rows in tables.items()}, INPUT_COUNTS)
    states = planned_states(primary)
    metadata = {(s["meta"]["run_id"], s["meta"]["step"]): s["meta"] for s in states}
    fractions = protocol["random_removal"]["removed_fractions"]
    draws = protocol["random_removal"]["mask_draws_per_fraction"]
    seeds = protocol["rotation_control"]["seeds"]
    expected = {
        "checkpoint_summary": set(metadata),
        "task_summary": {(*cell, task) for cell in metadata for task in tasks},
        "random_removal": {
            (*cell, task, fraction, draw)
            for cell in metadata
            for task in tasks
            for fraction in fractions
            for draw in range(draws)
        },
        "rotation_summary": {
            (*cell, task, seed)
            for cell in metadata
            if cell[1] in (0, protocol["inputs"]["checkpoint_stages"][-1])
            for task in tasks
            for seed in seeds
        },
    }
    canonical = {}
    for name, rows in tables.items():
        seen = {}
        for row in rows:
            if type(row.get("step")) is not int or not isinstance(row.get("run_id"), str):
                raise ValueError("Invalid functional run/stage identity")
            cell = row["run_id"], row["step"]
            if cell not in metadata:
                raise ValueError("Unknown or historical functional state")
            require_same({key: row[key] for key in metadata[cell]}, metadata[cell])
            key = cell
            if name != "checkpoint_summary":
                key += (row["task"],)
            if name == "random_removal":
                if type(row["draw"]) is not int:
                    raise ValueError("Mask draw must be an integer")
                key += (finite(row["removed_fraction"]), row["draw"])
                for metric in ("shortlist_ndcg", "relative_shortlist_ndcg", "margin"):
                    finite(row[metric])
                if row["relative_margin"] is not None:
                    finite(row["relative_margin"])
            elif name == "rotation_summary":
                if type(row["rotation_seed"]) is not int:
                    raise ValueError("Rotation seed must be an integer")
                key += (row["rotation_seed"],)
            elif name == "task_summary" and row.get("rotation_seed") is not None:
                raise ValueError("A native task row cannot be a rotated observation")
            if name != "random_removal":
                for feature in PRIMARY_FEATURES:
                    finite(row[feature])
            if key not in expected[name] or key in seen:
                raise ValueError("Duplicate, extra or unpaired functional cell")
            seen[key] = row
        if set(seen) != expected[name]:
            raise ValueError("Incomplete functional panel")
        canonical[name] = [seen[key] for key in sorted(seen)]
    # Checkpoint attribution summaries must actually equal the task-first means.
    grouped = defaultdict(list)
    for row in canonical["task_summary"]:
        grouped[row["run_id"], row["step"]].append(row)
    for row in canonical["checkpoint_summary"]:
        if type(row.get("tasks")) is not int or row["tasks"] != 14:
            raise ValueError("A checkpoint lacks all fourteen task means")
        for feature in PRIMARY_FEATURES:
            require_same(
                row[feature],
                float(np.mean([r[feature] for r in grouped[row["run_id"], row["step"]]])),
            )
    return canonical


def add_functional_features(primary, tables, original_panel):
    original = validate_panel(original_panel)
    state_metadata = {
        (s["meta"]["run_id"], s["stage"]): s["meta"] for s in planned_states(primary) if s["stage"]
    }
    if {(row["run_id"], row["stage"]) for row in original} != set(state_metadata):
        raise ValueError("Original bridge does not cover the same complete v3 states")
    checkpoints = {(r["run_id"], r["step"]): r for r in tables["checkpoint_summary"]}
    retention = defaultdict(list)
    for row in tables["random_removal"]:
        if row["optimizer"] in OPTIMIZERS and row["removed_fraction"] == 0.5:
            retention[row["run_id"], row["step"]].append(finite(row["relative_shortlist_ndcg"]))
    output = []
    for row in original:
        meta = state_metadata[row["run_id"], row["stage"]]
        require_same(
            {key: row[key] for key in ("run_id", "optimizer", "learning_rate")},
            {key: meta[key] for key in ("run_id", "optimizer", "learning_rate")},
        )
        if set(BRIDGE_FEATURES) & set(row):
            raise ValueError("A functional feature cannot overwrite an existing original column")
        key = meta["run_id"], meta["step"]
        if len(retention[key]) != 280:
            raise ValueError("Bridge retention must use all fourteen tasks and twenty shared masks")
        output.append(
            {
                **row,
                **{name: finite(checkpoints[key][name]) for name in PRIMARY_FEATURES},
                "random_50pct_relative_shortlist_ndcg": float(np.mean(retention[key])),
            }
        )
    return output


def figure_points(tables, protocol):
    """Preserve the original 68-point equal-cell estimator on an admitted balanced panel."""
    points = []
    steps = protocol["inputs"]["checkpoint_stages"]

    def append(feature, optimizer, step, x, rows, *, draws=0, identity=False):
        expected = (1 if optimizer == "pretrained" else 4) * 14 * (draws or 1)
        if not identity and len(rows) != expected:
            raise ValueError("Descriptive point lacks its complete all-rate task/draw population")
        values = [finite(row[feature]) for row in rows]
        points.append(
            {
                "feature": feature,
                "optimizer": optimizer,
                "step": step,
                "x": x,
                "mean": 1.0 if identity else math.fsum(values) / len(values),
                "run_count": 1 if optimizer == "pretrained" else 4,
                "task_count": 14,
                "mask_draws": draws,
                "role": "identity_reference" if identity else "descriptive_mean",
            }
        )

    for optimizer in ("pretrained", *OPTIMIZERS):
        step = 0 if optimizer == "pretrained" else steps[-1]
        append("relative_shortlist_ndcg", optimizer, step, 0.0, [], identity=True)
        for fraction in protocol["random_removal"]["removed_fractions"]:
            selected = [
                r
                for r in tables["random_removal"]
                if r["optimizer"] == optimizer
                and r["step"] == step
                and r["removed_fraction"] == fraction
            ]
            append(
                "relative_shortlist_ndcg",
                optimizer,
                step,
                100 * fraction,
                selected,
                draws=protocol["random_removal"]["mask_draws_per_fraction"],
            )
    for feature in PRIMARY_FEATURES:
        for optimizer in ("pretrained", *OPTIMIZERS):
            for step in [0] if optimizer == "pretrained" else steps:
                selected = [
                    r
                    for r in tables["task_summary"]
                    if r["optimizer"] == optimizer and r["step"] == step
                ]
                append(
                    feature,
                    optimizer,
                    step,
                    0.0 if step == 0 else 100 * (steps.index(step) + 1) / len(steps),
                    selected,
                )
    return points


def optimizer_stage_metrics(tables, protocol):
    """Report every native metric at every stage, never only the co-primary metrics."""
    ignored = {"run_id", "optimizer", "learning_rate", "step", "rotation_seed", "task"}
    columns = set(tables["task_summary"][0]) - ignored
    output = []
    for optimizer in OPTIMIZERS:
        for stage, step in enumerate(protocol["inputs"]["checkpoint_stages"], 1):
            rows = [
                r
                for r in tables["task_summary"]
                if r["optimizer"] == optimizer and r["step"] == step
            ]
            if len(rows) != 56 or any(set(r) - ignored != columns for r in rows):
                raise ValueError("Optimizer trajectory lacks complete equal task/rate metrics")
            output.append(
                {
                    "optimizer": optimizer,
                    "stage": stage,
                    "step": step,
                    "rates": 4,
                    "tasks": 14,
                    **{
                        name: math.fsum(finite(r[name]) for r in rows) / 56
                        for name in sorted(columns)
                    },
                }
            )
    return output


def summarize(primary, tables, original_panel, protocol, tasks):
    tables = validate_tables(primary, tables, protocol, tasks)
    contrasts, constructive = _primary_intervals(tables["task_summary"], protocol)
    rotations, stable = _rotation_contrasts(tables["rotation_summary"], protocol)
    retention = _random50(tables["random_removal"], protocol["inputs"]["checkpoint_stages"][-1])
    bridge, diagnostics = evaluate_named(
        add_functional_features(primary, tables, original_panel), BRIDGE_FEATURES
    )
    result = {
        **bridge,
        "primary_contrasts": contrasts,
        "rotation_contrasts": rotations,
        "figure_points": figure_points(tables, protocol),
        "optimizer_stage_metrics": optimizer_stage_metrics(tables, protocol),
    }
    require_same({name: len(rows) for name, rows in result.items()}, TABLE_COUNTS)
    return result, {
        "constructive_native_coordinate_use": constructive,
        "direction_stable_across_tested_rotations": stable,
        "random_50pct_relative_shortlist_ndcg": retention,
        "bridge_numerical_diagnostics": diagnostics,
        "rules": RULES,
        "scientific_completion": False,
    }
