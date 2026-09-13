"""Explicit exact-measurement columns and full-coverage AdamW comparators."""

from __future__ import annotations

import copy
import itertools
import math

from .bridge_exact_arithmetic import rational
from .bridge_numerics import validate_panel
from .primary_completion import integer
from .primary_contract import require_same

CHECKPOINT_FEATURES = {
    "exact_nonzero_saved_segment_stable_rank_fraction": (
        "exact_saved_segment_stable_rank_fraction_parameter_weighted_nonzero",
        "saved_segment",
        "saved_segment_stable_rank_fraction",
    ),
    "full_spectrum_nonzero_saved_segment_entropy_rank_fraction": (
        "exact_saved_segment_full_entropy_effective_rank_fraction_parameter_weighted_nonzero",
        "saved_segment",
        "saved_segment_sketch_effective_rank_fraction",
    ),
    "exact_nonzero_cumulative_stable_rank_fraction": (
        "exact_cumulative_stable_rank_fraction_parameter_weighted_nonzero",
        "cumulative",
        "cumulative_stable_rank_fraction",
    ),
}
PAIR_FEATURES = {
    "exact_mean_saved_segment_overlap_to_adamw": (
        "saved_segment",
        "mean_saved_segment_subspace_overlap_to_adamw",
    ),
    "exact_mean_cumulative_overlap_to_adamw": (
        "cumulative",
        "mean_cumulative_subspace_overlap_to_adamw",
    ),
}
FEATURES = (*CHECKPOINT_FEATURES, *PAIR_FEATURES)
MEASUREMENT_RULES = {
    "complete_population": "All twelve rates and all five stages; no selection by validation, BEIR, feature magnitude or support",
    "rank_denominator": "Parameter-weighted means over all nonzero hidden matrices; expose mass; never silently equate with original zero-imputed averages",
    "entropy": "Full singular-spectrum entropy, explicitly different from top-64 renormalized sketch entropy",
    "adamw_comparators": "All four AdamW rates for Muon/NorMuon; all other three AdamW rates for AdamW; exact equal rate weights",
    "undefined_comparator": "If any required complete pair mean is undefined, the whole focal mean is undefined; never use resolved-only conditional overlap or drop a comparator",
    "zero_or_unsafe": "Retain zero/unsupported/unresolved population mass and reasons; missing rows remain errors rather than an allowed undefined value",
    "nonspectral_features": "The original four nonspectral columns remain in the original family only; no duplicate hypothesis tests",
    "comparison": "Report all five original/exact correspondences and coverage, regardless of predictive result; differences can include named estimand changes, not just numerical approximation",
    "causal_boundary": "Weight displacement rank/subspaces alone are neither functional embedding-dimension utility nor a causal explanation of retrieval quality",
}


def bounded(value):
    q = rational(value)
    if not 0 <= q <= 1:
        raise ValueError("Feature fraction lies outside [0,1]")
    return float(q)


def checkpoint_features(rows, configs, steps):
    """Geometry-only mapping; supports explicitly labelled diagnostic populations too."""
    recipes = {r.run_id: r for r in configs}
    expected = {(run_id, stage) for run_id in recipes for stage in range(1, len(steps) + 1)}
    if len(rows) != len(expected) or len(recipes) != len(configs):
        raise ValueError("Incomplete or duplicate exact checkpoint population")
    result = {}
    for row in rows:
        stage = integer(row["stage"], minimum=1)
        key = (row["run_id"], stage)
        config = recipes.get(row["run_id"])
        if key not in expected or key in result or config is None:
            raise ValueError("Unknown or duplicate exact checkpoint identity")
        if (
            row["optimizer"] != config.optimizer.name
            or rational(row["learning_rate"]) != rational(config.optimizer.lr)
            or integer(row["step"], minimum=1) != steps[stage - 1]
            or row["progress_fraction"] != stage / len(steps)
        ):
            raise ValueError("Exact checkpoint identity/stage differs")
        total = integer(row["hidden_parameters"], minimum=1)
        values, coverage = {}, {}
        for kind in ("saved_segment", "cumulative"):
            prefix = f"exact_{kind}"
            counts = {
                status: integer(row[f"{prefix}_{status}_parameters"], minimum=0)
                for status in (
                    "zero",
                    "resolved",
                    "insufficient_signal_rank",
                    "unresolved_boundary",
                )
            }
            nonzero = integer(row[f"{prefix}_nonzero_parameters"], minimum=0)
            if (
                sum(counts.values()) != total
                or nonzero != total - counts["zero"]
                or bounded(row[f"{prefix}_nonzero_parameter_fraction"]) != nonzero / total
            ):
                raise ValueError("Exact nonzero/status denominator is inconsistent")
            coverage[kind] = {
                "hidden_parameters": total,
                "nonzero_parameters": nonzero,
                "nonzero_fraction": nonzero / total,
                **counts,
            }
        for name, (column, kind, _) in CHECKPOINT_FEATURES.items():
            value = row[column]
            if (value is None) != (coverage[kind]["nonzero_parameters"] == 0):
                raise ValueError("Exact spectral feature undefined status differs from denominator")
            values[name] = None if value is None else bounded(value)
        result[key] = {"values": values, "coverage": coverage}
    require_same(sorted(result), sorted(expected))
    return result


def pair_features(rows, configs, steps):
    recipes = {r.run_id: r for r in configs}
    if (
        len(recipes) != 12
        or any(
            sum(r.optimizer.name == name for r in configs) != 4
            for name in ("adamw", "muon", "normuon")
        )
        or len(steps) != 5
    ):
        raise ValueError("Exact comparator mapping requires the complete twelve-run primary grid")
    expected = {
        (tuple(sorted(pair)), stage, kind)
        for pair in itertools.combinations(recipes, 2)
        for stage in range(1, 6)
        for kind in ("saved_segment", "cumulative")
    }
    if len(rows) != len(expected):
        raise ValueError("Require all 660 exact pair rows")
    pairs = {}
    for row in rows:
        a, b, stage, kind = (
            row["first_run_id"],
            row["second_run_id"],
            integer(row["stage"], minimum=1),
            row["displacement_kind"],
        )
        key = (tuple(sorted((a, b))), stage, kind)
        if key not in expected or key in pairs:
            raise ValueError("Unknown/duplicate exact run-pair identity")
        for prefix, run_id in (("first", a), ("second", b)):
            config = recipes[run_id]
            if row[f"{prefix}_optimizer"] != config.optimizer.name or rational(
                row[f"{prefix}_learning_rate"]
            ) != rational(config.optimizer.lr):
                raise ValueError("Exact pair recipe identity differs")
        if (
            integer(row["step"], minimum=1) != steps[stage - 1]
            or row["progress_fraction"] != stage / 5
        ):
            raise ValueError("Exact pair retained stage differs")
        defined = integer(row["defined_parameters"], minimum=0)
        zero = integer(row["undefined_zero_parameters"], minimum=0)
        unsafe = integer(row["undefined_signal_rank_parameters"], minimum=0) + integer(
            row["undefined_boundary_parameters"], minimum=0
        )
        total = defined + zero + unsafe
        if (
            not total
            or bounded(row["defined_parameter_fraction"]) != defined / total
            or type(row["all_nonzero_pair_subspaces_resolved"]) is not bool
            or row["all_nonzero_pair_subspaces_resolved"] != (unsafe == 0)
        ):
            raise ValueError("Exact pair subspace coverage differs")
        value = row["mean_subspace_overlap"]
        if (value is None) != (unsafe > 0 or defined == 0):
            raise ValueError("A complete exact pair mean has an inconsistent undefined state")
        pairs[key] = {
            "value": None if value is None else bounded(value),
            "defined_parameters": defined,
            "zero_parameters": zero,
            "unsafe_parameters": unsafe,
            "total_parameters": total,
        }
    result = {}
    for run_id, config in recipes.items():
        others = sorted(
            r.run_id for r in configs if r.optimizer.name == "adamw" and r.run_id != run_id
        )
        assert len(others) == (3 if config.optimizer.name == "adamw" else 4)
        for stage in range(1, 6):
            values, coverage = {}, {}
            for feature, (kind, _) in PAIR_FEATURES.items():
                selected = [
                    pairs[(tuple(sorted((run_id, other))), stage, kind)] for other in others
                ]
                defined = all(r["value"] is not None for r in selected)
                values[feature] = (
                    math.fsum(r["value"] for r in selected) / len(selected) if defined else None
                )
                coverage[feature] = {
                    "comparators": others,
                    "complete_comparators": sum(r["value"] is not None for r in selected),
                    "required_comparators": len(selected),
                    "pair_coverage": selected,
                }
            result[(run_id, stage)] = {"values": values, "coverage": coverage}
    return result


def assemble_exact_panel(original, checkpoints, pairs, configs, steps):
    original = validate_panel(original)
    c, p = checkpoint_features(checkpoints, configs, steps), pair_features(pairs, configs, steps)
    expected = {(row["run_id"], row["stage"]) for row in original}
    if set(c) != expected or set(p) != expected:
        raise ValueError("Exact features and original bridge have different complete populations")
    output, comparisons, coverage = [], [], []
    for source in original:
        key = (source["run_id"], source["stage"])
        row = copy.deepcopy(source)
        if set(FEATURES) & set(row):
            raise ValueError(
                "Exact feature columns already exist; do not overwrite or relabel them"
            )
        row.update(c[key]["values"])
        row.update(p[key]["values"])
        output.append(row)
        for feature in FEATURES:
            old = (
                CHECKPOINT_FEATURES[feature][2]
                if feature in CHECKPOINT_FEATURES
                else PAIR_FEATURES[feature][1]
            )
            value = row[feature]
            relation = (
                "full_spectrum_and_nonzero_denominator"
                if "entropy" in feature
                else "exact_rank_and_nonzero_denominator"
                if feature in CHECKPOINT_FEATURES
                else "exact_projectors_complete_comparator_population"
            )
            comparisons.append(
                {
                    "run_id": key[0],
                    "stage": key[1],
                    "original_feature": old,
                    "exact_feature": feature,
                    "original_value": row[old],
                    "exact_value": value,
                    "exact_minus_original": None if value is None else value - row[old],
                    "relationship": relation,
                    "defined": value is not None,
                }
            )
        coverage.append(
            {
                "run_id": key[0],
                "stage": key[1],
                "checkpoint": c[key]["coverage"],
                "comparators": p[key]["coverage"],
            }
        )
    return output, comparisons, coverage
