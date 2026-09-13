"""Independent joins and previously accepted Torch/SymPy numerical references.

Inputs here are the full synthetic joint fixture. This oracle never trains or
encodes a model and never supplies a scientific optimizer conclusion.
"""

import math
from collections import defaultdict

from embed_optim.primary_contract import read_json
from embed_optim.primary_v3_bridge import TABLE_COUNTS as BRIDGE_COUNTS
from embed_optim.primary_v3_dimension_contract import TABLE_COUNTS as FEATURE_COUNTS
from embed_optim.primary_v3_dimension_inference import TABLE_COUNTS as INFERENCE_COUNTS
from embed_optim.primary_v3_exact_bridge import typed_csv
from scripts.audit_dense_v3_bridge import independent_full_systems
from scripts.dimension_inference_reference import close, exact_predictions, task_inference

DIRECT = {
    "saved_segment_stable_rank_fraction": "saved_segment_stable_rank_fraction_parameter_weighted",
    "saved_segment_sketch_effective_rank_fraction": "saved_segment_sketch_effective_rank_fraction_parameter_weighted",
    "saved_segment_row_norm_cv": "saved_segment_row_cv_parameter_weighted",
    "saved_segment_top_1pct_row_energy": "saved_segment_top_1pct_row_energy_parameter_weighted",
    "cumulative_displacement_to_weight_ratio": "cumulative_displacement_to_weight_ratio",
    "cumulative_stable_rank_fraction": "cumulative_stable_rank_fraction_parameter_weighted",
}
FUNCTIONAL = (
    "margin_helpful_mass_share",
    "margin_helpful_attribution_participation_ratio",
    "margin_degrading_attribution_mass",
)


def joins(original, functional, geometry, pairs, scores, features):
    """Direct identities/means, without calling production panel-assembly helpers."""

    def key(row):
        return row["run_id"], row["stage"]

    states = {key(row): row for row in geometry}
    assert len(states) == len(geometry) == len(original) == len(functional) == 60
    original_by_key = {key(row): row for row in original}
    assert set(states) == set(original_by_key) == {key(row) for row in functional}
    score_groups = defaultdict(list)
    for row in scores:
        score_groups[key(row)].append(row)
    assert len(score_groups) == 60 and sum(map(len, score_groups.values())) == 840
    checkpoints = {(r["run_id"], r["step"]): r for r in features["checkpoint_summary"]}
    retention = defaultdict(list)
    for row in features["random_removal"]:
        if row["removed_fraction"] == 0.5:
            retention[row["run_id"], row["step"]].append(row["relative_shortlist_ndcg"])
    for row in original:
        current = states[key(row)]
        task_scores = score_groups[key(row)]
        assert len(task_scores) == len({r["task"] for r in task_scores}) == 14
        assert all(
            r["optimizer"] == row["optimizer"] and r["learning_rate"] == row["learning_rate"]
            for r in task_scores
        )
        close(row["mean_ndcg_at_10"], math.fsum(r["ndcg_at_10"] for r in task_scores) / 14)
        close(
            row["log_saved_segment_to_weight_ratio"],
            math.log(current["saved_segment_to_weight_ratio"]),
        )
        for feature, source in DIRECT.items():
            assert row[feature] == current[source]
        for kind in ("saved_segment", "cumulative"):
            selected = []
            for pair in pairs:
                if pair["stage"] != row["stage"] or pair["displacement_kind"] != kind:
                    continue
                for focal, other in (("first", "second"), ("second", "first")):
                    if (
                        pair[focal + "_run_id"] == row["run_id"]
                        and pair[other + "_optimizer"] == "adamw"
                    ):
                        assert pair[other + "_run_id"] != row["run_id"]
                        selected.append(pair["mean_subspace_overlap"])
            assert len(selected) == (3 if row["optimizer"] == "adamw" else 4)
            close(
                row["mean_" + kind + "_subspace_overlap_to_adamw"],
                math.fsum(selected) / len(selected),
            )
    for row in functional:
        original_row = original_by_key[key(row)]
        assert {name: row[name] for name in original_row} == original_row
        state = states[key(row)]
        cell = row["run_id"], state["step"]
        for name in FUNCTIONAL:
            assert row[name] == checkpoints[cell][name]
        assert len(retention[cell]) == 280
        close(row["random_50pct_relative_shortlist_ndcg"], math.fsum(retention[cell]) / 280)
    return {
        "paired_states": 60,
        "geometry_feature_cells": 540,
        "functional_feature_cells": 240,
        "raw_beir_task_cells": 840,
    }


def audit(root):
    metadata = read_json(root / "manifest.json")["metadata"]
    assert (
        metadata["joint_fixture"]["all_upstream_measurements_and_primary_admissions_simulated"]
        is True
    )
    assert metadata["joint_fixture"]["vector_dimension"] == 768
    original = {name: typed_csv(root / "bridge" / (name + ".csv")) for name in BRIDGE_COUNTS}
    functional = {
        name: typed_csv(root / "inference" / (name + ".csv")) for name in INFERENCE_COUNTS
    }
    features = {name: typed_csv(root / "features" / (name + ".csv")) for name in FEATURE_COUNTS}
    alignment = joins(
        original["bridge_rows"],
        functional["bridge_rows"],
        typed_csv(root / "geometry/checkpoint_geometry.csv"),
        typed_csv(root / "geometry/run_pair_subspace_overlap.csv"),
        typed_csv(root / "outcomes/all_task_scores.csv"),
        features,
    )
    decisions = read_json(root / "inference/evidence.json")["decisions"]
    protocol = read_json(root / "source/configs/dense_dimension_utilization_protocol.json")
    return {
        "passed": True,
        "alignment": alignment,
        "original_bridge": independent_full_systems(original),
        "functional_bridge": exact_predictions(functional),
        "functional_task_family": task_inference(
            {"protocol": protocol, "tables": features}, functional, decisions
        ),
        "primary_admission_supplied": False,
        "scientific_completion": False,
    }
