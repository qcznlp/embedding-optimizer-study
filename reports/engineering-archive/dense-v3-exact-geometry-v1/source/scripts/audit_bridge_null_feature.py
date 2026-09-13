"""Synthetic diagnostic of the unchanged bridge's baseline-span numerical boundary.

Two fixed cases only, no search over outcomes, seeds or support thresholds. This
records old behavior; it does not repair or release the bridge implementation.
"""

import argparse
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embed_optim.corrected_retrieval_bridge import (
    FEATURES,
    _baseline_design,
    evaluate_bridge_features,
)
from embed_optim.primary_contract import file_identity, verify_file
from embed_optim.primary_v3_validation_io import write_new


def fixture(bounded):
    rows = []
    for optimizer in ("adamw", "muon", "normuon"):
        rates = [1e-6, 3e-6, 1e-5, 3e-5] if optimizer == "adamw" else [1e-4, 3e-4, 1e-3, 3e-3]
        center = float(np.mean([math.log10(rate) for rate in rates]))
        for dose, rate in enumerate(rates, start=1):
            z = math.log10(rate) - center
            for stage in range(1, 6):
                outcome = (
                    0.5 + stage / 64 + (optimizer == "muon") / 32 - (optimizer == "normuon") / 64
                    if bounded
                    else 0.4
                    + 0.013 * stage
                    + 0.05 * (optimizer == "muon")
                    - 0.03 * (optimizer == "normuon")
                    + 0.019 * z
                )
                rows.append(
                    {
                        "run_id": f"synthetic-{optimizer}-{dose}",
                        "optimizer": optimizer,
                        "learning_rate": rate,
                        "dose_index": dose,
                        "centered_log10_learning_rate": z,
                        "stage": stage,
                        "progress_fraction": stage / 5,
                        "mean_ndcg_at_10": outcome,
                        **{feature: stage / 8 if bounded else float(stage) for feature in FEATURES},
                    }
                )
    return rows


def run_case(bounded):
    rows = fixture(bounded)
    baseline = _baseline_design(rows)
    values = np.array([row[FEATURES[0]] for row in rows])
    # Stage = intercept + 1*stage2 + 2*stage3 + 3*stage4 + 4*stage5.
    coefficients = np.array([1.0, 0, 0, 1, 2, 3, 4, 0]) / (8 if bounded else 1)
    assert np.array_equal(baseline @ coefficients, values)
    outcome = np.array([row["mean_ndcg_at_10"] for row in rows])
    outcome_coefficients = (
        np.array([0.5 + 1 / 64, 1 / 32, -1 / 64, 1 / 64, 2 / 64, 3 / 64, 4 / 64, 0])
        if bounded
        else None
    )
    if bounded:
        assert np.array_equal(baseline @ outcome_coefficients, outcome)
        assert all(0 <= row[feature] <= 1 for row in rows for feature in FEATURES)
    folds, summaries, associations = evaluate_bridge_features(rows)
    return {
        "case": "bounded_dyadic_null" if bounded else "initial_stage_float_outcome",
        "rows": rows,
        "baseline_rank": int(np.linalg.matrix_rank(baseline)),
        "augmented_rank": int(np.linalg.matrix_rank(np.column_stack([baseline, values]))),
        "closed_form_feature_coefficients": coefficients.tolist(),
        "closed_form_outcome_coefficients": outcome_coefficients.tolist() if bounded else None,
        "feature_exactly_in_baseline_span": True,
        "outcome_exactly_in_baseline_span": True if bounded else None,
        "expected_feature_residual": "Exactly zero; residual correlations are mathematically undefined",
        "legacy_folds": folds,
        "legacy_summaries": summaries,
        "legacy_associations": associations,
        "legacy_predicted_useful_count": sum(row["predictively_useful"] for row in summaries),
        "legacy_finite_residual_associations": sum(
            row["pearson_residual_association"] is not None
            or row["spearman_residual_association"] is not None
            for row in associations
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require a new CPU-only synthetic diagnostic output")
    source = args.repository / "src/embed_optim/corrected_retrieval_bridge.py"
    bound = file_identity(source)
    assert bound["sha256"] == "347331a3c35d12a95188c0927a327b06bd30dd4ffa11b161c432117b4299eb10"
    cases = [run_case(False), run_case(True)]
    verify_file(source, bound)
    result = {
        "scope": "engineering_synthetic_bridge_null_feature_diagnosis",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
        "legacy_source": {"path": str(source), **bound},
        "numpy_version": np.__version__,
        "cases": cases,
        "bridge_correctness_accepted": False,
        "bridge_source_changed": False,
        "scientific_completion": False,
        "boundary": "Two mathematical-null synthetic panels, not retrieval observations or a repaired bridge. The initial panel preserves the first read-only probe; the bounded dyadic panel also has an exactly representable outcome in the baseline span.",
    }
    write_new(args.output, result)
    print(
        {
            "sha256": file_identity(args.output)["sha256"],
            "cases": [
                {
                    "case": case["case"],
                    "legacy_predicted_useful_count": case["legacy_predicted_useful_count"],
                    "finite_residual_associations": case["legacy_finite_residual_associations"],
                    "first_summary": case["legacy_summaries"][0],
                    "first_association": case["legacy_associations"][0],
                }
                for case in cases
            ],
        }
    )


if __name__ == "__main__":
    main()
