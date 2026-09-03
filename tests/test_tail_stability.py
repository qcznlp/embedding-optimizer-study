from __future__ import annotations

from pathlib import Path

import pytest

from embed_optim.geometry import _sha256
from embed_optim.tail_stability import (
    ALGORITHMS,
    FAMILIES,
    SHORT_BRANCH_FIELDS,
    _midrank_percentiles,
    _quantile,
    _short_branch_tail_rows,
    discovery_cross_tail_summary,
    discovery_family_contrasts,
    load_tail_stability_protocol,
    short_branch_contrasts,
)


def test_frozen_tail_stability_protocol_is_self_consistent():
    path, protocol = load_tail_stability_protocol("configs/tail_stability_analysis.json")

    assert path.name == "tail_stability_analysis.json"
    assert protocol["analysis_status"] == (
        "post_hoc_discovery_with_prospective_short_branch_confirmation"
    )
    assert protocol["freeze_context"]["preliminary_tail_diagnostic_inspected"] is True
    assert protocol["freeze_context"]["short_branch_results_available"] is False
    assert protocol["secondary_discovery_diagnostic"]["status"] == (
        "post_hoc_after_cross_tail_inspection"
    )
    assert protocol["amendments"][0]["prospective_confirmation_rule_changed"] is False
    assert protocol["short_branch_confirmation"]["expected_checkpoints"] == 90
    root = path.parent.parent
    sources = protocol["source_inputs"]
    for path_key, digest_key in (
        ("functional_intervention_spec", "functional_intervention_spec_sha256"),
        ("functional_intervention_summary", "functional_intervention_summary_sha256"),
        ("local_global_reversal_protocol", "local_global_reversal_protocol_sha256"),
        ("local_global_reversal_summary", "local_global_reversal_summary_sha256"),
        ("short_branch_protocol", "short_branch_protocol_sha256"),
        ("short_branch_matrix_manifest", "short_branch_matrix_manifest_sha256"),
        ("validation_spec", "validation_spec_sha256"),
        ("unseen_probe_spec", "unseen_probe_spec_sha256"),
    ):
        assert _sha256(root / sources[path_key]) == sources[digest_key]


def test_linear_quantile_matches_frozen_definition():
    values = [4.0, 1.0, 3.0, 2.0]

    assert _quantile(values, 0.0) == 1.0
    assert _quantile(values, 0.5) == 2.5
    assert _quantile(values, 0.95) == pytest.approx(3.85)
    assert _quantile(values, 1.0) == 4.0
    with pytest.raises(ValueError, match="finite non-empty"):
        _quantile([], 0.5)


def test_percentile_midranks_are_tie_invariant():
    ranks = _midrank_percentiles({30: 3.0, 10: 1.0, 20: 1.0, 40: 4.0})

    assert ranks[10] == ranks[20] == pytest.approx(1 / 6)
    assert ranks[30] == pytest.approx(2 / 3)
    assert ranks[40] == 1.0


def _discovery_rows() -> list[dict]:
    rows = []
    for family in FAMILIES:
        for anchor_index in range(10):
            anchor = f"{family}/anchor-{anchor_index}"
            for algorithm in ALGORITHMS:
                if algorithm == "adamw":
                    mean_margin, p05_margin, p95_loss, p99_loss = 0.002, -0.02, 0.10, 0.20
                elif algorithm == "muon":
                    mean_margin, p05_margin, p95_loss, p99_loss = 0.001, -0.01, 0.05, 0.10
                else:
                    mean_margin, p05_margin, p95_loss, p99_loss = 0.0005, -0.005, 0.04, 0.08
                rows.append(
                    {
                        "family": family,
                        "anchor": anchor,
                        "algorithm": algorithm,
                        "mean_delta_positive_margin": mean_margin,
                        "p05_delta_positive_margin": p05_margin,
                        "p95_delta_contrastive_loss": p95_loss,
                        "p99_delta_contrastive_loss": p99_loss,
                    }
                )
    return rows


def test_discovery_contrast_detects_mean_tail_tradeoff_and_leave_one_out_stability():
    contrasts = discovery_family_contrasts(_discovery_rows())

    assert len(contrasts) == 4
    assert all(row["mean_tail_tradeoff_observed"] is True for row in contrasts)
    assert all(row["mean_margin_challenger_losses"] == 10 for row in contrasts)
    assert all(row["p05_margin_challenger_wins"] == 10 for row in contrasts)
    assert all(row["p99_loss_challenger_wins"] == 10 for row in contrasts)
    assert all(row["p99_loss_leave_one_out_negative_fraction"] == 1.0 for row in contrasts)


def _cross_tail_rows() -> list[dict]:
    rows = []
    for family in FAMILIES:
        for challenger in ("muon", "normuon"):
            for anchor_index in range(10):
                rows.append(
                    {
                        "family": family,
                        "anchor": f"{family}/anchor-{anchor_index}",
                        "challenger": challenger,
                        "tail_fraction": 0.05,
                        "tail_size": 12,
                        "challenger_minus_adam_on_adam_tail_loss_mean": -0.1,
                        "challenger_minus_adam_on_challenger_tail_loss_mean": (
                            0.02 if family == "dense" else -0.05
                        ),
                        "adam_tail_challenger_win_fraction": 0.9,
                        "challenger_tail_challenger_win_fraction": (
                            0.4 if family == "dense" else 0.8
                        ),
                        "tail_intersection": 4 if family == "dense" else 10,
                        "tail_jaccard": 0.2 if family == "dense" else 10 / 14,
                        "adam_tail_baseline_margin_percentile_median": (
                            0.2 if family == "dense" else 0.04
                        ),
                    }
                )
    return rows


def test_cross_tail_summary_distinguishes_redistribution_from_severity_suppression():
    rows = discovery_cross_tail_summary(_cross_tail_rows())

    assert len(rows) == 4
    by_identity = {(row["family"], row["challenger"]): row for row in rows}
    for challenger in ("muon", "normuon"):
        dense = by_identity[("dense", challenger)]
        late = by_identity[("late", challenger)]
        assert dense["tail_identity_regime"] == "tail redistribution"
        assert dense["dual_selected_tail_advantage"] is False
        assert late["tail_identity_regime"] == "shared-tail severity suppression"
        assert late["dual_selected_tail_advantage"] is True
        assert late["challenger_tail_leave_one_out_negative_fraction"] == 1.0


def _short_branch_rows() -> list[dict]:
    rows = []
    seeds = (314159, 271828, 161803)
    for family in FAMILIES:
        for seed in seeds:
            for stage in range(1, 6):
                for operator in ALGORITHMS:
                    loss_offset = -0.1 if operator in {"muon", "normuon"} else 0.0
                    unseen_tail_offset = (
                        0.05 if operator == "muon" else -0.05 if operator == "normuon" else 0.0
                    )
                    row = {
                        "family": family,
                        "seed": seed,
                        "operator": operator,
                        "run_id": f"{operator}-scale-matched",
                        "stage": stage,
                        "fraction": stage / 5,
                        "step": stage * 10,
                        "validation_samples": 4096,
                        "validation_loss_mean": 1.0 + loss_offset,
                        "validation_loss_p95": 2.0 + loss_offset,
                        "validation_loss_p99": 3.0 + loss_offset,
                        "validation_margin_mean": 0.1,
                        "validation_margin_p05": -0.2,
                        "unseen_samples": 224,
                        "unseen_margin_mean": 0.2,
                        "unseen_margin_p05": -0.3 + unseen_tail_offset,
                        "unseen_pretrained_top1_agreement": 0.9,
                        "unseen_pretrained_score_drift_rms": 0.1,
                    }
                    assert list(row) == SHORT_BRANCH_FIELDS
                    rows.append(row)
    return rows


def test_short_branch_confirmation_applies_frozen_joint_support_rule():
    contrasts, final = short_branch_contrasts(_short_branch_rows())

    assert len(contrasts) == 60
    assert len(final) == 4
    decisions = {(row["family"], row["challenger"]): row for row in final}
    for family in FAMILIES:
        assert decisions[(family, "muon")]["tail_stability_decision"] == "supported"
        assert decisions[(family, "muon")]["validation_loss_p95_seed_wins"] == 3
        assert decisions[(family, "muon")]["unseen_margin_p05_seed_wins"] == 3
        assert decisions[(family, "normuon")]["tail_stability_decision"] == "mixed"


def test_complete_short_branch_manifest_corruption_fails_closed(tmp_path: Path):
    protocol_path = tmp_path / "configs" / "tail.json"
    protocol_path.parent.mkdir()
    report_manifest = tmp_path / "reports" / "short-branch" / "summary_manifest.json"
    report_manifest.parent.mkdir(parents=True)
    report_manifest.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError):
        _short_branch_tail_rows(
            protocol_path,
            {"source_inputs": {}, "short_branch_confirmation": {}},
            experiment_matrix=Path("unused"),
            matrix_dir=None,
            results_root=tmp_path / "results",
        )
