from __future__ import annotations

import json
from pathlib import Path

import pytest

from embed_optim.local_global_reversal import (
    ALGORITHMS,
    FAMILIES,
    _declared_file,
    _frontier_rows,
    _reversal_rows,
    build_local_global_reversal,
)


def _final_rows():
    output = {}
    for family_index, family in enumerate(FAMILIES):
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            rows = []
            for lr_index, learning_rate in enumerate((1e-4, 3e-4, 1e-3, 3e-3), start=1):
                optimizer_gain = algorithm_index * 0.01
                rows.append(
                    {
                        "family": family,
                        "optimizer": algorithm,
                        "run_id": f"{algorithm}-lr{lr_index}",
                        "learning_rate": learning_rate,
                        "unseen_margin": 0.1
                        + family_index * 0.01
                        + optimizer_gain
                        + lr_index / 1000,
                        "top1_agreement": 0.95 - lr_index / 100,
                        "score_drift": lr_index / 100,
                        "beir": 0.5
                        + family_index * 0.01
                        + optimizer_gain
                        + (0.01 if lr_index == 2 else 0),
                    }
                )
            output[(family, algorithm)] = rows
    return output


def test_reversal_requires_local_loss_but_long_horizon_gain():
    local = {
        (family, algorithm): 0.003 - algorithm_index * 0.001
        for family in FAMILIES
        for algorithm_index, algorithm in enumerate(ALGORITHMS)
    }
    rows = _reversal_rows(local, _final_rows(), 0.001)

    assert len(rows) == 4
    assert all(row["local_margin_contrast"] < 0 for row in rows)
    assert all(row["final_unseen_margin_contrast"] > 0 for row in rows)
    assert all(row["final_beir_contrast"] > 0 for row in rows)
    assert all(row["local_global_reversal"] is True for row in rows)


def test_frontier_keeps_validation_selection_separate_from_beir_oracle():
    selection = {
        (family, algorithm): {
            "run_id": f"{algorithm}-lr4",
            "learning_rate": 3e-3,
            "validation_loss": 0.1,
            "validation_margin": 0.2,
        }
        for family in FAMILIES
        for algorithm in ALGORITHMS
    }
    rows = _frontier_rows(_final_rows(), selection)

    assert len(rows) == 6
    assert all(row["best_discovery_beir_lr"] == 3e-4 for row in rows)
    assert all(row["selection_matches_beir_oracle"] is False for row in rows)
    assert all(row["discovery_beir_regret"] > 0 for row in rows)
    assert all(row["score_drift_excess"] > 0 for row in rows)


def test_declared_file_rejects_changed_bytes(tmp_path: Path):
    table = tmp_path / "table.csv"
    table.write_text("a\n1\n", encoding="utf-8")
    from embed_optim.geometry import _sha256

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "outputs": {
                    "table": {
                        "path": "table.csv",
                        "bytes": table.stat().st_size,
                        "sha256": _sha256(table),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    table.write_text("a\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its manifest"):
        _declared_file(
            manifest,
            "table",
            required={"schema_version": 1, "complete": True},
        )


def test_declared_file_relocates_unavailable_absolute_producer_path(tmp_path: Path):
    table = tmp_path / "table.csv"
    table.write_text("a\n1\n", encoding="utf-8")
    from embed_optim.geometry import _sha256

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "outputs": {
                    "table": {
                        "path": "/unavailable/producer/checkout/table.csv",
                        "bytes": table.stat().st_size,
                        "sha256": _sha256(table),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    resolved, _ = _declared_file(
        manifest,
        "table",
        required={"schema_version": 1, "complete": True},
    )

    assert resolved == table.resolve()


def test_repository_reversal_report_is_reproducible(tmp_path: Path):
    manifest = build_local_global_reversal(
        Path("configs/local_global_reversal.json"),
        Path("reports/functional-intervention/manifest.json"),
        Path("reports/mechanism-bridge/summary_manifest.json"),
        Path("reports/recipe-validation/manifest.json"),
        tmp_path,
    )

    assert manifest["complete"] is True
    assert manifest["analysis_status"] == "post_hoc_exploratory"
    assert manifest["reversal_contrasts"] == 4
    assert manifest["reversal_contrasts_observed"] == 4
    assert manifest["selection_oracle_matches"] == 2
