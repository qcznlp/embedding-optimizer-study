from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.config import load_matrix
from embed_optim.mechanism_bridge import (
    EVALUATION_FIELDS,
    _expected_checkpoints,
    build_mechanism_bridge,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_reports(tmp_path: Path, *, complete: bool = True) -> Path:
    reports = tmp_path / "reports"
    reports.mkdir()
    expected = _expected_checkpoints(load_matrix(Path("configs/experiment.yaml")))
    rows = []
    for metadata in expected.values():
        optimizer_offset = {"adamw": 0.0, "muon": 0.01, "normuon": 0.015}[metadata["optimizer"]]
        rows.append(
            {
                "model_family": metadata["model_family"],
                "optimizer": metadata["optimizer"],
                "learning_rate": metadata["learning_rate"],
                "run_id": metadata["run_id"],
                "stage": metadata["stage"],
                "fraction": metadata["fraction"],
                "checkpoint_step": metadata["step"],
                "mean_ndcg_at_10": 0.2 + metadata["stage"] * 0.01 + optimizer_offset,
                "tasks_completed": 14,
            }
        )
    _write_csv(reports / "checkpoint_summary.csv", rows, EVALUATION_FIELDS)
    valid = 1680 if complete else 1679
    (reports / "coverage.json").write_text(
        json.dumps(
            {
                "complete": complete,
                "evaluation_complete": complete,
                "expected_results": 1680,
                "observed_results": valid,
                "expected_checkpoint_summaries": 120,
                "observed_checkpoint_summaries": 120,
                "missing": [] if complete else ["missing"],
                "unexpected": [],
            }
        )
        + "\n"
    )
    return reports


def _representation_tier(tier: str):
    expected = _expected_checkpoints(load_matrix(Path("configs/experiment.yaml")))
    checkpoint_rows = []
    representation_rows = []
    for metadata in expected.values():
        family = metadata["model_family"]
        stage = metadata["stage"]
        checkpoint = {
            "family": family,
            "kind": "checkpoint",
            "optimizer": metadata["optimizer"],
            "learning_rate": str(metadata["learning_rate"]),
            "run_id": metadata["run_id"],
            "stage": str(stage),
            "fraction": str(metadata["fraction"]),
            "step": str(metadata["step"]),
            "margin_mean": str(0.1 + stage * 0.02),
            "margin_median": str(0.1 + stage * 0.02),
            "top1_accuracy": "0.7",
            "mean_reciprocal_rank": "0.8",
            "reference_mean_top_k_overlap": str(0.95 - stage * 0.03),
            "reference_top1_agreement": str(0.96 - stage * 0.04),
            "reference_score_drift_rms": str(stage * 0.01),
            "token_evidence_entropy_mean": "0.7" if family == "late" else "",
            "token_evidence_gini_mean": "0.2" if family == "late" else "",
            "document_token_coverage_mean": "0.6" if family == "late" else "",
            "repeated_token_dominance_mean": "0.3" if family == "late" else "",
        }
        checkpoint_rows.append(checkpoint)
        roles = (
            ("queries", "documents")
            if family == "dense"
            else (
                "query_tokens",
                "document_tokens",
            )
        )
        for index, role in enumerate(roles):
            representation_rows.append(
                {
                    **checkpoint,
                    "representation_role": role,
                    "normalized_effective_rank": str(0.4 + stage * 0.03 + index * 0.01),
                }
            )
    return (
        checkpoint_rows,
        representation_rows,
        {
            "summary_manifest": {"path": tier, "sha256": tier * 4},
            "probe": {"spec_sha256": f"{tier}-spec"},
        },
    )


def test_mechanism_bridge_strictly_joins_all_checkpoint_sources(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "embed_optim.mechanism_bridge._declared_checkpoint_steps",
        lambda config: [782, 1563, 2345, 3126, 3907],
    )
    training = _representation_tier("training")
    unseen = _representation_tier("unseen")

    def fake_summary(path, configs):
        del configs
        return training if Path(path).name == "training" else unseen

    monkeypatch.setattr("embed_optim.mechanism_bridge.load_representation_summary", fake_summary)
    output = tmp_path / "bridge"
    reports = _evaluation_reports(tmp_path)
    manifest = build_mechanism_bridge(
        Path("configs/experiment.yaml"),
        Path("reports/weight-space"),
        Path("training"),
        Path("unseen"),
        reports,
        output,
    )
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    repeated = build_mechanism_bridge(
        Path("configs/experiment.yaml"),
        Path("reports/weight-space"),
        Path("training"),
        Path("unseen"),
        reports,
        output,
    )

    assert manifest["complete"] is True
    assert repeated == manifest
    assert {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    } == first_bytes
    assert manifest["checkpoints"] == 120
    assert manifest["within_run_transitions"] == 96
    assert manifest["correlations"] == 216
    assert set(manifest["sources"]) == {
        "weight_space",
        "training_dynamics",
        "representation",
        "evaluation",
        "loss_retrieval_protocol",
    }
    with (output / "checkpoint_bridge.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 120
    assert {row["model_family"] for row in rows} == {"dense", "late"}
    assert all(float(row["mean_training_loss"]) >= 0 for row in rows)
    with (output / "within_run_changes.csv").open(newline="") as handle:
        changes = list(csv.DictReader(handle))
    assert len(changes) == 96
    assert changes[0]["delta_previous_checkpoint_displacement_frobenius_norm"] == ""
    assert json.loads((output / "summary_manifest.json").read_text()) == manifest


def test_mechanism_bridge_rejects_partial_beir_coverage(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "embed_optim.mechanism_bridge._declared_checkpoint_steps",
        lambda config: [782, 1563, 2345, 3126, 3907],
    )
    with pytest.raises(ValueError, match="not strictly complete"):
        build_mechanism_bridge(
            Path("configs/experiment.yaml"),
            Path("reports/weight-space"),
            Path("training"),
            Path("unseen"),
            _evaluation_reports(tmp_path, complete=False),
            tmp_path / "bridge",
        )
