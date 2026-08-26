from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.config import load_matrix
from embed_optim.geometry import _sha256
from embed_optim.representation_plot import (
    CHECKPOINT_FIELDS,
    REPRESENTATION_TABLE_FIELDS,
    TOKEN_FIELDS,
    plot_late_token_dynamics,
    plot_representation_dynamics,
)
from embed_optim.representation_summary import IDENTITY_FIELDS, SCORE_FIELDS


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _identity(
    family: str,
    kind: str,
    optimizer: str,
    learning_rate: float | str,
    run_id: str,
    stage: int,
) -> dict[str, object]:
    fraction = stage / 5 if stage else 0
    return {
        "family": family,
        "kind": kind,
        "seed": "",
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "run_id": run_id,
        "stage": stage,
        "fraction": fraction,
        "step": stage * 100,
        "label": f"{family}/{run_id}/stage-{stage}",
        "scorer": "cosine" if family == "dense" else "mean_maxsim_cosine",
    }


def _checkpoint_row(identity: dict[str, object]) -> dict[str, object]:
    stage = int(identity["stage"])
    family = str(identity["family"])
    optimizer_offset = {"": 0.0, "adamw": 0.0, "muon": 0.02, "normuon": 0.03}[
        str(identity["optimizer"])
    ]
    score = 0.2 + stage * 0.02 + optimizer_offset
    row = {
        **identity,
        "samples": 8,
        "candidates_per_sample": 8,
        "positive_score_mean": 0.8,
        "hardest_negative_score_mean": 0.8 - score,
        "margin_mean": score,
        "margin_median": score,
        "margin_std": 0.1,
        "top1_accuracy": 0.75,
        "mean_reciprocal_rank": 0.85,
        "mean_candidate_score_std": 0.2,
        "reference_top_k": "" if stage == 0 else 10,
        "reference_mean_top_k_overlap": "" if stage == 0 else 0.8,
        "reference_top1_agreement": "" if stage == 0 else 0.98 - stage * 0.05,
        "reference_score_drift_rms": "" if stage == 0 else stage * 0.01,
    }
    row.update({field: "" if family == "dense" else 0.7 - stage * 0.01 for field in TOKEN_FIELDS})
    return row


def _representation_rows(identity: dict[str, object]) -> list[dict[str, object]]:
    family = str(identity["family"])
    stage = int(identity["stage"])
    roles = (
        ("queries", "documents")
        if family == "dense"
        else ("query_tokens", "document_tokens", "pooled_queries", "pooled_documents")
    )
    output = []
    for role_index, role in enumerate(roles):
        rank = 0.45 + stage * 0.03 + role_index * 0.005
        output.append(
            {
                **identity,
                "representation_role": role,
                "original_vectors": 64,
                "analyzed_vectors": 64,
                "sampled": False,
                "dimension": 768 if family == "dense" else 128,
                "mean_norm": 1.0,
                "norm_cv": 0.1,
                "mean_pairwise_cosine": 0.05,
                "covariance_trace": 1.0,
                "entropy_effective_rank": rank * 100,
                "normalized_effective_rank": rank,
                "stable_rank": 10.0,
                "leading_variance_fraction": 0.2,
            }
        )
    return output


def _summary(tmp_path: Path, tier: str) -> Path:
    matrix = Path("configs/experiment.yaml").resolve()
    configs = load_matrix(matrix)
    root = tmp_path / tier
    summary = root / "summary"
    summary.mkdir(parents=True)
    probe = tmp_path / f"{tier}-probe"
    probe.mkdir()
    probe_manifest = probe / "manifest.json"
    probe_manifest.write_text(json.dumps({"count": 8, "task_counts": {"group": 8}}) + "\n")
    probe_spec = tmp_path / f"{tier}-spec.json"
    probe_spec.write_text(json.dumps({"tier": tier}) + "\n")

    identities = []
    for family in ("dense", "late"):
        identities.append(_identity(family, "reference", "", "", "pretrained", 0))
    for config in configs:
        for stage in range(1, 6):
            identities.append(
                _identity(
                    config.model_family,
                    "checkpoint",
                    config.optimizer.name,
                    config.optimizer.lr,
                    config.run_id,
                    stage,
                )
            )
    assert len(identities) == 122
    checkpoint_rows = [_checkpoint_row(identity) for identity in identities]
    representation_rows = [row for identity in identities for row in _representation_rows(identity)]
    group_rows = [
        {
            **{key: checkpoint_row[key] for key in [*IDENTITY_FIELDS, *SCORE_FIELDS]},
            "group": "group",
        }
        for checkpoint_row in checkpoint_rows
    ]
    outputs = {
        "checkpoint_metrics": (checkpoint_rows, CHECKPOINT_FIELDS),
        "representation_metrics": (representation_rows, REPRESENTATION_TABLE_FIELDS),
        "group_metrics": (group_rows, [*IDENTITY_FIELDS, "group", *SCORE_FIELDS]),
    }
    declarations = {}
    for name, (rows, fields) in outputs.items():
        path = summary / f"{name}.csv"
        _write_csv(path, rows, fields)
        declarations[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "rows": len(rows),
        }
    (summary / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "allow_partial": False,
                "expected_jobs": 122,
                "valid_jobs": 122,
                "missing_labels": [],
                "probe": {
                    "manifest_path": str(probe_manifest),
                    "manifest_sha256": _sha256(probe_manifest),
                    "spec_path": str(probe_spec),
                    "spec_sha256": _sha256(probe_spec),
                    "samples": 8,
                    "groups": {"group": 8},
                },
                "outputs": declarations,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return summary


def test_representation_plot_is_complete_and_deterministic(tmp_path: Path):
    training = _summary(tmp_path, "training")
    unseen = _summary(tmp_path, "unseen")
    output = tmp_path / "representation-dynamics.svg"

    first = plot_representation_dynamics(Path("configs/experiment.yaml"), training, unseen, output)
    first_bytes = output.read_bytes()
    second = plot_representation_dynamics(Path("configs/experiment.yaml"), training, unseen, output)

    assert first == second
    assert output.read_bytes() == first_bytes
    assert first["complete"] is True
    assert first["jobs"] == 244
    assert first["metrics"] == [
        "margin_mean",
        "normalized_effective_rank",
        "reference_top1_agreement",
    ]
    assert first_bytes.startswith(b"<?xml")
    sidecar = json.loads(output.with_suffix(".manifest.json").read_text())
    assert sidecar == first
    assert sidecar["output"]["sha256"] == _sha256(output)

    late_output = tmp_path / "late-token-dynamics.svg"
    late_first = plot_late_token_dynamics(
        Path("configs/experiment.yaml"), training, unseen, late_output
    )
    late_bytes = late_output.read_bytes()
    late_second = plot_late_token_dynamics(
        Path("configs/experiment.yaml"), training, unseen, late_output
    )
    assert late_first == late_second
    assert late_output.read_bytes() == late_bytes
    assert late_first["family"] == "late"
    assert late_first["jobs"] == 244
    assert late_first["metrics"] == [
        "token_evidence_entropy_mean",
        "token_evidence_gini_mean",
        "document_token_coverage_mean",
        "repeated_token_dominance_mean",
    ]
    assert json.loads(late_output.with_suffix(".manifest.json").read_text()) == late_first


def test_representation_plot_rejects_tampered_summary(tmp_path: Path):
    training = _summary(tmp_path, "training")
    unseen = _summary(tmp_path, "unseen")
    with (training / "checkpoint_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="differs from its representation summary manifest"):
        plot_representation_dynamics(
            Path("configs/experiment.yaml"),
            training,
            unseen,
            tmp_path / "representation-dynamics.svg",
        )
