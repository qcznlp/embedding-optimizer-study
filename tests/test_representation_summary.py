import csv
import json
from pathlib import Path

import pytest

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.probe_matrix import ProbeJob
from embed_optim.representation_summary import (
    ExpectedMetric,
    expected_probe_metrics,
    summarize_probe_metrics,
)


def _config(tmp_path: Path) -> RunConfig:
    config = RunConfig(
        run_id="adamw-fixture",
        model_family="dense",
        optimizer=OptimizerConfig(name="adamw", lr=1e-5),
        model_name="fixture/dense",
        dataset_path="fixture",
        output_root=str(tmp_path / "outputs"),
    )
    config.output_dir.mkdir(parents=True)
    steps = [2, 4, 6, 8, 10]
    (config.output_dir / "checkpoint_schedule.json").write_text(json.dumps({"steps": steps}))
    for step in steps:
        (config.output_dir / f"checkpoint-{step}").mkdir()
    return config


def _quantiles(value: float) -> dict[str, float]:
    return {
        "min": value,
        "p05": value,
        "p25": value,
        "median": value,
        "p75": value,
        "p95": value,
        "max": value,
        "mean": value,
        "std": 0.0,
    }


def _representation() -> dict[str, object]:
    return {
        "original_vectors": 2,
        "analyzed_vectors": 2,
        "sampled": False,
        "dimension": 2,
        "mean_norm": 1.0,
        "norm_cv": 0.0,
        "mean_pairwise_cosine": 0.0,
        "covariance_trace": 1.0,
        "entropy_effective_rank": 1.0,
        "normalized_effective_rank": 1.0,
        "stable_rank": 1.0,
        "leading_variance_fraction": 1.0,
    }


def _score(samples: int = 2) -> dict[str, object]:
    return {
        "samples": samples,
        "candidates_per_sample": 8,
        "positive_score": _quantiles(0.8),
        "hardest_negative_score": _quantiles(0.4),
        "positive_hardest_negative_margin": _quantiles(0.4),
        "top1_accuracy": 1.0,
        "mean_reciprocal_rank": 1.0,
        "mean_candidate_score_std": 0.2,
    }


def _payload(label: str) -> dict[str, object]:
    score = _score()
    score["by_group"] = {"a": _score(1), "b": _score(1)}
    return {
        "schema_version": 1,
        "family": "dense",
        "label": label,
        "input": {},
        "parameters": {"require_export_manifest": True, "positive_candidate_index": 0},
        "metrics": {
            "scorer": "cosine",
            "score_geometry": score,
            "representations": {
                "queries": _representation(),
                "documents": _representation(),
            },
        },
    }


def test_expected_probe_metrics_covers_reference_and_five_checkpoints(tmp_path: Path):
    config = _config(tmp_path)
    expected = expected_probe_metrics([config], tmp_path / "representation", ("manifest", "spec"))
    assert len(expected) == 6
    assert expected[0].job.label == "dense/pretrained"
    assert expected[0].stage == 0
    assert [item.stage for item in expected[1:]] == [1, 2, 3, 4, 5]
    assert [item.fraction for item in expected[1:]] == list(config.checkpoint_fractions)
    assert {item.job.probe_manifest_sha256 for item in expected} == {"manifest"}
    assert {item.job.probe_spec_sha256 for item in expected} == {"spec"}


def test_summary_writes_strict_checkpoint_representation_and_group_tables(
    tmp_path: Path, monkeypatch
):
    result_root = tmp_path / "results"
    metrics = result_root / "metrics" / "dense" / "pretrained.json"
    export = result_root / "exports" / "dense" / "pretrained.npz"
    metrics.parent.mkdir(parents=True)
    export.parent.mkdir(parents=True)
    export.write_bytes(b"fixture export")
    job = ProbeJob(
        kind="reference",
        family="dense",
        label="dense/pretrained",
        checkpoint=tmp_path,
        export=export,
        metrics=metrics,
        reference_export=None,
    )
    metrics.write_text(json.dumps(_payload(job.label)) + "\n")
    expected = [
        ExpectedMetric(
            job=job,
            optimizer="",
            learning_rate="",
            run_id="pretrained",
            stage=0,
            fraction=0.0,
            step=0,
        )
    ]
    probe = tmp_path / "probe"
    probe.mkdir()
    probe_manifest = probe / "manifest.json"
    probe_manifest.write_text(json.dumps({"count": 2, "task_counts": {"a": 1, "b": 1}}))
    probe_spec = tmp_path / "probe-spec.json"
    probe_spec.write_text("{}\n")
    monkeypatch.setattr("embed_optim.representation_summary.probe_job_complete", lambda job: True)

    manifest = summarize_probe_metrics(
        expected,
        tmp_path / "summary",
        probe_manifest_path=probe_manifest,
        probe_spec_path=probe_spec,
    )
    assert manifest["complete"] is True
    assert manifest["valid_jobs"] == 1
    assert manifest["outputs"]["checkpoint_metrics"]["rows"] == 1
    assert manifest["outputs"]["representation_metrics"]["rows"] == 2
    assert manifest["outputs"]["group_metrics"]["rows"] == 2
    with (tmp_path / "summary" / "checkpoint_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["margin_mean"] == "0.4"
    assert rows[0]["run_id"] == "pretrained"

    rogue = result_root / "metrics" / "rogue.json"
    rogue.write_text("{}\n")
    with pytest.raises(ValueError, match="Unexpected representation metric"):
        summarize_probe_metrics(
            expected,
            tmp_path / "rejected",
            probe_manifest_path=probe_manifest,
            probe_spec_path=probe_spec,
        )


def test_summary_rejects_incomplete_matrix(tmp_path: Path, monkeypatch):
    job = ProbeJob(
        kind="reference",
        family="dense",
        label="dense/pretrained",
        checkpoint=tmp_path,
        export=tmp_path / "exports" / "dense" / "pretrained.npz",
        metrics=tmp_path / "metrics" / "dense" / "pretrained.json",
        reference_export=None,
    )
    expected = [ExpectedMetric(job, "", "", "pretrained", 0, 0.0, 0)]
    probe_manifest = tmp_path / "manifest.json"
    probe_manifest.write_text(json.dumps({"count": 2, "quotas": {"a": 1, "b": 1}}))
    probe_spec = tmp_path / "spec.json"
    probe_spec.write_text("{}\n")
    monkeypatch.setattr("embed_optim.representation_summary.probe_job_complete", lambda job: False)
    with pytest.raises(ValueError, match="matrix is incomplete"):
        summarize_probe_metrics(
            expected,
            tmp_path / "summary",
            probe_manifest_path=probe_manifest,
            probe_spec_path=probe_spec,
        )
