from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.geometry import _sha256
from embed_optim.validation_data import load_validation_spec
from embed_optim.validation_evaluation import METRICS, _group_summaries
from embed_optim.validation_matrix import (
    _job_cli,
    build_validation_jobs,
    parse_args,
    validation_job_complete,
)
from embed_optim.validation_summary import _paired_lr_effects, select_recipes

RATES = {
    "adamw": (1e-6, 3e-6, 1e-5, 3e-5),
    "muon": (1e-4, 3e-4, 1e-3, 3e-3),
    "normuon": (1e-4, 3e-4, 1e-3, 3e-3),
}


def _configs(tmp_path: Path) -> list[RunConfig]:
    configs = []
    for family in ("dense", "late"):
        for optimizer, rates in RATES.items():
            for index, rate in enumerate(rates):
                config = RunConfig(
                    run_id=f"{optimizer}-rate-{index}",
                    model_family=family,
                    optimizer=OptimizerConfig(name=optimizer, lr=rate),
                    model_name=f"fixture/{family}",
                    model_revision="revision",
                    dataset_path="fixture",
                    output_root=str(tmp_path / "outputs"),
                )
                config.output_dir.mkdir(parents=True)
                steps = [2, 4, 6, 8, 10]
                (config.output_dir / "checkpoint_schedule.json").write_text(
                    json.dumps({"fractions": [0.2, 0.4, 0.6, 0.8, 1.0], "steps": steps})
                )
                for step in steps:
                    (config.output_dir / f"checkpoint-{step}").mkdir()
                configs.append(config)
    return configs


def test_group_summary_preserves_overall_and_sources():
    records = []
    for sample_id, group, loss in ((1, "fiqa", 1.0), (2, "nq", 3.0)):
        records.append(
            {
                "sample_id": sample_id,
                "group": group,
                **{metric: loss for metric in METRICS},
            }
        )
    summaries = _group_summaries(records)

    assert [row["group"] for row in summaries] == ["__all__", "fiqa", "nq"]
    assert summaries[0]["samples"] == 2
    assert summaries[0]["contrastive_loss"] == 2.0


def test_final_validation_jobs_cover_exact_main_matrix(tmp_path: Path):
    configs = _configs(tmp_path)
    jobs = build_validation_jobs(configs, tmp_path / "validation")

    assert len(jobs) == 24
    assert len({job.label for job in jobs}) == 24
    assert all(job.checkpoint.name == "checkpoint-10" for job in jobs)
    assert all(job.output_dir == (tmp_path / "validation" / job.label) for job in jobs)


def test_recipe_selection_uses_loss_then_margin_then_lower_lr(tmp_path: Path):
    jobs = build_validation_jobs(_configs(tmp_path), tmp_path / "validation")
    rows = []
    for job in jobs:
        index = int(job.config.run_id.rsplit("-", 1)[1])
        loss = 0.5 if index in {1, 2} else 1.0
        margin = 0.3 if index == 2 else 0.2
        if job.config.optimizer.name == "adamw":
            loss = 0.5
            margin = 0.2
        rows.append(
            {
                "family": job.config.model_family,
                "run_id": job.config.run_id,
                "contrastive_loss": loss,
                "positive_margin": margin,
            }
        )
    winners, selected = select_recipes(jobs, rows)

    assert len(selected) == 6
    assert all(
        winners[(family, "adamw")].config.optimizer.lr == 1e-6 for family in ("dense", "late")
    )
    assert all(
        winners[(family, optimizer)].config.run_id.endswith("-2")
        for family in ("dense", "late")
        for optimizer in ("muon", "normuon")
    )


def test_paired_lr_effects_preserve_sample_pairing(tmp_path: Path):
    jobs = build_validation_jobs(_configs(tmp_path), tmp_path / "validation")
    rows = [
        {
            "family": job.config.model_family,
            "run_id": job.config.run_id,
            "contrastive_loss": job.config.optimizer.lr,
            "positive_margin": -job.config.optimizer.lr,
        }
        for job in jobs
    ]
    winners, _ = select_recipes(jobs, rows)
    samples = {}
    for job in jobs:
        samples[job.label] = {
            sample_id: {metric: float(job.config.optimizer.lr + sample_id) for metric in METRICS}
            for sample_id in (10, 20)
        }
    effects = _paired_lr_effects(jobs, samples, winners)

    assert len(effects) == 18
    assert all(row["samples"] == 2 for row in effects)
    assert all(row["candidate_minus_selected_contrastive_loss"] > 0 for row in effects)


def _declared(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_validation_completion_and_worker_command(tmp_path: Path):
    jobs = build_validation_jobs(_configs(tmp_path), tmp_path / "validation")
    job = jobs[0]
    job.output_dir.mkdir(parents=True)
    sample = job.output_dir / "sample_metrics.jsonl"
    group = job.output_dir / "group_metrics.jsonl"
    sample.write_text("sample\n", encoding="utf-8")
    group.write_text("group\n", encoding="utf-8")
    spec = Path("configs/validation_probe.json").resolve()
    _, frozen = load_validation_spec(spec)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "family": job.config.model_family,
        "checkpoint": {"path": str(job.checkpoint)},
        "validation_spec": {"sha256": _sha256(spec)},
        "sample_records": frozen["evaluation"]["expected_sample_records_per_job"],
        "group_records": 8,
        "outputs": {
            "sample_metrics": _declared(sample, job.output_dir),
            "group_metrics": _declared(group, job.output_dir),
        },
    }
    (job.output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert validation_job_complete(job, spec, verify_hashes=True)
    command = _job_cli(
        job,
        Namespace(
            probe=tmp_path / "probe",
            validation_spec=spec,
        ),
    )
    assert command[:3] == [sys.executable, "-m", "embed_optim.validation_matrix"]
    parsed = parse_args(command[3:])
    assert parsed.worker is True
    assert parsed.label == job.label
    sample.write_text("changed\n", encoding="utf-8")
    assert not validation_job_complete(job, spec)


def test_recipe_selection_rejects_missing_run(tmp_path: Path):
    jobs = build_validation_jobs(_configs(tmp_path), tmp_path / "validation")
    rows = [
        {
            "family": job.config.model_family,
            "run_id": job.config.run_id,
            "contrastive_loss": 1.0,
            "positive_margin": 0.0,
        }
        for job in jobs[:-1]
    ]
    with pytest.raises(ValueError, match="exact job matrix"):
        select_recipes(jobs, rows)
