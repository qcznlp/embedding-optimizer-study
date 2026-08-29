from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.short_branch_evaluation import (
    audit_short_branch_training,
    build_short_branch_probe_jobs,
    build_short_branch_validation_jobs,
    parse_args,
)


def _configs(tmp_path: Path) -> dict[int, list[RunConfig]]:
    result = {}
    for seed in (314159, 271828, 161803):
        runs = []
        for family in ("dense", "late"):
            for algorithm in ("adamw", "muon", "normuon"):
                run_id = f"{algorithm}-scale-matched"
                config = RunConfig(
                    run_id=run_id,
                    model_family=family,
                    optimizer=OptimizerConfig(
                        name="hybrid_adamw" if algorithm == "adamw" else algorithm,
                        lr=1e-4,
                    ),
                    model_name="unused",
                    dataset_path="unused",
                    output_root=str(tmp_path / "outputs" / f"seed{seed}"),
                    seed=seed,
                )
                config.output_dir.mkdir(parents=True)
                steps = [10, 20, 30, 40, 50]
                (config.output_dir / "checkpoint_schedule.json").write_text(
                    json.dumps(
                        {
                            "steps": steps,
                            "fractions": [0.2, 0.4, 0.6, 0.8, 1.0],
                        }
                    ),
                    encoding="utf-8",
                )
                for step in steps:
                    (config.output_dir / f"checkpoint-{step}").mkdir()
                runs.append(config)
        result[seed] = runs
    return result


def test_short_branch_validation_covers_all_seed_checkpoint_pairs(tmp_path: Path):
    jobs = build_short_branch_validation_jobs(_configs(tmp_path), tmp_path / "validation")

    assert len(jobs) == 90
    assert len({job.label for job in jobs}) == 90
    assert {job.seed for job in jobs} == {314159, 271828, 161803}
    assert {job.step for job in jobs} == {10, 20, 30, 40, 50}
    assert all(f"seed{job.seed}" in str(job.output_dir) for job in jobs)


def test_short_branch_unseen_probe_adds_two_shared_references(tmp_path: Path):
    references = {"dense": tmp_path / "dense-base", "late": tmp_path / "late-base"}
    jobs = build_short_branch_probe_jobs(
        _configs(tmp_path),
        references,
        tmp_path / "unseen",
        ("probe-manifest", "probe-spec"),
    )

    reference_jobs = [job for job in jobs if job.kind == "reference"]
    checkpoint_jobs = [job for job in jobs if job.kind == "checkpoint"]
    assert len(jobs) == 92
    assert len(reference_jobs) == 2
    assert len(checkpoint_jobs) == 90
    assert len({job.label for job in jobs}) == 92
    assert all(job.probe_manifest_sha256 == "probe-manifest" for job in jobs)
    assert all(job.probe_spec_sha256 == "probe-spec" for job in jobs)
    assert all(job.reference_export is not None for job in checkpoint_jobs)


def test_short_branch_training_audit_covers_all_runs_and_checkpoints(tmp_path, monkeypatch):
    seeds = (314159, 271828, 161803)
    configs = {
        seed: [SimpleNamespace(run_id=f"run-{index}") for index in range(6)] for seed in seeds
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    protocol = {"training": {"order_seeds": list(seeds), "expected_runs": 18}}
    monkeypatch.setattr(
        "embed_optim.short_branch_evaluation.load_short_branch_protocol",
        lambda _path: (protocol_path, protocol),
    )
    dataset = {
        "rows": 50_000,
        "manifest_sha256": "a" * 64,
        "training_view_fingerprint": "shared-view",
    }
    monkeypatch.setattr(
        "embed_optim.short_branch_evaluation.audit_short_branch_subset",
        lambda _path: dataset,
    )
    calls = []

    def audit(runs, receipt, *, deep):
        calls.append((runs, receipt, deep))
        return {
            "complete": True,
            "verified_runs": 6,
            "verified_checkpoints": 30,
            "errors": [],
        }

    monkeypatch.setattr(
        "embed_optim.short_branch_evaluation.audit_derived_training_artifacts",
        audit,
    )

    result = audit_short_branch_training(protocol_path, configs)

    assert result["complete"] is True
    assert result["verified_runs"] == 18
    assert result["verified_checkpoints"] == 90
    assert result["dataset"] is dataset
    assert len(calls) == 3
    assert all(receipt is dataset and deep is True for _, receipt, deep in calls)


def test_short_branch_training_audit_rejects_silent_checkpoint_shortfall(tmp_path, monkeypatch):
    seeds = (314159, 271828, 161803)
    configs = {seed: [SimpleNamespace() for _ in range(6)] for seed in seeds}
    protocol = {"training": {"order_seeds": list(seeds), "expected_runs": 18}}
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "embed_optim.short_branch_evaluation.load_short_branch_protocol",
        lambda _path: (tmp_path / "protocol.json", protocol),
    )
    monkeypatch.setattr(
        "embed_optim.short_branch_evaluation.audit_short_branch_subset",
        lambda _path: {"rows": 50_000},
    )
    monkeypatch.setattr(
        "embed_optim.short_branch_evaluation.audit_derived_training_artifacts",
        lambda *_args, **_kwargs: {
            "complete": True,
            "verified_runs": 6,
            "verified_checkpoints": 29,
            "errors": [],
        },
    )

    result = audit_short_branch_training(tmp_path / "protocol.json", configs)

    assert result["complete"] is False
    assert "87/90 checkpoints" in result["errors"][0]


def test_short_branch_cli_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--audit-only", "--training-audit-only"])
