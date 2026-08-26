import json
from pathlib import Path
from types import SimpleNamespace

from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.probe_matrix import ProbeJob, build_probe_jobs, run_probe_job, run_probe_matrix


def _config(tmp_path: Path, family: str, run_id: str) -> RunConfig:
    config = RunConfig(
        run_id=run_id,
        model_family=family,
        optimizer=OptimizerConfig(name="adamw", lr=1e-5),
        model_name=f"fixture/{family}",
        dataset_path="fixture",
        output_root=str(tmp_path / "outputs"),
    )
    config.output_dir.mkdir(parents=True)
    steps = [2, 4, 6, 8, 10]
    (config.output_dir / "checkpoint_schedule.json").write_text(json.dumps({"steps": steps}))
    for step in steps:
        (config.output_dir / f"checkpoint-{step}").mkdir()
    return config


def test_build_probe_jobs_deduplicates_reference_and_covers_five_checkpoints(tmp_path: Path):
    first = _config(tmp_path, "dense", "adamw-a")
    second = _config(tmp_path, "dense", "muon-b")
    reference = tmp_path / "dense-pretrained"
    reference.mkdir()

    jobs = build_probe_jobs(
        [first, second],
        {"dense": reference},
        tmp_path / "representation",
    )

    assert len(jobs) == 11
    assert [job.kind for job in jobs].count("reference") == 1
    assert [job.kind for job in jobs].count("checkpoint") == 10
    assert jobs[0].label == "dense/pretrained"
    assert jobs[-1].label == "dense/muon-b/checkpoint-10"
    assert jobs[-1].reference_export == jobs[0].export


def test_probe_matrix_dry_run_only_lists_incomplete_jobs(tmp_path: Path, monkeypatch, capsys):
    jobs = [
        ProbeJob(
            kind="reference",
            family="dense",
            label="dense/pretrained",
            checkpoint=tmp_path / "checkpoint",
            export=tmp_path / "pretrained.npz",
            metrics=tmp_path / "pretrained.json",
            reference_export=None,
        ),
        ProbeJob(
            kind="checkpoint",
            family="dense",
            label="dense/adamw/checkpoint-2",
            checkpoint=tmp_path / "checkpoint-2",
            export=tmp_path / "checkpoint-2.npz",
            metrics=tmp_path / "checkpoint-2.json",
            reference_export=tmp_path / "pretrained.npz",
        ),
    ]
    monkeypatch.setattr(
        "embed_optim.probe_matrix.probe_job_complete",
        lambda job: job.kind == "reference",
    )
    args = SimpleNamespace(dry_run=True)

    assert run_probe_matrix(jobs, args) == 0
    assert capsys.readouterr().out.strip() == "dense/adamw/checkpoint-2"


def test_run_probe_job_resumes_valid_export_and_rewrites_metrics(tmp_path: Path, monkeypatch):
    job = ProbeJob(
        kind="checkpoint",
        family="late",
        label="late/muon/checkpoint-2",
        checkpoint=tmp_path / "checkpoint-2",
        export=tmp_path / "checkpoint-2.npz",
        metrics=tmp_path / "checkpoint-2.json",
        reference_export=tmp_path / "pretrained.npz",
    )
    calls = []
    monkeypatch.setattr("embed_optim.probe_matrix._valid_export", lambda *args: True)
    monkeypatch.setattr(
        "embed_optim.probe_matrix.export_probe",
        lambda *args, **kwargs: calls.append("export"),
    )
    monkeypatch.setattr(
        "embed_optim.probe_matrix.analyze_probe",
        lambda *args, **kwargs: calls.append(("analyze", kwargs["reference_source"])),
    )
    monkeypatch.setattr("embed_optim.probe_matrix.probe_job_complete", lambda *args: True)

    run_probe_job(
        job,
        probe=tmp_path / "probe",
        probe_spec=tmp_path / "spec.json",
        batch_size=32,
        model_dtype="bfloat16",
        storage_dtype="float16",
        device="cuda:0",
        flash_attention=True,
    )

    assert calls == [("analyze", job.reference_export)]
