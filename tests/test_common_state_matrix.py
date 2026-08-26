import json
import sys
from argparse import Namespace
from pathlib import Path

from embed_optim.common_state_matrix import (
    CommonStateJob,
    _job_cli,
    _load_protocol,
    build_common_state_jobs,
    common_state_job_complete,
    parse_args,
)
from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.geometry import _sha256

RUNS = (
    ("adamw-lr1e-5", "adamw", 1e-5),
    ("muon-lr1e-3", "muon", 1e-3),
    ("normuon-lr1e-3", "normuon", 1e-3),
)


def _config(tmp_path: Path, family: str, run_id: str, optimizer: str, lr: float) -> RunConfig:
    config = RunConfig(
        run_id=run_id,
        model_family=family,
        optimizer=OptimizerConfig(name=optimizer, lr=lr),
        model_name=f"fixture/{family}",
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
    return config


def _spec() -> dict:
    return {
        "schema_version": 1,
        "selection": {"gradient_steps": 8},
        "anchor_protocol": {
            "include_pretrained": True,
            "run_ids": [item[0] for item in RUNS],
            "checkpoint_fractions": [0.2, 0.6, 1.0],
            "expected_families": ["dense", "late"],
            "expected_anchors_per_family": 10,
            "expected_total_anchors": 20,
            "expected_hidden_partition": {"tensors": 88, "parameters": 110297088},
            "selection_basis": "fixture",
        },
    }


def test_frozen_common_state_protocol_is_self_consistent():
    path = Path("configs/common_state_probe.json")
    spec, anchor = _load_protocol(path)

    assert spec["selection"]["gradient_steps"] == 8
    assert anchor["expected_anchors_per_family"] == 10
    assert anchor["expected_total_anchors"] == 20
    assert anchor["freeze_context"]["strict_beir_valid_units"] == 98
    assert anchor["freeze_context"]["partial_results_already_observed"] is True
    assert "before any common-state GPU output" in anchor["freeze_context"]["protocol_amendment"]


def test_build_common_state_jobs_covers_frozen_anchor_grid(tmp_path: Path):
    configs = [
        _config(tmp_path, family, run_id, optimizer, lr)
        for family in ("dense", "late")
        for run_id, optimizer, lr in RUNS
    ]
    references = {}
    for family in ("dense", "late"):
        references[family] = tmp_path / f"{family}-pretrained"
        references[family].mkdir()

    jobs = build_common_state_jobs(configs, references, _spec(), tmp_path / "common-state")

    assert len(jobs) == 20
    assert [job.label for job in jobs[:4]] == [
        "dense/pretrained",
        "dense/adamw-lr1e-5/checkpoint-2",
        "dense/adamw-lr1e-5/checkpoint-6",
        "dense/adamw-lr1e-5/checkpoint-10",
    ]
    assert jobs[-1].label == "late/normuon-lr1e-3/checkpoint-10"
    assert all(job.gradient_steps == 8 for job in jobs)
    assert all(job.update_dir.parent == job.gradient_dir.parent for job in jobs)


def _declared(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_common_state_completion_checks_both_manifests_and_outputs(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    gradient_dir = tmp_path / "anchor" / "gradients"
    update_dir = tmp_path / "anchor" / "updates"
    gradient_dir.mkdir(parents=True)
    update_dir.mkdir(parents=True)
    spec = tmp_path / "common-state.json"
    spec.write_text('{"fixture":true}\n')
    shard = gradient_dir / "gradient-0000.safetensors"
    shard.write_bytes(b"gradient")
    gradient_manifest = gradient_dir / "manifest.json"
    gradient_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "checkpoint": {"path": str(checkpoint)},
                "common_state_spec": {"sha256": _sha256(spec)},
                "partition_summary": {"hidden": {"tensors": 1, "parameters": 4}},
                "gradient_shards": [_declared(shard, gradient_dir)],
            }
        )
    )
    output_paths = {
        "metrics": update_dir / "metrics.jsonl",
        "adamw_matched": update_dir / "adamw-matched.safetensors",
        "muon_matched": update_dir / "muon-matched.safetensors",
        "normuon_matched": update_dir / "normuon-matched.safetensors",
    }
    for label, path in output_paths.items():
        path.write_bytes(label.encode())
    update_manifest = update_dir / "manifest.json"
    update_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": {"path": str(checkpoint)},
                "common_state_spec": {"sha256": _sha256(spec)},
                "gradient_manifest": {
                    "path": str(gradient_manifest.resolve()),
                    "bytes": gradient_manifest.stat().st_size,
                    "sha256": _sha256(gradient_manifest),
                },
                "gradient_steps": 1,
                "tensors": 1,
                "parameters": 4,
                "outputs": {
                    label: _declared(path, update_dir) for label, path in output_paths.items()
                },
            }
        )
    )
    job = CommonStateJob(
        family="dense",
        label="dense/pretrained",
        checkpoint=checkpoint.resolve(),
        gradient_dir=gradient_dir,
        update_dir=update_dir,
        gradient_steps=1,
        hidden_tensors=1,
        hidden_parameters=4,
    )

    assert common_state_job_complete(job, spec, verify_hashes=True)
    output_paths["muon_matched"].write_bytes(b"changed")
    assert not common_state_job_complete(job, spec)


def test_common_state_worker_command_round_trips_through_parser(tmp_path: Path):
    job = CommonStateJob(
        family="late",
        label="late/muon-lr1e-3/checkpoint-2345",
        checkpoint=(tmp_path / "checkpoint-2345").resolve(),
        gradient_dir=(tmp_path / "gradients").resolve(),
        update_dir=(tmp_path / "updates").resolve(),
        gradient_steps=8,
        hidden_tensors=88,
        hidden_parameters=110_297_088,
    )
    command = _job_cli(
        job,
        Namespace(
            probe=tmp_path / "probe",
            probe_spec=tmp_path / "probe.json",
            common_state_spec=tmp_path / "common-state.json",
        ),
    )

    assert command[:3] == [sys.executable, "-m", "embed_optim.common_state_matrix"]
    parsed = parse_args(command[3:])
    assert parsed.worker is True
    assert parsed.family == "late"
    assert parsed.label == job.label
    assert parsed.checkpoint == job.checkpoint
    assert parsed.hidden_tensors == 88
    assert parsed.hidden_parameters == 110_297_088
