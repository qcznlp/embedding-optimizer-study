from __future__ import annotations

import json
from pathlib import Path

import pytest

from embed_optim.candidate_breadth_matrix import (
    _parse_gpus,
    _preflight_candidate_breadth_jobs,
    _verified_source_audit_receipt,
    candidate_breadth_jobs,
)
from embed_optim.geometry import _sha256


def test_candidate_breadth_matrix_covers_the_frozen_discovery_grid() -> None:
    _, protocol, jobs = candidate_breadth_jobs("configs/candidate_breadth_probe.json")
    assert len(jobs) == 12
    assert {job["run_id"].split("-lr", 1)[0] for job in jobs} == {
        "adamw",
        "muon",
        "normuon",
    }
    assert all(job["checkpoint"].name == "checkpoint-3907" for job in jobs)
    assert protocol["evaluation"]["baseline_root"] == "results/recipe-validation/dense"


def test_gpu_parser_requires_unique_integer_devices() -> None:
    assert _parse_gpus("0,2,7") == ["0", "2", "7"]
    for value in ("", "0,0", "cuda:0", "0,x"):
        with pytest.raises(ValueError, match="GPUs"):
            _parse_gpus(value)


def test_matrix_preflight_rejects_missing_inputs_before_gpu_launch(tmp_path: Path) -> None:
    jobs = [
        {
            "run_id": run_id,
            "checkpoint": tmp_path / "outputs" / run_id / "checkpoint-3907",
            "output_dir": tmp_path / "results" / run_id,
        }
        for run_id in ("adamw-lr1e-5", "muon-lr3e-4")
    ]
    baseline_root = tmp_path / "baseline"

    with pytest.raises(FileNotFoundError, match="before GPU launch") as error:
        _preflight_candidate_breadth_jobs(jobs, baseline_root=baseline_root)
    assert "checkpoint-3907" in str(error.value)
    assert "sample_metrics.jsonl" in str(error.value)

    for job in jobs:
        Path(job["checkpoint"]).mkdir(parents=True)
        baseline = baseline_root / str(job["run_id"]) / "sample_metrics.jsonl"
        baseline.parent.mkdir(parents=True)
        baseline.write_text("{}\n", encoding="utf-8")

    _preflight_candidate_breadth_jobs(jobs, baseline_root=baseline_root)


def test_matrix_binds_the_full_source_audit_receipt(tmp_path) -> None:
    protocol = tmp_path / "configs" / "candidate.json"
    protocol.parent.mkdir()
    protocol.write_text("{}\n", encoding="utf-8")
    data_audit = {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": _sha256(protocol),
        "manifest_sha256": "a" * 64,
        "upstream_reconstruction_verified": False,
    }
    full_audit = {**data_audit, "upstream_reconstruction_verified": True}
    receipt = tmp_path / "reports" / "candidate-breadth" / "data-audit.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(full_audit) + "\n", encoding="utf-8")

    identity = _verified_source_audit_receipt(
        receipt,
        root=tmp_path,
        protocol_path=protocol,
        data_audit=data_audit,
    )
    assert identity["audit"]["upstream_reconstruction_verified"] is True
    assert identity["sha256"] == _sha256(receipt)

    full_audit["manifest_sha256"] = "b" * 64
    receipt.write_text(json.dumps(full_audit) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match current data"):
        _verified_source_audit_receipt(
            receipt,
            root=tmp_path,
            protocol_path=protocol,
            data_audit=data_audit,
        )
