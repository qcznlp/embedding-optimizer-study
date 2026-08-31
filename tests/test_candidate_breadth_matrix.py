from __future__ import annotations

import pytest

from embed_optim.candidate_breadth_matrix import (
    _parse_gpus,
    candidate_breadth_jobs,
)


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
