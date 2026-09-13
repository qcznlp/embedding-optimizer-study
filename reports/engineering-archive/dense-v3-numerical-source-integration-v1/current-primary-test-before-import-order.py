"""Actual current-checkout source identity, without model loading or training."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from embed_optim import dense_run_contract
from embed_optim.primary_v3_contract import PrimaryV3Contract

ROOT = Path(__file__).resolve().parents[1]


def current_contract():
    return PrimaryV3Contract.load(ROOT / "configs/dense_primary_v3_protocol.json", ROOT, ROOT)


def test_current_tree_matches_all_twelve_original_primary_sources():
    contract = current_contract()
    source = dense_run_contract.source_identity()
    assert contract.repository == contract.training_root == ROOT
    assert len(contract.inputs["runs"]) == 12
    for run in contract.inputs["runs"]:
        assert contract.expected_identity(run["run_id"])["source"] == source


def test_integration_does_not_release_the_historical_draft():
    with pytest.raises(ValueError):
        current_contract().require_execution()


def test_actual_primary_inspect_cli_uses_one_checkout(tmp_path):
    run_id = "verified-v3-adamw-3e-5"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "embed_optim.primary_v3_training",
            "--protocol",
            str(ROOT / "configs/dense_primary_v3_protocol.json"),
            "--repository",
            str(ROOT),
            "--training-root",
            str(ROOT),
            "--experiment-root",
            str(tmp_path / "absent-experiment"),
            "--run-id",
            run_id,
            "--inspect",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["run_id"] == run_id
    assert observed["recipe"] == current_contract().expected_identity(run_id)["recipe"]
    assert observed["checkpoint_steps"] == [782, 1563, 2345, 3126, 3907]
    assert observed["training_executed"] is False
    assert observed["scientific_completion"] is False
    assert not (tmp_path / "absent-experiment").exists()
