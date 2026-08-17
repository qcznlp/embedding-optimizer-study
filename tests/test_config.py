from pathlib import Path

import pytest

from embed_optim.config import (
    MUON_NS_IMPLEMENTATION,
    OptimizerConfig,
    RunConfig,
    _resolve_matrix_path,
    load_matrix,
    matrix_runtime_spec,
    resolve_matrix_path,
    source_wandb_run_id,
)


def test_matrix_has_24_controlled_runs():
    path = Path(__file__).parents[1] / "configs" / "experiment.yaml"
    runs = load_matrix(path)
    assert len(runs) == 24
    for family in ("dense", "late"):
        family_runs = [run for run in runs if run.model_family == family]
        assert len(family_runs) == 12
        assert {run.optimizer.name for run in family_runs} == {"adamw", "muon", "normuon"}
        assert all(len(run.checkpoint_fractions) == 5 for run in family_runs)
        assert all(run.model_revision and len(run.model_revision) == 40 for run in family_runs)


def test_bundled_default_matrix_falls_back_to_wheel_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefix = tmp_path / "venv"
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / "experiment.yaml"
    installed.parent.mkdir(parents=True)
    installed.write_text("bundled: true\n")

    assert _resolve_matrix_path("configs/experiment.yaml", prefix) == installed
    assert resolve_matrix_path("configs/experiment.yaml", prefix) == installed
    assert _resolve_matrix_path("custom.yaml", prefix) == Path("custom.yaml")


def test_only_formal_matrix_declares_runtime_contract():
    root = Path(__file__).parents[1]

    assert matrix_runtime_spec(root / "configs" / "experiment.yaml") == (
        root / "configs" / "formal_runtime.json"
    )
    assert matrix_runtime_spec(root / "configs" / "smoke.yaml") is None


def test_muon_source_wandb_id_isolated_from_native_history():
    common = {
        "run_id": "optimizer-test",
        "model_family": "late",
        "model_name": "model",
        "dataset_path": "data",
    }
    adamw = RunConfig(optimizer=OptimizerConfig(name="adamw", lr=1e-6), **common)
    muon = RunConfig(optimizer=OptimizerConfig(name="muon", lr=1e-4), **common)

    assert source_wandb_run_id(adamw) == "study-v2-late-optimizer-test-seed42"
    assert source_wandb_run_id(muon) == (
        f"study-v3-late-optimizer-test-seed42-{MUON_NS_IMPLEMENTATION}"
    )
    assert "ns_implementation" not in adamw.as_dict()["optimizer"]
    assert muon.as_dict()["optimizer"]["ns_implementation"] == MUON_NS_IMPLEMENTATION
    assert RunConfig.from_dict(muon.as_dict()) == muon

    invalid = muon.as_dict()
    invalid["optimizer"]["ns_implementation"] = "native-addmm"
    with pytest.raises(ValueError, match="Unsupported optimizer implementation"):
        RunConfig.from_dict(invalid)
