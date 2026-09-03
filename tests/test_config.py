import hashlib
import json
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
        assert all(run.dense_can_flatten_inputs is True for run in family_runs)


def test_corrected_dense_matrix_is_isolated_and_explicitly_padded():
    path = Path(__file__).parents[1] / "configs" / "dense_no_packing_retrain.yaml"
    runs = load_matrix(path)

    assert len(runs) == 12
    assert {run.model_family for run in runs} == {"dense"}
    assert {run.optimizer.name for run in runs} == {"adamw", "muon", "normuon"}
    assert {
        optimizer: {run.optimizer.lr for run in runs if run.optimizer.name == optimizer}
        for optimizer in ("adamw", "muon", "normuon")
    } == {
        "adamw": {1e-6, 3e-6, 1e-5, 3e-5},
        "muon": {1e-4, 3e-4, 1e-3, 3e-3},
        "normuon": {1e-4, 3e-4, 1e-3, 3e-3},
    }
    assert all(run.run_id.startswith("padded-") for run in runs)
    assert all(run.output_root == "outputs/dense-no-packing-v1" for run in runs)
    assert all(run.dense_can_flatten_inputs is False for run in runs)
    assert all(run.dataset_path == "data/denseon-sft-500k-seed42" for run in runs)
    assert all(run.seed == 42 and len(run.checkpoint_fractions) == 5 for run in runs)


def test_public_docs_record_compute_and_acceleration_contract() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    blog = (root / "docs/blog.md").read_text(encoding="utf-8")
    gates = (root / "docs/completion-gates.md").read_text(encoding="utf-8")

    for document in (readme, blog, gates):
        normalized = document.lower()
        assert "eight h100-equivalent gpus" in normalized
        assert "two disjoint four-gpu" in normalized
    compact_blog = " ".join(blog.split())
    assert "task-parallel retrieval evaluation" in compact_blog
    assert "exact retrieval rather than approximate ranking" in compact_blog
    assert "fused CUDA AdamW" in compact_blog
    assert "decomposed-bfloat16" in compact_blog
    assert "None of these systems optimizations introduces in-batch negatives" in compact_blog


def test_hybrid_adamw_matrix_is_an_eight_run_routing_control():
    path = Path(__file__).parents[1] / "configs" / "hybrid_adamw.yaml"
    runs = load_matrix(path)

    assert len(runs) == 8
    assert {run.model_family for run in runs} == {"dense", "late"}
    assert {run.optimizer.name for run in runs} == {"hybrid_adamw"}
    assert {run.optimizer.lr for run in runs} == {1e-6, 3e-6, 1e-5, 3e-5}
    assert all(run.optimizer.aux_lr == 3e-6 for run in runs)
    assert all(run.output_root == "outputs/hybrid-adamw" for run in runs)

    protocol_path = path.with_name("hybrid_adamw_control.json")
    protocol = json.loads(protocol_path.read_text())
    assert protocol["status"] == "prospective_completion_lock"
    assert protocol["selection"]["expected_runs"] == 8
    assert protocol["selection"]["formal_beir_stages"] == [5]
    assert protocol["selection"]["expected_beir_units"] == 112
    assert protocol["observed_before_freeze"]["strict_beir"]["valid_units"] == 140
    assert protocol["observed_before_freeze"]["hybrid_training_outputs_exist"] is False
    assert (
        protocol["sources"]["control_matrix"]["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_matrix_supports_family_specific_explicit_runs(tmp_path: Path):
    path = tmp_path / "explicit.yaml"
    path.write_text(
        """
common:
  dataset_path: data
  seed: 314159
models:
  dense:
    model_name: dense-model
  late:
    model_name: late-model
runs:
  - id: adamw-selected
    model_family: dense
    optimizer: {name: adamw, lr: 1.0e-5}
  - id: adamw-selected
    model_family: late
    optimizer: {name: adamw, lr: 3.0e-5}
""".lstrip(),
        encoding="utf-8",
    )

    runs = load_matrix(path)

    assert len(runs) == 2
    assert {run.model_family for run in runs} == {"dense", "late"}
    assert {run.model_family: run.optimizer.lr for run in runs} == {
        "dense": 1e-5,
        "late": 3e-5,
    }
    assert all(run.seed == 314159 for run in runs)


def test_matrix_rejects_mixed_grid_and_explicit_runs(tmp_path: Path):
    path = tmp_path / "mixed.yaml"
    path.write_text(
        """
common: {dataset_path: data}
models: {dense: {model_name: model}}
optimizers: [{id: adamw, name: adamw, lr: 1.0e-5}]
runs: [{id: adamw, model_family: dense, optimizer: {name: adamw, lr: 1.0e-5}}]
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both explicit runs"):
        load_matrix(path)


def test_bundled_default_matrix_falls_back_to_wheel_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefix = tmp_path / "venv"
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / "experiment.yaml"
    installed.parent.mkdir(parents=True)
    installed.write_text("bundled: true\n")

    assert _resolve_matrix_path("configs/experiment.yaml", prefix) == installed
    assert resolve_matrix_path("configs/experiment.yaml", prefix) == installed
    assert _resolve_matrix_path("custom.yaml", prefix) == Path("custom.yaml")


def test_nested_bundled_matrix_loads_from_wheel_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefix = tmp_path / "venv"
    installed = (
        prefix
        / "share"
        / "embedding-optimizer-study"
        / "configs"
        / "generated"
        / "confirmatory"
        / "seed314159.yaml"
    )
    installed.parent.mkdir(parents=True)
    installed.write_text(
        """
formal_runtime: ../../formal_runtime.json
common:
  dataset_path: data
  seed: 314159
models:
  dense:
    model_name: model
runs:
  - id: adamw-selected
    model_family: dense
    optimizer:
      name: adamw
      lr: 3.0e-5
""".lstrip(),
        encoding="utf-8",
    )
    runtime = prefix / "share" / "embedding-optimizer-study" / "configs" / "formal_runtime.json"
    runtime.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("embed_optim.config.sys.prefix", str(prefix))

    matrix = "configs/generated/confirmatory/seed314159.yaml"
    runs = load_matrix(matrix)

    assert len(runs) == 1
    assert runs[0].run_id == "adamw-selected"
    assert resolve_matrix_path(matrix) == installed
    resolved_runtime = matrix_runtime_spec(matrix)
    assert resolved_runtime is not None
    assert resolved_runtime.resolve() == runtime
    assert resolved_runtime.is_file()


def test_bundled_matrix_fallback_rejects_parent_traversal(tmp_path):
    prefix = tmp_path / "venv"
    escaped = prefix / "share" / "embedding-optimizer-study" / "outside.yaml"
    escaped.parent.mkdir(parents=True)
    escaped.write_text("escaped: true\n", encoding="utf-8")
    requested = Path("configs/../outside.yaml")

    assert _resolve_matrix_path(requested, prefix) == requested


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
    hybrid = RunConfig(
        optimizer=OptimizerConfig(name="hybrid_adamw", lr=1e-5, aux_lr=3e-6), **common
    )
    muon = RunConfig(optimizer=OptimizerConfig(name="muon", lr=1e-4), **common)

    assert source_wandb_run_id(adamw) == "study-v2-late-optimizer-test-seed42"
    assert source_wandb_run_id(hybrid) == "study-v4-late-optimizer-test-seed42"
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
