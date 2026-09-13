import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "hf_cleanup", Path(__file__).parents[1] / "scripts/hf_obsolete_cleanup.py"
)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


@pytest.mark.parametrize(
    "kind,path",
    [
        ("model", "dense/muon-lr3e-4/checkpoint-782/model.safetensors"),
        ("model", "quarantine/native-addmm-20260818/late/model.safetensors"),
        ("dataset", "decontaminated-beir/dense/muon/task.json"),
        ("dataset", "representation-space/training/exports/dense/pretrained.npz"),
        ("dataset", "representation-space/decontaminated-beir/summary/summary_manifest.json"),
        ("dataset", "project/reports/outcome-summary.md"),
        ("dataset", "project/logs/training/dense-muon.log"),
    ],
)
def test_invalidated_paths_selected(kind, path):
    assert cleanup.removal_reason(kind, path)


@pytest.mark.parametrize(
    "kind,path",
    [
        ("model", "corrected-dense-no-packing-v1/dense/padded-muon/model.safetensors"),
        ("model", "late/muon-lr3e-4/checkpoint-782/model.safetensors"),
        ("model", "dense-new/model.safetensors"),
        ("dataset", "project/data/candidate-breadth/samples.jsonl"),
        ("dataset", "project/configs/experiment.yaml"),
        ("dataset", "project/reports/confirmatory-data/receipt.json"),
        ("dataset", "project/logs/training/late-muon.log"),
        ("dataset", "representation-space/decontaminated-beir/exports/dense/pretrained.npz"),
        ("dataset", "primary-dimension/v1/abc/figure_points.csv"),
        ("dataset", "unexpected-new-results/task.json"),
    ],
)
def test_preserve_current_shared_late_baseline_and_unknown(kind, path):
    assert cleanup.removal_reason(kind, path) is None


@pytest.mark.parametrize("path", ["/", "/dense/model", "dense/../late/model", "dense//model"])
def test_reject_unsafe_paths(path):
    with pytest.raises(ValueError, match="Unsafe"):
        cleanup.removal_reason("model", path)


def test_changed_target_fails_before_any_remote_write(monkeypatch, tmp_path):
    old = {"path": "dense/run/model", "size": 12, "blob_id": "abc", "lfs_sha256": None}
    snapshot = {
        "repo_id": cleanup.REPOSITORIES["model"],
        "repo_type": "model",
        "revision": "old",
        "files": [old],
    }
    plan = cleanup.build_plan({"repositories": [snapshot]}, "inventory-sha")
    monkeypatch.setattr(
        cleanup, "inventory", lambda *a: {**snapshot, "files": [{**old, "size": 13}]}
    )
    api = SimpleNamespace(create_commit=lambda **kw: pytest.fail("Must not write"))
    with pytest.raises(ValueError, match="changed"):
        cleanup.execute(api, plan, tmp_path, tmp_path)


def test_protected_path_cannot_be_injected_into_plan(monkeypatch, tmp_path):
    row = {
        "path": "corrected-dense-no-packing-v1/dense/model",
        "size": 12,
        "blob_id": "abc",
        "lfs_sha256": None,
    }
    snapshot = {
        "repo_id": cleanup.REPOSITORIES["model"],
        "repo_type": "model",
        "revision": "old",
        "files": [row],
    }
    plan = {"repositories": [{**snapshot, "delete": [dict(row, reason="injected")]}]}
    monkeypatch.setattr(cleanup, "inventory", lambda *a: snapshot)
    api = SimpleNamespace(create_commit=lambda **kw: pytest.fail("Must not write"))
    with pytest.raises(ValueError, match="protected"):
        cleanup.execute(api, plan, tmp_path, tmp_path)
