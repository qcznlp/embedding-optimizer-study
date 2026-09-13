from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from embed_optim import primary_dimension_probe as primary
from embed_optim import probe_export
from embed_optim.config import load_matrix
from embed_optim.dimension_utilization import _atomic_json, _identity, _sha256

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    can_flatten_inputs = True
    max_seq_length = 8192

    def __init__(self):
        self.calls = []

    def _first_module(self):
        return self

    def encode(self, texts, **kwargs):
        assert self.can_flatten_inputs is False
        assert kwargs["normalize_embeddings"] is True
        self.calls.append((list(texts), kwargs))
        values = np.zeros((len(texts), 768), dtype=np.float32)
        values[:, 0] = 1
        return values


@pytest.fixture
def export_fixture(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    (repository / "configs").mkdir(parents=True)
    for name in (primary.PROTOCOL, primary.SCIENTIFIC_PROTOCOL):
        _atomic_json(repository / name, {"test": str(name)})
    rows = [
        {
            "sample_id": i,
            "source": f"task-{i // 16}",
            "query": f"query {i}",
            "positive": f"positive {i}",
            **{f"negative_{j}": f"negative {i}-{j}" for j in range(7)},
        }
        for i in range(224)
    ]
    probe = tmp_path / "probe"
    Dataset.from_list(rows).save_to_disk(str(probe / "dataset"))
    dataset = Dataset.load_from_disk(str(probe / "dataset"))
    selection = probe / "selection.jsonl"
    selection.write_text("fixed selection\n")
    digest = hashlib.sha256("".join(f"{i}\n" for i in range(224)).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "count": 224,
        "negative_candidates": 7,
        "positive_candidate_index": 0,
        "selection_sha256": _sha256(selection),
        "selected_sample_ids_sha256": digest,
        "serialized_probe_dataset_fingerprint": dataset._fingerprint,
    }
    _atomic_json(probe / "manifest.json", manifest)
    expected = {
        "manifest_sha256": _sha256(probe / "manifest.json"),
        "selection_sha256": manifest["selection_sha256"],
        "selected_sample_ids_sha256": digest,
        "task_counts": {f"task-{i}": 16 for i in range(14)},
    }
    _atomic_json(repository / "configs/beir_representation_probe.json", {"expected": expected})
    checkpoint = tmp_path / "run" / "checkpoint-782"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"synthetic checkpoint")
    _atomic_json(checkpoint.parent / "run_config.json", {"model_family": "dense"})
    _atomic_json(checkpoint.parent / "completed.json", {"step": 3907})
    job = primary.ExportJob(
        "padded-muon-1e-4/checkpoint-782",
        checkpoint,
        tmp_path / "exports/padded-muon-1e-4/checkpoint-782.npz",
        checkpoint.parent / "completed.json",
    )
    model = FakeModel()
    monkeypatch.setattr(probe_export, "_load_model", lambda *a, **k: model)
    monkeypatch.setattr(probe_export.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(probe_export.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(probe_export.torch.cuda, "get_device_name", lambda *a: "fixture")
    return repository, probe, job, model


def test_exact_matrix_plan_has_61_unique_states():
    configs = load_matrix(ROOT / primary.MATRIX)
    cells = primary.planned_cells(configs, [782, 1563, 2345, 3126, 3907])
    assert len(cells) == len(set(cells)) == 61
    assert cells[0] == "pretrained"
    with pytest.raises(ValueError, match="complete 12-by-5"):
        primary.planned_cells(configs[:-1], [782, 1563, 2345, 3126, 3907])
    with pytest.raises(ValueError, match="complete 12-by-5"):
        primary.planned_cells(configs, [782, 1563, 2345, 3126, 9999])


def test_source_lock_matches_current_implementation():
    contract = primary.load_contract(ROOT)
    assert contract["model"]["revision"] == "0edbd55684eb782bce55ee74c95b25c97cbe7f43"
    assert contract["encoding"]["storage_dtype"] == "float32"


def test_export_real_encoder_wrapper_and_cache_resume(export_fixture):
    repository, probe, job, model = export_fixture
    original_loader = probe_export._load_model
    output = primary.run_job(job, repository, probe)
    assert probe_export._load_model is original_loader
    assert model.calls[0][1]["prompt"] == "query: "
    assert model.calls[1][1]["prompt"] == "document: "
    assert model.calls[1][0][:2] == ["positive 0", "negative 0-0"]
    assert primary.run_job(job, repository, probe) == output
    assert len(model.calls) == 2
    receipt = json.loads(job.receipt.read_text())
    assert receipt["status"] == "complete"
    assert receipt["observed_input_execution"] == primary.INPUT_EXECUTION
    with np.load(job.export) as arrays:
        assert arrays["query_embeddings"].shape == (224, 768)
        assert arrays["document_embeddings"].dtype == np.float32


@pytest.mark.parametrize("mutation", ["weights", "run_config", "completion", "probe", "protocol"])
def test_resume_rejects_changed_request_inputs(export_fixture, mutation):
    repository, probe, job, _ = export_fixture
    primary.run_job(job, repository, probe)
    paths = {
        "weights": job.checkpoint / "model.safetensors",
        "run_config": job.checkpoint.parent / "run_config.json",
        "completion": job.completion,
        "probe": probe / "selection.jsonl",
        "protocol": repository / primary.PROTOCOL,
    }
    path = paths[mutation]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="request changed"):
        primary.run_job(job, repository, probe)


@pytest.mark.parametrize(
    "mutation", ["checkpoint_identity", "encoding", "observed_execution", "payload"]
)
def test_rehashed_cache_cannot_bypass_execution_or_model_identity(export_fixture, mutation):
    repository, probe, job, _ = export_fixture
    primary.run_job(job, repository, probe)
    manifest_path = job.export.with_suffix(".npz.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(job.receipt.read_text())
    if mutation == "checkpoint_identity":
        manifest["checkpoint_inputs"][0]["sha256"] = "0" * 64
    elif mutation == "encoding":
        manifest["encoding"]["dense_query_prompt"] = "document: "
    elif mutation == "observed_execution":
        receipt["observed_input_execution"]["sentence_transformers_can_flatten_inputs"] = True
    else:
        with np.load(job.export) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["sample_ids"] = arrays["sample_ids"][::-1]
        np.savez(job.export, **arrays)
        manifest["output"].update(bytes=job.export.stat().st_size, sha256=_sha256(job.export))
    _atomic_json(manifest_path, manifest)
    receipt["outputs"] = {
        "export": _identity(job.export),
        "export_manifest": _identity(manifest_path),
    }
    _atomic_json(job.receipt, receipt)
    with pytest.raises(ValueError):
        primary.run_job(job, repository, probe)


def test_resume_can_seal_verified_outputs_after_interruption(export_fixture):
    repository, probe, job, model = export_fixture
    output = primary.run_job(job, repository, probe)
    receipt = json.loads(job.receipt.read_text())
    receipt["status"] = "model_verified"
    receipt.pop("outputs")
    _atomic_json(job.receipt, receipt)
    assert primary.run_job(job, repository, probe) == output
    assert len(model.calls) == 2
    assert json.loads(job.receipt.read_text())["status"] == "complete"


def test_untagged_cache_is_preserved_and_rejected(export_fixture):
    repository, probe, job, model = export_fixture
    job.export.parent.mkdir(parents=True)
    job.export.write_bytes(b"keep this unrelated output")
    with pytest.raises(FileExistsError, match="Untagged"):
        primary.run_job(job, repository, probe)
    assert job.export.read_bytes() == b"keep this unrelated output"
    assert model.calls == []


def test_loader_failure_restores_original_and_never_completes(export_fixture, monkeypatch):
    repository, probe, job, model = export_fixture
    model.max_seq_length = 512
    original_loader = probe_export._load_model
    with pytest.raises(ValueError, match="loading contract"):
        primary.run_job(job, repository, probe)
    assert probe_export._load_model is original_loader
    assert json.loads(job.receipt.read_text())["status"] == "in_progress"
    assert not job.export.exists()


def test_dry_run_is_read_only_and_reports_missing_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        primary, "_training_gate", lambda *a: pytest.fail("must not audit training")
    )
    args = Namespace(
        repository=ROOT,
        artifact_root=tmp_path,
        export_root=tmp_path / "exports",
        dry_run=True,
        audit_only=False,
        gpu="0",
    )
    output = primary.run(args)
    assert len(output["missing_completed_runs"]) == 12
    assert output["gpu_work_started"] is False
    assert not args.export_root.exists()
    args.dry_run = False
    with pytest.raises(RuntimeError, match="waits for all 12"):
        primary.run(args)
    assert not args.export_root.exists()


def test_matrix_audit_rejects_extra_archive(tmp_path):
    (tmp_path / "unexpected.npz").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="exact 61"):
        primary._audit_matrix([], ROOT, tmp_path, tmp_path)
