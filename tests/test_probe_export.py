import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from embed_optim.geometry import _sha256
from embed_optim.probe_export import (
    _load_model,
    encode_late_probe,
    export_probe,
    pack_variable_embeddings,
    pad_variable_embeddings,
)
from embed_optim.representation_geometry import analyze_probe


def _rows() -> list[dict]:
    rows = []
    for sample_id, source in ((11, "fiqa"), (19, "nq")):
        row = {
            "sample_id": sample_id,
            "source": source,
            "query": f"query {sample_id}",
            "positive": f"positive {sample_id}",
        }
        for index in range(7):
            row[f"negative_{index}"] = f"negative {sample_id} {index}"
        rows.append(row)
    return rows


def _probe_fixture(root: Path) -> Path:
    probe = root / "probe"
    dataset = Dataset.from_list(_rows())
    dataset.save_to_disk(str(probe / "dataset"))
    serialized = Dataset.load_from_disk(str(probe / "dataset"))
    digest = hashlib.sha256()
    for sample_id in serialized["sample_id"]:
        digest.update(f"{int(sample_id)}\n".encode())
    manifest = {
        "schema_version": 1,
        "count": len(serialized),
        "negative_candidates": 7,
        "positive_candidate_index": 0,
        "selected_sample_ids_sha256": digest.hexdigest(),
        "selection_sha256": "",
        "serialized_probe_dataset_fingerprint": serialized._fingerprint,
    }
    (probe / "selection.jsonl").write_text("fixture selection\n")
    manifest["selection_sha256"] = _sha256(probe / "selection.jsonl")
    (probe / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return probe


class FakeDense:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        embeddings = np.zeros((len(texts), 768), dtype=np.float32)
        embeddings[:, 0] = 1
        embeddings[:, 1] = np.arange(len(texts)) % 2
        return embeddings


class FakeLate:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        offset = 1 if kwargs["is_query"] else 2
        return [
            np.full((offset + index % 3, 128), index + 1, dtype=np.float32)
            for index in range(len(texts))
        ]


class FakeLoadedLate:
    def __init__(self, **kwargs):
        self.query_length = kwargs.get("query_length")
        self.document_length = kwargs.get("document_length")
        self.do_query_expansion = kwargs.get("do_query_expansion")
        self.can_flatten_inputs = True
        self.device = None
        self.training = True

    def _first_module(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self


def test_late_loader_overrides_pretrained_context_contract(tmp_path: Path, monkeypatch):
    calls = []

    class FakeColBERT(FakeLoadedLate):
        def __init__(self, checkpoint, **kwargs):
            calls.append((checkpoint, dict(kwargs)))
            super().__init__(**kwargs)

    fake_models = type("FakeModels", (), {"ColBERT": FakeColBERT})
    monkeypatch.setattr(
        "embed_optim.pylate_compat.configure_pylate_compatibility",
        lambda: fake_models,
    )

    checkpoint = tmp_path / "late-pretrained"
    model = _load_model(
        "late",
        checkpoint,
        dtype=torch.float32,
        device="cpu",
        flash_attention=False,
    )

    assert calls == [
        (
            str(checkpoint),
            {
                "query_length": 8192,
                "document_length": 8192,
                "do_query_expansion": False,
                "trust_remote_code": True,
                "model_kwargs": {"dtype": torch.float32, "attn_implementation": "sdpa"},
            },
        )
    ]
    assert model.query_length == 8192
    assert model.document_length == 8192
    assert model.do_query_expansion is False
    assert model.can_flatten_inputs is False
    assert model.device == "cpu"
    assert model.training is False


def test_late_loader_rejects_ignored_context_override(tmp_path: Path, monkeypatch):
    class FakeColBERT(FakeLoadedLate):
        def __init__(self, checkpoint, **kwargs):
            del checkpoint, kwargs
            super().__init__(
                query_length=32,
                document_length=300,
                do_query_expansion=False,
            )

    fake_models = type("FakeModels", (), {"ColBERT": FakeColBERT})
    monkeypatch.setattr(
        "embed_optim.pylate_compat.configure_pylate_compatibility",
        lambda: fake_models,
    )

    with pytest.raises(ValueError, match="query=32, document=300"):
        _load_model(
            "late",
            tmp_path / "late-pretrained",
            dtype=torch.float32,
            device="cpu",
            flash_attention=False,
        )


def test_pad_variable_embeddings_preserves_lengths_and_dtype():
    values = [np.ones((2, 3)), np.full((1, 3), 2)]
    embeddings, mask = pad_variable_embeddings(values, storage_dtype=np.dtype("float16"))

    assert embeddings.shape == (2, 2, 3)
    assert embeddings.dtype == np.float16
    assert mask.tolist() == [[True, True], [True, False]]
    np.testing.assert_array_equal(embeddings[1, 1], np.zeros(3))


def test_pack_variable_embeddings_preserves_lengths_without_padding():
    values = [np.ones((2, 3)), np.full((1, 3), 2)]
    embeddings, offsets = pack_variable_embeddings(values, storage_dtype=np.dtype("float16"))

    assert embeddings.shape == (3, 3)
    assert embeddings.dtype == np.float16
    assert offsets.tolist() == [0, 2, 3]
    np.testing.assert_array_equal(embeddings[2], np.full(3, 2))


def test_late_encoder_keeps_positive_first_and_emits_offsets():
    dataset = Dataset.from_list(_rows())
    model = FakeLate()

    arrays = encode_late_probe(
        model,
        dataset,
        batch_size=4,
        storage_dtype=np.dtype("float16"),
    )

    assert arrays["query_embeddings"].shape == (3, 128)
    assert arrays["query_offsets"].tolist() == [0, 1, 3]
    assert arrays["document_embeddings"].shape == (47, 128)
    assert np.diff(arrays["document_offsets"][:9]).tolist() == [2, 3, 4, 2, 3, 4, 2, 3]
    assert model.calls[0][1]["is_query"] is True
    assert model.calls[1][1]["is_query"] is False
    assert model.calls[1][0][0] == "positive 11"
    assert model.calls[1][0][1] == "negative 11 0"


def test_dense_export_is_atomic_hashed_and_analyzer_compatible(tmp_path: Path, monkeypatch):
    probe = _probe_fixture(tmp_path)
    checkpoint = tmp_path / "outputs" / "dense-run" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
    (checkpoint / "config.json").write_text('{"hidden_size": 768}\n')
    (checkpoint.parent / "run_config.json").write_text('{"model_family": "dense"}\n')
    model = FakeDense()
    monkeypatch.setattr("embed_optim.probe_export._load_model", lambda *args, **kwargs: model)
    output = tmp_path / "exports" / "dense.npz"
    probe_spec = tmp_path / "probe-spec.json"
    probe_spec.write_text(
        json.dumps(
            {
                "expected": {
                    "manifest_sha256": _sha256(probe / "manifest.json"),
                }
            }
        )
        + "\n"
    )

    exported, manifest_path = export_probe(
        checkpoint,
        probe,
        output,
        family="dense",
        batch_size=2,
        model_dtype="float32",
        storage_dtype="float32",
        device="cpu",
        flash_attention=False,
        probe_spec=probe_spec,
    )

    with np.load(exported, allow_pickle=False) as archive:
        assert archive["sample_ids"].tolist() == [11, 19]
        assert archive["sample_groups"].tolist() == ["fiqa", "nq"]
        assert archive["query_embeddings"].shape == (2, 768)
        assert archive["document_embeddings"].shape == (2, 8, 768)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["output"]["sha256"] == _sha256(exported)
    assert manifest["probe"]["manifest_sha256"] == _sha256(probe / "manifest.json")
    assert manifest["probe"]["frozen_spec"]["sha256"] == _sha256(probe_spec)
    assert {item["path"] for item in manifest["checkpoint_inputs"]} == {
        "config.json",
        "model.safetensors",
    }
    assert model.calls[0][1]["prompt"] == "query: "
    assert model.calls[1][1]["prompt"] == "document: "

    repeated, repeated_manifest_path = export_probe(
        checkpoint,
        probe,
        tmp_path / "exports" / "dense-repeated.npz",
        family="dense",
        batch_size=2,
        model_dtype="float32",
        storage_dtype="float32",
        device="cpu",
        flash_attention=False,
        probe_spec=probe_spec,
    )
    assert _sha256(repeated) == _sha256(exported)

    metrics = analyze_probe(
        exported,
        tmp_path / "metrics.json",
        family="dense",
        require_export_manifest=True,
        reference_source=repeated,
    )
    assert set(metrics["metrics"]["score_geometry"]["by_group"]) == {"fiqa", "nq"}
    assert metrics["input"]["export_manifest"]["sha256"] == _sha256(manifest_path)
    assert metrics["metrics"]["score_geometry"]["reference_ranking"]["mean_top_k_overlap"] == 1.0
    assert metrics["input"]["reference"]["sha256"] == _sha256(repeated)

    repeated_manifest = json.loads(repeated_manifest_path.read_text())
    repeated_manifest["probe"]["selection_sha256"] = "f" * 64
    repeated_manifest_path.write_text(json.dumps(repeated_manifest) + "\n")
    with pytest.raises(ValueError, match="probe_selection_sha256"):
        analyze_probe(
            exported,
            tmp_path / "mismatched-reference.json",
            family="dense",
            require_export_manifest=True,
            reference_source=repeated,
        )

    manifest["output"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        analyze_probe(exported, tmp_path / "corrupt.json", family="dense")


def test_late_export_uses_ragged_storage_and_is_analyzer_compatible(tmp_path: Path, monkeypatch):
    probe = _probe_fixture(tmp_path)
    checkpoint = tmp_path / "outputs" / "late-run" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
    (checkpoint / "config.json").write_text('{"hidden_size": 768}\n')
    (checkpoint.parent / "run_config.json").write_text('{"model_family": "late"}\n')
    monkeypatch.setattr("embed_optim.probe_export._load_model", lambda *args, **kwargs: FakeLate())

    exported, manifest_path = export_probe(
        checkpoint,
        probe,
        tmp_path / "exports" / "late.npz",
        family="late",
        batch_size=2,
        model_dtype="float32",
        storage_dtype="float32",
        device="cpu",
        flash_attention=False,
    )

    with np.load(exported, allow_pickle=False) as archive:
        assert "query_offsets" in archive.files
        assert "document_offsets" in archive.files
        assert "query_mask" not in archive.files
        assert archive["document_embeddings"].shape == (47, 128)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["encoding"]["late_storage"] == "ragged_offsets"

    metrics = analyze_probe(
        exported,
        tmp_path / "late-metrics.json",
        family="late",
        require_export_manifest=True,
        reference_source=exported,
    )
    assert metrics["metrics"]["storage"] == "ragged_offsets"
    assert metrics["metrics"]["score_geometry"]["reference_ranking"]["top1_agreement"] == 1.0
