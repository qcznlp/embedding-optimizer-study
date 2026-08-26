import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset

from embed_optim.geometry import _sha256
from embed_optim.probe_export import (
    encode_late_probe,
    export_probe,
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


def test_pad_variable_embeddings_preserves_lengths_and_dtype():
    values = [np.ones((2, 3)), np.full((1, 3), 2)]
    embeddings, mask = pad_variable_embeddings(values, storage_dtype=np.dtype("float16"))

    assert embeddings.shape == (2, 2, 3)
    assert embeddings.dtype == np.float16
    assert mask.tolist() == [[True, True], [True, False]]
    np.testing.assert_array_equal(embeddings[1, 1], np.zeros(3))


def test_late_encoder_keeps_positive_first_and_emits_masks():
    dataset = Dataset.from_list(_rows())
    model = FakeLate()

    arrays = encode_late_probe(
        model,
        dataset,
        batch_size=4,
        storage_dtype=np.dtype("float16"),
    )

    assert arrays["query_embeddings"].shape == (2, 2, 128)
    assert arrays["query_mask"].sum(axis=1).tolist() == [1, 2]
    assert arrays["document_embeddings"].shape == (2, 8, 4, 128)
    assert arrays["document_mask"].sum(axis=2)[0].tolist() == [2, 3, 4, 2, 3, 4, 2, 3]
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

    repeated, _ = export_probe(
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
    )
    assert set(metrics["metrics"]["score_geometry"]["by_group"]) == {"fiqa", "nq"}
    assert metrics["input"]["export_manifest"]["sha256"] == _sha256(manifest_path)

    manifest["output"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        analyze_probe(exported, tmp_path / "corrupt.json", family="dense")
