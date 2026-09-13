"""Synthetic encoder/storage controls, never primary scientific outputs."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from embed_optim import primary_v3_dimension_vector_io as vectors
from embed_optim.primary_contract import digest, file_identity, read_json


def arrays_fixture():
    rng = np.random.default_rng(20260907)
    identities = [
        {
            "sample_id": i,
            "source": f"task-{i // 16:02}",
            "query_id": f"q{i}",
            "candidate_ids_positive_first": [f"d{i}-{j}" for j in range(8)],
        }
        for i in range(224)
    ]
    arrays = {
        "sample_ids": np.arange(224, dtype=np.int64),
        "sample_groups": np.asarray([r["source"] for r in identities]),
        "query_embeddings": rng.standard_normal((224, 768)).astype(np.float32),
        "document_embeddings": rng.standard_normal((224, 8, 768)).astype(np.float32),
    }
    return arrays, identities


def model_fixture(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    original = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7}
    save_file(original, checkpoint / "model.safetensors")
    (checkpoint / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
            ]
        )
    )
    parameter = torch.nn.Parameter(original["weight"].to(torch.bfloat16))
    model = SimpleNamespace(
        training=False,
        max_seq_length=8192,
        _first_module=lambda: SimpleNamespace(can_flatten_inputs=False),
        parameters=lambda: [parameter],
        state_dict=lambda: {"0.model.weight": parameter},
    )
    return model, checkpoint, parameter


def saved(tmp_path):
    model, checkpoint, _ = model_fixture(tmp_path)
    before = vectors.verify_loaded(model, checkpoint)
    observed = {"before": before, "after": before, "parameters_unchanged": True}
    arrays, identities = arrays_fixture()
    output = tmp_path / "vectors"
    plan = {"scope": "synthetic-vector-storage-control", "scientific_completion": False}
    receipt = vectors.save_vectors(output, plan, arrays, observed, identities, checkpoint)
    return output, plan, identities, checkpoint, receipt, arrays


def test_observed_loading_exact_prefix_and_bf16_conversion(tmp_path):
    model, checkpoint, _ = model_fixture(tmp_path)
    value = vectors.verify_loaded(model, checkpoint)
    assert value["saved_weights_matched"] is True
    assert list(value["state_tensors"]) == ["0.model.weight"]
    assert value["state_tensors_sha256"] == digest(value["state_tensors"])


@pytest.mark.parametrize(
    "kind", ["training", "length", "flatten", "fp32", "weight", "extra", "bare", "former_prefix"]
)
def test_loaded_model_mismatch_refuses(tmp_path, kind):
    model, checkpoint, parameter = model_fixture(tmp_path)
    if kind == "training":
        model.training = True
    elif kind == "length":
        model.max_seq_length = 512
    elif kind == "flatten":
        model._first_module = lambda: SimpleNamespace(can_flatten_inputs=True)
    elif kind == "fp32":
        model.parameters = lambda: [parameter.float()]
    elif kind == "weight":
        with torch.no_grad():
            parameter[0, 0] += 1
    elif kind == "extra":
        model.state_dict = lambda: {"0.model.weight": parameter, "extra": parameter}
    elif kind == "former_prefix":
        model.state_dict = lambda: {"0.auto_model.weight": parameter}
    else:
        model.state_dict = lambda: {"weight": parameter}
    with pytest.raises(ValueError):
        vectors.verify_loaded(model, checkpoint)


def test_plain_directory_without_module_identity_is_not_adopted(tmp_path):
    model, checkpoint, _ = model_fixture(tmp_path)
    (checkpoint / "modules.json").rename(checkpoint / "preserved_modules.json")
    with pytest.raises(ValueError, match="module-0 namespace"):
        vectors.verify_loaded(model, checkpoint)


@pytest.mark.parametrize(
    "kind",
    [
        "rows",
        "dimension",
        "candidate_count",
        "fp16",
        "nan",
        "zero",
        "id_order",
        "group_order",
        "duplicate",
        "extra",
    ],
)
def test_raw_vector_shape_dtype_identity_refuses(kind):
    arrays, identities = arrays_fixture()
    if kind == "rows":
        arrays["query_embeddings"] = arrays["query_embeddings"][:-1]
    elif kind == "dimension":
        arrays["query_embeddings"] = arrays["query_embeddings"][:, :128]
    elif kind == "candidate_count":
        arrays["document_embeddings"] = arrays["document_embeddings"][:, :7]
    elif kind == "fp16":
        arrays["document_embeddings"] = arrays["document_embeddings"].astype(np.float16)
    elif kind == "nan":
        arrays["document_embeddings"][0, 0, 0] = np.nan
    elif kind == "zero":
        arrays["query_embeddings"][0] = 0
    elif kind == "id_order":
        arrays["sample_ids"] = arrays["sample_ids"][::-1]
    elif kind == "group_order":
        arrays["sample_groups"] = arrays["sample_groups"][::-1]
    elif kind == "duplicate":
        arrays["sample_ids"][0] = arrays["sample_ids"][1]
    else:
        arrays["unbound"] = np.zeros(1)
    with pytest.raises(ValueError):
        vectors.validate_arrays(arrays, identities)


def test_new_vectors_relocate_with_external_anchor(tmp_path):
    output, plan, identities, checkpoint, receipt, arrays = saved(tmp_path)
    relocated = tmp_path / "relocated"
    output.rename(relocated)
    actual, checked = vectors.inspect_vectors(
        relocated,
        plan,
        identities,
        checkpoint,
        expected_manifest_sha256=receipt["manifest"]["sha256"],
    )
    for key in arrays:
        np.testing.assert_array_equal(actual[key], arrays[key])
    assert checked["model_encoding_repeated"] is False


@pytest.mark.parametrize(
    "kind", ["vectors", "plan", "loaded", "candidate_order", "no_anchor", "extra", "symlink"]
)
def test_rehashed_vector_cache_cannot_create_its_own_trust(tmp_path, kind):
    output, plan, identities, checkpoint, receipt, arrays = saved(tmp_path)
    manifest = read_json(output / "manifest.json")
    anchor = receipt["manifest"]["sha256"]
    if kind == "vectors":
        arrays["document_embeddings"][:, [0, 1]] = arrays["document_embeddings"][:, [1, 0]]
        np.savez(output / "vectors.npz", **arrays)
        manifest["output"] = {"path": "vectors.npz", **file_identity(output / "vectors.npz")}
    elif kind == "plan":
        manifest["plan"]["scope"] = "historical"
    elif kind == "loaded":
        manifest["observed_loading"]["before"]["state_tensors"] = {}
    elif kind == "candidate_order":
        plan = {**plan, "candidate_order_changed": True}
    elif kind == "no_anchor":
        anchor = None
    elif kind == "extra":
        (output / "unbound").write_text("extra")
    else:
        target = output / "vectors.npz"
        target.rename(output / "payload.npz")
        target.symlink_to(output / "payload.npz")
    if kind in {"vectors", "plan", "loaded"}:
        (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        vectors.inspect_vectors(
            output, plan, identities, checkpoint, expected_manifest_sha256=anchor
        )


def test_actual_wrapper_preserves_arguments_order_and_detects_mutation(tmp_path, monkeypatch):
    from embed_optim import probe_export

    model, checkpoint, parameter = model_fixture(tmp_path)
    arrays, identities = arrays_fixture()
    dataset = {"sample_id": arrays["sample_ids"], "source": arrays["sample_groups"]}
    calls = []

    def load(family, source, **kwargs):
        assert family == "dense" and source == checkpoint
        assert kwargs == {"dtype": torch.bfloat16, "device": "cuda:0", "flash_attention": True}
        calls.append("load")
        return model

    def encode(current, data, **kwargs):
        assert current is model and data is dataset and torch.is_inference_mode_enabled()
        assert kwargs == {"batch_size": 8, "storage_dtype": np.dtype("float32")}
        calls.append("encode")
        return {key: arrays[key] for key in ("query_embeddings", "document_embeddings")}

    monkeypatch.setattr(probe_export, "_load_model", load)
    monkeypatch.setattr(probe_export, "encode_dense_probe", encode)
    actual, observation = vectors.encode_state(checkpoint, dataset, identities)
    assert calls == ["load", "encode"]
    assert observation["parameters_unchanged"] is True
    for key in arrays:
        np.testing.assert_array_equal(actual[key], arrays[key])
    original = copy.deepcopy(parameter.detach())

    def bad_encode(*args, **kwargs):
        result = encode(*args, **kwargs)
        with torch.no_grad():
            parameter.copy_(original + 1)
        return result

    monkeypatch.setattr(probe_export, "encode_dense_probe", bad_encode)
    with pytest.raises(ValueError):
        vectors.encode_state(checkpoint, dataset, identities)
