"""Observed model loading and trusted-anchor raw-vector bundles for fixed probes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .geometry import TensorStore
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_v3_dimension_contract import ENCODING, INPUT_EXECUTION
from .primary_v3_validation_io import write_new

ARRAYS = {"sample_ids", "sample_groups", "query_embeddings", "document_embeddings"}
MODEL_KEY_PREFIX = "0.model."


def _tensor_record(value):
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("Non-finite loaded model state")
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def saved_state_fingerprints(checkpoint):
    with TensorStore(Path(checkpoint)) as store:
        result = {}
        for name in store.keys():
            value = store.tensor(name)
            if value.is_floating_point():
                value = value.to(torch.bfloat16)
            # TensorStore's SentenceTransformer-directory reader already adds
            # the module name (0.). The pinned runtime owns its Transformer as
            # 0.model, not the former auto_model attribute. No arbitrary stripping.
            if not name.startswith("0."):
                raise ValueError("Expected the saved SentenceTransformer module-0 namespace")
            result[MODEL_KEY_PREFIX + name[2:]] = _tensor_record(value)
        return result


def verify_loaded(model, checkpoint):
    if (
        model.training
        or model.max_seq_length != ENCODING["max_length"]
        or bool(model._first_module().can_flatten_inputs)
    ):
        raise ValueError("Loaded model mode, context or input execution differs")
    if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError("Export model parameters must use the declared BF16 dtype")
    expected = saved_state_fingerprints(checkpoint)
    actual = {name: _tensor_record(value) for name, value in sorted(model.state_dict().items())}
    require_same(actual, expected)
    return {
        "model_mode": "eval",
        "max_length": 8192,
        "input_execution": INPUT_EXECUTION,
        "saved_dtype_conversion": "bfloat16",
        "state_tensors": actual,
        "state_tensors_sha256": digest(actual),
        "saved_weights_matched": True,
    }


def validate_arrays(arrays, identities):
    if set(arrays) != ARRAYS or len(identities) != 224:
        raise ValueError("Require the complete four-array, 224-row probe export")
    shapes = {
        "sample_ids": (224,),
        "sample_groups": (224,),
        "query_embeddings": (224, 768),
        "document_embeddings": (224, 8, 768),
    }
    for name, shape in shapes.items():
        if not isinstance(arrays[name], np.ndarray) or arrays[name].shape != shape:
            raise ValueError("Probe vector/sample shape differs")
    if arrays["sample_ids"].dtype != np.int64 or arrays["sample_groups"].dtype.kind != "U":
        raise ValueError("Probe identity dtypes differ")
    require_same(arrays["sample_ids"].tolist(), [row["sample_id"] for row in identities])
    require_same(arrays["sample_groups"].tolist(), [row["source"] for row in identities])
    if len(np.unique(arrays["sample_ids"])) != 224:
        raise ValueError("Duplicated probe sample IDs")
    for name in ("query_embeddings", "document_embeddings"):
        value = arrays[name]
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(
                "Raw vectors must be finite FP32, without historical FP16 archive rounding"
            )
        if np.any(np.linalg.norm(value.astype(np.float64), axis=-1) == 0):
            raise ValueError("Raw vector export contains a zero-norm embedding")
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(arrays.items())
    }


def encode_state(checkpoint, dataset, identities):
    """Actual encoder wrapper; only the reviewed top-level producer may call it formally."""
    from . import probe_export
    from .corrected_input_execution import require_independently_padded_dense

    model = probe_export._load_model(
        "dense",
        Path(checkpoint),
        dtype=torch.bfloat16,
        device=ENCODING["device"],
        flash_attention=True,
    )
    try:
        require_same(require_independently_padded_dense(model), INPUT_EXECUTION)
        before = verify_loaded(model, checkpoint)
        with torch.inference_mode():
            embeddings = probe_export.encode_dense_probe(
                model, dataset, batch_size=ENCODING["batch_size"], storage_dtype=np.dtype("float32")
            )
        after = verify_loaded(model, checkpoint)
        require_same(after, before)
        arrays = {
            "sample_ids": np.asarray(dataset["sample_id"], dtype=np.int64),
            "sample_groups": np.asarray(dataset["source"], dtype=str),
            **embeddings,
        }
        validate_arrays(arrays, identities)
        return arrays, {"before": before, "after": after, "parameters_unchanged": True}
    finally:
        del model


def inspect_vectors(output, plan, identities, checkpoint, *, expected_manifest_sha256):
    """Hash must come from trusted production/matrix receipt, never from this directory."""
    output = Path(output)
    if (
        output.is_symlink()
        or not output.is_dir()
        or not isinstance(expected_manifest_sha256, str)
        or len(expected_manifest_sha256) != 64
    ):
        raise ValueError("Require an ordinary vector bundle and an explicit trusted manifest hash")
    manifest_path = output / "manifest.json"
    if file_identity(manifest_path)["sha256"] != expected_manifest_sha256:
        raise ValueError("Vector manifest differs from the trusted production anchor")
    manifest = read_json(manifest_path)
    if (
        set(manifest) != {"status", "plan", "observed_loading", "output", "array_metadata"}
        or manifest["status"] != "complete"
    ):
        raise ValueError("Incomplete raw-vector manifest")
    require_same(manifest["plan"], plan)
    if {path.name for path in output.iterdir()} != {"vectors.npz", "manifest.json"} or manifest[
        "output"
    ]["path"] != "vectors.npz":
        raise ValueError("Unexpected raw-vector inventory or path")
    verify_file(output / "vectors.npz", manifest["output"])
    observed = manifest["observed_loading"]
    if (
        set(observed) != {"before", "after", "parameters_unchanged"}
        or observed["parameters_unchanged"] is not True
    ):
        raise ValueError("Missing immutable-loading observation")
    require_same(observed["before"], observed["after"])
    expected = saved_state_fingerprints(checkpoint)
    require_same(
        observed["before"],
        {
            "model_mode": "eval",
            "max_length": 8192,
            "input_execution": INPUT_EXECUTION,
            "saved_dtype_conversion": "bfloat16",
            "state_tensors": expected,
            "state_tensors_sha256": digest(expected),
            "saved_weights_matched": True,
        },
    )
    with np.load(output / "vectors.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    require_same(manifest["array_metadata"], validate_arrays(arrays, identities))
    verify_file(output / "vectors.npz", manifest["output"])
    if file_identity(manifest_path)["sha256"] != expected_manifest_sha256:
        raise ValueError("Vector manifest changed during readback")
    return arrays, {
        "manifest": {"path": str(manifest_path), **file_identity(manifest_path)},
        "output": manifest["output"],
        "array_metadata": manifest["array_metadata"],
        "model_encoding_repeated": False,
        "scientific_completion": False,
    }


def save_vectors(output, plan, arrays, observed, identities, checkpoint):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Use a new vector bundle; never overwrite or adopt partial exports")
    metadata = validate_arrays(arrays, identities)
    # Validate before writing, including the independently reconstructed loaded state.
    expected = saved_state_fingerprints(checkpoint)
    before = {
        "model_mode": "eval",
        "max_length": 8192,
        "input_execution": INPUT_EXECUTION,
        "saved_dtype_conversion": "bfloat16",
        "state_tensors": expected,
        "state_tensors_sha256": digest(expected),
        "saved_weights_matched": True,
    }
    require_same(observed, {"before": before, "after": before, "parameters_unchanged": True})
    output.mkdir(parents=True, exist_ok=False)
    with (output / "vectors.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    manifest = {
        "status": "complete",
        "plan": plan,
        "observed_loading": observed,
        "output": {"path": "vectors.npz", **file_identity(output / "vectors.npz")},
        "array_metadata": metadata,
    }
    write_new(output / "manifest.json", manifest)
    sha = file_identity(output / "manifest.json")["sha256"]
    _, receipt = inspect_vectors(output, plan, identities, checkpoint, expected_manifest_sha256=sha)
    return receipt
