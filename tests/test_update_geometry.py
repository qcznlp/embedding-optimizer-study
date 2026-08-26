from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from embed_optim.geometry import _sha256
from embed_optim.optimizers import EmbeddingOptimizer
from embed_optim.probe_export import _checkpoint_inputs
from embed_optim.update_geometry import (
    ALGORITHMS,
    UpdateOperatorConfig,
    analyze_common_state_updates,
    replay_update_directions,
)


def _optimizer(algorithm: str, parameter: nn.Parameter, lr: float) -> EmbeddingOptimizer:
    if algorithm == "adamw":
        group = {
            "params": [parameter],
            "algorithm": "adamw",
            "lr": lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
        }
    else:
        group = {
            "params": [parameter],
            "algorithm": algorithm,
            "lr": lr,
            "weight_decay": 0.0,
            "momentum": 0.95,
            "beta2": 0.95,
            "ns_steps": 5,
            "adjust_lr_fn": "original",
        }
    return EmbeddingOptimizer([group])


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_replayed_direction_matches_training_optimizer(algorithm: str):
    generator = torch.Generator().manual_seed(101)
    gradients = [torch.randn(8, 4, generator=generator) for _ in range(3)]
    original_gradients = [gradient.clone() for gradient in gradients]
    parameter = nn.Parameter(torch.randn(8, 4, generator=generator))
    # A moderate LR keeps float32 subtraction error below the optimizer comparison tolerance.
    lr = 1e-2
    optimizer = _optimizer(algorithm, parameter, lr)

    observed = None
    for gradient in gradients:
        before = parameter.detach().clone()
        parameter.grad = gradient.clone()
        optimizer.step()
        observed = (before - parameter.detach()) / lr

    expected = replay_update_directions(gradients)[algorithm]
    torch.testing.assert_close(observed, expected, rtol=2e-4, atol=2e-5)
    for before, after in zip(original_gradients, gradients, strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(
        {
            "encoder.layers.0.weight": torch.tensor(
                [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float32
            ),
            "encoder.embeddings.weight": torch.ones(4, 2),
        },
        checkpoint / "model.safetensors",
    )
    (checkpoint / "config.json").write_text('{"fixture": true}\n', encoding="utf-8")

    gradient_root = tmp_path / "gradients"
    gradient_root.mkdir()
    gradient_shards = []
    for index, gradient in enumerate(
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, -1.0]]),
            torch.tensor([[2.0, 1.0], [1.0, 3.0], [-1.0, 2.0]]),
        )
    ):
        path = gradient_root / f"gradient-{index:04d}.safetensors"
        save_file({"encoder.layers.0.weight": gradient}, path)
        gradient_shards.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "sample_ids": [index],
            }
        )
    manifest_path = gradient_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "checkpoint": {
                    "path": str(checkpoint),
                    "inputs": _checkpoint_inputs(checkpoint),
                },
                "gradient_shards": gradient_shards,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, manifest_path


def test_common_state_analysis_is_hashed_scale_matched_and_resumable(tmp_path: Path):
    checkpoint, gradient_manifest = _fixture(tmp_path)
    output = tmp_path / "analysis"
    manifest = analyze_common_state_updates(
        checkpoint,
        gradient_manifest,
        output,
        sketch_rank=2,
        oversample=0,
        power_iterations=0,
    )

    assert manifest["gradient_steps"] == 2
    assert manifest["tensors"] == 1
    assert manifest["parameters"] == 6
    record = json.loads((output / "metrics.jsonl").read_text(encoding="utf-8"))
    assert set(record["algorithms"]) == set(ALGORITHMS)
    assert set(record["pairwise_cosine"]) == {
        "adamw__muon",
        "adamw__normuon",
        "muon__normuon",
    }
    weight_norm = record["weight_frobenius_norm"]
    for algorithm in ALGORITHMS:
        path = output / f"{algorithm}-matched.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            update = handle.get_tensor("encoder.layers.0.weight")
            assert handle.metadata()["algorithm"] == algorithm
        assert torch.linalg.vector_norm(update).item() == pytest.approx(weight_norm, rel=1e-6)
        declared = manifest["outputs"][f"{algorithm}_matched"]
        assert declared["sha256"] == _sha256(path)

    before = (output / "metrics.jsonl").stat().st_mtime_ns
    resumed = analyze_common_state_updates(
        checkpoint,
        gradient_manifest,
        output,
        sketch_rank=2,
        oversample=0,
        power_iterations=0,
    )
    assert resumed == manifest
    assert (output / "metrics.jsonl").stat().st_mtime_ns == before


def test_common_state_analysis_rejects_changed_gradient_input(tmp_path: Path):
    checkpoint, gradient_manifest = _fixture(tmp_path)
    payload = json.loads(gradient_manifest.read_text(encoding="utf-8"))
    payload["gradient_shards"][0]["sha256"] = "0" * 64
    gradient_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Gradient shard identity mismatch"):
        analyze_common_state_updates(checkpoint, gradient_manifest, tmp_path / "analysis")


def test_update_operator_config_rejects_invalid_state_settings():
    with pytest.raises(ValueError, match="adam_beta2"):
        replay_update_directions([torch.ones(2, 2)], UpdateOperatorConfig(adam_beta2=1.0))
