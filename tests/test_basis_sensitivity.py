from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from embed_optim.basis_sensitivity import (
    HEAD_METRICS,
    RECORD_METRICS,
    _direction_metrics,
    _load_protocol,
    _validate_model_config,
    analyze_basis_sensitivity,
    audit_basis_sensitivity,
    functional_invariance_error,
    rope_commuting_angles,
    rotate_qk_rows,
)
from embed_optim.geometry import _sha256
from embed_optim.update_geometry import ALGORITHMS, UpdateOperatorConfig, replay_update_directions


def test_frozen_basis_protocol_uses_sentence_transformers_qkv_namespace():
    _, protocol = _load_protocol("configs/basis_sensitivity.json")

    assert protocol["architecture"]["qkv_tensor_template"] == ("0.layers.{layer}.attn.Wqkv.weight")
    assert protocol["metrics"]["full_tensor"] == list(RECORD_METRICS)
    assert protocol["metrics"]["selected_qk_heads"] == list(HEAD_METRICS)
    assert protocol["freeze_context"]["formal_basis_output_visible"] is False


def test_model_config_validation_accepts_legacy_modernbert_rope_keys(tmp_path: Path):
    architecture = {
        "model_type": "modernbert",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "attention_bias": False,
        "max_position_embeddings": 8192,
        "rope_bases": [10_000.0, 160_000.0],
    }
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "model_type": "modernbert",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "attention_bias": False,
            "max_position_embeddings": 8192,
            "local_rope_theta": 10_000.0,
            "global_rope_theta": 160_000.0,
        },
    )

    _validate_model_config(config, architecture)

    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["global_rope_theta"] = 80_000.0
    _write_json(config, payload)
    with pytest.raises(ValueError, match="architecture differs"):
        _validate_model_config(config, architecture)


def test_rope_commuting_transform_is_invertible_and_preserves_attention_logits():
    generator = torch.Generator().manual_seed(41)
    heads, head_dim, columns = 3, 8, 11
    hidden_size = heads * head_dim
    matrix = torch.randn(3 * hidden_size, columns, generator=generator, dtype=torch.float64)
    angles = rope_commuting_angles(heads, head_dim, 717, dtype=torch.float64)

    transformed = rotate_qk_rows(matrix, angles)
    restored = rotate_qk_rows(transformed, angles, inverse=True)

    torch.testing.assert_close(restored, matrix, rtol=0, atol=2e-15)
    torch.testing.assert_close(
        transformed[2 * hidden_size :], matrix[2 * hidden_size :], rtol=0, atol=0
    )
    assert (
        functional_invariance_error(
            angles,
            vector_seed=991,
            rope_bases=[10_000.0, 160_000.0],
            position_pairs=[[0, 1], [17, 53], [4096, 8191]],
        )
        < 1e-12
    )


def test_replayed_optimizers_have_distinct_basis_sensitivity():
    generator = torch.Generator().manual_seed(123)
    heads, head_dim, columns = 3, 4, 7
    gradients = [torch.randn(3 * heads * head_dim, columns, generator=generator) for _ in range(8)]
    angles = rope_commuting_angles(heads, head_dim, 2026082701)
    original = replay_update_directions(gradients)
    rotated_gradients = [rotate_qk_rows(gradient, angles) for gradient in gradients]
    rotated = replay_update_directions(rotated_gradients)
    errors = {
        algorithm: _direction_metrics(
            original[algorithm],
            rotate_qk_rows(rotated[algorithm], angles, inverse=True),
        )["mapped_relative_frobenius_error"]
        for algorithm in ALGORITHMS
    }

    # The actual training implementation uses a bfloat16 Newton--Schulz polynomial, so Muon's
    # equivariance is approximate rather than machine-precision exact. Coordinatewise Adam state
    # and NorMuon's rowwise state are substantially more sensitive on this fixed anisotropic case.
    assert errors["muon"] < 0.05
    assert errors["adamw"] > errors["muon"] + 0.08
    assert errors["normuon"] > errors["muon"] + 0.03

    # The exact polar reference is left-orthogonally equivariant; this isolates implementation
    # rounding from the mathematical property being diagnosed.
    final = gradients[-1].double()
    transformed_final = rotate_qk_rows(final, angles.double())
    left, _, right = torch.linalg.svd(final, full_matrices=False)
    transformed_left, _, transformed_right = torch.linalg.svd(
        transformed_final, full_matrices=False
    )
    polar = left @ right
    transformed_polar = transformed_left @ transformed_right
    mapped_polar = rotate_qk_rows(transformed_polar, angles.double(), inverse=True)
    torch.testing.assert_close(mapped_polar, polar, rtol=1e-11, atol=1e-11)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _basis_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    common_spec = tmp_path / "common-state.json"
    _write_json(
        common_spec,
        {
            "schema_version": 1,
            "selection": {"gradient_steps": 2},
            "operator_protocol": asdict(UpdateOperatorConfig()),
        },
    )
    common_root = tmp_path / "common-state"
    tensor_template = "0.layers.{layer}.attn.Wqkv.weight"
    tensor_name = tensor_template.format(layer=0)
    first_shard = None
    for family_index, family in enumerate(("dense", "late")):
        checkpoint = tmp_path / "checkpoints" / family
        _write_json(
            checkpoint / "config.json",
            {
                "model_type": "modernbert",
                "hidden_size": 4,
                "num_attention_heads": 2,
                "attention_bias": False,
                "max_position_embeddings": 64,
                "rope_parameters": {
                    "full_attention": {"rope_theta": 160_000.0},
                    "sliding_attention": {"rope_theta": 10_000.0},
                },
            },
        )
        gradient_root = common_root / family / "pretrained" / "gradients"
        gradient_root.mkdir(parents=True)
        generator = torch.Generator().manual_seed(100 + family_index)
        shards = []
        for step_index in range(2):
            shard = gradient_root / f"gradient-{step_index:04d}.safetensors"
            save_file({tensor_name: torch.randn(12, 4, generator=generator)}, shard)
            first_shard = shard if first_shard is None else first_shard
            shards.append(
                {
                    "path": shard.name,
                    "bytes": shard.stat().st_size,
                    "sha256": _sha256(shard),
                    "step_index": step_index,
                }
            )
        _write_json(
            gradient_root / "manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "checkpoint": {"path": str(checkpoint)},
                "common_state_spec": {"sha256": _sha256(common_spec)},
                "config": {"gradient_steps": 2},
                "gradient_shards": shards,
            },
        )

    protocol = tmp_path / "basis.json"
    _write_json(
        protocol,
        {
            "schema_version": 1,
            "status": "prospective_appendix_diagnostic",
            "common_state": {
                "root": str(common_root),
                "spec": str(common_spec),
                "spec_sha256": _sha256(common_spec),
                "expected_anchors": 2,
                "expected_anchors_per_family": 1,
                "gradient_steps": 2,
            },
            "architecture": {
                "model_type": "modernbert",
                "hidden_size": 4,
                "num_attention_heads": 2,
                "head_dim": 2,
                "attention_bias": False,
                "max_position_embeddings": 64,
                "qkv_layout": "contiguous_q_k_v",
                "qkv_tensor_template": tensor_template,
                "rotary_pairing": "split_half",
                "rope_bases": [10_000.0, 160_000.0],
            },
            "selection": {
                "families": ["dense", "late"],
                "layers": [0],
                "heads": [0, 1],
                "rotation_seeds": [11, 13],
                "optimizers": list(ALGORITHMS),
                "expected_tensor_sequences": 2,
                "expected_records": 12,
                "expected_head_records": 48,
                "expected_summary_rows": 6,
            },
            "transformation": {
                "group": "split-half rotary planes",
                "angle_distribution": "seeded uniform",
                "query_key_share_angles": True,
                "value_rows_unchanged": True,
                "map_rotated_updates_back_before_comparison": True,
                "function_preserving_reason": "commutes with RoPE",
            },
            "functional_calibration": {
                "dtype": "float64",
                "vector_seed": 17,
                "position_pairs": [[0, 1], [7, 31]],
                "maximum_absolute_logit_error": 1e-10,
            },
            "metrics": {
                "full_tensor": list(RECORD_METRICS),
                "selected_qk_heads": list(HEAD_METRICS),
                "causal_boundary": "coordinate diagnostic, not retrieval intervention",
            },
            "freeze_context": {
                "frozen_at_utc": "2026-08-27T06:20:09Z",
                "strict_beir_valid_units": 1,
                "strict_beir_expected_units": 1680,
                "complete_retrieval_matrix_visible": False,
                "common_state_output_visible": False,
                "formal_basis_output_visible": False,
                "completed_weight_trajectories_visible": True,
                "note": "test fixture frozen before outputs",
            },
        },
    )
    assert first_shard is not None
    return protocol, first_shard, common_root


def test_basis_analysis_is_complete_resumable_and_strictly_audited(tmp_path: Path):
    protocol, first_shard, _ = _basis_fixture(tmp_path)
    output = tmp_path / "reports"

    manifest = analyze_basis_sensitivity(
        protocol,
        output_dir=output,
        device="cpu",
        verify_inputs=True,
    )

    assert manifest["coverage"] == {
        "anchors": 2,
        "tensor_sequences": 2,
        "records": 12,
        "head_records": 48,
        "summary_rows": 6,
    }
    with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert {(row["family"], row["optimizer"]) for row in summary} == {
        (family, optimizer) for family in ("dense", "late") for optimizer in ALGORITHMS
    }
    assert audit_basis_sensitivity(protocol, output_dir=output, verify_inputs=True) == manifest

    before = (output / "records.csv").stat().st_mtime_ns
    resumed = analyze_basis_sensitivity(
        protocol,
        output_dir=output,
        device="cpu",
        verify_inputs=True,
    )
    assert resumed == manifest
    assert (output / "records.csv").stat().st_mtime_ns == before

    records_path = output / "records.csv"
    original_records = records_path.read_bytes()
    records_path.write_bytes(original_records + b"\n")
    with pytest.raises(ValueError, match="Basis output differs"):
        audit_basis_sensitivity(protocol, output_dir=output, verify_inputs=True)
    records_path.write_bytes(original_records)

    save_file(
        {"0.layers.0.attn.Wqkv.weight": torch.full((12, 4), 9.0)},
        first_shard,
    )
    with pytest.raises(ValueError, match="Gradient shard differs"):
        audit_basis_sensitivity(protocol, output_dir=output, verify_inputs=True)
