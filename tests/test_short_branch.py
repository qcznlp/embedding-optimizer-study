from __future__ import annotations

import json
import math
from pathlib import Path

from embed_optim.geometry import _sha256
from embed_optim.short_branch import (
    _load_update_metrics,
    _optimizer_payload,
    _selected_source_ids,
    load_short_branch_protocol,
)


def test_frozen_short_branch_uses_shared_start_scale_and_order_seeds():
    path, protocol = load_short_branch_protocol("configs/short_branch_protocol.json")

    assert path.name == "short_branch_protocol.json"
    assert protocol["subset"]["count"] == 50_000
    assert protocol["shared_start"]["fraction"] == 0.6
    assert protocol["shared_start"]["checkpoint_step"] == 2345
    assert protocol["scale_calibration"]["target_global_hidden_update_to_weight"] == 5e-4
    assert protocol["training"]["order_seeds"] == [314159, 271828, 161803]
    assert protocol["training"]["expected_runs"] == 18
    assert protocol["evaluation"]["full_corpus_beir"] is False
    assert protocol["freeze_context"]["common_state_outputs_visible"] is False


def test_short_branch_subset_selection_is_fixed_and_proportional():
    _, protocol = load_short_branch_protocol("configs/short_branch_protocol.json")
    counts = {
        "fiqa": 10,
        "hotpotqa": 20,
        "msmarco": 30,
        "nq": 40,
        "fever": 50,
        "squadv2": 60,
        "trivia": 70,
    }
    rows = []
    for source, count in counts.items():
        rows.extend({"sample_id": len(rows), "source": source} for _ in range(count))
    protocol["subset"]["count"] = 28

    first, quotas = _selected_source_ids(rows, {"quotas": counts}, protocol)
    repeated, repeated_quotas = _selected_source_ids(rows, {"quotas": counts}, protocol)

    assert first == repeated
    assert quotas == repeated_quotas
    assert len(first) == 28
    assert first == sorted(set(first))
    assert sum(quotas.values()) == 28


def test_scale_calibration_uses_global_frobenius_ratio(tmp_path: Path):
    _, protocol = load_short_branch_protocol("configs/short_branch_protocol.json")
    checkpoint = tmp_path / "checkpoint-2345"
    checkpoint.mkdir()
    common_spec = tmp_path / "common.json"
    common_spec.write_text("{}\n", encoding="utf-8")
    update_root = tmp_path / "updates"
    update_root.mkdir()
    metrics_path = update_root / "metrics.jsonl"
    total_parameters = 110_297_088
    rows = []
    for index in range(88):
        parameters = total_parameters - 87 if index == 0 else 1
        rows.append(
            {
                "tensor": f"tensor.{index}",
                "parameters": parameters,
                "weight_frobenius_norm": 2.0,
                "algorithms": {
                    "adamw": {"frobenius_norm": 1.0},
                    "muon": {"frobenius_norm": 2.0},
                    "normuon": {"frobenius_norm": 4.0},
                },
            }
        )
    with metrics_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "schema_version": 1,
        "checkpoint": {"path": str(checkpoint)},
        "common_state_spec": {"sha256": _sha256(common_spec)},
        "gradient_steps": 8,
        "analysis_config": {"weight_decay_included": False},
        "outputs": {
            "metrics": {
                "path": "metrics.jsonl",
                "bytes": metrics_path.stat().st_size,
                "sha256": _sha256(metrics_path),
            }
        },
    }
    (update_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    protocol["shared_start"]["checkpoints"]["dense"] = str(checkpoint)
    protocol["scale_calibration"]["common_state_spec"] = str(common_spec)
    protocol["scale_calibration"]["update_metrics"]["dense"] = str(metrics_path)

    identity, ratios = _load_update_metrics("dense", protocol)

    assert identity["tensors"] == 88
    assert identity["parameters"] == total_parameters
    assert math.isclose(ratios["adamw"], 0.5)
    assert math.isclose(ratios["muon"], 1.0)
    assert math.isclose(ratios["normuon"], 2.0)


def test_short_branch_adamw_is_routed_hybrid_control():
    adamw = _optimizer_payload("adamw", 1e-5, 3e-6)
    muon = _optimizer_payload("muon", 1e-3, 3e-6)

    assert adamw["name"] == "hybrid_adamw"
    assert adamw["lr"] == 1e-5
    assert adamw["aux_lr"] == 3e-6
    assert "ns_implementation" not in adamw
    assert muon["name"] == "muon"
    assert muon["ns_implementation"] == "unfused-bfloat16-v1"
