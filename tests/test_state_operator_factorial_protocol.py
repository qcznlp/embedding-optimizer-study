from __future__ import annotations

import hashlib
import json
from pathlib import Path

from embed_optim.config import load_matrix

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/dense_no_packing_state_operator_factorial_protocol.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_state_operator_factorial_protocol_is_self_consistent() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["schema_version"] == 1
    assert protocol["status"] == "prospective_scientific_lock_pending_implementation"

    for binding in protocol["parent_bindings"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]

    matrix = {
        config.run_id: config
        for config in load_matrix(ROOT / protocol["parent_bindings"]["matrix"]["path"])
    }
    states = protocol["source_states"]["states"]
    assert {state["label"] for state in states} == {"adamw_state", "muon_state"}
    assert protocol["source_states"]["checkpoint_step"] == 2345
    for state in states:
        config = matrix[state["run_id"]]
        assert config.optimizer.name == state["optimizer"]
        assert config.optimizer.lr == state["learning_rate"]
        assert config.dense_can_flatten_inputs is False
        assert Path(state["checkpoint"]) == (
            config.output_dir / f"checkpoint-{protocol['source_states']['checkpoint_step']}"
        )

    branch = protocol["branch_data"]
    branch_root = ROOT / branch["path"]
    manifest_path = branch_root / "manifest.json"
    ledger_path = branch_root / "rows.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert _sha256(manifest_path) == branch["manifest_sha256"]
    assert _sha256(ledger_path) == branch["row_ledger_sha256"]
    assert manifest["rows"] == branch["rows"] == 50_000
    assert manifest["selected_sample_ids_sha256"] == branch["selected_sample_ids_sha256"]

    design = protocol["factorial_design"]
    factors = design["factors"]
    seeds = branch["order_seeds"]
    assert design["expected_runs"] == (
        len(factors["weight_state"]) * len(factors["continuation_operator"]) * len(seeds)
    )
    assert factors == {
        "weight_state": ["adamw_state", "muon_state"],
        "continuation_operator": ["adamw", "muon"],
    }
    assert len(seeds) == len(set(seeds)) == 3
    assert design["optimizer_state_at_branch_start"].startswith("reset to zero")
    assert design["dense_input_execution"] == {
        "mode": "independently_padded",
        "sentence_transformers_can_flatten_inputs": False,
    }
    assert design["training"]["expected_optimizer_steps"] == 391
    assert design["training"]["expected_checkpoint_steps"] == [79, 157, 235, 313, 391]
    assert protocol["scale_calibration"]["target_global_hidden_update_to_weight"] == 5e-4
    assert protocol["evaluation"]["final_checkpoint"]["primary_metric"] == (
        "task-macro mean nDCG@10"
    )
    assert set(protocol["estimands"]) == {
        "coding",
        "weight_state_effect",
        "operator_effect",
        "state_operator_interaction",
        "uncertainty",
        "multiplicity",
    }
