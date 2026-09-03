"""Validate the source-bound implementation lock for the state-by-operator factorial."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import resolve_matrix_path
from .geometry import SCHEMA_VERSION, _sha256

IMPLEMENTATION_PROTOCOL = Path(
    "configs/dense_no_packing_state_operator_factorial_implementation_protocol.json"
)


def require_factorial_implementation(
    path: str | Path = IMPLEMENTATION_PROTOCOL,
) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_matrix_path(path).resolve()
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported factorial implementation protocol schema")
    if protocol.get("status") != "prospective_state_operator_factorial_implementation_lock":
        raise ValueError("Factorial implementation is not prospectively locked")
    for group in ("parent_bindings", "source_bindings", "configuration_bindings"):
        bindings = protocol.get(group)
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError(f"Factorial implementation protocol lacks {group}")
        for label, identity in bindings.items():
            source = Path(identity["path"]).resolve()
            if (
                not source.is_file()
                or source.stat().st_size != identity.get("bytes")
                or _sha256(source) != identity.get("sha256")
            ):
                raise ValueError(f"Factorial {group} identity differs: {label}")
    cardinalities = protocol.get("cardinalities", {})
    if cardinalities != {
        "source_states": 2,
        "continuation_operators": 2,
        "order_seeds": 3,
        "training_runs": 12,
        "matrices": 6,
        "checkpoints_per_run": 5,
        "probe_checkpoint_jobs": 60,
        "probe_reference_jobs": 1,
        "final_beir_task_units": 168,
        "estimand_seed_task_cells": 126,
        "co_primary_estimands": 3,
    }:
        raise ValueError("Factorial implementation cardinalities differ")
    source_states = protocol.get("source_states", [])
    if [item.get("label") for item in source_states] != ["adamw_state", "muon_state"]:
        raise ValueError("Factorial implementation source-state identities differ")
    if any(item.get("checkpoint_step") != 2345 for item in source_states):
        raise ValueError("Factorial implementation source checkpoint step differs")
    if protocol.get("execution", {}).get("requires_clean_source_content_receipt") is not True:
        raise ValueError("Factorial implementation does not content-address source checkpoints")
    return resolved, protocol
