"""Explicit synthetic authoring for portable feature tests, never primary evidence.

Only source/probe identities and pretrained loading metadata are real. All
checkpoint admissions and raw embeddings are synthetic, and inference/outcome
authoring is simulated. The full-width mode uses the actual unchanged 768D
kernel at every state. The separately named reduced mode is only fast wiring.
"""

import copy
from pathlib import Path

import numpy as np

from embed_optim import primary_v3_dimensions as features
from embed_optim import reconstruction_files as files
from embed_optim.dimension_intervention_io import save_state
from embed_optim.dimension_interventions import compute_state
from embed_optim.primary_contract import digest, file_identity, read_json
from embed_optim.primary_v3_dimension_contract import (
    ENCODING,
    INPUT_EXECUTION,
    SCOPE,
    planned_states,
)
from embed_optim.primary_v3_dimension_inference import SCOPE as INFERENCE_SCOPE
from embed_optim.primary_v3_dimension_vector_io import validate_arrays
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_reconstruction_authoring import ACCEPTANCE, BOUNDARY, require_parent
from embed_optim.primary_v3_reconstruction_inputs import probe_identity, vector_admission
from embed_optim.primary_v3_reconstruction_sources import source_selection
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.runtime import runtime_snapshot


def reduced_kernel(meta, arrays, protocol):
    """8D numerical wiring with repeated attribution columns; explicitly not 768D numerics."""
    reduced = {
        key: value[..., :8] if key.endswith("embeddings") else value
        for key, value in arrays.items()
    }
    spec = copy.deepcopy(protocol)
    spec["inputs"]["embedding_dimension"] = 8
    result = compute_state(meta, reduced, spec)
    for values in [result["attributions"], *result["rotated_attributions"]]:
        for key in ("ndcg_removal_gain", "margin_removal_gain"):
            values[key] = np.tile(values[key], (1, 96))
    return result


def admitted_fixture(contract):
    root = contract.primary.repository
    actual = read_json(root / "reports/engineering-archive/dense-v3-dimension-chain-v1/audit.json")[
        "actual_loading"
    ]
    reference = actual["reference"]
    observed = {
        "before": actual["observation"],
        "after": actual["observation"],
        "parameters_unchanged": True,
    }
    common = {
        "scope": SCOPE,
        "dimension_protocol_sha256": contract.dimensions.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "probe": probe_identity(contract.dimensions),
        "encoding": ENCODING,
        "input_execution": INPUT_EXECUTION,
        "scientific_completion": False,
    }
    runs = {}
    for row in contract.primary.inputs["runs"]:
        run_id = row["run_id"]
        runs[run_id] = {
            "run_identity_sha256": digest(contract.primary.expected_identity(run_id)),
            "whole_run_artifacts_verified": True,
            "steps": contract.primary.payload["checkpoint_steps"],
            "checkpoints": [
                {"step": step, "files": reference["files"]}
                for step in contract.primary.payload["checkpoint_steps"]
            ],
            "simulated_admission": True,
            "scientific_completion": False,
        }
    states = []
    for state in planned_states(contract.primary):
        if state["cell"] == "pretrained":
            model = {"kind": "immutable_pretrained", "reference": reference}
        else:
            checked = runs[state["meta"]["run_id"]]
            model = {
                "kind": "complete_primary_checkpoint",
                "run_identity_sha256": checked["run_identity_sha256"],
                "complete_run_sha256": digest(checked),
                "checkpoint": checked["checkpoints"][state["stage"] - 1],
            }
        states.append({**common, "state": state, "model": model})
    admitted = {**common, "reference": reference, "complete_runs": runs, "states": states}
    vector_admission(contract.dimensions, admitted)
    return admitted, observed


def make(root, contract, *, kernel=compute_state, progress=None):
    root = Path(root)
    root.mkdir()
    producer = root / "producer"
    vectors, feature_root = producer / "vectors", producer / "features"
    vectors.mkdir(parents=True)
    feature_root.mkdir()
    admitted, observed = admitted_fixture(contract)
    identities = admitted["probe"]["row_identities"]
    write_new(vectors / "admission.json", admitted)
    records = {}
    for index, plan in enumerate(admitted["states"]):
        rng = np.random.default_rng(2026090700 + index)
        arrays = {
            "sample_ids": np.asarray([r["sample_id"] for r in identities], dtype=np.int64),
            "sample_groups": np.asarray([r["source"] for r in identities]),
            "query_embeddings": rng.standard_normal((224, 768)).astype(np.float32),
            "document_embeddings": rng.standard_normal((224, 8, 768)).astype(np.float32),
        }
        cell = plan["state"]["cell"]
        target = vectors / "states" / cell
        target.mkdir(parents=True)
        with (target / "vectors.npz").open("xb") as stream:
            np.savez(stream, **arrays)
        write_new(
            target / "manifest.json",
            {
                "status": "complete",
                "plan": plan,
                "observed_loading": observed,
                "output": {"path": "vectors.npz", **file_identity(target / "vectors.npz")},
                "array_metadata": validate_arrays(arrays, identities),
            },
        )
        records[cell] = {
            "path": f"states/{cell}/manifest.json",
            **file_identity(target / "manifest.json"),
        }
    write_new(
        vectors / "manifest.json",
        {
            "status": "complete",
            "admission": {"path": "admission.json", **file_identity(vectors / "admission.json")},
            "states": records,
            "scientific_completion": False,
        },
    )
    anchor = file_identity(vectors / "manifest.json")["sha256"]
    matrix_plan = features.feature_plan(contract.dimensions, admitted, anchor)
    write_new(feature_root / "admission.json", matrix_plan)
    aggregate = {name: [] for name in features.TABLE_COUNTS}
    states = {}
    for index, plan in enumerate(admitted["states"], 1):
        cell, state = plan["state"]["cell"], plan["state"]
        with np.load(vectors / "states" / cell / "vectors.npz", allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        result = kernel(state["meta"], arrays, contract.dimensions.scientific)
        features.check_result(result, state, sorted({r["source"] for r in identities}))
        target = feature_root / "states" / cell
        target.parent.mkdir(parents=True, exist_ok=True)
        save_state(target, features.state_plan(matrix_plan, {"plan": plan}, records[cell]), result)
        states[cell] = {
            "path": f"states/{cell}/manifest.json",
            **file_identity(target / "manifest.json"),
        }
        for name, rows in result["tables"].items():
            aggregate[name].extend(rows)
        if progress:
            progress({"fixture_states": index, "total_states": 61, "cell": cell})
    tables = {}
    for name, rows in aggregate.items():
        path = feature_root / (name + ".csv")
        with path.open("xb") as stream:
            stream.write(csv_bytes(rows))
        tables[name] = {"path": path.name, "rows": len(rows), **file_identity(path)}
    write_new(
        feature_root / "manifest.json",
        {"status": "complete", "plan": matrix_plan, "states": states, "tables": tables},
    )
    evidence = {
        "upstream_primary_admission_simulated": True,
        "synthetic_dimension_mode": "full_768" if kernel is compute_state else "reduced_8d_wiring",
        "producer_path_must_not_be_read": str(producer),
        "dimension_feature_reader": {
            "status": "complete",
            "manifest_sha256": file_identity(feature_root / "manifest.json")["sha256"],
            "plan": matrix_plan,
            "table_counts": features.TABLE_COUNTS,
            "raw_vector_states_recomputed": 61,
            "model_encoding_repeated": False,
            "scientific_completion": False,
        },
        "scientific_completion": False,
    }
    plan = {
        "scope": INFERENCE_SCOPE,
        "primary_protocol_sha256": contract.primary.sha256,
        "inference_protocol_sha256": contract.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "manuscript_installation_authorized": False,
    }
    inference = producer / "inference"
    inference.mkdir()
    write_new(inference / "evidence.json", evidence)
    write_new(inference / "manifest.json", {"plan": plan})
    selected = source_selection(contract, {})
    parent = require_parent(contract)
    files.select(selected, "source", ACCEPTANCE[0], parent, file_identity(parent))
    for role, location in (
        ("vectors", vectors),
        ("features", feature_root),
        ("inference", inference),
    ):
        for name in files.inventory(location):
            files.select(selected, role, name, location / name, file_identity(location / name))
    metadata = {
        "scope": "dense_primary_v3_checkpoint_backed_reconstruction_selection",
        "primary_protocol_sha256": contract.primary.sha256,
        "inference_protocol_sha256": contract.sha256,
        "inference_acceptance_sha256": ACCEPTANCE[1],
        "dimension_protocol_sha256": contract.dimensions.sha256,
        "trusted_vector_manifest_sha256": anchor,
        "authoring_plan": plan,
        "authoring_evidence_sha256": digest(evidence),
        "original_location_fields_preserved": True,
        "boundary": BOUNDARY,
        "authoring_runtime": runtime_snapshot(
            [
                "torch",
                "numpy",
                "scipy",
                "sympy",
                "sentence-transformers",
                "transformers",
                "accelerate",
                "mteb",
            ]
        ),
        "full_raw_vector_reconstruction_repeated_by_transport": False,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    receipt = files.write(root / "archive", selected, metadata)
    producer.rename(root / "preserved-producer-unavailable-at-original-location")
    return root / "archive", receipt["manifest_sha256"]
