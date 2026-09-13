"""Full-width simulated vectors/features under one externally supplied test admission."""

import numpy as np

from embed_optim import primary_v3_dimensions as features
from embed_optim.dimension_intervention_io import save_state
from embed_optim.dimension_interventions import compute_state
from embed_optim.primary_contract import file_identity, require_same
from embed_optim.primary_v3_dimension_vector_io import validate_arrays
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_reconstruction_inputs import vector_admission
from embed_optim.primary_v3_validation_io import write_new


def make(producer, contract, admitted, observed, *, progress=None):
    """No encoder, no reduced-width kernel, no independently invented run identity."""
    vector_admission(contract.dimensions, admitted)
    vectors, feature_root = producer / "vectors", producer / "features"
    vectors.mkdir()
    feature_root.mkdir()
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
    tasks = sorted({r["source"] for r in identities})
    for index, plan in enumerate(admitted["states"], 1):
        state = plan["state"]
        cell = state["cell"]
        with np.load(vectors / "states" / cell / "vectors.npz", allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        result = compute_state(state["meta"], arrays, contract.dimensions.scientific)
        features.check_result(result, state, tasks)
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
            progress({"joint_fixture_vector_states": index, "total_states": 61})
    require_same({name: len(rows) for name, rows in aggregate.items()}, features.TABLE_COUNTS)
    tables = {}
    for name, rows in aggregate.items():
        path = feature_root / (name + ".csv")
        with path.open("xb") as stream:
            stream.write(csv_bytes(rows))
        tables[name] = {"path": path.name, "rows": len(rows), **file_identity(path)}
    write_new(
        feature_root / "manifest.json",
        {
            "status": "complete",
            "plan": matrix_plan,
            "states": states,
            "tables": tables,
        },
    )
    reader = {
        "status": "complete",
        "manifest_sha256": file_identity(feature_root / "manifest.json")["sha256"],
        "plan": matrix_plan,
        "table_counts": features.TABLE_COUNTS,
        "raw_vector_states_recomputed": 61,
        "model_encoding_repeated": False,
        "scientific_completion": False,
    }
    return anchor, reader, aggregate
