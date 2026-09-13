"""Complete literal-dimension feature matrix, freshly reconstructed from trusted vectors.

This is an engineering input/feature interface, not a publication or mechanism
claim. Four-family inference and the separate named predictor bridge are later
consumers. Every per-task coordinate array and every mask/rotation is retained.
"""

from __future__ import annotations

from pathlib import Path

from .dimension_intervention_io import inspect_state, save_state
from .dimension_interventions import compute_state
from .primary_contract import (
    digest,
    file_identity,
    read_json,
    relative_path,
    require_same,
    verify_file,
)
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_dimension_contract import SCOPE, TABLE_COUNTS, DimensionContract
from .primary_v3_dimension_exports import (
    _manifest,
    admission,
    cli_arguments,
    inventory,
    iter_vectors,
    recheck_contract,
)
from .primary_v3_outcomes import csv_bytes
from .primary_v3_validation_io import write_new


def feature_plan(contract, admitted, vector_manifest_sha256):
    return {
        "scope": SCOPE,
        "dimension_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "input_admission_sha256": digest(admitted),
        "trusted_vector_manifest_sha256": vector_manifest_sha256,
        "numerical_policy": contract.payload["numerical_policy"],
        "table_counts": TABLE_COUNTS,
        "complete_state_count": 61,
        "scientific_completion": False,
        "publication_inference_verified": False,
    }


def state_plan(plan, job, raw_manifest):
    return {
        "matrix_plan": plan,
        "state": job["plan"]["state"],
        "raw_vector_manifest": raw_manifest,
    }


def check_result(result, state, task_names):
    """Exact per-state population before any aggregate can hide missing rows."""
    rotations = 3 if state["stage"] in (0, 5) else 0
    expected_counts = {
        "checkpoint_summary": 1,
        "task_summary": 14,
        "random_removal": 1120,
        "rotation_summary": 14 * rotations,
    }
    require_same({key: len(rows) for key, rows in result["tables"].items()}, expected_counts)
    require_same(sorted(row["task"] for row in result["tables"]["task_summary"]), task_names)
    require_same(result["attributions"]["task_groups"].tolist(), task_names)
    if (
        len(result["rotation_checks"]) != rotations
        or len(result["rotated_attributions"]) != rotations
    ):
        raise ValueError("Endpoint rotations are incomplete")
    for values in [result["attributions"], *result["rotated_attributions"]]:
        require_same(values["task_groups"].tolist(), task_names)
        for name in ("ndcg_removal_gain", "margin_removal_gain"):
            array = values[name]
            if array.shape != (14, 768) or str(array.dtype) != "float64":
                raise ValueError("All FP64 task-by-coordinate attributions must be retained")


def _feature_manifest(output, plan, jobs):
    output = Path(output)
    value = read_json(output / "manifest.json")
    if set(value) != {"status", "plan", "states", "tables"} or value["status"] != "complete":
        raise ValueError("Incomplete functional feature matrix")
    require_same(value["plan"], plan)
    require_same(read_json(output / "admission.json"), plan)
    cells = [job["plan"]["state"]["cell"] for job in jobs]
    if (
        set(value["states"]) != set(cells)
        or len(cells) != 61
        or set(value["tables"]) != set(TABLE_COUNTS)
    ):
        raise ValueError("Feature population or complete table family differs")
    names = ["manifest.json", "admission.json"]
    for cell in cells:
        record = value["states"][cell]
        if record["path"] != f"states/{cell}/manifest.json":
            raise ValueError("Feature state path differs")
        verify_file(output / record["path"], record)
        child = read_json(output / record["path"])
        for name, binding in child["outputs"].items():
            if binding["path"] != name:
                raise ValueError("Feature payload path differs")
            verify_file(output / "states" / cell / relative_path(name), binding)
        names.extend(
            f"states/{cell}/{name}"
            for name in [
                "manifest.json",
                "coordinate_attribution.npz",
                "numerical_evidence.json",
                *(k + ".csv" for k in TABLE_COUNTS),
            ]
        )
    for key, record in value["tables"].items():
        if record["path"] != key + ".csv" or record["rows"] != TABLE_COUNTS[key]:
            raise ValueError("Feature aggregate path or declared count differs")
        verify_file(output / record["path"], record)
        names.append(record["path"])
    require_same(inventory(output), sorted(names))
    return value


def process(
    contract,
    experiment_root,
    reference,
    probe_root,
    vectors,
    output,
    *,
    expected_vector_manifest_sha256,
    write=False,
):
    """CPU-only full reconstruction. Saved summaries never stand in for raw vectors."""
    recheck_contract(contract)
    _, identities, jobs, admitted = admission(contract, experiment_root, reference, probe_root)
    manifest = _manifest(vectors, admitted, expected_vector_manifest_sha256)
    plan = feature_plan(contract, admitted, expected_vector_manifest_sha256)
    output = Path(output)
    if write:
        if output.exists() or output.is_symlink():
            raise ValueError(
                "Use a new feature matrix; preserve every partial or prior calculation"
            )
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "admission.json", plan)
    else:
        _feature_manifest(output, plan, jobs)
    tables = {key: [] for key in TABLE_COUNTS}
    state_receipts = {}
    task_names = sorted({row["source"] for row in identities})
    for job, arrays, _ in iter_vectors(vectors, manifest, jobs, identities):
        state = job["plan"]["state"]
        cell = state["cell"]
        result = compute_state(state["meta"], arrays, contract.scientific)
        check_result(result, state, task_names)
        current = state_plan(plan, job, manifest["states"][cell])
        location = output / "states" / cell
        if write:
            location.parent.mkdir(parents=True, exist_ok=True)
            save_state(location, current, result)
        else:
            inspect_state(location, current, result)
        state_receipts[cell] = {
            "path": f"states/{cell}/manifest.json",
            **file_identity(location / "manifest.json"),
        }
        for key, rows in result["tables"].items():
            tables[key].extend(rows)
        del arrays, result
    require_same({key: len(rows) for key, rows in tables.items()}, TABLE_COUNTS)
    # Every raw vector/model/data identity is checked again before accepting the matrix.
    _, after_ids, _, after = admission(contract, experiment_root, reference, probe_root)
    require_same(after_ids, identities)
    require_same(after, admitted)
    _manifest(vectors, admitted, expected_vector_manifest_sha256)
    recheck_contract(contract)
    records = {}
    for key, rows in tables.items():
        data = csv_bytes(rows)
        path = output / (key + ".csv")
        if write:
            with path.open("xb") as stream:
                stream.write(data)
        elif path.read_bytes() != data:
            raise ValueError(f"Aggregate differs from full raw-vector reconstruction: {key}")
        records[key] = {"path": key + ".csv", "rows": len(rows), **file_identity(path)}
    expected = {"status": "complete", "plan": plan, "states": state_receipts, "tables": records}
    if write:
        write_new(output / "manifest.json", expected)
    require_same(_feature_manifest(output, plan, jobs), expected)
    return {
        "status": "complete",
        "manifest_sha256": file_identity(output / "manifest.json")["sha256"],
        "plan": plan,
        "table_counts": TABLE_COUNTS,
        "raw_vector_states_recomputed": len(jobs),
        "model_encoding_repeated": False,
        "scientific_completion": False,
    }


def main():
    parser = cli_arguments(__doc__)
    parser.add_argument("action", choices=("compute", "inspect"))
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--expected-vector-manifest-sha256", required=True)
    args = parser.parse_args()
    primary = PrimaryV3Contract.load(
        args.repository / "configs/dense_primary_v3_protocol.json",
        args.repository,
        args.training_root,
    )
    contract = DimensionContract.load(args.protocol, primary)
    print(
        process(
            contract,
            args.experiment_root,
            args.reference,
            args.probe_root,
            args.vectors,
            args.output,
            expected_vector_manifest_sha256=args.expected_vector_manifest_sha256,
            write=args.action == "compute",
        )
    )


if __name__ == "__main__":
    main()
