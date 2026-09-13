"""Complete v3 raw-vector matrix with externally anchored, relocatable readback.

The inspector verifies saved files and loading provenance, not a second encoding.
All twelve whole-run gates precede any vector or feature output. No old checkpoint
name or self-rehashed cache can supply the missing primary identity.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .primary_contract import (
    digest,
    file_identity,
    read_json,
    relative_path,
    require_same,
    verify_file,
)
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_dimension_contract import (
    ENCODING,
    INPUT_EXECUTION,
    SCOPE,
    DimensionContract,
    planned_states,
)
from .primary_v3_dimension_vector_io import encode_state, inspect_vectors, save_vectors
from .primary_v3_validation_io import write_new
from .primary_weight_entries import reference_identity


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Require an ordinary matrix directory")
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Matrix inventory contains a symlink")
    return [p.relative_to(root).as_posix() for p in paths if p.is_file()]


def recheck_contract(contract):
    checked = DimensionContract.load(contract.path, contract.primary)
    require_same(checked.sha256, contract.sha256)
    return checked


def admission(contract, experiment_root, reference, probe_root):
    """All primary states first; plans contain only content identities, not host paths."""
    states = planned_states(contract.primary)
    require_same(states, contract.payload["states"])
    runs = contract.admit_runs(experiment_root)
    wanted_runs = {row["run_id"] for row in contract.primary.inputs["runs"]}
    if set(runs) != wanted_runs or len(runs) != 12:
        raise ValueError("Export requires every distinct declared primary run, without extras")
    for run_id, item in runs.items():
        checked = item["checked"]
        if checked["whole_run_artifacts_verified"] is not True or checked[
            "run_identity_sha256"
        ] != digest(contract.primary.expected_identity(run_id)):
            raise ValueError("Export lacks complete identity-bound primary admission")
    reference = Path(reference)
    if reference.is_symlink() or not reference.is_dir():
        raise ValueError("Use an ordinary authenticated reference copy")
    reference_bound = reference_identity(contract.primary, reference)
    dataset, probe = contract.probe(probe_root)
    common = {
        "scope": SCOPE,
        "dimension_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "probe": probe,
        "encoding": ENCODING,
        "input_execution": INPUT_EXECUTION,
        "scientific_completion": False,
    }
    jobs, checked_runs = [], {}
    for run_id, item in sorted(runs.items()):
        checked_runs[run_id] = item["checked"]
        require_same(
            [r["step"] for r in item["checked"]["checkpoints"]],
            contract.primary.payload["checkpoint_steps"],
        )
    for state in states:
        if state["cell"] == "pretrained":
            checkpoint = reference
            identity = {"kind": "immutable_pretrained", "reference": reference_bound}
        else:
            item = runs[state["meta"]["run_id"]]
            checkpoint = item["root"] / f"checkpoint-{state['meta']['step']}"
            checked = item["checked"]
            identity = {
                "kind": "complete_primary_checkpoint",
                "run_identity_sha256": checked["run_identity_sha256"],
                "complete_run_sha256": digest(checked),
                "checkpoint": checked["checkpoints"][state["stage"] - 1],
            }
        jobs.append(
            {"checkpoint": checkpoint, "plan": {**common, "state": state, "model": identity}}
        )
    admitted = {
        **common,
        "reference": reference_bound,
        "complete_runs": checked_runs,
        "states": [j["plan"] for j in jobs],
    }
    return dataset, probe["row_identities"], jobs, admitted


def _manifest(root, admitted, expected_manifest_sha256):
    root = Path(root)
    if not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64:
        raise ValueError("A trusted export matrix manifest SHA-256 is required")
    manifest_path = root / "manifest.json"
    if file_identity(manifest_path)["sha256"] != expected_manifest_sha256:
        raise ValueError("Export matrix differs from the trusted production anchor")
    manifest = read_json(manifest_path)
    if (
        set(manifest) != {"status", "admission", "states", "scientific_completion"}
        or manifest["status"] != "complete"
        or manifest["scientific_completion"] is not False
    ):
        raise ValueError("Incomplete export matrix")
    require_same(
        manifest["admission"], {"path": "admission.json", **file_identity(root / "admission.json")}
    )
    require_same(read_json(root / "admission.json"), admitted)
    cells = [state["state"]["cell"] for state in admitted["states"]]
    if set(manifest["states"]) != set(cells) or len(cells) != 61:
        raise ValueError("Export matrix requires exactly the complete 61-state population")
    expected_files = ["admission.json", "manifest.json"]
    for cell in cells:
        name = f"states/{relative_path(cell)}/manifest.json"
        if manifest["states"][cell]["path"] != name:
            raise ValueError("A state manifest path differs from the admitted cell")
        verify_file(root / name, manifest["states"][cell])
        child = read_json(root / name)
        if child["output"]["path"] != "vectors.npz":
            raise ValueError("Unexpected raw-vector payload path")
        verify_file(root / "states" / cell / "vectors.npz", child["output"])
        expected_files.extend([name, f"states/{cell}/vectors.npz"])
    require_same(inventory(root), sorted(expected_files))
    return manifest


def iter_vectors(root, manifest, jobs, identities):
    """One state at a time: bounded memory, all samples in the accepted order."""
    for job in jobs:
        cell = job["plan"]["state"]["cell"]
        arrays, checked = inspect_vectors(
            Path(root) / "states" / cell,
            job["plan"],
            identities,
            job["checkpoint"],
            expected_manifest_sha256=manifest["states"][cell]["sha256"],
        )
        yield job, arrays, checked


def inspect_export(
    contract, experiment_root, reference, probe_root, output, *, expected_manifest_sha256
):
    recheck_contract(contract)
    _, identities, jobs, admitted = admission(contract, experiment_root, reference, probe_root)
    manifest = _manifest(output, admitted, expected_manifest_sha256)
    verified = []
    for job, _, checked in iter_vectors(output, manifest, jobs, identities):
        cell = job["plan"]["state"]["cell"]
        verified.append(
            {
                "cell": cell,
                "manifest": manifest["states"][cell],
                "array_metadata": checked["array_metadata"],
            }
        )
    _, after_ids, _, after = admission(contract, experiment_root, reference, probe_root)
    require_same(after_ids, identities)
    require_same(after, admitted)
    _manifest(output, admitted, expected_manifest_sha256)
    recheck_contract(contract)
    return {
        "status": "complete",
        "admission_sha256": digest(admitted),
        "manifest_sha256": expected_manifest_sha256,
        "states": verified,
        "model_encoding_repeated": False,
        "scientific_completion": False,
    }


def write_matrix(contract, experiment_root, reference, probe_root, output):
    """Internal producer under execute's release/runtime/lease guard, never a CLI bypass."""
    contract.require_execution()
    dataset, identities, jobs, admitted = admission(
        contract, experiment_root, reference, probe_root
    )
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Use a new complete matrix namespace; partial exports remain preserved")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "admission.json", admitted)
    states = {}
    for job in jobs:
        arrays, observed = encode_state(job["checkpoint"], dataset, identities)
        cell = job["plan"]["state"]["cell"]
        saved = save_vectors(
            output / "states" / cell, job["plan"], arrays, observed, identities, job["checkpoint"]
        )
        states[cell] = {
            "path": f"states/{cell}/manifest.json",
            "bytes": saved["manifest"]["bytes"],
            "sha256": saved["manifest"]["sha256"],
        }
        del arrays
    # Recheck all checkpoint and probe bytes before declaring the entire output complete.
    _, after_ids, _, after = admission(contract, experiment_root, reference, probe_root)
    require_same(after_ids, identities)
    require_same(after, admitted)
    contract.require_execution()
    write_new(
        output / "manifest.json",
        {
            "status": "complete",
            "admission": {"path": "admission.json", **file_identity(output / "admission.json")},
            "states": states,
            "scientific_completion": False,
        },
    )
    trusted_sha = file_identity(output / "manifest.json")["sha256"]
    return inspect_export(
        contract,
        experiment_root,
        reference,
        probe_root,
        output,
        expected_manifest_sha256=trusted_sha,
    )


def execute(contract, experiment_root, reference, probe_root, output):
    contract = contract.require_execution()  # Must fail before runtime, leases, model or output.
    from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens
    from .runtime import verify_runtime_spec

    tokens = parse_gpu_tokens(os.environ.get("CUDA_VISIBLE_DEVICES", ""), expected_count=1)
    if not set(tokens).issubset(
        {g for pool in contract.primary.payload["gpu_pools"] for g in pool}
    ):
        raise ValueError("Functional export requested an undeclared GPU")
    verify_runtime_spec(contract.primary.repository / "configs/formal_runtime.json")
    admission(contract, experiment_root, reference, probe_root)
    if Path(output).exists() or Path(output).is_symlink():
        raise ValueError("Export requires a new output namespace")
    with acquire_gpu_lease(
        tokens,
        lock_dir=contract.primary.payload["gpu_lease_root"],
        timeout_seconds=60,
        purpose="primary-v3-dimension-export",
    ):
        return write_matrix(contract, experiment_root, reference, probe_root, output)


def cli_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    for name in (
        "repository",
        "training-root",
        "experiment-root",
        "protocol",
        "reference",
        "probe-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main():
    parser = cli_arguments(__doc__)
    parser.add_argument("action", choices=("plan", "export", "inspect"))
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    primary = PrimaryV3Contract.load(
        args.repository / "configs/dense_primary_v3_protocol.json",
        args.repository,
        args.training_root,
    )
    contract = DimensionContract.load(args.protocol, primary)
    inputs = (contract, args.experiment_root, args.reference, args.probe_root, args.output)
    if args.action == "plan":
        _, _, jobs, admitted = admission(*inputs[:-1])
        print(
            {
                "states": len(jobs),
                "admission_sha256": digest(admitted),
                "scientific_completion": False,
            }
        )
    elif args.action == "export":
        print(execute(*inputs))
    else:
        print(inspect_export(*inputs, expected_manifest_sha256=args.expected_manifest_sha256))


if __name__ == "__main__":
    main()
