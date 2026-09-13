"""Fresh-process readback of all real diagnostic geometry and rehashed corruptions."""

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from safetensors.torch import load_file, save_file
from threadpoolctl import threadpool_limits

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_geometry import tables_from_verified_runs
from embed_optim.primary_v3_geometry_io import inspect_run
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_v3_geometry import DIAG_SCOPE, binding, setup

PROTOCOL_SHA = "9f11cbcdfa8b75593cc3ebd22336e5a34073531426431da88cf904f6e470f358"


def replay(args):
    assert file_identity(args.rehearsal)["sha256"] == args.rehearsal_sha256
    rehearsal = read_json(args.rehearsal)
    assert rehearsal["actual_geometry_audit_passed"] is True
    assert rehearsal["geometry_protocol"]["sha256"] == PROTOCOL_SHA
    for row in (*rehearsal["unchanged_inputs_and_sources"], *rehearsal["artifacts"]):
        verify_file(row["path"], row)
    contract, _, _, reference, ref, admitted = setup(args.repository.resolve(), args.training_root)
    assert contract.sha256 == PROTOCOL_SHA
    require_same(
        [row["run_id"] for row in rehearsal["records"]],
        [row[1]["recipe"]["run_id"] for row in admitted],
    )
    results, checked_rows = [], []
    output_root = args.rehearsal.parent / "runs"
    for run, expected, checked in admitted:
        run_id = expected["recipe"]["run_id"]
        result = inspect_run(
            output_root / run_id,
            run,
            expected,
            checked,
            reference,
            ref,
            contract.payload["settings"],
            scope=DIAG_SCOPE,
            contract_sha=contract.sha256,
        )
        expected_readback = next(
            row["numeric_readback"] for row in rehearsal["records"] if row["run_id"] == run_id
        )
        require_same(result, expected_readback)
        checked_rows.append(result)
        results.append(
            {
                "run_id": run_id,
                "checkpoints": 3,
                "raw_matrix_records": 264,
                "numeric_readback_passed": True,
            }
        )
        print({"fresh_numeric_geometry_readback": run_id}, flush=True)
    tables = tables_from_verified_runs(admitted, output_root, ref, checked_rows)
    for name, rows in tables.items():
        assert (args.rehearsal.parent / f"{name}.csv").read_bytes() == csv_bytes(rows)
    changed_cases = []
    run, expected, checked = admitted[0]
    run_id = expected["recipe"]["run_id"]
    for kind in ("raw_metric", "nonzero_basis"):
        target = args.workdir / f"invalid-{kind}"
        shutil.copytree(output_root / run_id, target)
        manifest = read_json(target / "manifest.json")
        if kind == "raw_metric":
            bound = manifest["outputs"][0]["records"]
            path = target / bound["path"]
            value = read_json(path)
            value["records"][0]["weight"]["frobenius_norm"] += 1.0
            path.write_text(json.dumps(value))
            bound.update(file_identity(path))
        else:
            bound = manifest["outputs"][1]["bases"]
            path = target / bound["path"]
            values = load_file(path)
            values[sorted(values)[0]][0, 0] += 0.125
            save_file(
                values,
                path,
                metadata={
                    "scope": DIAG_SCOPE,
                    "run_identity_sha256": manifest["plan"]["run_identity_sha256"],
                },
            )
            bound.update(file_identity(path))
        (target / "manifest.json").write_text(json.dumps(manifest))
        try:
            inspect_run(
                target,
                run,
                expected,
                checked,
                reference,
                ref,
                contract.payload["settings"],
                scope=DIAG_SCOPE,
                contract_sha=contract.sha256,
            )
        except ValueError as error:
            changed_cases.append(
                {
                    "case": kind,
                    "rejected": True,
                    "error": str(error),
                    "mutated_file": binding(path),
                    "manifest": binding(target / "manifest.json"),
                }
            )
        else:
            raise AssertionError(f"Rehashed real diagnostic corruption accepted: {kind}")
        print({"real_rehashed_corruption_rejected": kind}, flush=True)
    for row in (*rehearsal["unchanged_inputs_and_sources"], *rehearsal["artifacts"]):
        verify_file(row["path"], row)
    require_same(handoff(), rehearsal["post_execution_dispatchers"])
    return {
        "scope": "engineering_dense_v3_geometry_fresh_cpu_replay",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "fresh_geometry_replay_passed": True,
        "rehearsal": binding(args.rehearsal),
        "source": binding(Path(__file__)),
        "models": 3,
        "checkpoints": 9,
        "raw_matrix_records": 792,
        "records": results,
        "changed_cases": changed_cases,
        "table_counts": {name: len(rows) for name, rows in tables.items()},
        "artifacts": [binding(path) for path in sorted(args.workdir.rglob("*")) if path.is_file()],
        "post_execution_dispatchers": handoff(),
        "model_updates": 0,
        "gpu_workers": 0,
        "formal_geometry_produced": False,
        "scientific_completion": False,
        "boundary": "Exact same-source/runtime CPU replay of short diagnostic trajectories and two rehashed semantic corruption controls, not primary geometry, cross-host equivalence, useful dimensions or optimizer findings.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "rehearsal", "workdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rehearsal-sha256", required=True)
    args = parser.parse_args()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not args.workdir.is_dir()
        or any(args.workdir.iterdir())
    ):
        raise ValueError("Require new CPU-only replay namespace")
    with threadpool_limits(limits=4):
        try:
            value = replay(args)
        except Exception as error:
            write_new(
                args.workdir / "failure.json",
                {
                    "scope": DIAG_SCOPE,
                    "observed_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "source": binding(Path(__file__)),
                    "replay_passed": False,
                    "scientific_completion": False,
                },
            )
            raise
    write_new(args.workdir / "result.json", value)
    print({"fresh_geometry_replay_passed": True, **file_identity(args.workdir / "result.json")})


if __name__ == "__main__":
    main()
