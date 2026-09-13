"""Actual CPU geometry integration and independent full-spectrum diagnostic audit."""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from embed_optim.corrected_geometry_summary import (
    _basis_seed,
    _checkpoint_rows,
    _optimizer_pair_rows,
    _pair_subspace_rows,
    _top_bases,
)
from embed_optim.geometry import TensorStore, analyze_run
from embed_optim.primary_completion import inspect_complete_run
from embed_optim.primary_contract import digest, file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_geometry import SOURCES, GeometryContract, tables_from_verified_runs
from embed_optim.primary_v3_geometry_io import inspect_run, produce_run, read_stage_bases
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.primary_weight_entries import reference_identity
from scripts.audit_dense_natural_data import handoff

DIAG_SCOPE = "engineering_dense_v3_geometry_diagnostic"
NATURAL_SHA = "bf6e893e3b9e244242871227e28a3729acc17477dd3c989f7c647308bf332d36"
EXACT_NAMES = tuple(
    f"0.layers.0.{suffix}.weight" for suffix in ("attn.Wo", "attn.Wqkv", "mlp.Wi", "mlp.Wo")
)
SCALAR_RTOL, SCALAR_ATOL = 2e-5, 1e-10


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), **file_identity(path)}


def setup(root, training_root):
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, training_root
    )
    contract = GeometryContract.load(
        root / "configs/dense_primary_v3_geometry_protocol.json", primary
    )
    torch.set_num_threads(contract.payload["settings"]["cpu_threads"])
    natural_path = Path("/tmp/dense-natural-readiness.a4xZzR/result.json")
    assert file_identity(natural_path)["sha256"] == NATURAL_SHA
    natural = read_json(natural_path)
    for row in natural["artifacts"]:
        verify_file(row["path"], row)
    reference = Path("/tmp/dense-weight-entry-reference.Um8TxW")
    reference_bound = reference_identity(primary, reference)
    admitted = []
    require_same([row["algorithm"] for row in natural["records"]], ["adamw", "muon", "normuon"])
    for source in natural["records"]:
        run = Path(source["run_root"])
        expected = read_json(run / "dense_run_contract.json")
        assert digest(expected) == source["checked"]["run_identity_sha256"]
        require_same(expected["model"], primary.inputs["common_identity"]["model"])
        checked = inspect_complete_run(run, expected, [1, 2, 3])
        admitted.append((run, expected, checked))
    return contract, natural_path, natural, reference, reference_bound, admitted


def views(admitted):
    return [
        SimpleNamespace(
            run_id=expected["recipe"]["run_id"],
            model_family="dense",
            output_dir=run,
            optimizer=SimpleNamespace(
                **{key: expected["recipe"]["optimizer"][key] for key in ("name", "lr")}
            ),
            checkpoint_fractions=tuple(
                stage / len(checked["steps"]) for stage in range(1, len(checked["steps"]) + 1)
            ),
        )
        for run, expected, checked in admitted
    ]


def compare_scalar(actual, expected):
    assert math.isclose(actual, expected, rel_tol=SCALAR_RTOL, abs_tol=SCALAR_ATOL), (
        actual,
        expected,
    )


def full_spectrum_checks(admitted, output_root, reference, reference_bound, settings):
    results = []
    for run, expected, checked in admitted:
        step = checked["steps"][-1]
        run_id = expected["recipe"]["run_id"]
        payload = read_json(output_root / run_id / f"checkpoint-{step}.json")
        records = {row["tensor"]: row for row in payload["records"]}
        stored_bases = read_stage_bases(
            output_root / run_id, step, reference_bound["hidden_shapes"], "cumulative"
        )
        with TensorStore(run / f"checkpoint-{step}") as current, TensorStore(reference) as initial:
            for name in EXACT_NAMES:
                delta = current.tensor(name).float() - initial.tensor(name).float()
                matrix = delta.numpy().astype(np.float64)
                singular = np.linalg.svd(matrix, compute_uv=False)
                norm = float(np.linalg.norm(matrix))
                stable = float(np.sum(singular**2) / singular[0] ** 2)
                probabilities = singular / np.sum(singular)
                full_entropy_rank = float(
                    np.exp(
                        -np.sum(
                            probabilities[probabilities > 0]
                            * np.log(probabilities[probabilities > 0])
                        )
                    )
                )
                exact_top_energy = float(np.sum(singular[:64] ** 2) / np.sum(singular**2))
                record = records[name]["delta_from_reference"]
                compare_scalar(record["frobenius_norm"], norm)
                row_norms = np.sqrt(np.sum(matrix**2, axis=1))
                compare_scalar(
                    record["row_norms"]["cv"], float(np.std(row_norms) / np.mean(row_norms))
                )
                assert record["spectral_norm"] <= singular[0] * (1 + SCALAR_RTOL) + SCALAR_ATOL
                assert record["approx_stable_rank"] >= stable * (1 - SCALAR_RTOL) - SCALAR_ATOL
                assert (
                    record["captured_frobenius_energy"]
                    <= exact_top_energy * (1 + SCALAR_RTOL) + SCALAR_ATOL
                )
                assert 0 < record["sketched_entropy_effective_rank"] <= 64 * (1 + SCALAR_RTOL)
                old = _top_bases(
                    delta,
                    rank=16,
                    oversample=8,
                    power_iterations=2,
                    seed=_basis_seed("cumulative", 3, name, settings["seed"]),
                )
                assert all(torch.equal(a, b) for a, b in zip(old, stored_bases[name], strict=True))
                for value in stored_bases[name]:
                    basis = value.numpy().astype(np.float64)
                    assert np.max(np.abs(basis.T @ basis - np.eye(16))) <= 2e-5
                results.append(
                    {
                        "run_id": run_id,
                        "step": step,
                        "tensor": name,
                        "shape": list(matrix.shape),
                        "exact_frobenius_norm": norm,
                        "exact_spectral_norm": float(singular[0]),
                        "sketched_spectral_norm": record["spectral_norm"],
                        "relative_spectral_underestimate": float(
                            (singular[0] - record["spectral_norm"]) / singular[0]
                        ),
                        "exact_stable_rank": stable,
                        "approx_stable_rank": record["approx_stable_rank"],
                        "full_spectrum_entropy_effective_rank": full_entropy_rank,
                        "top64_sketch_entropy_effective_rank": record[
                            "sketched_entropy_effective_rank"
                        ],
                        "exact_top64_captured_energy": exact_top_energy,
                        "sketched_top64_captured_energy": record["captured_frobenius_energy"],
                        "exact_relative_rank16_boundary_gap": float(
                            (singular[15] - singular[16]) / singular[15]
                        ),
                        "selected_old_bases_equal_exactly": True,
                        "variational_bounds_passed": True,
                        "scientific_completion": False,
                    }
                )
    assert len(results) == 12
    return results


def run_audit(root, training_root, work):
    contract, natural_path, natural, reference, ref, admitted = setup(root, training_root)
    original = handoff()
    bound = [binding(root / name) for name in SOURCES]
    bound += [binding(contract.path), binding(natural_path), binding(Path(__file__))]
    bound += natural["artifacts"]
    bound += [binding(reference / row["path"]) for row in ref["files"]]
    (work / "runs").mkdir()
    (work / "legacy").mkdir()
    checked_rows, records = [], []
    for run, expected, checked in admitted:
        run_id = expected["recipe"]["run_id"]
        algorithm = expected["recipe"]["optimizer"]["name"]
        output = work / "runs" / run_id
        produce_run(
            run,
            expected,
            checked,
            reference,
            ref,
            contract.payload["settings"],
            output,
            scope=DIAG_SCOPE,
            contract_sha=contract.sha256,
        )
        readback = inspect_run(
            output,
            run,
            expected,
            checked,
            reference,
            ref,
            contract.payload["settings"],
            scope=DIAG_SCOPE,
            contract_sha=contract.sha256,
        )
        checked_rows.append(readback)
        parent_census_path = (
            root / f"reports/engineering-archive/dense-v3-geometry-v1/{algorithm}-census.json"
        )
        parent_census = read_json(parent_census_path)
        assert parent_census["run_identity_sha256"] == digest(expected)
        bound.append(binding(parent_census_path))
        legacy = work / "legacy" / "dense" / f"{run_id}-rank64"
        analyze_run(
            run,
            legacy,
            reference=reference,
            **{
                key: contract.payload["settings"][key]
                for key in ("sketch_rank", "oversample", "power_iterations", "seed")
            },
        )
        for stage, step in enumerate(checked["steps"]):
            value = read_json(output / f"checkpoint-{step}.json")
            old = [
                json.loads(line)
                for line in (legacy / f"records/checkpoint-{step}.jsonl").read_text().splitlines()
            ]
            require_same(value["records"], old)
            require_same(value["entry_records"], parent_census["checkpoints"][stage]["records"])
        records.append(
            {
                "run_root": str(run),
                "run_id": run_id,
                "algorithm": algorithm,
                "output": str(output),
                "numeric_readback": readback,
            }
        )
        print({"actual_legacy_spectral_records_agree": run_id, "records": 264}, flush=True)
    tables = tables_from_verified_runs(admitted, work / "runs", ref, checked_rows)
    actual_counts = {name: len(rows) for name, rows in tables.items()}
    require_same(
        actual_counts,
        {
            "checkpoint_geometry": 9,
            "run_pair_subspace_overlap": 18,
            "optimizer_pair_subspace_summary": 18,
            "subspace_health": 1584,
        },
    )
    diagnostic_views = views(admitted)
    old_rows, _ = _checkpoint_rows(work / "legacy", diagnostic_views, sketch_rank=64)
    for old, new in zip(old_rows, tables["checkpoint_geometry"], strict=True):
        require_same(
            {key: new[key] for key in old if key != "saved_segment_nonzero_parameter_fraction"},
            {
                key: value
                for key, value in old.items()
                if key != "saved_segment_nonzero_parameter_fraction"
            },
        )
        assert (
            old["saved_segment_nonzero_parameter_fraction"]
            == new["saved_segment_parameter_mass_in_nonzero_matrices"]
        )
    old_pairs = _pair_subspace_rows(
        diagnostic_views,
        reference,
        rank=16,
        oversample=8,
        power_iterations=2,
        seed=contract.payload["settings"]["seed"],
    )

    def pair_key(row):
        return (row["stage"], row["first_run_id"], row["second_run_id"], row["displacement_kind"])

    require_same(
        sorted(old_pairs, key=pair_key),
        sorted(
            [
                {key: value for key, value in row.items() if key != "step"}
                for row in tables["run_pair_subspace_overlap"]
            ],
            key=pair_key,
        ),
    )
    require_same(_optimizer_pair_rows(old_pairs), tables["optimizer_pair_subspace_summary"])
    exact = full_spectrum_checks(
        admitted, work / "runs", reference, ref, contract.payload["settings"]
    )
    write_new(
        work / "full-spectrum-controls.json",
        {
            "selection": list(EXACT_NAMES),
            "stage": 3,
            "kind": "cumulative",
            "rtol": SCALAR_RTOL,
            "atol": SCALAR_ATOL,
            "records": exact,
            "scientific_completion": False,
        },
    )
    for name, rows in tables.items():
        with (work / f"{name}.csv").open("xb") as stream:
            stream.write(csv_bytes(rows))
    changed_cases = []
    for name, keys, value in (
        ("zero_imputation", ("subspace_validity", "zero"), "Impute zero"),
        (
            "drop_nonzero_rank",
            ("subspace_validity", "nonzero_signal_rank"),
            "Drop deficient matrices",
        ),
        ("best_rate", ("weight_space_rules", "rate_pair_policy"), "Choose best BEIR rate"),
        ("rank", ("settings", "subspace_rank"), 8),
        ("seed", ("settings", "seed"), 42),
        ("sources", ("sources",), {}),
        ("primary", ("parents", "primary", "sha256"), "0" * 64),
        ("release", ("formal_execution_authorized",), True),
    ):
        value_dict = copy.deepcopy(contract.payload)
        node = value_dict
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
        path = work / f"invalid-{name}.json"
        write_new(path, value_dict)
        try:
            GeometryContract.load(path, contract.primary)
        except ValueError as error:
            changed_cases.append(
                {"case": name, "rejected": True, "error": str(error), "artifact": binding(path)}
            )
        else:
            raise AssertionError(f"Changed geometry contract accepted: {name}")
    cli = []
    for action in ("produce", "inspect"):
        output = work / f"must-not-exist-{action}"
        argv = [
            sys.executable,
            "-B",
            "-m",
            "embed_optim.primary_v3_geometry",
            "--repository",
            str(root),
            "--training-root",
            str(training_root),
            "--experiment-root",
            "/root/embedding-optimizer-study",
            "--primary-protocol",
            str(contract.primary.path),
            "--geometry-protocol",
            str(contract.path),
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--action",
            action,
        ]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=180, check=False)
        assert (
            result.returncode != 0 and "Require an ordinary retained run directory" in result.stderr
        )
        assert not output.exists()
        cli.append(
            {
                "action": action,
                "argv": argv,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    for row in bound:
        verify_file(row["path"], row)
    require_same(handoff(), original)
    nonzero = [row for row in tables["subspace_health"] if not row["zero_displacement"]]
    zeros = [row for row in tables["subspace_health"] if row["zero_displacement"]]
    assert len(nonzero) == 1056 and len(zeros) == 528
    assert all(row["retained_signal_supported"] is True for row in nonzero)
    assert all(row["retained_signal_supported"] is None for row in zeros)
    return {
        "scope": DIAG_SCOPE,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "actual_geometry_audit_passed": True,
        "geometry_protocol": binding(contract.path),
        "models": 3,
        "checkpoints": 9,
        "raw_hidden_matrix_records": 792,
        "exact_old_raw_record_comparisons": 792,
        "entry_amendment_record_comparisons": 792,
        "table_counts": actual_counts,
        "nonzero_rank16_supported_cases": 1056,
        "zero_undefined_cases": 528,
        "records": records,
        "full_spectrum_controls": exact,
        "independent_full_spectrum_cases": 12,
        "scalar_rtol": SCALAR_RTOL,
        "scalar_atol": SCALAR_ATOL,
        "changed_cases": changed_cases,
        "actual_missing_primary_cli": cli,
        "unchanged_inputs_and_sources": bound,
        "artifacts": [binding(path) for path in sorted(work.rglob("*")) if path.is_file()],
        "post_execution_dispatchers": handoff(),
        "model_updates": 0,
        "gpu_workers": 0,
        "network_uploads": 0,
        "formal_geometry_produced": False,
        "full_primary_geometry_observed": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "source_published": False,
        "boundary": "Actual source integration on three short diagnostic trajectories, with twelve preselected full-spectrum CPU controls. Not the full primary population, all learning rates, retrieval effects, dimension utility or a released manuscript mechanism.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "workdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not args.workdir.is_dir()
        or any(args.workdir.iterdir())
    ):
        raise ValueError("Require a fresh CPU-only diagnostic namespace")
    with threadpool_limits(limits=4):
        try:
            result = run_audit(args.repository.resolve(), args.training_root, args.workdir)
        except Exception as error:
            write_new(
                args.workdir / "failure.json",
                {
                    "scope": DIAG_SCOPE,
                    "recorded_at_utc": datetime.now(UTC).isoformat(),
                    "source": binding(Path(__file__)),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "audit_passed": False,
                    "scientific_completion": False,
                },
            )
            raise
    write_new(args.workdir / "result.json", result)
    print({"actual_geometry_audit_passed": True, **file_identity(args.workdir / "result.json")})


if __name__ == "__main__":
    main()
