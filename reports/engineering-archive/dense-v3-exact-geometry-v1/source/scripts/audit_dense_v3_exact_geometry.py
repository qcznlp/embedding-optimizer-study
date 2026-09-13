"""All-hidden-matrix exact geometry on existing diagnostics, plus independent controls."""

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from threadpoolctl import threadpool_limits

from embed_optim.geometry import TensorStore
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_exact_geometry import (
    SOURCES,
    ExactGeometryContract,
    tables_from_verified_runs,
)
from embed_optim.primary_v3_exact_geometry_io import inspect_run, produce_run
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_v3_geometry import binding, setup

SCOPE = "engineering_dense_v3_exact_geometry_diagnostic"
RTOL, ATOL, PROJECTOR_ATOL = 2e-8, 1e-10, 2e-7


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def independent_run(run, expected, checked, reference, output, settings):
    counts = {
        "matrix_kind_records": 0,
        "zero_records": 0,
        "nonzero_spectra": 0,
        "resolved_projectors": 0,
        "unresolved_nonzero": 0,
    }
    max_singular = max_projector = 0.0
    stages = []
    for stage, step in enumerate(checked["steps"], start=1):
        before_path = run / f"checkpoint-{checked['steps'][stage - 2]}" if stage > 1 else reference
        payload = read_json(output / f"checkpoint-{step}.json")
        arrays = load_file(output / f"checkpoint-{step}-exact.safetensors")
        with (
            TensorStore(run / f"checkpoint-{step}") as current,
            TensorStore(reference) as base,
            TensorStore(before_path) as before,
        ):
            norms, weights = {}, {}
            for record in payload["records"]:
                name, kind = record["tensor"], record["displacement_kind"]
                weight = current.tensor(name).float()
                anchor = (
                    before.tensor(name).float()
                    if kind == "saved_segment"
                    else base.tensor(name).float()
                )
                matrix = (weight - anchor).double()
                weights[name] = float(torch.linalg.vector_norm(weight.double()))
                counts["matrix_kind_records"] += 1
                key = f"{kind}|{name}"
                norm = float(torch.linalg.vector_norm(matrix))
                norms[kind, name] = norm
                close(record["frobenius_norm"], norm)
                if not bool((matrix != 0).any()):
                    counts["zero_records"] += 1
                    assert record["basis_status"] == "zero"
                    assert record["stable_rank"] is record["full_entropy_effective_rank"] is None
                    assert not bool((arrays[f"{key}|singular"] != 0).any())
                    assert all(f"{key}|{side}" not in arrays for side in ("left", "right"))
                    continue
                counts["nonzero_spectra"] += 1
                left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
                actual_singular = arrays[f"{key}|singular"]
                close(actual_singular.numpy(), singular.numpy())
                max_singular = max(
                    max_singular, float((actual_singular - singular).abs().max() / singular[0])
                )
                close(record["spectral_norm"], float(singular[0]))
                close(record["stable_rank"], float(matrix.square().sum() / singular[0].square()))
                probabilities = singular / singular.sum()
                probabilities = probabilities[probabilities > 0]
                close(
                    record["full_entropy_effective_rank"],
                    float((-torch.sum(probabilities * torch.log(probabilities))).exp()),
                )
                rank = min(settings["subspace_rank"], min(matrix.shape))
                close(
                    record["top_rank_energy"],
                    float(singular[:rank].square().sum() / matrix.square().sum()),
                )
                threshold = torch.finfo(torch.float64).eps * max(matrix.shape) * singular[0]
                numerical_rank = int((singular > threshold).sum())
                assert record["numerical_rank"] == numerical_rank
                gap = (
                    float((singular[rank - 1] - singular[rank]) / singular[0])
                    if rank < len(singular)
                    else None
                )
                expected_status = (
                    "insufficient_signal_rank"
                    if numerical_rank < rank
                    else "unresolved_boundary"
                    if gap is not None and gap <= settings["boundary_gap_to_leading_tolerance"]
                    else "resolved"
                )
                assert record["basis_status"] == expected_status
                if gap is not None:
                    close(record["boundary_gap_to_leading"], gap)
                if expected_status == "resolved":
                    counts["resolved_projectors"] += 1
                    for side, basis in (("left", left[:, :rank]), ("right", right_h[:rank].T)):
                        actual = arrays[f"{key}|{side}"]
                        # Full projectors are a slower independent control for the fast cross-Gram path.
                        distance = float(
                            torch.linalg.matrix_norm(actual @ actual.T - basis @ basis.T)
                            / math.sqrt(2 * rank)
                        )
                        assert distance <= PROJECTOR_ATOL
                        max_projector = max(max_projector, distance)
                else:
                    counts["unresolved_nonzero"] += 1
                    assert all(f"{key}|{side}" not in arrays for side in ("left", "right"))
            row = payload["checkpoint_row"]
            close(row["weight_frobenius_norm"], np.linalg.norm(list(weights.values())))
            for kind in ("saved_segment", "cumulative"):
                selected = [
                    value for value in payload["records"] if value["displacement_kind"] == kind
                ]
                nonzero = [value for value in selected if value["basis_status"] != "zero"]
                masses = np.array([value["parameters"] for value in nonzero], dtype=np.float64)
                prefix = f"exact_{kind}"
                assert row[f"{prefix}_nonzero_parameters"] == int(masses.sum())
                close(
                    row[f"{prefix}_frobenius_norm"],
                    np.linalg.norm([norms[kind, value["tensor"]] for value in selected]),
                )
                for field in ("stable_rank", "full_entropy_effective_rank", "top_rank_energy"):
                    key = f"{prefix}_{field}_parameter_weighted_nonzero"
                    if nonzero:
                        close(
                            row[key],
                            float(np.average([value[field] for value in nonzero], weights=masses)),
                        )
                    else:
                        assert row[key] is None
        stages.append(
            {"step": step, "independent_records": len(payload["records"]), "passed": True}
        )
        print(
            {"independent_full_svd_stage": expected["recipe"]["run_id"], "step": step, **counts},
            flush=True,
        )
    return {
        "run_id": expected["recipe"]["run_id"],
        "stages": stages,
        **counts,
        "max_singular_error_relative_to_leading": max_singular,
        "max_projector_rms_sine": max_projector,
    }


def execute(args):
    root = args.repository.resolve()
    original_geometry, natural_path, natural, reference, ref, admitted = setup(
        root, args.training_root
    )
    contract = ExactGeometryContract.load(args.protocol, original_geometry.primary)
    original = handoff()
    bound = [binding(root / name) for name in SOURCES]
    bound += [binding(args.protocol), binding(Path(__file__)), binding(natural_path)] + natural[
        "artifacts"
    ]
    bound += [binding(reference / row["path"]) for row in ref["files"]]
    if args.replay is not None:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        prior = read_json(args.replay)
        assert prior["actual_exact_geometry_verified"] is True
        require_same(prior["protocol"], binding(args.protocol))
        for row in (*prior["inputs"], *prior["artifacts"]):
            verify_file(row["path"], row)
        outputs = args.replay.parent / "runs"
    else:
        outputs = args.workdir / "runs"
        outputs.mkdir()
    readers, independent = [], []
    for run, expected, checked in admitted:
        output = outputs / expected["recipe"]["run_id"]
        if args.replay is None:
            produce_run(
                run,
                expected,
                checked,
                reference,
                ref,
                contract.payload["settings"],
                output,
                scope=SCOPE,
                contract_sha=contract.sha256,
            )
        readers.append(
            inspect_run(
                output,
                run,
                expected,
                checked,
                reference,
                ref,
                contract.payload["settings"],
                scope=SCOPE,
                contract_sha=contract.sha256,
            )
        )
        if args.replay is None:
            independent.append(
                independent_run(
                    run, expected, checked, reference, output, contract.payload["settings"]
                )
            )
        print(
            {"exact_geometry_numeric_readback": expected["recipe"]["run_id"], "checkpoints": 3},
            flush=True,
        )
    tables = tables_from_verified_runs(
        admitted, outputs, ref, readers, contract.payload["settings"]
    )
    counts = {name: len(rows) for name, rows in tables.items()}
    require_same(
        counts,
        {
            "checkpoint_exact_geometry": 9,
            "run_pair_exact_subspace_overlap": 18,
            "optimizer_pair_exact_subspace_summary": 18,
            "exact_matrix_geometry": 1584,
        },
    )
    if args.replay is not None:
        require_same(readers, prior["readers"])
        for name, rows in tables.items():
            assert (args.replay.parent / f"{name}.csv").read_bytes() == csv_bytes(rows)
    else:
        for name, rows in tables.items():
            with (args.workdir / f"{name}.csv").open("xb") as stream:
                stream.write(csv_bytes(rows))
    changes, cli = [], []
    if args.replay is None:
        for label, keys, value in (
            ("rank", ("settings", "subspace_rank"), 8),
            ("boundary", ("settings", "boundary_gap_to_leading_tolerance"), 1e-3),
            ("partial", ("table_counts", "exact_matrix_geometry"), 5280),
            ("imputation", ("measurement_rules", "undefined"), "Impute zero"),
            ("selected_pairs", ("measurement_rules", "optimizer_mean"), "Use only defined"),
            ("old_features", ("measurement_rules", "old_features"), "Replace original features"),
            ("sources", ("sources",), {}),
            ("release", ("formal_execution_authorized",), True),
        ):
            changed = copy.deepcopy(contract.payload)
            node = changed
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = value
            path = args.workdir / f"invalid-{label}.json"
            write_new(path, changed)
            try:
                ExactGeometryContract.load(path, contract.primary)
            except ValueError as error:
                changes.append(
                    {
                        "case": label,
                        "rejected": True,
                        "error": str(error),
                        "artifact": binding(path),
                    }
                )
            else:
                raise AssertionError("Altered exact geometry plan accepted")
        for action in ("produce", "inspect"):
            output = args.workdir / f"missing-primary-{action}"
            argv = [
                sys.executable,
                "-B",
                "-m",
                "embed_optim.primary_v3_exact_geometry",
                "--repository",
                str(root),
                "--training-root",
                str(args.training_root),
                "--experiment-root",
                "/root/embedding-optimizer-study",
                "--primary-protocol",
                str(contract.primary.path),
                "--exact-protocol",
                str(args.protocol),
                "--reference",
                str(reference),
                "--output",
                str(output),
                "--action",
                action,
            ]
            result = subprocess.run(argv, capture_output=True, text=True, timeout=180, check=False)
            assert result.returncode != 0 and "ordinary retained run" in result.stderr
            assert not output.exists()
            cli.append(
                {
                    "action": action,
                    "argv": argv,
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                    "output_created": False,
                }
            )
    else:
        run, expected, checked = admitted[0]
        for kind in ("zero_scalar", "nonzero_basis"):
            target = args.workdir / f"invalid-{kind}"
            shutil.copytree(outputs / expected["recipe"]["run_id"], target)
            manifest = read_json(target / "manifest.json")
            if kind == "zero_scalar":
                bound_file = manifest["outputs"][0]["records"]
                path = target / bound_file["path"]
                value = read_json(path)
                value["records"][0]["frobenius_norm"] = 0.125
                path.write_text(json.dumps(value))
            else:
                bound_file = manifest["outputs"][1]["arrays"]
                path = target / bound_file["path"]
                values = load_file(path)
                key = next(name for name in sorted(values) if name.endswith("|left"))
                values[key][0, 0] += 0.125
                save_file(
                    values,
                    path,
                    metadata={
                        "scope": SCOPE,
                        "run_identity_sha256": manifest["plan"]["run_identity_sha256"],
                    },
                )
            bound_file.update(file_identity(path))
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
                    scope=SCOPE,
                    contract_sha=contract.sha256,
                )
            except ValueError as error:
                changes.append(
                    {"case": kind, "rejected": True, "error": str(error), "artifact": binding(path)}
                )
            else:
                raise AssertionError("Rehashed exact numerical corruption accepted")
    for row in bound:
        verify_file(row["path"], row)
    ExactGeometryContract.load(args.protocol, contract.primary)
    require_same(handoff(), original)
    return {
        "scope": SCOPE,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "actual_exact_geometry_verified": True,
        "fresh_numeric_replay_passed": True if args.replay else None,
        "protocol": binding(args.protocol),
        "replay_parent": binding(args.replay) if args.replay else None,
        "models": 3,
        "checkpoints": 9,
        "table_counts": counts,
        "readers": readers,
        "independent_full_matrix_controls": independent,
        "basis_status_counts": {
            status: sum(row["basis_status"] == status for row in tables["exact_matrix_geometry"])
            for status in ("zero", "resolved", "insufficient_signal_rank", "unresolved_boundary")
        },
        "changed_cases": changes,
        "actual_missing_primary_cli": cli,
        "inputs": bound,
        "artifacts": [binding(path) for path in sorted(args.workdir.rglob("*")) if path.is_file()],
        "post_execution_dispatchers": original,
        "model_updates": 0,
        "gpu_workers": 0,
        "formal_exact_geometry_produced": False,
        "scientific_completion": False,
        "boundary": "Complete exact-geometry implementation and all hidden matrices of three existing short diagnostic trajectories; not full primary results, functional dimensions, bridge validation, training release or optimizer quality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "protocol", "workdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not args.workdir.is_dir()
        or any(args.workdir.iterdir())
    ):
        raise ValueError("Require a new CPU-only exact geometry diagnostic namespace")
    if (args.replay is None) != (args.replay_sha256 is None):
        raise ValueError("Fresh replay requires the exact trusted audit hash")
    with threadpool_limits(limits=4):
        try:
            result = execute(args)
        except Exception as error:
            write_new(
                args.workdir / "failure.json",
                {
                    "scope": SCOPE,
                    "observed_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "source": binding(Path(__file__)),
                    "actual_exact_geometry_verified": False,
                    "scientific_completion": False,
                },
            )
            raise
    write_new(args.workdir / "result.json", result)
    print(
        {"actual_exact_geometry_verified": True, **file_identity(args.workdir / "result.json")},
        flush=True,
    )


if __name__ == "__main__":
    main()
