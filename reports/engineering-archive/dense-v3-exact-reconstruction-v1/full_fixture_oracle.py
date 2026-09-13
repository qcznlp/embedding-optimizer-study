"""Independent scalar/set and SymPy checks on the new complete synthetic exact panel."""

import argparse
import itertools
import math
import os
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from safetensors import safe_open

from embed_optim import primary_v3_exact_bridge as exact
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, require_same
from embed_optim.primary_v3_publication import read_tables
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_v3_exact_bridge import independent_systems

KINDS = ("saved_segment", "cumulative")
ORDER = {"adamw": 0, "muon": 1, "normuon": 2}


def close(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert type(actual) in (int, float) and math.isfinite(actual)
        assert math.isclose(actual, float(expected), rel_tol=2e-12, abs_tol=2e-14), (
            actual,
            expected,
        )


def scalar(singular, shape):
    assert len(singular) == min(shape) and all(math.isfinite(s) and s >= 0 for s in singular)
    assert singular == sorted(singular, reverse=True)
    if singular[0] == 0:
        return {
            "basis_status": "zero",
            "stable_rank": None,
            "full_entropy_effective_rank": None,
            "frobenius_norm": 0.0,
            "spectral_norm": 0.0,
            "numerical_rank": 0,
            "top_rank_energy": None,
        }
    relative = [value / singular[0] for value in singular]
    total, energy = math.fsum(relative), math.fsum(value * value for value in relative)
    probabilities = [value / total for value in relative if value > 0]
    signal = sum(value > sys.float_info.epsilon * max(shape) * singular[0] for value in singular)
    gap = (singular[15] - singular[16]) / singular[0]
    status = (
        "insufficient_signal_rank"
        if signal < 16
        else "unresolved_boundary"
        if gap <= 2e-8
        else "resolved"
    )
    return {
        "basis_status": status,
        "stable_rank": energy,
        "full_entropy_effective_rank": math.exp(-math.fsum(p * math.log(p) for p in probabilities)),
        "frobenius_norm": singular[0] * math.sqrt(energy),
        "spectral_norm": singular[0],
        "numerical_rank": signal,
        "top_rank_energy": math.fsum(v * v for v in relative[:16]) / energy,
    }


def audit(root, anchor, output):
    if sys.flags.optimize or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or output.exists():
        raise ValueError("Require assertions, CPU-only operation and a new oracle output")
    manifest = files.inspect(root, anchor)
    assert manifest["metadata"]["exact_fixture"]["upstream_primary_admission_simulated"] is True
    geometry, bridge = root / "provenance/exact-geometry", root / "provenance/exact-bridge"
    shapes = read_json(root / "vectors/admission.json")["reference"]["hidden_shapes"]
    assert len(shapes) == 88 and sum(math.prod(shape) for shape in shapes.values()) == 110297088
    manifests = {p.name: read_json(p / "manifest.json") for p in (geometry / "runs").iterdir()}
    order = sorted(
        manifests,
        key=lambda run: (
            ORDER[manifests[run]["plan"]["recipe"]["optimizer"]["name"]],
            manifests[run]["plan"]["recipe"]["optimizer"]["lr"],
            run,
        ),
    )
    assert len(order) == 12
    raw, coordinates, statuses = {}, {}, Counter()
    basis_count = 0
    for run in order:
        assert len(manifests[run]["outputs"]) == 5
        for stage, item in enumerate(manifests[run]["outputs"], 1):
            record = read_json(geometry / "runs" / run / item["records"]["path"])
            assert len(record["records"]) == 176
            raw[run, stage] = record["records"]
            with safe_open(
                geometry / "runs" / run / item["arrays"]["path"], framework="numpy"
            ) as handle:
                for row in record["records"]:
                    name, kind = row["tensor"], row["displacement_kind"]
                    prefix = kind + "|" + name + "|"
                    singular = handle.get_tensor(prefix + "singular")
                    assert singular.dtype == np.float64
                    metrics = scalar(singular.tolist(), shapes[name])
                    for key, value in metrics.items():
                        if key == "basis_status":
                            assert row[key] == value
                        else:
                            close(row[key], value)
                    statuses[metrics["basis_status"]] += 1
                    sides = []
                    for axis, side in enumerate(("left", "right")):
                        key = prefix + side
                        if metrics["basis_status"] != "resolved":
                            assert key not in handle.keys()
                            sides.append(None)
                            continue
                        value = handle.get_tensor(key)
                        assert value.dtype == np.float64 and value.shape == (shapes[name][axis], 16)
                        assert np.isin(value, [0.0, 1.0]).all()
                        assert (value.sum(axis=0) == 1).all() and (value.sum(axis=1) <= 1).all()
                        sides.append(frozenset(int(n) for n in np.nonzero(value)[0]))
                        basis_count += 1
                    coordinates[run, stage, kind, name] = None if sides[0] is None else sides
    geometry_tables = read_tables(geometry, exact.exact_geometry.TABLE_COUNTS)
    for row in geometry_tables["checkpoint_exact_geometry"]:
        records = raw[row["run_id"], row["stage"]]
        for kind in KINDS:
            selected = [r for r in records if r["displacement_kind"] == kind]
            nonzero = [r for r in selected if r["basis_status"] != "zero"]
            mass = sum(r["parameters"] for r in nonzero)
            prefix = "exact_" + kind
            assert row[prefix + "_nonzero_parameters"] == mass
            for status in ("resolved", "zero", "insufficient_signal_rank", "unresolved_boundary"):
                assert row[prefix + "_" + status + "_parameters"] == sum(
                    r["parameters"] for r in selected if r["basis_status"] == status
                )
            for field in ("stable_rank", "full_entropy_effective_rank", "top_rank_energy"):
                value = (
                    sum((r["parameters"] * Fraction(r[field]) for r in nonzero), Fraction(0)) / mass
                    if mass
                    else None
                )
                close(row[prefix + "_" + field + "_parameter_weighted_nonzero"], value)
                if field != "top_rank_energy":
                    value = (
                        sum(
                            (
                                r["parameters"] * Fraction(r[field]) / min(r["shape"])
                                for r in nonzero
                            ),
                            Fraction(0),
                        )
                        / mass
                        if mass
                        else None
                    )
                    close(row[prefix + "_" + field + "_fraction_parameter_weighted_nonzero"], value)
    pairs = geometry_tables["run_pair_exact_subspace_overlap"]
    keys = [
        (stage, kind, a, b)
        for stage in range(1, 6)
        for kind in KINDS
        for a, b in itertools.combinations(order, 2)
    ]
    assert [
        (r["stage"], r["displacement_kind"], r["first_run_id"], r["second_run_id"]) for r in pairs
    ] == keys
    for row, (stage, kind, first, second) in zip(pairs, keys, strict=True):
        total, mass, count, sums = 0, 0, 0, [Fraction(0), Fraction(0)]
        for name, shape in shapes.items():
            n = math.prod(shape)
            total += n
            a, b = (coordinates[run, stage, kind, name] for run in (first, second))
            if a is None or b is None:
                continue
            mass += n
            count += 1
            for side in (0, 1):
                sums[side] += n * Fraction(len(a[side] & b[side]), 16)
        assert row["defined_parameters"] == mass and row["defined_tensors"] == count
        assert row["undefined_zero_parameters"] == total - mass
        assert row["undefined_signal_rank_parameters"] == row["undefined_boundary_parameters"] == 0
        close(row["defined_parameter_fraction"], Fraction(mass, total))
        for field, value in (
            ("left_subspace_overlap", sums[0]),
            ("right_subspace_overlap", sums[1]),
            ("mean_subspace_overlap", sum(sums) / 2),
            ("resolved_only_mean_subspace_overlap", sum(sums) / 2),
        ):
            close(row[field], value / mass if mass else None)
    for row in geometry_tables["optimizer_pair_exact_subspace_summary"]:
        selected = [
            r
            for r in pairs
            if r["stage"] == row["stage"]
            and r["displacement_kind"] == row["displacement_kind"]
            and sorted([r["first_optimizer"], r["second_optimizer"]])
            == sorted([row["first_optimizer"], row["second_optimizer"]])
        ]
        values = [r["mean_subspace_overlap"] for r in selected]
        assert row["run_pairs"] == len(selected)
        close(
            row["mean_subspace_overlap"],
            sum(map(Fraction, values)) / len(values)
            if all(v is not None for v in values)
            else None,
        )
    tables = read_tables(bridge, exact.TABLE_COUNTS)
    # Prove the unmodified positive consumer before relying on mutation refusals.
    exact.inspect_bundle(
        bridge,
        read_json(bridge / "manifest.json")["plan"],
        read_json(bridge / "evidence.json"),
        tables,
    )
    independent = independent_systems(tables)
    require_same(files.inspect(root, anchor), manifest)
    write_new(
        output,
        {
            "scope": "engineering_complete_synthetic_exact_oracles",
            "passed": True,
            "external_manifest_sha256": anchor,
            "basis_status_counts": dict(statuses),
            "native_coordinate_bases_checked": basis_count,
            "spectrum_rows": sum(statuses.values()),
            "checkpoint_rows": 60,
            "run_pair_rows": len(pairs),
            "optimizer_pair_rows": 60,
            "independent_sympy_systems": independent,
            "unaltered_exact_bundle_inspection_passed": True,
            "geometry_oracle": "Independent scalar sums and exact set intersections; no production spectrum/aggregation/overlap function called",
            "retrieval_oracle": "Unchanged independent SymPy full normal equations, not production rational FWL",
            "oracle_source": {
                "path": str(Path(__file__).resolve()),
                **file_identity(Path(__file__)),
            },
            "reused_sympy_source": {
                "path": str(Path(sys.modules[independent_systems.__module__].__file__)),
                **file_identity(Path(sys.modules[independent_systems.__module__].__file__)),
            },
            "upstream_primary_admission_simulated": True,
            "scientific_completion": False,
        },
    )
    print({"complete_synthetic_exact_oracles_passed": True, **file_identity(output)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.archive_root, args.expected_manifest_sha256, args.output)
