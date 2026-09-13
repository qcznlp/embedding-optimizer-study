"""Independent scalar/set oracle for the explicitly synthetic coordinate-basis fixture.

No production aggregation, overlap, entry-census or fixture-generating function is
called. Set intersections give exact canonical overlap for the verified one-hot
bases; rational accumulation checks counts, norms and parameter-weighted metrics.
This does not independently validate spectra of real checkpoint weights.
"""

import csv
import itertools
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from safetensors import safe_open

KINDS = ("saved_segment", "cumulative")
ORDER = {"adamw": 0, "muon": 1, "normuon": 2}


def read(path):
    return json.loads(Path(path).read_text())


def table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    if expected is None:
        assert actual == "", (actual, expected)
    else:
        actual = float(actual)
        assert math.isfinite(actual) and math.isclose(
            actual, float(expected), rel_tol=2e-14, abs_tol=2e-15
        ), (actual, expected)


def weighted(records, field, keys, *, normalize=False):
    numerator, denominator = Fraction(0), 0
    for record in records:
        value = record[field]
        for key in keys:
            value = value[key]
        value = Fraction(value)
        if normalize:
            value /= min(record["shape"])
        count = math.prod(record["shape"])
        numerator += count * value
        denominator += count
    return numerator / denominator


def checkpoint(row, raw, shapes):
    records, entries = raw["records"], raw["entry_records"]
    total = sum(math.prod(shape) for shape in shapes.values())
    assert int(row["hidden_tensors"]) == 88 and int(row["hidden_parameters"]) == total
    norms = {}
    segment = "delta_from_reference" if int(row["stage"]) == 1 else "delta_from_previous"
    for field in ("weight", "delta_from_reference", segment):
        energy = sum((Fraction(r[field]["frobenius_norm"]) ** 2 for r in records), Fraction(0))
        norms[field] = math.sqrt(float(energy))
    for label, value in {
        "weight_frobenius_norm": norms["weight"],
        "saved_segment_frobenius_norm": norms[segment],
        "saved_segment_to_weight_ratio": norms[segment] / norms["weight"],
        "cumulative_displacement_frobenius_norm": norms["delta_from_reference"],
        "cumulative_displacement_to_weight_ratio": norms["delta_from_reference"] / norms["weight"],
        "saved_segment_row_cv_parameter_weighted": weighted(records, segment, ("row_norms", "cv")),
        "saved_segment_top_1pct_row_energy_parameter_weighted": weighted(
            records, segment, ("top_1pct_row_energy",)
        ),
    }.items():
        close(row[label], value)
    for kind, field in (("saved_segment", segment), ("cumulative", "delta_from_reference")):
        changed = sum(r[kind + "_nonzero_parameters"] for r in entries)
        mass = sum(math.prod(r["shape"]) for r in entries if r[kind + "_nonzero_parameters"] > 0)
        assert int(row[kind + "_nonzero_parameters"]) == changed
        close(row[kind + "_nonzero_parameter_fraction"], Fraction(changed, total))
        close(row[kind + "_parameter_mass_in_nonzero_matrices"], Fraction(mass, total))
        for label, key in (
            ("stable_rank", "approx_stable_rank"),
            ("sketch_effective_rank", "sketched_entropy_effective_rank"),
        ):
            close(row[f"{kind}_{label}_parameter_weighted"], weighted(records, field, (key,)))
            close(
                row[f"{kind}_{label}_fraction_parameter_weighted"],
                weighted(records, field, (key,), normalize=True),
            )
        close(
            row[f"{kind}_sketch_captured_energy_parameter_weighted"],
            weighted(records, field, ("captured_frobenius_energy",)),
        )


def audit(root):
    root = Path(root)
    evidence = read(root / "inference/evidence.json")
    assert evidence["upstream_primary_admission_simulated"] is True
    assert evidence["native_axis_coordinate_bases_synthetic"] is True
    assert evidence["scientific_completion"] is False
    shapes = read(root / "vectors/admission.json")["reference"]["hidden_shapes"]
    total = sum(math.prod(shape) for shape in shapes.values())
    assert len(shapes) == 88 and total == 110297088
    geometry = root / "geometry"
    manifests = {p.parent.name: read(p) for p in (geometry / "runs").glob("*/manifest.json")}
    order = sorted(
        manifests,
        key=lambda run: (
            ORDER[manifests[run]["plan"]["recipe"]["optimizer"]["name"]],
            manifests[run]["plan"]["recipe"]["optimizer"]["lr"],
            run,
        ),
    )
    assert len(order) == 12
    coordinates, raw_rows, basis_count = {}, {}, 0
    for run in order:
        manifest = manifests[run]
        assert len(manifest["outputs"]) == 5
        for stage, output in enumerate(manifest["outputs"], 1):
            raw = read(geometry / "runs" / run / output["records"]["path"])
            raw_rows[run, stage] = raw
            assert len(raw["records"]) == len(raw["entry_records"]) == 88
            assert len(raw["subspace_health"]) == 176
            with safe_open(
                geometry / "runs" / run / output["bases"]["path"], framework="numpy"
            ) as handle:
                for name, shape in sorted(shapes.items()):
                    for kind in KINDS:
                        key = run, stage, kind, name
                        pair = []
                        for axis, side in enumerate(("left", "right")):
                            tensor_key = f"{kind}|{name}|{side}"
                            if tensor_key not in handle.keys():
                                pair.append(None)
                                continue
                            value = handle.get_tensor(tensor_key)
                            assert value.dtype == np.float32 and value.shape == (shape[axis], 16)
                            assert np.isin(value, [0.0, 1.0]).all()
                            assert (value.sum(axis=0) == 1).all()
                            assert (value.sum(axis=1) <= 1).all()
                            pair.append(frozenset(int(n) for n in np.nonzero(value)[0]))
                            basis_count += 1
                        assert (pair[0] is None) == (pair[1] is None)
                        coordinates[key] = None if pair[0] is None else pair
    rows = table(geometry / "checkpoint_geometry.csv")
    assert len(rows) == 60
    assert len({(r["run_id"], int(r["stage"])) for r in rows}) == 60
    for row in rows:
        checkpoint(row, raw_rows[row["run_id"], int(row["stage"])], shapes)
    pairs = table(geometry / "run_pair_subspace_overlap.csv")
    expected_keys = [
        (stage, kind, left, right)
        for stage in range(1, 6)
        for kind in KINDS
        for left, right in itertools.combinations(order, 2)
    ]
    assert [
        (int(r["stage"]), r["displacement_kind"], r["first_run_id"], r["second_run_id"])
        for r in pairs
    ] == expected_keys
    groups, undefined_pairs = defaultdict(list), 0
    for row, (stage, kind, left, right) in zip(pairs, expected_keys, strict=True):
        defined, tensors, sums = 0, 0, [Fraction(0), Fraction(0)]
        for name, shape in sorted(shapes.items()):
            first, second = (
                coordinates[left, stage, kind, name],
                coordinates[right, stage, kind, name],
            )
            if first is None or second is None:
                continue
            count = math.prod(shape)
            for axis in (0, 1):
                sums[axis] += count * Fraction(len(first[axis] & second[axis]), 16)
            defined += count
            tensors += 1
        assert int(row["defined_tensors"]) == tensors
        assert int(row["defined_parameters"]) == defined
        assert int(row["undefined_zero_parameters"]) == total - defined
        close(row["defined_parameter_fraction"], Fraction(defined, total))
        means = [value / defined if defined else None for value in sums]
        mean = sum(means) / 2 if defined else None
        for label, value in zip(
            ("left_subspace_overlap", "right_subspace_overlap", "mean_subspace_overlap"),
            [*means, mean],
            strict=True,
        ):
            close(row[label], value)
        undefined_pairs += not bool(defined)
        groups[(stage, kind, row["first_optimizer"], row["second_optimizer"])].append(
            (Fraction(defined, total), mean)
        )
    summaries = table(geometry / "optimizer_pair_subspace_summary.csv")
    assert len(summaries) == len(groups) == 60
    for row in summaries:
        members = groups[
            int(row["stage"]),
            row["displacement_kind"],
            row["first_optimizer"],
            row["second_optimizer"],
        ]
        defined = [mean for _, mean in members if mean is not None]
        assert int(row["rate_pairs"]) == len(members)
        assert int(row["defined_rate_pairs"]) == len(defined)
        close(row["mean_defined_parameter_fraction"], sum(x for x, _ in members) / len(members))
        close(
            row["mean_subspace_overlap_across_rate_pairs"],
            sum(defined) / len(defined) if defined else None,
        )
    assert undefined_pairs == 132 and basis_count > 0
    return {
        "scope": "engineering_independent_synthetic_coordinate_geometry_oracle",
        "passed": True,
        "checkpoints": 60,
        "raw_matrix_records": 5280,
        "native_coordinate_bases_checked": basis_count,
        "exact_set_overlap_pairs": 660,
        "undefined_pairs": undefined_pairs,
        "optimizer_groups": 60,
        "rational_oracle_relative_tolerance": 2e-14,
        "rational_oracle_absolute_tolerance": 2e-15,
        "production_replay_tolerance_changed": False,
        "real_weight_spectra_verified": False,
        "primary_scientific_admission": False,
        "scientific_completion": False,
    }
