"""CPU-only exact spectra/projectors and bounded approximation sensitivity.

The same command's --replay path fully recomputes raw numerical measurements;
it cannot accept a result merely because its output hashes are consistent.
"""

import argparse
import copy
import itertools
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from threadpoolctl import threadpool_limits

from embed_optim.corrected_geometry_summary import _basis_seed
from embed_optim.geometry import TensorStore
from embed_optim.geometry_robustness import (
    REFERENCE_ATOL,
    REFERENCE_RTOL,
    exact_reference,
    projector_comparison,
    subspace_quality,
)
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_geometry_kernels import audited_top_bases
from embed_optim.primary_v3_geometry_io import read_stage_bases
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_v3_geometry import binding, setup
from scripts.prepare_dense_geometry_robustness import PARENTS, load_protocol


def compare_arrays(first, second):
    if set(first) != set(second) or any(
        not torch.equal(first[name], second[name]) for name in first
    ):
        raise ValueError("Numerically recomputed robustness arrays differ")


def pair_measure(first, second):
    sides = [projector_comparison(a, b) for a, b in zip(first, second, strict=True)]
    return {
        "left": sides[0],
        "right": sides[1],
        "mean_overlap": sum(x["overlap"] for x in sides) / 2,
    }


def compute(args):
    root = args.repository.resolve()
    plan = load_protocol(args.protocol, root)
    _, natural_path, natural, reference, ref, admitted = setup(root, args.training_root)
    original = handoff()
    parent_path = root / PARENTS["diagnostic_geometry"][0]
    parent = read_json(parent_path)
    for bound in (*parent["unchanged_inputs_and_sources"], *parent["artifacts"]):
        verify_file(bound["path"], bound)
    geometry_root = Path("/tmp/dense-v3-geometry-retry.Pysj3k/runs")
    inputs = [binding(root / name) for name in plan["sources"]]
    inputs += [binding(root / row["path"]) for row in plan["parents"].values()]
    inputs += [binding(args.protocol), binding(natural_path)] + natural["artifacts"]
    inputs += [binding(reference / row["path"]) for row in ref["files"]]
    inputs += parent["artifacts"]
    variants = plan["settings"]["variants"]
    exact_rows, variant_rows, seed_rows, pair_rows, arrays, spaces = [], [], [], [], {}, {}
    prior_exact = {(row["run_id"], row["tensor"]): row for row in parent["full_spectrum_controls"]}
    for run, expected, checked in admitted:
        run_id = expected["recipe"]["run_id"]
        algorithm = expected["recipe"]["optimizer"]["name"]
        assert checked["steps"] == [1, 2, 3]
        saved = read_stage_bases(geometry_root / run_id, 3, ref["hidden_shapes"], "cumulative")
        with TensorStore(run / "checkpoint-3") as current, TensorStore(reference) as initial:
            for name in plan["selection"]["tensor_names"]:
                delta = current.tensor(name).float() - initial.tensor(name).float()
                matrix = delta.numpy().astype(np.float64)
                exact, summary = exact_reference(matrix)
                independent, torch_summary = exact_reference(matrix, backend="torch")
                np.testing.assert_allclose(
                    exact[1], independent[1], rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
                )
                for key in (
                    "spectral_norm",
                    "frobenius_norm",
                    "stable_rank",
                    "full_entropy_effective_rank",
                ):
                    np.testing.assert_allclose(
                        summary[key], torch_summary[key], rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
                    )
                prior = prior_exact[run_id, name]
                for key, old_key in (
                    ("spectral_norm", "exact_spectral_norm"),
                    ("frobenius_norm", "exact_frobenius_norm"),
                    ("stable_rank", "exact_stable_rank"),
                    ("full_entropy_effective_rank", "full_spectrum_entropy_effective_rank"),
                ):
                    np.testing.assert_allclose(
                        summary[key], prior[old_key], rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
                    )
                identity = {"run_id": run_id, "algorithm": algorithm, "tensor": name, "step": 3}
                rank_controls = []
                for rank in (8, 16, 32, 64):
                    control = pair_measure(
                        (exact[0][:, :rank], exact[2][:, :rank]),
                        (independent[0][:, :rank], independent[2][:, :rank]),
                    )
                    # Numerical agreement does not establish uniqueness at a repeated boundary.
                    gap = float((exact[1][rank - 1] - exact[1][rank]) / exact[1][0])
                    resolved = gap > REFERENCE_RTOL
                    if resolved:
                        assert (
                            max(control[side]["projector_rms_sine"] for side in ("left", "right"))
                            <= 2e-7
                        )
                    rank_controls.append(
                        {
                            "rank": rank,
                            "gap_to_leading": gap,
                            "numerically_resolved_boundary": resolved,
                            **control,
                        }
                    )
                exact_rows.append(
                    {
                        **identity,
                        "shape": list(matrix.shape),
                        **summary,
                        "torch_reference": torch_summary,
                        "independent_projectors": rank_controls,
                        "legacy_approx_stable_rank": prior["approx_stable_rank"],
                        "legacy_relative_stable_rank_error": prior["approx_stable_rank"]
                        / summary["stable_rank"]
                        - 1,
                    }
                )
                prefix = f"{run_id}|{name}"
                arrays[f"{prefix}|exact_singular"] = torch.from_numpy(exact[1].copy())
                for side, value in (("left", exact[0]), ("right", exact[2])):
                    arrays[f"{prefix}|exact_{side}"] = torch.from_numpy(value[:, :64].copy())
                sketches = {}
                for variant in variants:
                    seed = (
                        _basis_seed("cumulative", 3, name, plan["settings"]["base_seed"])
                        + variant["seed_offset"]
                    )
                    bases, health = audited_top_bases(
                        delta,
                        rank=variant["rank"],
                        oversample=8,
                        power_iterations=variant["power_iterations"],
                        seed=seed,
                    )
                    assert health["retained_signal_supported"] is True
                    if variant["id"] == "r16-s0-p2":
                        assert all(
                            torch.equal(a, b) for a, b in zip(bases, saved[name], strict=True)
                        )
                    candidate = tuple(value.numpy().astype(np.float64) for value in bases)
                    sketches[variant["id"]] = candidate
                    quality = subspace_quality(matrix, exact, candidate, rank=variant["rank"])
                    variant_rows.append(
                        {**identity, "variant": variant["id"], "seed": seed, **quality}
                    )
                    for side, value in zip(("left", "right"), bases, strict=True):
                        arrays[f"{prefix}|{variant['id']}|{side}"] = value.contiguous()
                for first, second in itertools.combinations(variants, 2):
                    if (
                        first["rank"] == second["rank"]
                        and first["power_iterations"] == second["power_iterations"]
                    ):
                        seed_rows.append(
                            {
                                **identity,
                                "rank": first["rank"],
                                "first": first["id"],
                                "second": second["id"],
                                **pair_measure(sketches[first["id"]], sketches[second["id"]]),
                            }
                        )
                spaces[algorithm, name] = {
                    "exact": (exact[0][:, :64], exact[2][:, :64]),
                    "sketches": sketches,
                }
                print(
                    {
                        "exact_and_approximate_projectors_checked": algorithm,
                        "tensor": name,
                        "variants": len(variants),
                    },
                    flush=True,
                )
    for first, second in itertools.combinations(plan["selection"]["algorithms"], 2):
        for name in plan["selection"]["tensor_names"]:
            for variant in variants:
                a, b = spaces[first, name], spaces[second, name]
                rank = variant["rank"]
                exact_pair = pair_measure(
                    tuple(x[:, :rank] for x in a["exact"]), tuple(x[:, :rank] for x in b["exact"])
                )
                approximate_pair = pair_measure(
                    a["sketches"][variant["id"]], b["sketches"][variant["id"]]
                )
                pair_rows.append(
                    {
                        "first_algorithm": first,
                        "second_algorithm": second,
                        "tensor": name,
                        "variant": variant["id"],
                        "rank": rank,
                        "exact": exact_pair,
                        "approximate": approximate_pair,
                        "mean_overlap_error": approximate_pair["mean_overlap"]
                        - exact_pair["mean_overlap"],
                    }
                )
    assert (
        len(exact_rows) == 12
        and len(variant_rows) == len(pair_rows) == 156
        and len(seed_rows) == 144
    )
    for bound in inputs:
        verify_file(bound["path"], bound)
    require_same(handoff(), original)
    return (
        {
            "exact_cases": exact_rows,
            "approximation_cases": variant_rows,
            "same_matrix_seed_comparisons": seed_rows,
            "cross_optimizer_pair_comparisons": pair_rows,
        },
        arrays,
        inputs,
        original,
    )


def execute(args):
    details, arrays, inputs, original = compute(args)
    replay_cases = []
    if args.replay is not None:
        if file_identity(args.replay)["sha256"] != args.replay_sha256:
            raise ValueError("Changed trusted replay receipt")
        prior = read_json(args.replay)
        require_same(prior["protocol"], binding(args.protocol))
        require_same(prior["inputs"], inputs)
        for bound in prior["artifacts"]:
            verify_file(bound["path"], bound)
        expected = read_json(args.replay.parent / "details.json")
        require_same(details, expected)
        compare_arrays(arrays, load_file(args.replay.parent / "arrays.safetensors"))
        for key in ("exact_cases", "approximation_cases", "cross_optimizer_pair_comparisons"):
            altered = copy.deepcopy(expected)
            row = altered[key][0]
            field = {
                "exact_cases": "stable_rank",
                "approximation_cases": "left_captured_energy",
                "cross_optimizer_pair_comparisons": "mean_overlap_error",
            }[key]
            row[field] += 0.125
            path = args.workdir / f"invalid-{key}.json"
            write_new(path, altered)
            try:
                require_same(details, read_json(path))
            except ValueError:
                replay_cases.append({"case": key, "rehashed": binding(path), "rejected": True})
            else:
                raise AssertionError("Numerically false robustness output accepted")
        altered = {name: value.clone() for name, value in arrays.items()}
        altered[sorted(altered)[0]].flatten()[0] += 0.125
        path = args.workdir / "invalid-arrays.safetensors"
        save_file(altered, path)
        try:
            compare_arrays(arrays, load_file(path))
        except ValueError:
            replay_cases.append({"case": "arrays", "rehashed": binding(path), "rejected": True})
        else:
            raise AssertionError("Numerically false robustness arrays accepted")
    write_new(args.workdir / "details.json", details)
    save_file(arrays, args.workdir / "arrays.safetensors")
    return {
        "scope": "engineering_dense_geometry_robustness",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "numerical_controls_passed": True,
        "fresh_numeric_replay_passed": True if args.replay is not None else None,
        "replay_parent": binding(args.replay) if args.replay is not None else None,
        "protocol": binding(args.protocol),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "counts": {key: len(rows) for key, rows in details.items()},
        "changed_cases": replay_cases,
        "inputs": inputs,
        "artifacts": [binding(path) for path in sorted(args.workdir.iterdir()) if path.is_file()],
        "post_execution_dispatchers": original,
        "approximation_accuracy_pass_declared": False,
        "primary_features_changed": False,
        "formal_geometry_produced": False,
        "scientific_completion": False,
        "model_updates": 0,
        "gpu_workers": 0,
        "boundary": "Twelve inherited short diagnostic matrices only; exact numerical references and measured approximation sensitivity, not primary results, useful dimensions or a retrieval/causal explanation.",
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
        raise ValueError("Require new CPU-only diagnostic namespace")
    if (args.replay is None) != (args.replay_sha256 is None):
        raise ValueError("Replay requires an explicit trusted receipt SHA")
    with threadpool_limits(limits=4):
        try:
            result = execute(args)
        except Exception as error:
            write_new(
                args.workdir / "failure.json",
                {
                    "scope": "engineering_dense_geometry_robustness",
                    "observed_at_utc": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "source": binding(Path(__file__)),
                    "numerical_controls_passed": False,
                    "scientific_completion": False,
                },
            )
            raise
    write_new(args.workdir / "result.json", result)
    print(
        {"numerical_controls_passed": True, **file_identity(args.workdir / "result.json")},
        flush=True,
    )


if __name__ == "__main__":
    main()
