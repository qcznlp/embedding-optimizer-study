"""Post-hoc diagnostic of shared random probes, never a primary feature revision.

Motivated by the exact-projector audit. Before reading the new contrasts, fix
all 3x3 seed pairings, all three optimizer pairs and all four inherited matrices.
Independent seeds here are distinct deterministic RNG seeds, not training seeds.
"""

import argparse
import itertools
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from threadpoolctl import threadpool_limits

from embed_optim.geometry_robustness import projector_comparison
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_v3_geometry import binding

PARENT_SHA = "ca916d9169d2398bc20e2db1d1393b162671850315930d975d313d1c2d49960c"
PLAN = {
    "scope": "engineering_dense_geometry_shared_probe_followup",
    "rank": 16,
    "power_iterations": 2,
    "seed_offsets": [0, 1, 2],
    "algorithms": ["adamw", "muon", "normuon"],
    "tensor_names": [
        f"0.layers.0.{part}.weight" for part in ("attn.Wo", "attn.Wqkv", "mlp.Wi", "mlp.Wo")
    ],
    "selection": "Every 3x3 seed pairing for each of the 12 inherited matrix/optimizer-pair cells; no selection, exclusions or significance test",
    "motivation": "Post-hoc diagnostic after observing poor exact-projector agreement and inflated same-seed Muon/NorMuon overlaps",
    "primary_features_changed": False,
    "scientific_completion": False,
}


def compare_seed_pairings(spaces, exact_spaces, algorithms, names, *, rank):
    if len(set(algorithms)) != len(algorithms) or len(set(names)) != len(names):
        raise ValueError("Duplicate populations")
    population = set(itertools.product(algorithms, names))
    if set(spaces) != population or set(exact_spaces) != population:
        raise ValueError("Incomplete matrix population")
    for key in population:
        if set(spaces[key]) != {0, 1, 2}:
            raise ValueError("Require all three declared seeds and no extras")
        for bases in [exact_spaces[key], *spaces[key].values()]:
            if len(bases) != 2 or any(value.ndim != 2 or value.shape[1] != rank for value in bases):
                raise ValueError("Wrong declared basis rank")
    rows, summaries = [], []
    for first, second in itertools.combinations(algorithms, 2):
        for name in names:
            exact = [
                projector_comparison(a, b)["overlap"]
                for a, b in zip(exact_spaces[first, name], exact_spaces[second, name], strict=True)
            ]
            group = []
            for first_seed, second_seed in itertools.product(range(3), repeat=2):
                values = [
                    projector_comparison(a, b)["overlap"]
                    for a, b in zip(
                        spaces[first, name][first_seed],
                        spaces[second, name][second_seed],
                        strict=True,
                    )
                ]
                row = {
                    "first_algorithm": first,
                    "second_algorithm": second,
                    "tensor": name,
                    "first_seed_offset": first_seed,
                    "second_seed_offset": second_seed,
                    "same_probe_seed": first_seed == second_seed,
                    "left_overlap": values[0],
                    "right_overlap": values[1],
                    "mean_overlap": sum(values) / 2,
                }
                group.append(row)
                rows.append(row)
            same = [row["mean_overlap"] for row in group if row["same_probe_seed"]]
            different = [row["mean_overlap"] for row in group if not row["same_probe_seed"]]
            summaries.append(
                {
                    "first_algorithm": first,
                    "second_algorithm": second,
                    "tensor": name,
                    "exact_mean_overlap": sum(exact) / 2,
                    "same_seed_pairs": len(same),
                    "different_seed_pairs": len(different),
                    "same_seed_mean_overlap": float(np.mean(same)),
                    "different_seed_mean_overlap": float(np.mean(different)),
                    "same_minus_different": float(np.mean(same) - np.mean(different)),
                }
            )
    return {"rows": rows, "summaries": summaries}


def execute(args):
    assert file_identity(args.parent)["sha256"] == PARENT_SHA
    parent = read_json(args.parent)
    assert parent["numerical_controls_passed"] is True
    sources = [
        binding(Path(__file__)),
        binding(args.repository / "src/embed_optim/geometry_robustness.py"),
    ]
    for row in (*parent["inputs"], *parent["artifacts"]):
        verify_file(row["path"], row)
    original = handoff()
    details = read_json(args.parent.parent / "details.json")
    arrays = load_file(args.parent.parent / "arrays.safetensors")
    exact_rows = {(row["algorithm"], row["tensor"]): row for row in details["exact_cases"]}
    assert len(exact_rows) == 12
    spaces, exact_spaces = {}, {}
    for algorithm, name in itertools.product(PLAN["algorithms"], PLAN["tensor_names"]):
        row = exact_rows[algorithm, name]
        prefix = f"{row['run_id']}|{name}"
        spaces[algorithm, name] = {
            offset: tuple(
                arrays[f"{prefix}|r16-s{offset}-p2|{side}"].numpy().astype(np.float64)
                for side in ("left", "right")
            )
            for offset in (0, 1, 2)
        }
        exact_spaces[algorithm, name] = tuple(
            arrays[f"{prefix}|exact_{side}"].numpy()[:, :16] for side in ("left", "right")
        )
    results = compare_seed_pairings(
        spaces, exact_spaces, PLAN["algorithms"], PLAN["tensor_names"], rank=16
    )
    assert len(results["rows"]) == 108 and len(results["summaries"]) == 12
    expected = {
        (row["first_algorithm"], row["second_algorithm"], row["tensor"], row["variant"]): row
        for row in details["cross_optimizer_pair_comparisons"]
    }
    for row in results["rows"]:
        if row["same_probe_seed"]:
            key = (
                row["first_algorithm"],
                row["second_algorithm"],
                row["tensor"],
                f"r16-s{row['first_seed_offset']}-p2",
            )
            assert row["mean_overlap"] == expected[key]["approximate"]["mean_overlap"]
    for row in results["summaries"]:
        key = (row["first_algorithm"], row["second_algorithm"], row["tensor"], "r16-s0-p2")
        assert row["exact_mean_overlap"] == expected[key]["exact"]["mean_overlap"]
    if args.replay is not None:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        previous = read_json(args.replay)
        require_same(previous["plan"], PLAN)
        require_same(previous["sources"], sources)
        require_same(previous["results"], results)
    for row in (*parent["artifacts"], *sources):
        verify_file(row["path"], row)
    require_same(handoff(), original)
    return {
        "scope": PLAN["scope"],
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "plan": PLAN,
        "sources": sources,
        "parent": binding(args.parent),
        "results": results,
        "fresh_replay_passed": True if args.replay else None,
        "replay_parent": binding(args.replay) if args.replay else None,
        "same_seed_parent_agreements": 36,
        "exact_parent_agreements": 12,
        "model_updates": 0,
        "gpu_workers": 0,
        "scientific_completion": False,
        "post_execution_dispatchers": original,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "parent", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if sys.flags.optimize or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require a new CPU-only diagnostic result")
    if (args.replay is None) != (args.replay_sha256 is None):
        raise ValueError("Replay requires its exact trusted hash")
    with threadpool_limits(limits=4):
        result = execute(args)
    write_new(args.output, result)
    print(file_identity(args.output))


if __name__ == "__main__":
    main()
