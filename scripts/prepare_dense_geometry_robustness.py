"""Prepare a bounded diagnostic plan without changing primary geometry features."""

import argparse
from pathlib import Path

from embed_optim.primary_contract import file_identity, require_same, verify_file
from embed_optim.primary_v3_geometry import SOURCES as GEOMETRY_SOURCES
from embed_optim.primary_v3_validation_io import write_new

PARENTS = {
    "geometry_acceptance": (
        "reports/engineering-archive/dense-v3-geometry-chain-v1/validation.json",
        "053866a23b34d7323cbfe7d97ceb0d9ae3f1473227ae01cf1fa0b83b32c39226",
    ),
    "geometry_protocol": (
        "configs/dense_primary_v3_geometry_protocol_v2.json",
        "9b6dad688072c08e8994e45f80c897a7889ab5fbcbccab27b643944c85ae625b",
    ),
    "diagnostic_geometry": (
        "reports/engineering-archive/dense-v3-geometry-chain-v1/rehearsal-v2.json",
        "0bd4d9290fd672229ee2d3a5cd4c891854c3e951fc47286bf8e8424190c5c665",
    ),
}
SOURCES = tuple(
    sorted(
        set(GEOMETRY_SOURCES)
        | {
            "src/embed_optim/geometry_robustness.py",
            "scripts/prepare_dense_geometry_robustness.py",
            "scripts/audit_dense_geometry_robustness.py",
            "scripts/audit_dense_v3_geometry.py",
            "scripts/audit_dense_natural_data.py",
            "tests/test_geometry_robustness.py",
        }
    )
)
NAMES = [f"0.layers.0.{suffix}.weight" for suffix in ("attn.Wo", "attn.Wqkv", "mlp.Wi", "mlp.Wo")]
VARIANTS = [
    {"id": f"r{rank}-s{offset}-p2", "rank": rank, "seed_offset": offset, "power_iterations": 2}
    for rank in (8, 16, 32, 64)
    for offset in (0, 1, 2)
] + [{"id": "r16-s0-p8", "rank": 16, "seed_offset": 0, "power_iterations": 8}]


def payload(root):
    parents = {}
    for key, (name, sha) in PARENTS.items():
        bound = file_identity(root / name)
        require_same(bound["sha256"], sha)
        parents[key] = {"path": name, **bound}
    return {
        "schema_version": 1,
        "scope": "engineering_dense_geometry_robustness",
        "status": "DRAFT_DIAGNOSTIC_ONLY",
        "parents": parents,
        "sources": {name: file_identity(root / name) for name in SOURCES},
        "selection": {
            "algorithms": ["adamw", "muon", "normuon"],
            "tensor_names": NAMES,
            "step": 3,
            "displacement_kind": "cumulative",
            "matrices": 12,
            "origin": "Inherit all twelve prior full-spectrum controls; no further selection or exclusions",
            "prior_information": "Known optimizer-dependent spectral sketch error motivated this diagnostic; it is not a blind confirmatory mechanism test",
        },
        "settings": {
            "variants": VARIANTS,
            "oversample": 8,
            "base_seed": 20260903,
            "seed_mapping": "Unchanged cumulative _basis_seed at step 3 plus declared offset",
            "cpu_threads": 4,
            "input": "Identical primary FP32 displacement promoted losslessly to FP64 for exact controls",
            "exact_backends": ["numpy", "torch"],
            "reference_rtol": 2e-8,
            "reference_atol": 1e-10,
            "original_basis_orthogonality_atol": 2e-5,
            "projector_reference_atol": 2e-7,
        },
        "interpretation": {
            "full_spectrum": "Stable rank and singular-value entropy use the complete spectrum, not a truncated renormalized sketch",
            "projectors": "FP64 QR for comparison only; overlap is tr(PQ)/r, RMS sine is ||P-Q||_F/sqrt(2r); retain boundary gaps and captured energy",
            "degeneracy": "A repeated singular value crossing a retained-rank boundary makes that individual top-r subspace non-unique; report gaps, never impute uniqueness",
            "approximation_gate": "No pass threshold for approximation quality, ranking or seed stability; report every case and distinguish correctness from precision adequacy",
            "primary_features_changed": False,
            "useful_embedding_dimensions_demonstrated": False,
            "retrieval_or_causal_claim_authorized": False,
        },
        "scientific_completion": False,
        "formal_execution_authorized": False,
    }


def load_protocol(path, root):
    from embed_optim.primary_contract import read_json

    value = read_json(path)
    require_same(value, payload(root))
    for name, bound in value["sources"].items():
        verify_file(root / name, bound)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_new(args.output, payload(args.repository.resolve()))
    print(file_identity(args.output))


if __name__ == "__main__":
    main()
