"""Independent CPU checks of every intervention on one retained pretrained export."""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from embed_optim import dimension_interventions as interventions
from embed_optim.dimension_intervention_io import inspect_state, save_state
from embed_optim.dimension_utilization import (
    _cosine_scores,
    _leave_one_out_metrics,
    _orthogonal_matrix,
    _query_metrics,
    _shared_masks,
    _validate_export,
)
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dimension_deletion_boundary import cases, control, scalar_reference

PARENT = "reports/engineering-archive/dense-v3-exact-bridge-v1/validation.json"
PARENT_SHA = "f43a9049a4cc2022ef6a6262c18fa80f4f2088d0ee465da8a3d708bbd848a077"
SOURCES = (
    "scripts/audit_dimension_interventions.py",
    "scripts/audit_dimension_deletion_boundary.py",
    "src/embed_optim/dimension_interventions.py",
    "src/embed_optim/dimension_intervention_io.py",
    "src/embed_optim/dimension_utilization.py",
    "src/embed_optim/primary_contract.py",
    "src/embed_optim/primary_v3_outcomes.py",
    "src/embed_optim/primary_v3_validation_io.py",
    "configs/dense_dimension_utilization_protocol.json",
    "configs/beir_representation_probe.json",
    "tests/test_dimension_interventions.py",
    "tests/test_dimension_intervention_io.py",
)
PLAN = {
    "scope": "engineering_retained_pretrained_dimension_interventions",
    "selection": "The one retained pretrained 224-query/768D export; no optimizer state or outcome selected",
    "samples": 224,
    "dimension": 768,
    "candidates": 8,
    "bases": ["native", "rotation_314159", "rotation_271828", "rotation_161803"],
    "numeric_comparison": {"rtol": 1e-9, "atol": 1e-12},
    "rank_ndcg_comparison": "exact equality; no numeric tie band",
    "reference": "Independent CPU Torch float64 literal deletion, vector normalization, dot product and positive ranking",
    "numerical_policy": interventions.NUMERICAL_POLICY,
    "historical_float16_input_is_primary": False,
    "model_encoding_repeated": False,
    "checkpoint_revalidated": False,
    "upstream_probe_text_revalidated": False,
    "primary_pipeline_integrated": False,
    "scientific_completion": False,
}


def binding(path):
    return {"path": str(Path(path).resolve()), **file_identity(path)}


def reference_scores(q, d, keep=None):
    q, d = (torch.as_tensor(x, dtype=torch.float64) for x in (q, d))
    if keep is not None:
        q, d = q[:, keep], d[:, :, keep]
    qnorm, dnorm = torch.linalg.vector_norm(q, dim=-1), torch.linalg.vector_norm(d, dim=-1)
    if bool(torch.any(qnorm == 0) or torch.any(dnorm == 0)):
        raise ValueError("Independent reference rejects zero residual norm")
    return torch.sum((q / qnorm[:, None])[:, None, :] * (d / dnorm[:, :, None]), dim=-1)


def reference_metrics(scores):
    positive = scores[:, 0]
    ranks = 1 + torch.sum(scores[:, 1:] > positive[:, None], dim=1)
    ndcg = 1 / torch.log2(ranks.to(torch.float64) + 1)
    margin = positive - torch.max(scores[:, 1:], dim=1).values
    # Use NumPy's display convention for nDCG, but decide ranks independently.
    return 1 / np.log2(ranks.numpy() + 1), margin.numpy(), ranks.numpy(), ndcg.numpy()


def agreement(actual, expected):
    np.testing.assert_allclose(actual, expected, **PLAN["numeric_comparison"])
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def independent_basis(q, d):
    actual = interventions.leave_one_out_metrics(q, d)
    expected_ndcg, expected_margin = (np.empty_like(actual[0]) for _ in range(2))
    for coordinate in range(q.shape[1]):
        keep = np.arange(q.shape[1]) != coordinate
        values = reference_metrics(reference_scores(q, d, keep))
        expected_ndcg[:, coordinate], expected_margin[:, coordinate] = values[:2]
        agreement(values[0], values[3])
    np.testing.assert_array_equal(actual[0], expected_ndcg)
    error = agreement(actual[1], expected_margin)
    actual_full = _query_metrics(_cosine_scores(q, d))
    expected_full = reference_metrics(reference_scores(q, d))
    np.testing.assert_array_equal(actual_full[2], expected_full[2])
    agreement(actual_full[1], expected_full[1])
    return (
        {
            "all_positive_ranks_identical": True,
            "coordinate_query_rank_checks": int(expected_ndcg.size),
            "maximum_margin_absolute_error": error,
        },
        expected_ndcg - expected_full[0][:, None],
        expected_margin - expected_full[1][:, None],
    )


def independent_checks(arrays, protocol, computed):
    q, d = (arrays[key].astype(np.float64) for key in ("query_embeddings", "document_embeddings"))
    groups = arrays["sample_groups"].astype(str)
    bases = {}
    for label, seed in [
        ("native", None),
        *((f"rotation_{seed}", seed) for seed in protocol["rotation_control"]["seeds"]),
    ]:
        rotation = None if seed is None else _orthogonal_matrix(q.shape[1], seed)
        current_q, current_d = (q, d) if seed is None else (q @ rotation, d @ rotation)
        check, ndcg_gain, margin_gain = independent_basis(current_q, current_d)
        saved = (
            computed["attributions"]
            if seed is None
            else next(row for row in computed["rotated_attributions"] if row["seed"] == seed)
        )
        check["task_attribution_errors"] = {
            key: agreement(
                saved[key],
                np.stack(
                    [
                        np.mean(values[groups == group], axis=0)
                        for group in sorted(set(groups.tolist()))
                    ]
                ),
            )
            for key, values in (
                ("ndcg_removal_gain", ndcg_gain),
                ("margin_removal_gain", margin_gain),
            )
        }
        bases[label] = check
        print({"independent_basis": label, **check}, flush=True)
    mask_errors = []
    for mask in _shared_masks(q.shape[1], protocol):
        actual = _query_metrics(_cosine_scores(q, d, mask["keep"]))
        expected = reference_metrics(reference_scores(q, d, mask["keep"]))
        np.testing.assert_array_equal(actual[2], expected[2])
        mask_errors.append(agreement(actual[1], expected[1]))
    return {
        "bases": bases,
        "coordinate_query_rank_checks": sum(
            row["coordinate_query_rank_checks"] for row in bases.values()
        ),
        "shared_mask_query_rank_checks": len(mask_errors) * len(q),
        "maximum_random_removal_margin_error": max(mask_errors),
        "all_rank_checks_passed": True,
    }


def altered_cases(output, state, computed, plan):
    results = []
    for index, name in enumerate(
        (
            "task_summary.csv",
            "random_removal.csv",
            "rotation_summary.csv",
            "coordinate_attribution.npz",
            "numerical_evidence.json",
        )
    ):
        clone = output / f"altered-{index}"
        shutil.copytree(state, clone)
        path = clone / name
        if name.endswith(".npz"):
            with np.load(path, allow_pickle=False) as src:
                arrays = {key: src[key] for key in src.files}
            arrays["margin_removal_gain"][0, 0] += 0.0001
            np.savez_compressed(path, **arrays)
        else:
            path.write_bytes(path.read_bytes() + b"\n")
        manifest = read_json(clone / "manifest.json")
        manifest["outputs"][name] = {"path": name, **file_identity(path)}
        (clone / "manifest.json").write_text(json.dumps(manifest))
        try:
            inspect_state(clone, plan, computed)
        except ValueError as exc:
            if "fresh numerical" not in str(exc):
                raise
            results.append({"file": name, "rejected": True, "reason": str(exc)})
        else:
            raise AssertionError("A changed and rehashed output was accepted")
    return results


def run(args):
    root = args.repository.resolve()
    assert file_identity(root / PARENT)["sha256"] == PARENT_SHA
    source_bindings = [binding(root / name) for name in SOURCES]
    path = args.export.resolve()
    inputs = [
        binding(path),
        binding(path.with_suffix(".npz.manifest.json")),
        binding(root / PARENT),
    ]
    arrays, manifest = _validate_export(
        path, 768, probe_spec=root / "configs/beir_representation_probe.json"
    )
    if (
        manifest["checkpoint_run_config"] is not None
        or manifest["encoding"]["storage_dtype"] != "float16"
    ):
        raise ValueError(
            "Require the declared historical pretrained export, not a selected optimizer"
        )
    model_rows = {row["path"]: row for row in manifest["checkpoint_inputs"]}
    assert (
        model_rows["model.safetensors"]["sha256"]
        == "87b1a9f9fe402bded82712e5e06c5d1425cbd5b5e9224d0365324e10812d3bc2"
    )
    original = handoff()
    protocol = read_json(root / "configs/dense_dimension_utilization_protocol.json")
    meta = {"run_id": "pretrained", "optimizer": "pretrained", "learning_rate": None, "step": 0}
    start = time.perf_counter()
    computed = interventions.compute_state(meta, arrays, protocol)
    elapsed = time.perf_counter() - start
    state = args.output_root / "state"
    receipt = save_state(state, PLAN, computed)
    independent = independent_checks(arrays, protocol, computed)
    boundary = control()
    try:
        interventions.leave_one_out_metrics(*cases()["zero_remaining_norm"])
    except ValueError as exc:
        refusal = str(exc)
    else:
        raise AssertionError("The zero residual direction was silently scored")
    q, d = cases()["nonzero_dyadic_remaining_direction"]
    fixed = interventions.leave_one_out_metrics(q, d)
    exact = scalar_reference(q, d, 0)
    np.testing.assert_array_equal(fixed[0][:, 0], exact["ndcg"])
    agreement(fixed[1][:, 0], exact["margin"])
    old = _leave_one_out_metrics(arrays["query_embeddings"], arrays["document_embeddings"])
    new = interventions.leave_one_out_metrics(
        arrays["query_embeddings"], arrays["document_embeddings"]
    )
    legacy_comparison = {
        "native_ndcg_changed_cells": int(np.count_nonzero(old[0] != new[0])),
        "maximum_native_margin_absolute_difference": float(np.max(np.abs(old[1] - new[1]))),
        "population": "one historical pretrained vector export; no optimizer contrasts",
    }
    if args.replay is not None:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        parent = read_json(args.replay)
        for key, value in (
            ("plan", PLAN),
            ("sources", source_bindings),
            ("inputs", inputs),
            ("independent", independent),
            ("legacy_comparison", legacy_comparison),
        ):
            require_same(parent[key], value)
        for record in parent["artifacts"]:
            verify_file(record["path"], record)
        inspect_state(args.replay.parent / "state", PLAN, computed)
    altered = altered_cases(args.output_root, state, computed, PLAN)
    for row in (*inputs, *source_bindings):
        verify_file(row["path"], row)
    require_same(handoff(), original)
    return {
        "scope": PLAN["scope"],
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "rehearsal_passed": True,
        "plan": PLAN,
        "sources": source_bindings,
        "inputs": inputs,
        "artifacts": [binding(p) for p in sorted(args.output_root.rglob("*")) if p.is_file()],
        "independent": independent,
        "legacy_boundary": boundary,
        "new_zero_remainder_refusal": refusal,
        "legacy_comparison": legacy_comparison,
        "bundle": receipt,
        "altered_cases": altered,
        "feature_computation_wall_seconds": elapsed,
        "fresh_replay": args.replay is not None,
        "replay_parent": binding(args.replay) if args.replay else None,
        "primary_pipeline_integrated": False,
        "scientific_completion": False,
        "model_updates": 0,
        "gpu_workers": 0,
        "post_execution_dispatchers": original,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "export", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not args.output_root.is_dir()
        or any(args.output_root.iterdir())
    ):
        raise ValueError("Require a fresh explicit CPU-only diagnostic directory")
    if (args.replay is None) != (args.replay_sha256 is None):
        raise ValueError("A replay requires its exact parent hash")
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        result = run(args)
    write_new(args.output_root / "result.json", result)
    print(
        {
            "passed": True,
            "ranks": result["independent"]["coordinate_query_rank_checks"],
            **file_identity(args.output_root / "result.json"),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
