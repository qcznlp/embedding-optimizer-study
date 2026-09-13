"""Localize a failed global FP32 norm reduction against independent scalar summation."""

import argparse
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from embed_optim.geometry import TensorStore
from embed_optim.primary_contract import read_json, verify_file
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_v3_geometry import EXACT_NAMES, SCALAR_ATOL, SCALAR_RTOL, binding, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require CPU-only new diagnostic output")
    rows = []
    with threadpool_limits(limits=4):
        contract, _, natural, reference, ref, admitted = setup(args.repository, args.training_root)
        assert contract.sha256 == "9f11cbcdfa8b75593cc3ebd22336e5a34073531426431da88cf904f6e470f358"
        for run, expected, checked in admitted:
            run_id = expected["recipe"]["run_id"]
            stored = read_json(
                Path("/tmp/dense-v3-geometry.75bnhK/runs") / run_id / "checkpoint-3.json"
            )
            stored_records = {row["tensor"]: row for row in stored["records"]}
            with TensorStore(run / "checkpoint-3") as current, TensorStore(reference) as initial:
                for name in EXACT_NAMES:
                    delta = current.tensor(name).float() - initial.tensor(name).float()
                    array = delta.numpy().astype(np.float64)
                    scalar = math.sqrt(math.fsum(float(x) * float(x) for x in array.ravel()))
                    numpy_norm = float(np.linalg.norm(array))
                    original = float(torch.linalg.vector_norm(delta))
                    promoted = float(torch.linalg.vector_norm(delta, dtype=torch.float64))
                    assert (
                        original == stored_records[name]["delta_from_reference"]["frobenius_norm"]
                    )
                    assert math.isclose(promoted, scalar, rel_tol=1e-12, abs_tol=1e-14)
                    assert math.isclose(numpy_norm, scalar, rel_tol=1e-12, abs_tol=1e-14)
                    rows.append(
                        {
                            "run_id": run_id,
                            "tensor": name,
                            "parameters": array.size,
                            "fp32_vector_norm": original,
                            "fp64_vector_norm": promoted,
                            "numpy_fp64_norm": numpy_norm,
                            "scalar_fsum_norm": scalar,
                            "fp32_relative_error": abs(original - scalar) / scalar,
                            "fp64_relative_error": abs(promoted - scalar) / scalar,
                            "original_gate_passed": math.isclose(
                                original, scalar, rel_tol=SCALAR_RTOL, abs_tol=SCALAR_ATOL
                            ),
                        }
                    )
        for row in natural["artifacts"]:
            verify_file(row["path"], row)
        for row in ref["files"]:
            verify_file(reference / row["path"], row)
    value = {
        "scope": "engineering_global_norm_precision_localization",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "source": binding(Path(__file__)),
        "geometry_protocol": binding(contract.path),
        "failed_attempt": "/tmp/dense-v3-geometry.75bnhK",
        "selected_cases": 12,
        "rows": rows,
        "original_rtol": SCALAR_RTOL,
        "original_atol": SCALAR_ATOL,
        "tolerance_changed": False,
        "inputs_changed": False,
        "model_updates": 0,
        "gpu_workers": 0,
        "scientific_completion": False,
        "boundary": "A scalar geometry-reduction precision failure on identical FP32 displacement inputs, not a gradient, optimizer or retrieval finding.",
    }
    write_new(args.output, value)
    print(
        {
            "selected_cases": len(rows),
            "failed_original_gate_cases": sum(not row["original_gate_passed"] for row in rows),
            "max_fp32_relative_error": max(row["fp32_relative_error"] for row in rows),
            "max_fp64_relative_error": max(row["fp64_relative_error"] for row in rows),
            **binding(args.output),
        }
    )


if __name__ == "__main__":
    main()
