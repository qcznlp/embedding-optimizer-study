"""Read retained real diagnostic spectra with the new primitive; never relabel primary runs."""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from threadpoolctl import threadpool_limits

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_exact_geometry_primitives import spectrum_metrics
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / "reports/engineering-archive/dense-v3-exact-geometry-v1/validation.json"
PARENT_SHA = "63076c3fcabbca857601b72f5dd55f83698d25134980b54f71304dbf517fa993"
SOURCE = Path("/tmp/dense-v3-exact-geometry.CBVst9/result.json")
METADATA = {
    "run_id",
    "optimizer",
    "learning_rate",
    "stage",
    "step",
    "displacement_kind",
    "tensor",
    "shape",
    "parameters",
}


def binding(path):
    return {"path": str(path), **file_identity(path)}


def run(output):
    if sys.flags.optimize or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or output.exists():
        raise ValueError("Require assertions, CPU-only verification and a new output")
    before = handoff()
    assert file_identity(PARENT)["sha256"] == PARENT_SHA
    parent = read_json(PARENT)
    assert parent["artifact_validation_passed"] is True
    assert file_identity(SOURCE)["sha256"] == parent["accepted_audit_sha256"]
    source = read_json(SOURCE)
    assert (
        source["actual_exact_geometry_verified"] is True
        and source["scientific_completion"] is False
    )
    for row in source["artifacts"]:
        verify_file(row["path"], row)
    manifests = [
        Path(row["path"])
        for row in source["artifacts"]
        if Path(row["path"]).name == "manifest.json" and "/runs/" in row["path"]
    ]
    assert len(manifests) == 3
    counts, stages, inputs, max_error = Counter(), [], [], 0.0
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for path in manifests:
            manifest = read_json(path)
            assert manifest["plan"]["scope"] == "engineering_dense_v3_exact_geometry_diagnostic"
            assert len(manifest["outputs"]) == 3
            inputs.append(binding(path))
            for item in manifest["outputs"]:
                record_file, array_file = (
                    path.parent / item[key]["path"] for key in ("records", "arrays")
                )
                for key, file in (("records", record_file), ("arrays", array_file)):
                    verify_file(file, item[key])
                    inputs.append(binding(file))
                record = read_json(record_file)
                assert len(record["records"]) == 176
                names = set()
                with safe_open(array_file, framework="pt", device="cpu") as handle:
                    require_same(
                        handle.metadata(),
                        {
                            "scope": manifest["plan"]["scope"],
                            "run_identity_sha256": manifest["plan"]["run_identity_sha256"],
                        },
                    )
                    for row in record["records"]:
                        prefix = row["displacement_kind"] + "|" + row["tensor"] + "|"
                        names.add(prefix + "singular")
                        tensor = handle.get_tensor(prefix + "singular")
                        assert tensor.dtype == torch.float64
                        metrics = spectrum_metrics(
                            tensor.numpy(),
                            row["shape"],
                            manifest["plan"]["settings"],
                            row["relative_reconstruction_residual"],
                        )
                        require_same(
                            {key: value for key, value in row.items() if key not in METADATA},
                            metrics,
                        )
                        counts[metrics["basis_status"]] += 1
                        if metrics["basis_status"] == "resolved":
                            rank = metrics["retained_rank"]
                            for side, size in zip(("left", "right"), row["shape"], strict=True):
                                key = prefix + side
                                names.add(key)
                                basis = handle.get_tensor(key)
                                assert basis.dtype == torch.float64 and list(basis.shape) == [
                                    size,
                                    rank,
                                ]
                                assert bool(torch.isfinite(basis).all())
                                array = basis.numpy()
                                error = float(np.max(np.abs(array.T @ array - np.eye(rank))))
                                assert error <= 1e-10
                                max_error = max(max_error, error)
                    require_same(sorted(handle.keys()), sorted(names))
                stages.append(
                    {
                        "run_id": manifest["plan"]["recipe"]["run_id"],
                        "step": item["step"],
                        "spectrum_defined_metrics_exact": True,
                        "matrix_kind_records": 176,
                    }
                )
    require_same(dict(counts), {"zero": 528, "resolved": 1056})
    for row in inputs:
        verify_file(row["path"], row)
    assert file_identity(PARENT)["sha256"] == PARENT_SHA
    assert file_identity(SOURCE)["sha256"] == parent["accepted_audit_sha256"]
    require_same(handoff(), before)
    write_new(
        output,
        {
            "scope": "engineering_real_retained_diagnostic_spectrum_readback",
            "passed": True,
            "parent_acceptance": binding(PARENT),
            "original_diagnostic": binding(SOURCE),
            "new_primitive": binding(
                ROOT / "src/embed_optim/primary_v3_exact_geometry_primitives.py"
            ),
            "audit_source": binding(Path(__file__).resolve()),
            "stages": stages,
            "basis_status_counts": dict(counts),
            "maximum_basis_gram_error": max_error,
            "inputs": inputs,
            "all_1584_spectrum_defined_records_match_exactly": True,
            "saved_weights_loaded": False,
            "full_svd_repeated": False,
            "reconstruction_residual_remeasured": False,
            "complete_primary_admission": False,
            "scientific_completion": False,
            "post_execution_dispatchers": before,
        },
    )
    print({"real_diagnostic_spectrum_readback_passed": True, **file_identity(output)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
