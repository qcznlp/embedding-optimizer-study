"""Demonstrate old-publication/v3-numerics interface gaps with synthetic inputs."""

import argparse
import copy
import os
import runpy
import sys
from fractions import Fraction
from pathlib import Path

from embed_optim import corrected_publication as old
from embed_optim.bridge_numerics import error_columns, evaluate
from embed_optim.primary_contract import file_identity, require_same, verify_file
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff


def run(root, output):
    if sys.flags.optimize or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require assertions and CPU-only diagnostic execution")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Use a new diagnostic result, never replace prior evidence")
    names = (
        "src/embed_optim/corrected_publication.py",
        "src/embed_optim/bridge_numerics.py",
        "src/embed_optim/bridge_exact_arithmetic.py",
        "src/embed_optim/corrected_retrieval_bridge.py",
        "tests/test_bridge_numerics.py",
    )
    sources = [{"path": str(root / name), **file_identity(root / name)} for name in names]
    sources.append({"path": str(Path(__file__).absolute()), **file_identity(__file__)})
    before = handoff()
    panel = runpy.run_path(str(root / "tests/test_bridge_numerics.py"))["panel"]
    cases = []
    for kind in ("signal", "constant", "train_only", "near_null"):
        tables, _ = evaluate(panel(kind))
        try:
            old._bridge_table(tables["feature_prediction_summary"], tables["residual_associations"])
            accepted, error = True, None
        except ValueError as exc:
            accepted, error = False, str(exc)
        assert accepted is (kind in ("signal", "constant"))
        cases.append(
            {
                "kind": kind,
                "complete_kernel_tables": tables,
                "old_publication_accepted": accepted,
                "old_publication_error": error,
            }
        )
    # A scalar exact-error display boundary, not an OLS fit or primary artifact.
    exact = error_columns([Fraction(0)], [Fraction(1)], [Fraction(1) - Fraction(1, 10**400)])
    assert exact["feature_improves"] is True and exact["rmse_reduction"] == 0.0
    table = copy.deepcopy(cases[0]["complete_kernel_tables"])
    for row in table["feature_prediction_summary"]:
        for suffix in (
            "baseline_rmse",
            "feature_rmse",
            "rmse_reduction",
            "baseline_mse_exact",
            "feature_mse_exact",
            "mse_reduction_exact",
        ):
            row["pooled_" + suffix] = exact[suffix]
    try:
        old._bridge_table(table["feature_prediction_summary"], table["residual_associations"])
    except ValueError as exc:
        underflow_error = str(exc)
        assert "support rule mismatch" in underflow_error
    else:
        raise AssertionError("Old display-sign boundary was not reproduced")
    for row in sources:
        verify_file(row["path"], row)
    require_same(handoff(), before)
    write_new(
        output,
        {
            "scope": "synthetic_v3_primary_rendering_interface_probe",
            "passed": True,
            "sources": sources,
            "cases": cases,
            "scalar_error_display_boundary": {
                "values": exact,
                "old_publication_error": underflow_error,
                "complete_ols_case": False,
            },
            "primary_admission_supplied": False,
            "manuscript_installed": False,
            "scientific_completion": False,
            "post_execution_dispatchers": before,
        },
    )
    print(
        {
            "passed": True,
            "cases": len(cases),
            "scientific_completion": False,
            **file_identity(output),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.repository.resolve(), args.output.absolute())
