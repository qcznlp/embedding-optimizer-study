"""Authenticate diagnostic checkpoints and independently census every hidden saved parameter."""

import argparse
import copy
import os
import subprocess
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from embed_optim.corrected_geometry_summary import (
    _basis_overlap,
    _nonzero_parameter_fraction,
    _top_bases,
)
from embed_optim.geometry import TensorStore
from embed_optim.primary_completion import inspect_complete_run
from embed_optim.primary_contract import digest, file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.primary_weight_entries import (
    SOURCES,
    load_amendment,
    reference_identity,
    stream_run_census,
)
from scripts.audit_dense_natural_data import handoff


def binding(path):
    return {"path": str(Path(path).resolve()), **file_identity(path)}


def reference_bindings(reference, records):
    return [{**row, "path": str(reference / row["path"])} for row in records]


def numpy_census(run, reference, result):
    """Separate array-value comparisons, integer reductions and global denominators."""
    totals, comparisons = [], 0
    for checkpoint in result["checkpoints"]:
        step, previous_step = checkpoint["step"], checkpoint["previous_step"]
        previous_path = run / f"checkpoint-{previous_step}" if previous_step else reference
        with ExitStack() as stack:
            current = stack.enter_context(TensorStore(run / f"checkpoint-{step}"))
            previous = stack.enter_context(TensorStore(previous_path))
            initial = stack.enter_context(TensorStore(reference))
            counts = {"saved_segment": 0, "cumulative": 0}
            old_records = []
            denominator = 0
            for record in checkpoint["records"]:
                name = record["tensor"]
                weight_tensor = current.tensor(name)
                weight = weight_tensor.numpy().astype(np.float64)
                anchors = {
                    "saved_segment": previous.tensor(name),
                    "cumulative": initial.tensor(name),
                }
                legacy = {"parameters": weight.size}
                for kind, anchor_tensor in anchors.items():
                    anchor = anchor_tensor.numpy().astype(np.float64)
                    actual = int(np.sum(np.not_equal(weight, anchor), dtype=np.int64))
                    assert actual == record[f"{kind}_nonzero_parameters"]
                    counts[kind] += actual
                    comparisons += 1
                    legacy[kind] = {
                        "frobenius_norm": float(
                            torch.linalg.vector_norm(
                                weight_tensor.float() - anchor_tensor.float()
                            ).item()
                        )
                    }
                denominator += weight.size
                old_records.append(legacy)
            assert denominator == 110297088
            row = {"step": step, "parameters": denominator}
            for kind, count in counts.items():
                fraction = count / denominator
                assert fraction == checkpoint["summary"][f"{kind}_nonzero_parameter_fraction"]
                old_fraction = _nonzero_parameter_fraction(old_records, kind)
                assert (
                    old_fraction
                    == checkpoint["summary"][f"{kind}_parameter_mass_in_nonzero_matrices"]
                )
                row[kind] = {
                    "individual_entries_changed": count,
                    "entry_fraction": fraction,
                    "legacy_matrix_mass_fraction": old_fraction,
                }
            totals.append(row)
    return totals, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError("Require CPU-only fresh diagnostic namespace")
    torch.set_num_threads(2)
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    amendment_path = root / "configs/dense_weight_entry_measurement_amendment.json"
    assert (
        file_identity(amendment_path)["sha256"]
        == "a4b4f6839b185931d2de7f0771487123b971a1e7eb69ef34bb5f9e4350060ec5"
    )
    amendment = load_amendment(amendment_path, primary)
    producer_path = Path("/tmp/dense-natural-readiness.a4xZzR/result.json")
    assert (
        file_identity(producer_path)["sha256"]
        == "bf6e893e3b9e244242871227e28a3729acc17477dd3c989f7c647308bf332d36"
    )
    producer = read_json(producer_path)
    for row in producer["artifacts"]:
        verify_file(row["path"], row)
    reference = args.reference.resolve()
    reference_bound = reference_identity(primary, reference)
    snapshot = Path(primary.inputs["initial_model"]["cached_root"])
    source_reference = []
    for row in reference_bound["files"]:
        resolved = (snapshot / row["path"]).resolve(strict=True)
        verify_file(resolved, row)
        source_reference.append({**row, "path": str(resolved)})
    original = handoff()
    bound = [binding(root / name) for name in SOURCES]
    bound += [binding(amendment_path), binding(producer_path), binding(Path(__file__).resolve())]
    bound += producer["artifacts"]
    bound += reference_bindings(reference, reference_bound["files"])
    bound += source_reference
    results, comparisons = [], 0
    for source in producer["records"]:
        run = Path(source["run_root"])
        expected = read_json(run / "dense_run_contract.json")
        assert digest(expected) == source["checked"]["run_identity_sha256"]
        require_same(expected["model"], primary.inputs["common_identity"]["model"])
        checked = inspect_complete_run(run, expected, [1, 2, 3])
        census = stream_run_census(
            run,
            expected,
            checked,
            reference,
            reference_bound,
            scope="engineering_diagnostic_weight_entry_census",
        )
        path = work / f"{source['algorithm']}-census.json"
        write_new(path, census)
        reopened = read_json(path)
        independent, count = numpy_census(run, reference, reopened)
        comparisons += count
        results.append(
            {
                "algorithm": source["algorithm"],
                "run_root": str(run),
                "census": binding(path),
                "independent_numpy_comparisons": count,
                "independent_totals": independent,
            }
        )
        print(
            {
                "algorithm": source["algorithm"],
                "matrix_count_comparisons": count,
                "scientific_completion": False,
            },
            flush=True,
        )
    one_changed = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    old_proxy = _nonzero_parameter_fraction(
        [{"parameters": 4, "delta": {"frobenius_norm": 1.0}}], "delta"
    )
    basis_settings = dict(rank=2, oversample=0, power_iterations=0, seed=20260903)
    second = torch.diag(torch.tensor([0.0, 1.0]))
    subspace = _basis_overlap(
        _top_bases(one_changed, **basis_settings), _top_bases(second, **basis_settings)
    )
    counterexample = {
        "scope": "engineering_synthetic_measurement_counterexample",
        "legacy_fraction": old_proxy,
        "actual_entry_fraction": 0.25,
        "rank_deficient_full_basis_overlap": list(subspace),
        "subspace_boundary": "Full fixed-rank bases include null directions when signal rank is smaller; this is not evidence about the actual primary models.",
        "scientific_completion": False,
    }
    write_new(work / "counterexample.json", counterexample)
    assert old_proxy == 1 and subspace == (1.0, 1.0, 1.0)
    rejected = []
    for name, keys, value in (
        ("old_primary", ("primary", "sha256"), "0" * 64),
        ("no_sources", ("sources",), {}),
        ("matrix_proxy", ("measurement", "numerator"), "Count whole nonzero matrices"),
        ("tolerance", ("measurement", "comparison"), "Absolute difference > 1e-6"),
        ("wrong_anchor", ("measurement", "saved_segment_anchor"), "Always previous source run"),
        ("false_completion", ("scientific_completion",), True),
    ):
        changed = copy.deepcopy(amendment)
        node = changed
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
        path = work / f"invalid-{name}.json"
        write_new(path, changed)
        try:
            load_amendment(path, primary)
        except (ValueError, KeyError) as error:
            rejected.append(
                {"case": name, "rejected": True, "error": str(error), "artifact": binding(path)}
            )
        else:
            raise AssertionError(f"Changed census accepted: {name}")
    argv = [
        sys.executable,
        "-B",
        "-m",
        "embed_optim.primary_weight_entries",
        "--primary-protocol",
        str(primary.path),
        "--amendment",
        str(amendment_path),
        "--repository",
        str(root),
        "--training-root",
        str(args.training_root),
        "--experiment-root",
        "/root/embedding-optimizer-study",
        "--reference",
        str(reference),
        "--output",
        str(work / "must-not-exist.json"),
    ]
    attempt = subprocess.run(argv, capture_output=True, text=True, timeout=180, check=False)
    assert (
        attempt.returncode != 0 and "Require an ordinary retained run directory" in attempt.stderr
    )
    assert not (work / "must-not-exist.json").exists()
    for row in bound:
        verify_file(row["path"], row)
    require_same(handoff(), original)
    result = {
        "scope": "engineering_dense_changed_entry_measurement_audit",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "actual_diagnostic_census_passed": True,
        "models": 3,
        "checkpoints": 9,
        "hidden_matrices_per_checkpoint": 88,
        "hidden_parameters_per_checkpoint": 110297088,
        "reference_copy": {
            "immutable_snapshot": str(snapshot),
            "ordinary_file_copy": str(reference),
            "all_declared_files_match": True,
            "symlink_policy_relaxed": False,
        },
        "independent_numpy_count_comparisons": comparisons,
        "records": results,
        "counterexample": counterexample,
        "changed_cases": rejected,
        "actual_missing_primary_cli": {
            "argv": argv,
            "returncode": attempt.returncode,
            "stdout": attempt.stdout,
            "stderr": attempt.stderr,
        },
        "unchanged_input_and_source_bindings": bound,
        "artifacts": [binding(path) for path in sorted(work.iterdir()) if path.is_file()],
        "post_execution_dispatchers": handoff(),
        "model_updates": 0,
        "gpu_workers": 0,
        "network_uploads": 0,
        "formal_geometry_produced": False,
        "full_geometry_integration_complete": False,
        "formal_replication_ready": False,
        "scientific_completion": False,
        "source_published": False,
        "boundary": "Exact entry census on three short diagnostic trajectories, not primary weight-space evidence, dimension use, retrieval findings or a completed geometry/subspace pipeline. Original matrix-mass proxy/source/artifacts remain unchanged.",
    }
    assert comparisons == 3 * 3 * 88 * 2
    write_new(work / "result.json", result)
    print(
        {
            "actual_diagnostic_census_passed": True,
            "independent_count_comparisons": comparisons,
            **file_identity(work / "result.json"),
        }
    )


if __name__ == "__main__":
    main()
