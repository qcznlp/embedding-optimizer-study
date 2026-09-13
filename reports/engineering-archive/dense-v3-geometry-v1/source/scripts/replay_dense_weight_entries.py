"""Fresh CPU semantic replay of the complete diagnostic changed-entry census."""

import argparse
import copy
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from embed_optim.primary_completion import inspect_complete_run
from embed_optim.primary_contract import digest, file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.primary_weight_entries import (
    DEFINITION,
    load_amendment,
    reference_identity,
    summarize_entries,
)
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_weight_entries import binding, numpy_census

REHEARSAL_SHA = "a269aa6f60ad931a5bd7998c6009b9af8f5a5e93d7f23acc33fd3ea586f84b49"


def inspect_metadata(census, expected, checked, reference_bound):
    require_same(census["scope"], "engineering_diagnostic_weight_entry_census")
    require_same(census["scientific_completion"], False)
    require_same(census["run_id"], expected["recipe"]["run_id"])
    require_same(census["run_identity_sha256"], digest(expected))
    require_same(census["definition"], DEFINITION)
    require_same(census["reference"], reference_bound)
    require_same([row["step"] for row in census["checkpoints"]], checked["steps"])
    for index, row in enumerate(census["checkpoints"]):
        require_same(row["previous_step"], checked["steps"][index - 1] if index else 0)
        require_same(row["cumulative_anchor_step"], 0)
        require_same(
            row["summary"], summarize_entries(row["records"], reference_bound["hidden_shapes"])
        )


def replay(root, rehearsal_path, training_root, work):
    assert file_identity(rehearsal_path)["sha256"] == REHEARSAL_SHA
    rehearsal = read_json(rehearsal_path)
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, training_root
    )
    load_amendment(root / "configs/dense_weight_entry_measurement_amendment.json", primary)
    for row in (*rehearsal["artifacts"], *rehearsal["unchanged_input_and_source_bindings"]):
        verify_file(row["path"], row)
    reference = Path(rehearsal["reference_copy"]["ordinary_file_copy"])
    reference_bound = reference_identity(primary, reference)
    records, total, changed_cases = [], 0, []
    require_same([row["algorithm"] for row in rehearsal["records"]], ["adamw", "muon", "normuon"])
    for source in rehearsal["records"]:
        run = Path(source["run_root"])
        expected = read_json(run / "dense_run_contract.json")
        require_same(expected["model"], primary.inputs["common_identity"]["model"])
        checked = inspect_complete_run(run, expected, [1, 2, 3])
        census = read_json(source["census"]["path"])
        inspect_metadata(census, expected, checked, reference_bound)
        values, count = numpy_census(run, reference, census)
        require_same(values, source["independent_totals"])
        total += count
        records.append({"algorithm": source["algorithm"], "numpy_count_comparisons": count})
        if source["algorithm"] == "adamw":
            for name in ("anchor", "denominator", "duplicate", "missing", "definition", "count"):
                changed = copy.deepcopy(census)
                second = changed["checkpoints"][1]
                if name == "anchor":
                    second["previous_step"] = 0
                elif name == "denominator":
                    second["summary"]["hidden_parameters"] -= 1
                elif name == "duplicate":
                    second["records"][1] = copy.deepcopy(second["records"][0])
                elif name == "missing":
                    second["records"].pop()
                elif name == "definition":
                    changed["definition"]["comparison"] = "Thresholded norm"
                else:
                    # An internally consistent false count must fail against actual tensor values.
                    second["records"][0]["cumulative_nonzero_parameters"] -= 1
                    second["summary"] = summarize_entries(
                        second["records"], reference_bound["hidden_shapes"]
                    )
                path = work / f"invalid-census-{name}.json"
                write_new(path, changed)
                try:
                    reopened = read_json(path)
                    inspect_metadata(reopened, expected, checked, reference_bound)
                    numpy_census(run, reference, reopened)
                except (ValueError, AssertionError) as error:
                    changed_cases.append(
                        {
                            "case": name,
                            "rejected": True,
                            "error_type": type(error).__name__,
                            "artifact": binding(path),
                        }
                    )
                else:
                    raise AssertionError(f"Changed census accepted: {name}")
        print({"fresh_cpu_readback": source["algorithm"], "count_comparisons": count}, flush=True)
    assert total == 1584 and len(changed_cases) == 6
    for row in (*rehearsal["artifacts"], *rehearsal["unchanged_input_and_source_bindings"]):
        verify_file(row["path"], row)
    require_same(handoff(), rehearsal["post_execution_dispatchers"])
    return {
        "scope": "engineering_dense_weight_entry_fresh_cpu_replay",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "fresh_cpu_replay_passed": True,
        "rehearsal": binding(rehearsal_path),
        "source": binding(Path(__file__).resolve()),
        "models": 3,
        "checkpoints": 9,
        "independent_numpy_count_comparisons": total,
        "records": records,
        "changed_cases": changed_cases,
        "artifacts": [binding(path) for path in sorted(work.iterdir()) if path.is_file()],
        "post_execution_dispatchers": handoff(),
        "model_updates": 0,
        "gpu_workers": 0,
        "formal_geometry_produced": False,
        "scientific_completion": False,
        "boundary": "Independent array comparisons on all saved diagnostic entries, not primary outcomes or functional dimension utility.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "rehearsal", "training-root", "workdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not args.workdir.is_dir()
        or any(args.workdir.iterdir())
    ):
        raise ValueError("Require CPU-only fresh replay output")
    torch.set_num_threads(2)
    value = replay(args.repository.resolve(), args.rehearsal, args.training_root, args.workdir)
    write_new(args.workdir / "result.json", value)
    print({"fresh_cpu_replay_passed": True, **file_identity(args.workdir / "result.json")})


if __name__ == "__main__":
    main()
