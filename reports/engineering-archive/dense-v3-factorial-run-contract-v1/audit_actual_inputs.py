"""Read the new actual branch view and full source tensors, with CUDA hidden.

This does not calibrate rates, run Trainer or claim any factorial run. Prior
input admission is consumed as an authenticated parent, not repeated wholesale.
"""

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from embed_optim import factorial_v3_run_contract as contract
from embed_optim.primary_contract import file_identity, read_json, verify_file


def main():
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require hidden CUDA and a new output path")
    os.nice(10)
    torch.set_num_threads(1)
    if Path(contract.__file__).resolve().parents[2] != args.source.resolve():
        raise ValueError("Unexpected imported run component")
    parent = (
        args.repository
        / "reports/engineering-archive/dense-v3-factorial-calibration-v1/actual/actual-inputs-third.json"
    )
    if file_identity(parent)["sha256"] != contract.CALIBRATION_SOURCE_SHA:
        raise ValueError("Changed genuine input parent")
    previous = read_json(parent)
    sources = contract.source_identity(args.repository, args.source)
    dataset, view = contract.branch_dataset(previous["inputs"])
    if len(dataset) != 50000 or len(sources) != 68:
        raise ValueError("Incomplete actual branch view or source assembly")
    weights = {}
    for state, source in previous["inputs"]["sources"].items():
        if (
            contract.digest({k: v for k, v in source.items() if k != "checkpoint"})
            != contract.SOURCE_RECORD_SHA[state]
        ):
            raise ValueError("Changed genuine source provenance")
        recorded = next(
            r for r in source["native_checkpoint"]["files"] if r["path"] == "model.safetensors"
        )
        path = Path(source["checkpoint"]) / "model.safetensors"
        verify_file(path, recorded)
        numeric = contract.inspect_model_weights(path)
        verify_file(path, recorded)
        weights[state] = {
            "checkpoint": source["checkpoint"],
            "payload": recorded,
            "numeric": numeric,
        }
    if torch.cuda.is_initialized():
        raise ValueError("A CPU-only input read initialized CUDA")
    result = {
        "scope": "actual-factorial-run-view-and-weight-reader-check",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source),
        "sources": sources,
        "audit_source": {"path": str(Path(__file__).resolve()), **file_identity(__file__)},
        "input_parent": {"path": str(parent), **file_identity(parent)},
        "complete_actual_branch_view": view,
        "actual_source_weights": weights,
        "gpu_initialized": False,
        "forward_or_backward_executed": False,
        "optimizer_or_calibration_executed": False,
        "run_identity_prepared": False,
        "formal_run_admitted": False,
        "scientific_completion": False,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                **file_identity(args.output),
                "branch_rows": len(dataset),
                "full_source_models_read": len(weights),
                "cuda_initialized": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
