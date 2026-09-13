"""Actual corrected input/native-checkpoint admission; no model or GPU execution."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from embed_optim import factorial_v3_calibration as calibration
from embed_optim import factorial_v3_inputs as inputs
from embed_optim.primary_contract import file_identity, read_json, verify_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        raise ValueError("Require hidden CUDA and a new non-overwriting output")
    os.nice(10)
    repository = Path("/root/embedding-optimizer-story-refactor")
    parent_path = (
        repository / "reports/engineering-archive/dense-v3-primary-launch-v1/source-assembly.json"
    )
    if (
        file_identity(parent_path)["sha256"]
        != "e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8"
    ):
        raise ValueError("Original primary source assembly changed")
    parent = read_json(parent_path)
    files = {}
    for relative, binding in parent["files"].items():
        verify_file(args.source / relative, binding["identity"])
        verify_file(Path("/root/embedding-optimizer-primary-v3") / relative, binding["identity"])
        files[relative] = file_identity(args.source / relative)
    extra = (
        "factorial_v3_inputs",
        "factorial_v3_calibration",
        "factorial_v3_optimizer",
        "factorial_v3_batches",
        "factorial_v3_trainer",
        "factorial_v3_checkpoint",
        "gradient_probe",
        "probe_export",
        "update_geometry",
        "probes",
    )
    for name in extra:
        relative = f"src/embed_optim/{name}.py"
        expected = file_identity(repository / relative)
        verify_file(args.source / relative, expected)
        files[relative] = expected
    if Path(inputs.__file__).parent != args.source / "src/embed_optim":
        raise ValueError("Imported another input component")
    sources = calibration.require_sources()
    locations = inputs.Locations(
        repository,
        Path("/root/embedding-optimizer-primary-v3"),
        Path("/root/embedding-optimizer-v3-experiment"),
        Path("/root/embedding-optimizer-study/data"),
        repository / inputs.EVIDENCE,
    )
    actual = inputs.load_inputs(locations)
    requests = {state: inputs.calibration_request(actual, state) for state in inputs.STATES}
    if len(requests) != 2 or any(len(r["selection"]) != 32 for r in requests.values()):
        raise ValueError("Actual calibration request coverage differs")
    # Both states must see the same ordered records and normalization protocol.
    if requests["adamw_state"]["selection"] != requests["muon_state"]["selection"]:
        raise ValueError("Two states use different calibration groups")
    result = {
        "scope": "actual-v3-factorial-input-consumer-readback",
        "audit_source": {"path": str(Path(__file__).resolve()), **file_identity(__file__)},
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source),
        "source_files": files,
        "primary_files_unchanged": len(parent["files"]),
        "loaded_numerical_sources": sources,
        "inputs": actual,
        "requests": requests,
        "genuine_source_states": 2,
        "branch_rows": 50000,
        "calibration_rows": 32,
        "actual_model_or_gradient_execution": False,
        "formal_calibration_or_branch_admission": False,
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
                "source_files": len(files),
                "genuine_sources": 2,
                "gpu_execution": False,
                "scientific_completion": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
