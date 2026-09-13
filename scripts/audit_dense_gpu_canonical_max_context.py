"""Apply the isolated canonical control to the unchanged 8192-token oracle."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from scripts import audit_dense_gpu_canonical_replay as control
from scripts import audit_dense_gpu_max_context as stress
from scripts.audit_dense_gpu_replay_update import identity


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control-output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output, rank = args.control_output.absolute(), int(os.environ["RANK"])
    if (
        output.parent != Path("/tmp")
        or not output.name.startswith("dense-canonical-replay.")
        or not output.is_dir()
        or output.is_symlink()
        or (rank == 0 and any(output.iterdir()))
    ):
        raise ValueError("Require a new empty mktemp control namespace")
    report = {
        "scope": "engineering_canonical_control_8192_token_gradient_oracle",
        "scientific_completion": False,
        "production_deployed": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source": identity(Path(__file__).resolve()),
        "command_arguments": remaining,
        "observations": {"attention_controls": [], "reducers": []},
        "boundary": "Unchanged fixed 32-long/256-short synthetic stress fixture and manual-gradient tolerance. Apply the same owned deterministic-backward and canonical-reduction control as the short-input audit. The existing stress parent enforces the disjoint GPUs 4,5,6,7 and exact paused project handoff. No numerical production file or primary state changes; no timing or scientific comparison.",
    }
    if rank == 0:
        with (output / "started.json").open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    try:
        with (
            control.controls(report["observations"]),
            patch.object(sys, "argv", [sys.argv[0], *remaining]),
        ):
            stress.main()
    finally:
        with (output / f"observations-rank-{rank}.json").open("x") as handle:
            json.dump({"rank": rank, "observations": report["observations"]}, handle, indent=2)
        if rank == 0:
            parent = Path(remaining[remaining.index("--workdir") + 1]) / "result.json"
            payload = json.loads(parent.read_text()) if parent.is_file() else {}
            report.update(
                finished_at_utc=datetime.now(UTC).isoformat(),
                audit_execution_complete=payload.get("audit_execution_complete", False),
                acceptance_passed=payload.get("normalization_acceptance_passed", False),
                parent_result=identity(parent) if parent.is_file() else None,
                source_bindings=[
                    identity(Path(inspect.getsourcefile(module)))
                    for module in (control, control.canonical, stress, control.change_owned_model)
                ],
            )
            with (output / "result.json").open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")


if __name__ == "__main__":
    main()
