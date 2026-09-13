"""Read-only rehearsal of new primary readers against authenticated real artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import (
    PrimaryContract,
    file_identity,
    inspect_sealed_checkpoint,
    read_json,
    require_same,
    verify_file,
)

ENTRY_SHA = "0a773e0906a635ae513e8322a803bffcb1437372e873abae0aaf9a744f28f4e9"
ADMISSION_SHA = "30ddc47b5509618d3dece4688255f368bb735b5918622f7a83f62638bd764c9b"


def rejected(call):
    try:
        call()
    except ValueError as exc:
        return str(exc)
    raise RuntimeError("Invalid artifact was accepted")


def audit(root, experiment, output):
    if output.exists() or output.is_symlink():
        raise ValueError("Audit output must be new")
    archive = root / "reports/engineering-archive/dense-full-identity-v1"
    protocol = root / "configs/dense_primary_v2_protocol.json"
    training = archive / "candidate-source"
    contract = PrimaryContract.load(protocol, root, training)
    entry_path, admission_path = (
        archive / "entrypoint/result.json",
        archive / "admission/result.json",
    )
    if file_identity(entry_path)["sha256"] != ENTRY_SHA:
        raise ValueError("Untrusted real checkpoint producer receipt")
    if file_identity(admission_path)["sha256"] != ADMISSION_SHA:
        raise ValueError("Untrusted real corrupted-copy receipt")
    entry, admission = read_json(entry_path), read_json(admission_path)
    records, identities = [], {}
    for producer in entry["records"]:
        done = producer["completion"]
        identity_path = Path(done["full_run_identity"]["path"])
        verify_file(identity_path, done["full_run_identity"])
        diagnostic = read_json(identity_path)
        identities[producer["run_id"]] = diagnostic
        for stage, original_files in done["checkpoint_files"].items():
            step = int(stage)
            checkpoint = identity_path.parent / f"checkpoint-{step}"
            checked = inspect_sealed_checkpoint(checkpoint, diagnostic, step)
            expected_files = [
                {
                    "path": Path(row["path"]).relative_to(checkpoint).as_posix(),
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
                for row in original_files
            ]
            require_same(sorted(checked["files"], key=lambda row: row["path"]), expected_files)
            reason = rejected(
                lambda: inspect_sealed_checkpoint(
                    checkpoint, contract.expected_identity(producer["run_id"]), step
                )
            )
            records.append(
                {
                    "checkpoint": str(checkpoint),
                    "diagnostic_identity_accepted": True,
                    "primary_identity_rejected": reason,
                    "producer_payloads_unchanged": True,
                    "checked": checked,
                }
            )
    old = []
    for row in contract.inputs["runs"]:
        old_id = row["run_id"].replace("verified-", "padded-", 1)
        for step in contract.payload["checkpoint_steps"]:
            checkpoint = (
                experiment / "outputs/dense-no-packing-v1/dense" / old_id / f"checkpoint-{step}"
            )
            if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
                raise ValueError("Expected retained old checkpoint is absent")
            old.append(
                {
                    "checkpoint": str(checkpoint),
                    "rejected": rejected(
                        lambda: contract.checkpoint(checkpoint, row["run_id"], step)
                    ),
                }
            )
    corrupt = []
    for row in admission["records"]:
        if row["case"] != "changed_real_optimizer_payload":
            continue
        checkpoint = Path(row["diagnostic_copy"])
        corrupt.append(
            {
                "checkpoint": str(checkpoint),
                "rejected": rejected(
                    lambda: inspect_sealed_checkpoint(checkpoint, identities[row["run_id"]], 2)
                ),
            }
        )
    base = [
        sys.executable,
        "-B",
        "-m",
        "embed_optim.primary_training",
        "--protocol",
        str(protocol),
        "--repository",
        str(root),
        "--training-root",
        str(training),
        "--experiment-root",
        str(experiment),
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root / "src"),
    }
    cli = []
    for row in contract.inputs["runs"]:
        command = [*base, "--run-id", row["run_id"], "--inspect"]
        result = subprocess.run(
            command, env=env, cwd=root, capture_output=True, text=True, timeout=30
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        plan = json.loads(result.stdout)
        require_same(plan["recipe"], contract.expected_identity(row["run_id"])["recipe"])
        if plan["training_executed"] is not False:
            raise RuntimeError("Inspect mode claims execution")
        cli.append({"command": command, "returncode": result.returncode, "plan": plan})
    # Exercise actual CLI rejection, not a substituted release backend.
    denied_command = [*base, "--run-id", "verified-muon-3e-4", "--execute"]
    denied = subprocess.run(
        denied_command, env=env, cwd=root, capture_output=True, text=True, timeout=30
    )
    if denied.returncode == 0 or "not execution authorized" not in denied.stderr:
        raise RuntimeError("Draft CLI did not reject before training")
    if (experiment / contract.payload["output_root"]).exists():
        raise RuntimeError("Formal output unexpectedly exists")
    if any(name in sys.modules for name in ("torch", "transformers", "sentence_transformers")):
        raise RuntimeError("Read-only rehearsal imported model/device stack")
    require_same(PrimaryContract.load(protocol, root, training).payload, contract.payload)
    result = {
        "scope": "engineering_dense_primary_consumer_rehearsal",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "read_only_rehearsal_passed": True,
        "scientific_completion": False,
        "production_deployed": False,
        "model_or_pickle_loaded": False,
        "model_updates_executed": 0,
        "hf_writes_executed": 0,
        "evaluation_workers_executed": 0,
        "protocol": {"path": str(protocol), **file_identity(protocol)},
        "sources": {
            name: file_identity(root / name)
            for name in (
                "scripts/audit_dense_primary_consumers.py",
                "src/embed_optim/primary_contract.py",
                "src/embed_optim/primary_io.py",
                "src/embed_optim/primary_training.py",
            )
        },
        "real_diagnostic_checkpoints": records,
        "old_primary_checkpoints_rejected": old,
        "real_corrupt_optimizer_copies_rejected": corrupt,
        "actual_read_only_cli_plans": cli,
        "actual_draft_execution_rejection": {
            "command": denied_command,
            "returncode": denied.returncode,
            "stderr": denied.stderr,
        },
        "boundary": "The generic reader accepts 12 prior real diagnostic payloads only under their authenticated diagnostic identities. It rejects those identities for the new primary and all 60 old primary checkpoints. No primary run, real upload, actual scoring or full-run acceptance was performed; network behavior is tested separately with a mocked backend.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "passed": True,
                "diagnostic_checkpoints": len(records),
                "old_rejected": len(old),
                "corrupt_rejected": len(corrupt),
                "cli_plans": len(cli),
                "output": str(output),
                **file_identity(output),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.repository.resolve(), args.experiment_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
