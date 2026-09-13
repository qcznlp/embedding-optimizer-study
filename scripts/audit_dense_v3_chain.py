"""Exercise real v3 CPU admission/CLI plans without formal training, evaluation or uploads."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from embed_optim import primary_v3_io as io
from embed_optim import primary_v3_training as training
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from scripts.audit_dense_natural_data import handoff, write_new

PROTOCOL_SHA = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not work.is_dir() or any(work.iterdir()):
        raise ValueError("Require a new explicit CPU-only audit namespace")
    path = root / "configs/dense_primary_v3_protocol.json"
    if file_identity(path)["sha256"] != PROTOCOL_SHA:
        raise ValueError("The declared v3 proposal differs")
    contract = PrimaryV3Contract.load(path, root, args.training_root)
    amendment = read_json(root / contract.payload["data_amendment"]["path"])
    data = {
        partition: contract.dataset(amendment["datasets"][partition]["root"], partition)
        for partition in ("training", "validation")
    }
    records, bindings = [], []

    def refused(label, action):
        try:
            action()
        except (ValueError, FileNotFoundError) as error:
            records.append({"case": label, "rejected": True, "error": str(error)})
        else:
            raise AssertionError(f"Invalid case accepted: {label}")

    old_data = Path("/root/embedding-optimizer-study/data")
    for partition, old_name in (
        ("training", "denseon-sft-500k-seed42"),
        ("validation", "validation-4096-seed20260826"),
    ):
        refused(
            f"old_{partition}_input",
            lambda p=partition, n=old_name: contract.dataset(old_data / n, p),
        )
    for row in contract.inputs["runs"]:
        run_id = row["run_id"]
        old_id = run_id.replace("verified-v3-", "padded-", 1)
        old_root = (
            Path("/root/embedding-optimizer-study/outputs/dense-no-packing-v1/dense") / old_id
        )
        for step in contract.payload["checkpoint_steps"]:
            checkpoint = old_root / f"checkpoint-{step}"
            marker = checkpoint / "trainer_state.json"
            bindings.append({"path": str(marker), **file_identity(marker)})
            refused(
                f"old_checkpoint:{old_id}:{step}",
                lambda c=checkpoint, r=run_id, s=step: contract.checkpoint(c, r, s),
            )

    natural_path = root / "reports/engineering-archive/dense-revised-natural-v1/natural-result.json"
    if (
        file_identity(natural_path)["sha256"]
        != "bf6e893e3b9e244242871227e28a3729acc17477dd3c989f7c647308bf332d36"
    ):
        raise ValueError("Actual natural diagnostic receipt differs")
    natural = read_json(natural_path)
    for row in natural["records"]:
        run_id = (
            f"verified-v3-{row['algorithm']}-{'3e-5' if row['algorithm'] == 'adamw' else '3e-4'}"
        )
        marker = Path(row["run_root"]) / "dense_run_contract.json"
        bindings.append({"path": str(marker), **file_identity(marker)})
        refused(
            f"natural_diagnostic_whole_run:{row['algorithm']}",
            lambda r=row, i=run_id: contract.complete_run(r["run_root"], i),
        )

    dummy = work / "must-remain-absent"
    run_id = "verified-v3-muon-3e-4"
    refused("actual_training_entrypoint_draft", lambda: training.execute(contract, run_id, dummy))
    refused(
        "actual_backup_entrypoint_draft",
        lambda: io.backup(contract, dummy, run_id, 782, dummy / "receipt.json"),
    )
    refused(
        "actual_beir_entrypoint_draft",
        lambda: io.evaluate(contract, dummy, run_id, 782, dummy, "0"),
    )
    if dummy.exists():
        raise AssertionError("A draft consumer created an output")

    cli = []
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{root / 'src'}:{root}",
        "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for row in contract.inputs["runs"]:
        command = [
            sys.executable,
            "-B",
            "-m",
            "embed_optim.primary_v3_training",
            "--protocol",
            str(path),
            "--repository",
            str(root),
            "--training-root",
            str(args.training_root.resolve()),
            "--experiment-root",
            str(dummy),
            "--run-id",
            row["run_id"],
            "--inspect",
        ]
        result = subprocess.run(
            command, env=env, cwd=root, text=True, capture_output=True, timeout=120, check=False
        )
        if result.returncode:
            raise ValueError(f"Read-only CLI failed: {result.stderr[-2000:]}")
        observed = json.loads(result.stdout)
        require_same(observed["recipe"], contract.expected_identity(row["run_id"])["recipe"])
        if observed["training_executed"] is not False:
            raise ValueError("Inspection claims a training side effect")
        cli.append({"command": command, "returncode": result.returncode, "observed": observed})

    command = [
        sys.executable,
        "-B",
        "-m",
        "embed_optim.primary_v3_completion",
        "matrix",
        "--protocol",
        str(path),
        "--repository",
        str(root),
        "--training-root",
        str(args.training_root.resolve()),
        "--experiment-root",
        str(dummy),
        "--results-root",
        str(dummy),
    ]
    result = subprocess.run(
        command, env=env, cwd=root, text=True, capture_output=True, timeout=120, check=False
    )
    if result.returncode == 0 or "Require an ordinary retained run directory" not in result.stderr:
        raise ValueError("Missing actual v3 grid did not fail at its first complete-run gate")
    if dummy.exists():
        raise AssertionError("Read-only CLI created missing primary outputs")
    for row in bindings:
        verify_file(row["path"], row)
    source_names = (
        *contract.payload["consumer_sources"],
        "scripts/audit_dense_v3_chain.py",
        "tests/test_primary_v3_chain.py",
    )
    receipt = {
        "scope": "engineering_dense_v3_core_chain_rehearsal",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "actual_input_partitions_accepted": data,
        "actual_rejections": records,
        "actual_inspection_cli": cli,
        "missing_primary_grid_cli": {
            "command": command,
            "returncode": result.returncode,
            "stderr": result.stderr,
        },
        "input_markers_preserved": bindings,
        "source_bindings": [
            {"path": str(root / n), **file_identity(root / n)} for n in source_names
        ],
        "protocol": {"path": str(path), **file_identity(path)},
        "post_execution_dispatchers": handoff(),
        "model_updates": 0,
        "gpu_workers": 0,
        "network_uploads": 0,
        "scientific_completion": False,
        "formal_replication_ready": False,
        "source_published": False,
        "boundary": "Actual revised datasets, 12 actual CLI plans, 60 old checkpoints and 3 successful natural diagnostic run directories exercise the new v3 core boundary. Accepted full primary checkpoints/whole runs/grid do not yet exist. No formal training, scoring, upload, controller transition or downstream scientific result is produced.",
    }
    write_new(work / "result.json", receipt)
    print(
        {
            "accepted_data_partitions": len(data),
            "rejected_actual_cases": len(records),
            "inspection_cli_passed": len(cli),
            "missing_primary_grid_rejected": True,
            "formal_replication_ready": False,
            **file_identity(work / "result.json"),
        }
    )


if __name__ == "__main__":
    main()
