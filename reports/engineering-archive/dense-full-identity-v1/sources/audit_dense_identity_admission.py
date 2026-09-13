"""Exercise actual full-identity admission against authenticated GPU-produced files.

No model, optimizer, pickle, device setup, seeding or training is allowed. Real
input data may be read by the actual admission path. Changes are constructed only
in new temporary configurations/directories; all producer files stay untouched.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from embed_optim import dense_run_contract as contract
from embed_optim import train
from embed_optim.config import RunConfig
from scripts.audit_dense_gpu_replay_update import identity
from scripts.audit_prepared_dense_gpu import RUNS, check_sources


class AcceptedBeforeSetup(RuntimeError):
    pass


def forbid(*args, **kwargs):
    raise RuntimeError("Admission probe crossed its no-side-effects boundary")


def admission(config, checkpoint, max_steps=3):
    def stop(*args, **kwargs):
        raise AcceptedBeforeSetup("Identity accepted; stopped before TrainingArguments setup")

    with (
        patch.dict(os.environ, {"EMBED_OPTIM_MAX_STEPS": str(max_steps)}),
        patch.object(train, "_training_arguments", side_effect=stop),
        patch.object(train, "set_seed", side_effect=forbid),
        patch.object(train, "_load_model_and_loss", side_effect=forbid),
        patch.object(train.torch, "load", side_effect=forbid),
    ):
        try:
            train.run_training(config, str(checkpoint))
        except AcceptedBeforeSetup:
            return {"accepted": True, "stopped_before_setup": True}
        except ValueError as error:
            return {"accepted": False, "rejection": str(error)}
    raise RuntimeError("Admission unexpectedly returned")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--entry-result", type=Path, required=True)
    parser.add_argument("--entry-sha256", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    work, root = args.workdir.absolute(), args.candidate_root.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("WORLD_SIZE") != "4":
        raise ValueError("Require CPU-only visibility and four-rank declared argument identity")
    if work.parent != Path("/tmp") or not work.name.startswith("dense-identity-admission."):
        raise ValueError("Require a new temporary admission namespace")
    if work.is_symlink() or not work.is_dir() or any(work.iterdir()):
        raise ValueError("Admission namespace must be empty and ordinary")
    manifest = check_sources(args.source_manifest, args.source_sha256, root)
    if identity(args.entry_result)["sha256"] != args.entry_sha256:
        raise ValueError("GPU producer receipt is not authenticated")
    entry = json.loads(args.entry_result.read_text())
    if not entry["entrypoint_acceptance_passed"] or len(entry["records"]) != 6:
        raise ValueError("Require the completed actual entrypoint campaign")
    authenticated = []
    for row in entry["records"]:
        completion = row["completion"]
        for item in (
            completion["completion"],
            completion["resolved_config"],
            completion["full_run_identity"],
        ):
            if identity(Path(item["path"])) != item:
                raise ValueError("GPU producer metadata changed")
            authenticated.append(item)
        for files in completion["checkpoint_files"].values():
            for item in files:
                if identity(Path(item["path"])) != item:
                    raise ValueError("GPU producer checkpoint changed")
                authenticated.append(item)
    entry_root = args.entry_result.parent
    changed_data = work / "changed-data"
    rows = Dataset.load_from_disk(str(entry_root / "diagnostic-dataset")).to_list()
    rows[0]["query"] += " explicitly changed diagnostic content"
    Dataset.from_list(rows).save_to_disk(str(changed_data))
    records = []
    for run_id in RUNS:
        source = entry_root / "baseline/dense" / run_id
        config = RunConfig.from_dict(json.loads((source / "run_config.json").read_text()))
        config = replace(config, output_root=str(work / "unused-output"))
        checkpoint = source / "checkpoint-2"
        cases = {
            "original": (config, checkpoint, 3, True),
            "identical_relocation": (
                replace(
                    config,
                    dataset_path=str(entry_root / "relocated-data"),
                    model_name=str(entry_root / "relocated-base"),
                ),
                checkpoint,
                3,
                True,
            ),
            "seed": (replace(config, seed=123), checkpoint, 3, False),
            "dataset_content": (
                replace(config, dataset_path=str(changed_data)),
                checkpoint,
                3,
                False,
            ),
            "epochs": (replace(config, epochs=2.0), checkpoint, 3, False),
            "context_length": (replace(config, max_length=512), checkpoint, 3, False),
            "temperature": (replace(config, temperature=0.03), checkpoint, 3, False),
            "gradient_clipping": (replace(config, max_grad_norm=0.5), checkpoint, 3, False),
            "global_batch_same_accumulation": (
                replace(config, global_batch_size=64, micro_batch_size=4),
                checkpoint,
                3,
                False,
            ),
            "warmup": (replace(config, warmup_ratio=0.2), checkpoint, 3, False),
            "declared_base_revision": (
                replace(config, model_revision="f" * 40),
                checkpoint,
                3,
                False,
            ),
            "runtime_horizon_override": (config, checkpoint, 4, False),
            "optimizer_lr": (
                replace(config, optimizer=replace(config.optimizer, lr=0.123)),
                checkpoint,
                3,
                False,
            ),
            "accumulation": (replace(config, global_batch_size=256), checkpoint, 3, False),
        }
        for case, (requested, sealed, steps, accepted) in cases.items():
            observed = admission(requested, sealed, steps)
            if observed["accepted"] is not accepted:
                raise ValueError(f"Wrong actual admission decision: {run_id}/{case}")
            records.append(
                {
                    "run_id": run_id,
                    "case": case,
                    "expected_accepted": accepted,
                    "requested_config": requested.as_dict(),
                    "max_steps": steps,
                    "observed": observed,
                }
            )
        # Corrupt an independently copied real optimizer payload, retaining the
        # producer's original seal. This must fail before any deserialization.
        copied = work / f"corrupted-{run_id}" / "checkpoint-2"
        shutil.copytree(checkpoint, copied)
        with (copied / "optimizer.pt").open("r+b") as handle:
            first = handle.read(1)
            handle.seek(0)
            handle.write(bytes([first[0] ^ 1]))
        observed = admission(config, copied)
        if observed["accepted"]:
            raise ValueError("A changed real optimizer payload was admitted")
        records.append(
            {
                "run_id": run_id,
                "case": "changed_real_optimizer_payload",
                "expected_accepted": False,
                "observed": observed,
                "diagnostic_copy": str(copied),
            }
        )
    if [identity(Path(x["path"])) for x in authenticated] != authenticated:
        raise ValueError("An original GPU-produced file was modified")
    if check_sources(args.source_manifest, args.source_sha256, root) != manifest:
        raise ValueError("Prepared source changed")
    result = {
        "scope": "engineering_actual_full_identity_admission",
        "audit_execution_complete": True,
        "admission_checks_passed": True,
        "scientific_completion": False,
        "production_deployed": False,
        "model_updates_executed": 0,
        "model_or_pickle_loaded": False,
        "accepted_positive_controls": 6,
        "rejected_changed_identities": 36,
        "rejected_changed_real_optimizer_payloads": 3,
        "all_original_producer_files_unchanged": True,
        "source_manifest": identity(args.source_manifest),
        "entry_result": identity(args.entry_result),
        "authenticated_files": authenticated,
        "records": records,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "source_bindings": [
            identity(Path(inspect.getsourcefile(obj)).resolve())
            for obj in (train, contract, check_sources)
        ]
        + [identity(Path(__file__).resolve())],
    }
    with (work / "result.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "admission_checks_passed",
                    "accepted_positive_controls",
                    "rejected_changed_identities",
                    "rejected_changed_real_optimizer_payloads",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
