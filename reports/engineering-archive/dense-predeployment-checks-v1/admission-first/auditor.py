"""Read-only admission counterexamples and a full-run identity proposal.

Authenticates real diagnostic checkpoint metadata before any deserialization.
Every actual run_training call is stopped before TrainingArguments setup, random
seeding, output writes, dataset loading or model creation. No update is executed.
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

import torch
from datasets import Dataset

from embed_optim import dense_numerical_contract as numerical
from embed_optim import train
from embed_optim.collators import TEXT_COLUMNS
from embed_optim.config import RunConfig
from scripts import dense_run_identity_proposal as proposal
from scripts.audit_dense_gpu_replay_update import identity
from scripts.audit_prepared_dense_gpu import RUNS, check_sources

RUNTIME_KEYS = (
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "lr_scheduler_type",
    "warmup_steps",
    "max_grad_norm",
    "seed",
    "data_seed",
    "bf16",
    "tf32",
    "fp16",
    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",
    "dataloader_num_workers",
    "dataloader_persistent_workers",
    "dataloader_prefetch_factor",
    "dataloader_drop_last",
    "remove_unused_columns",
    "ddp_find_unused_parameters",
)
PARENT_SHA = "df1210f30ad57e8950050a6530eba755fe34267f88074389fb655a6b6e7aff7f"


class StoppedBeforeSetup(RuntimeError):
    pass


def forbid(*args, **kwargs):
    raise RuntimeError("Read-only admission probe crossed its safety boundary")


def admission(config, checkpoint):
    def stop(*args, **kwargs):
        raise StoppedBeforeSetup("Observed actual post-admission/pre-setup boundary")

    with (
        patch.object(train, "_training_arguments", side_effect=stop),
        patch.object(train, "set_seed", side_effect=forbid),
        patch.object(train.Dataset, "load_from_disk", side_effect=forbid),
        patch.object(train, "_load_model_and_loss", side_effect=forbid),
    ):
        try:
            train.run_training(config, str(checkpoint))
        except StoppedBeforeSetup:
            return {"accepted_preload_gate": True, "stopped_before_setup": True}
        except ValueError as error:
            return {"accepted_preload_gate": False, "rejection": str(error)}
    raise ValueError("Probe unexpectedly returned from training")


def requested_runtime(config, max_steps):
    with (
        patch.dict(os.environ, {"EMBED_OPTIM_MAX_STEPS": str(max_steps)}),
        patch.object(train, "SentenceTransformerTrainingArguments", side_effect=lambda **kw: kw),
    ):
        values = train._training_arguments(config)
    return {**{key: values[key] for key in RUNTIME_KEYS}, "world_size": 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--parent-validation", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    work, root = args.workdir.absolute(), args.candidate_root.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("WORLD_SIZE") != "4":
        raise ValueError("Require CPU-only visibility and explicit declared four-rank arguments")
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-resume-identity.")
        or work.is_symlink()
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError("Require a new empty temporary namespace")
    manifest = check_sources(args.source_manifest, args.source_sha256, root)
    if identity(args.parent_validation)["sha256"] != PARENT_SHA:
        raise ValueError("Untrusted prior checkpoint audit")
    parent = json.loads(args.parent_validation.read_text())
    trusted = {x["path"]: x for x in parent["external_fingerprints_and_checkpoints"]}
    source_before, records = [], []
    prototype_checks = 0
    source_identity = {
        "candidate_manifest_sha256": args.source_sha256,
        "guard_proposal_sha256": identity(Path(inspect.getsourcefile(proposal)))["sha256"],
    }
    baseline_root = Path("/tmp/dense-prepared-entry.IrjSyV/baseline/dense")
    original_data = Path("/tmp/dense-prepared-entry.IrjSyV/diagnostic-dataset")
    original_files = proposal.dataset_files(original_data)
    relocated = work / "identical-relocated-data"
    shutil.copytree(original_data, relocated)
    changed = work / "changed-data"
    rows = Dataset.load_from_disk(str(original_data)).to_list()
    rows[0]["query"] += " deliberately different content"
    Dataset.from_list(rows).save_to_disk(str(changed))
    columns = [*TEXT_COLUMNS, "length"]

    def data(path):
        dataset = Dataset.load_from_disk(str(path)).select_columns(columns)
        return {
            "rows": len(dataset),
            "selected_columns": columns,
            "files": proposal.dataset_files(path),
        }

    for run_id in RUNS:
        run_root = baseline_root / run_id
        checkpoint = run_root / "checkpoint-2"
        metadata = [
            run_root / "run_config.json",
            checkpoint / numerical.RECEIPT_NAME,
            checkpoint / "training_args.bin",
        ]
        for path in metadata:
            actual = identity(path)
            if actual != trusted[str(path)]:
                raise ValueError("Real diagnostic metadata changed")
            source_before.append(actual)
        config = RunConfig.from_dict(json.loads(metadata[0].read_text()))
        saved_numerical = json.loads(metadata[1].read_text())
        # The entire pickle digest is authenticated against the prior completed audit above.
        saved_arguments = torch.load(metadata[2], map_location="cpu", weights_only=False)
        saved_runtime = {
            **{key: getattr(saved_arguments, key) for key in RUNTIME_KEYS},
            "world_size": 4,
        }
        expected = proposal.build_identity(
            config.as_dict(), data(original_data), saved_runtime, saved_numerical, source_identity
        )

        def proposed(request, path, max_steps=3):
            runtime = requested_runtime(request, max_steps)
            accumulation = request.global_batch_size // (request.micro_batch_size * 4)
            return proposal.build_identity(
                request.as_dict(),
                data(path),
                runtime,
                numerical.receipt(request.optimizer, accumulation),
                source_identity,
            )

        proposal.require_identity(expected, proposed(config, original_data))
        if not admission(config, checkpoint)["accepted_preload_gate"]:
            raise ValueError("Original valid checkpoint unexpectedly rejected")
        moved = replace(
            config, dataset_path=str(relocated), output_root=str(work / "relocated-output")
        )
        proposal.require_identity(expected, proposed(moved, relocated))
        prototype_checks += 2
        variants = [
            ("seed", replace(config, seed=43), original_data, 3),
            ("dataset_content", replace(config, dataset_path=str(changed)), changed, 3),
            ("epochs", replace(config, epochs=2.0), original_data, 3),
            ("context_length", replace(config, max_length=4096), original_data, 3),
            ("temperature", replace(config, temperature=0.04), original_data, 3),
            ("gradient_clipping", replace(config, max_grad_norm=0.5), original_data, 3),
            (
                "global_batch_same_accumulation",
                replace(config, global_batch_size=256, micro_batch_size=16),
                original_data,
                3,
            ),
            ("warmup", replace(config, warmup_ratio=0.2), original_data, 3),
            ("declared_base_revision", replace(config, model_revision="0" * 40), original_data, 3),
            ("runtime_horizon_override", config, original_data, 6),
            (
                "optimizer_lr_negative_control",
                replace(config, optimizer=replace(config.optimizer, lr=config.optimizer.lr * 2)),
                original_data,
                3,
            ),
            (
                "accumulation_negative_control",
                replace(config, global_batch_size=256),
                original_data,
                3,
            ),
        ]
        for label, request, dataset_path, horizon in variants:
            with patch.dict(os.environ, {"EMBED_OPTIM_MAX_STEPS": str(horizon)}):
                observed = admission(request, checkpoint)
            candidate = proposed(request, dataset_path, horizon)
            try:
                proposal.require_identity(expected, candidate)
            except ValueError:
                rejected = True
            else:
                raise ValueError("Proposal failed to reject a changed complete identity")
            if observed["accepted_preload_gate"] == label.endswith("negative_control"):
                raise ValueError("Actual admission result differs from the diagnosed boundary")
            prototype_checks += 1
            records.append(
                {
                    "run_id": run_id,
                    "change": label,
                    "actual_preload_gate": observed,
                    "proposed_full_identity_rejected": rejected,
                }
            )
    if proposal.dataset_files(original_data) != original_files:
        raise ValueError("Original diagnostic data changed")
    if [identity(Path(x["path"])) for x in source_before] != source_before:
        raise ValueError("Original checkpoint metadata changed")
    if check_sources(args.source_manifest, args.source_sha256, root) != manifest:
        raise ValueError("Candidate source changed")
    report = {
        "scope": "engineering_prepared_resume_identity_admission_audit",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": True,
        "full_recipe_resume_contract_passed": False,
        "unchecked_change_cases": sum(
            x["actual_preload_gate"]["accepted_preload_gate"] for x in records
        ),
        "negative_controls_rejected": sum(
            not x["actual_preload_gate"]["accepted_preload_gate"] for x in records
        ),
        "proposal_comparison_checks_passed": prototype_checks,
        "proposal_integrated_into_trainer_or_launcher": False,
        "model_updates_executed": 0,
        "source_metadata_and_data_unchanged": True,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "parent_validation": identity(args.parent_validation),
        "source_manifest": identity(args.source_manifest),
        "authenticated_source_metadata": source_before,
        "source_bindings": [
            identity(Path(inspect.getsourcefile(obj)))
            for obj in (train, numerical, proposal, check_sources)
        ]
        + [identity(Path(__file__).resolve())],
        "records": records,
        "boundary": "Three actual saved entrypoint checkpoint metadata sets, CPU-only. Actual run_training admission is intercepted before any runtime setup, seeding, output write, dataset/model load or update. Ten changed recipe/runtime/data cases per optimizer pass the current numerical-only gate; optimizer-LR and accumulation negative controls reject. A separate undeployed full-identity comparator rejects all changed cases while accepting original identity and byte-identical data/output relocation. This is a confirmed missing predeployment guard, not evidence of wrong-data historical training, a numerical defect, or integrated full-run acceptance.",
    }
    with (work / "result.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "audit_execution_complete",
                    "full_recipe_resume_contract_passed",
                    "unchecked_change_cases",
                    "negative_controls_rejected",
                    "proposal_comparison_checks_passed",
                )
            }
        )
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
