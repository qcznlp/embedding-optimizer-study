"""Observe revised primary/validation inputs through the unchanged prepared Trainer contract."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset

from embed_optim import aggregate, train
from embed_optim import dense_numerical_contract as numerical
from embed_optim import dense_run_contract as contract
from embed_optim.config import load_matrix
from scripts.audit_prepared_dense_gpu import check_sources

ACCEPTANCE_SHA = "3b2cd1902e02a43242b16fee41127cea3c8671e389606d9363c8776a0ef6a52c"
PRODUCER_SHA = "cec0ccee8c3ebda9bf24690badc50c153c58523373664fcdc79ed270c04fde49"
ORIGINAL_INPUT_SHA = "02561536a7ac691290f127dc07ddef41cede4b15e9bfbabb97eee0b569a0fa59"
SOURCE_SHA = "1c6163c5aeee9d0735a6859c0e2f56a701c59d5111b553de6fb5c758da138dfd"


def identity(path):
    return {"path": str(path), **contract.file_identity(path)}


def trusted(path, sha):
    if contract.file_identity(path)["sha256"] != sha:
        raise ValueError("An explicit parent/evidence identity changed")
    return contract.load_json(path)


def verify(row):
    if contract.file_identity(row["path"]) != {k: row[k] for k in ("bytes", "sha256")}:
        raise ValueError("A revised input payload changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root, candidate, prepared, work = (
        p.resolve()
        for p in (args.repository, args.candidate_root, args.prepared_data, args.workdir)
    )
    expected_env = {
        "CUDA_VISIBLE_DEVICES": "",
        "WORLD_SIZE": "4",
        "EMBED_OPTIM_MAX_STEPS": "-1",
        "EMBED_OPTIM_STOP_AFTER_STEP": "-1",
    }
    if any(os.environ.get(k) != v for k, v in expected_env.items()):
        raise ValueError("Require CPU-only full-horizon identity preparation")
    if not work.is_dir() or any(work.iterdir()) or args.workdir.is_symlink():
        raise ValueError("Require a fresh ordinary work directory")
    archive = root / "reports/engineering-archive/dense-data-candidate-v1"
    acceptance = trusted(archive / "validation.json", ACCEPTANCE_SHA)
    if not acceptance["independent_source_and_materialization_replay_passed"]:
        raise ValueError("Require the independent complete-data reconstruction")
    producer = trusted(prepared / "result.json", PRODUCER_SHA)
    original_path = root / "reports/dense-primary-v2/input-bindings.json"
    original = trusted(original_path, ORIGINAL_INPUT_SHA)
    source_path = root / "reports/engineering-archive/dense-full-identity-v1/prepared-source.json"
    source = check_sources(source_path, SOURCE_SHA, candidate)
    input_files = [r for part in producer["partitions"].values() for r in part["files"]]
    for row in input_files:
        verify(row)
    for partition in ("training", "validation"):
        actual = sorted(str(p) for p in (prepared / partition).rglob("*") if p.is_file())
        if actual != sorted(r["path"] for r in producer["partitions"][partition]["files"]):
            raise ValueError("Prepared parent directory does not match the audited inventory")
    configs = [
        replace(
            c,
            run_id=c.run_id.replace("verified-", "verified-v3-", 1),
            dataset_path=str(prepared / "training"),
            output_root=str(work / "unused-output"),
        )
        for c in load_matrix(candidate / "configs/dense_correctness_candidate.yaml")
    ]
    linkage = aggregate.audit_dataset_artifacts(configs)
    if linkage.get("complete") is not True or linkage["verified_rows"] != 500000:
        raise ValueError("Actual generic data consumer rejects the revised training inputs")
    first, dataset = contract.prepare(configs[0], train._training_argument_values(configs[0]), 4)
    if len(dataset) != 500000:
        raise ValueError("The actual Trainer view is not 500k rows")
    for field in ("source", "model"):
        contract.require_equal(first[field], original["common_identity"][field])
    if first["data"] == original["common_identity"]["data"]:
        raise ValueError("Revised data did not receive a distinct observed identity")
    common = {k: first[k] for k in ("schema_version", "scope", "source", "data", "model")}
    old_runs = {r["run_id"]: r for r in original["runs"]}
    runs = []
    for config in configs:
        arguments = train._training_argument_values(config)
        fields = {
            "recipe": contract.recipe_identity(config),
            "execution": contract.execution_identity(arguments, 4),
            "numerical_contract": numerical.receipt(
                config.optimizer, arguments["gradient_accumulation_steps"]
            ),
        }
        old_id = config.run_id.replace("verified-v3-", "verified-", 1)
        old_fields = old_runs[old_id]["identity_fields"]
        for key in fields:
            comparable = dict(fields[key])
            if key == "recipe":
                comparable["run_id"] = old_id
            contract.require_equal(comparable, old_fields[key])
        runs.append(
            {
                "run_id": config.run_id,
                "parent_run_id": old_id,
                "identity_fields": fields,
                "expected_identity_sha256": contract.digest({**common, **fields}),
            }
        )
    if len(runs) != 12 or len({r["expected_identity_sha256"] for r in runs}) != 12:
        raise ValueError("The full twelve-recipe identity grid is missing")
    validation = Dataset.load_from_disk(str(prepared / "validation/dataset"))
    if len(validation) != 4096:
        raise ValueError("The actual validation data is not 4096 rows")
    proposal = contract.load_json(
        root / "reports/engineering-archive/dense-data-partition-v1/revision-proposal.json"
    )
    train_full = Dataset.load_from_disk(str(prepared / "training/dataset"))
    derived = []
    for subset in proposal["derived_subset_impact"]:
        for row in subset["files"]:
            verify(row)
        arrow = next(
            Path(r["path"]).parent for r in subset["files"] if r["path"].endswith(".arrow")
        )
        values = Dataset.load_from_disk(str(arrow))
        parent = (
            validation if subset["name"] == "historical_candidate_breadth_queries" else train_full
        )
        columns = [c for c in values.column_names if c in parent.column_names]
        selected = parent.select(list(values["sample_id"])).select_columns(columns)
        contract.require_equal(selected.to_dict(), values.select_columns(columns).to_dict())
        derived.append({**subset, "every_value_matches_revised_parent": True})
    for row in input_files:
        verify(row)
    if check_sources(source_path, SOURCE_SHA, candidate) != source or any(
        c.output_dir.exists() for c in configs
    ):
        raise ValueError("Preparation changed source or created a training run")
    result = {
        "scope": "prepared_dense_revised_primary_actual_inputs_v3",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "preparation_passed": True,
        "scientific_completion": False,
        "production_deployed": False,
        "protocol_activated": False,
        "model_loaded": False,
        "model_updates_executed": 0,
        "parent_inputs": identity(original_path),
        "data_acceptance": identity(archive / "validation.json"),
        "data_producer": identity(prepared / "result.json"),
        "source_manifest": identity(source_path),
        "candidate_root": str(candidate),
        "source": identity(Path(__file__).resolve()),
        "common_identity": common,
        "runs": runs,
        "initial_model": original["initial_model"],
        "data_linkage_audit": linkage,
        "data_manifest": identity(prepared / "training/manifest.json"),
        "row_manifest": identity(prepared / "training/rows.jsonl"),
        "validation_inputs": {
            "root": str(prepared / "validation"),
            "rows": len(validation),
            "files": producer["partitions"]["validation"]["files"],
        },
        "derived_subset_revised_parent_checks": derived,
        "original_training": original["original_training"],
        "original_analysis": original["original_analysis"],
        "original_evaluation": original["original_evaluation"],
        "boundary": "Actual common data/model/source observed; twelve full-horizon recipe identities derived without training. Revised data and new v3 run IDs differ; model/source/runtime/hyperparameters do not. Old protocols remain unchanged and ineligible for new data. Derived subset values match the revised parents but no scientific consumer is deployed by this receipt.",
    }
    with (work / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "preparation_passed": True,
                "actual_training_rows": len(dataset),
                "actual_validation_rows": len(validation),
                "recipe_identities": len(runs),
                "derived_subsets": len(derived),
                "model_updates": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
