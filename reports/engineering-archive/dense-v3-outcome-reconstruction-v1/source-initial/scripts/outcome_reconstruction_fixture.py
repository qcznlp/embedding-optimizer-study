"""Complete raw-outcome reconstruction fixture, with explicitly simulated upstream state.

Reuse the preserved raw candidate scores and pinned task files. Upgrade only new
synthetic checkpoint metadata and shared-file topology; never edit the first
fixture, encode a model, perform retrieval, or pretend its checkpoints exist.
"""

import copy
import json
import shutil
from pathlib import Path

from embed_optim import reconstruction_files as files
from embed_optim.primary_completion import INFERENCE_FILES, task_score
from embed_optim.primary_contract import (
    RUN_RECEIPT,
    SEAL,
    digest,
    file_identity,
    read_json,
    require_same,
    verify_file,
)
from embed_optim.primary_v3_contract import SCOPE as PRIMARY_SCOPE
from embed_optim.primary_v3_dimension_contract import planned_states
from embed_optim.primary_v3_dimension_inference import SCOPE as INFERENCE_SCOPE
from embed_optim.primary_v3_outcomes import BOUNDARY as OUTCOME_BOUNDARY
from embed_optim.primary_v3_outcomes import SCOPE as OUTCOME_SCOPE
from embed_optim.primary_v3_outcomes import outcome_tables, save_bundle, system_rows
from embed_optim.primary_v3_reconstruction_authoring import ACCEPTANCE, BOUNDARY, require_parent
from embed_optim.primary_v3_reconstruction_sources import source_selection
from embed_optim.primary_v3_validation import METRICS, select_recipes
from embed_optim.primary_v3_validation_io import read_record_file, save_scoring, write_new
from embed_optim.runtime import runtime_snapshot
from scripts.vector_reconstruction_fixture import admitted_fixture

RAW_FIXTURE = "reports/engineering-archive/dense-v3-outcome-reconstruction-v1/fixture.json"
RAW_SHA = "aa71790c4a7a3c803d921593c72570b70889bf957a84ed09a90f1dcd72e27ff9"


def raw_source(contract):
    path = contract.primary.repository / RAW_FIXTURE
    if file_identity(path)["sha256"] != RAW_SHA:
        raise ValueError("Changed preserved raw-scoring fixture")
    value = read_json(path)
    assert value["upstream_primary_admission_simulated"] is True
    assert value["scientific_completion"] is False and value["validation_score_records"] == 49152
    for row in (*value["sources"], *value["artifacts"]):
        verify_file(row["path"], row)
    return Path(value["original_producer"])


def upgraded_runs(primary, source):
    runs = read_json(source / "synthetic_complete_runs.json")
    for run_id, run in runs.items():
        expected = primary.expected_identity(run_id)
        stages = []
        for step in run["steps"]:
            names = {
                RUN_RECEIPT,
                "dense_numerical_contract.json",
                "optimizer.pt",
                "scheduler.pt",
                "trainer_state.json",
                "training_args.bin",
                *INFERENCE_FILES,
                *(f"rng_state_{r}.pth" for r in range(expected["execution"]["world_size"])),
            }
            rows = [
                {
                    "path": name,
                    "bytes": 1000 + index,
                    "sha256": digest(
                        {"SIMULATED_NO_PAYLOAD": True, "run": run_id, "step": step, "name": name}
                    ),
                }
                for index, name in enumerate(sorted(names))
            ]
            seal = {"bytes": 2048, "sha256": digest({"SIMULATED_SEAL_NO_PAYLOAD": rows})}
            stages.append(
                {
                    "step": step,
                    "run_identity_sha256": digest(expected),
                    "checkpoint_seal": seal,
                    "files": [*rows, {"path": SEAL, **seal}],
                }
            )
        run["checkpoints"] = stages
        for key in ("checkpoint_bytes", "optimizer_state_bytes"):
            run["system_metrics"][key] = {
                f"checkpoint-{row['step']}": sum(
                    item["bytes"]
                    for item in row["files"]
                    if key == "checkpoint_bytes" or item["path"] == "optimizer.pt"
                )
                for row in stages
            }
        assert run["simulated_admission"] is True
    return runs


def validation_files(primary, source, target, runs, old):
    target.mkdir()
    data = read_json(source / "validation_identity.json")
    rows, receipts = [], []
    for recipe, previous in zip(primary.inputs["runs"], old["source_receipts"], strict=True):
        run_id = recipe["run_id"]
        plan = copy.deepcopy(previous["plan"])
        plan["checkpoint"] = runs[run_id]["checkpoints"][-1]
        path = Path(previous["manifest"]["path"]).parent / "sample_scores.jsonl"
        records = read_record_file(path)
        result = save_scoring(target / digest(plan), plan, records, data["row_identities"])
        overall = next(row for row in result["summary"]["groups"] if row["group"] == "__all__")
        assert overall["samples"] == 4096
        optimizer = primary.expected_identity(run_id)["recipe"]["optimizer"]
        rows.append(
            {
                "run_id": run_id,
                "optimizer": optimizer["name"],
                "learning_rate": optimizer["lr"],
                **{k: overall[k] for k in METRICS},
            }
        )
        receipts.append(result)
    result = {
        **old,
        "selected": select_recipes(rows, primary.inputs["runs"]),
        "run_metrics": rows,
        "source_receipts": receipts,
    }
    require_same(result["run_metrics"], old["run_metrics"])
    require_same(result["selected"], old["selected"])
    return data, result


def beir_files(primary, target, runs, old):
    target.mkdir()
    versions = read_json(primary.repository / "configs/formal_runtime.json")["packages"]
    evaluations, scores = [], []
    for previous in old["evaluations"]:
        run_id, stage, step = (previous[key] for key in ("run_id", "stage", "step"))
        checkpoint = {
            "scope": PRIMARY_SCOPE,
            "protocol_sha256": primary.sha256,
            "run_id": run_id,
            **runs[run_id]["checkpoints"][stage - 1],
        }
        key = digest({"protocol": primary.sha256, "checkpoint": checkpoint})
        directory = target / key
        shared = directory / "dense/shared"
        shared.mkdir(parents=True)
        plan = {
            **previous["plan"],
            "checkpoint": checkpoint,
            "cache_key": key,
            "results_root": str(directory),
        }
        write_new(directory / "primary_admission.json", plan)
        meta = read_json(previous["tasks"][0]["files"][1]["path"])
        write_new(shared / "model_meta.json", meta)
        settings = []
        for task in previous["tasks"]:
            require_same(read_json(task["files"][1]["path"]), meta)
            settings.append(read_json(task["files"][2]["path"]))
            old_path = Path(task["files"][0]["path"])
            shutil.copy2(old_path, shared / old_path.name)
        with (shared / "run_settings.jsonl").open("x") as stream:
            for setting in settings:
                stream.write(json.dumps(setting, sort_keys=True, allow_nan=False) + "\n")
        parsed = []
        expected = primary.expected_identity(run_id)
        optimizer = expected["recipe"]["optimizer"]
        for task in plan["tasks"]:
            value = task_score(
                shared / (task + "Decontaminated.json"),
                task,
                run_id,
                step,
                8192,
                primary.payload["beir_task_revisions"][task]["revision"],
                versions,
            )
            parsed.append(value)
            scores.append(
                {
                    "run_id": run_id,
                    "optimizer": optimizer["name"],
                    "learning_rate": optimizer["lr"],
                    "model_family": "dense",
                    "stage": stage,
                    "step": step,
                    "fraction": expected["recipe"]["checkpoint_fractions"][stage - 1],
                    "task": task,
                    "ndcg_at_10": value["ndcg_at_10"],
                }
            )
        evaluations.append({**previous, "plan": plan, "tasks": parsed})
    require_same(scores, old["score_rows"])
    return {
        **old,
        "runs": [runs[row["run_id"]] for row in primary.inputs["runs"]],
        "evaluations": evaluations,
        "score_rows": scores,
    }


def make(root, contract):
    root = Path(root)
    root.mkdir()
    source = raw_source(contract)
    primary = contract.primary
    old = read_json(source / "outcomes/evidence.json")
    producer = root / "producer"
    producer.mkdir()
    runs = upgraded_runs(primary, source)
    data, selection = validation_files(
        primary, source, producer / "validation", runs, old["validation_selection"]
    )
    grid = beir_files(primary, producer / "beir", runs, old["grid"])
    original_evidence = {"grid": grid, "validation_selection": selection}
    outcome = contract.bridge.outcomes
    original_plan = {
        "scope": OUTCOME_SCOPE,
        "outcome_protocol_sha256": outcome.sha256,
        "primary_protocol_sha256": primary.sha256,
        "validation_protocol_sha256": outcome.validation.sha256,
        "evidence_sha256": digest(original_evidence),
        "scientific_completion": False,
        "scientific_rules": outcome.payload["scientific_rules"],
        "boundary": OUTCOME_BOUNDARY,
    }
    tables = outcome_tables(primary, grid["score_rows"], selection)
    tables["system_metrics"] = system_rows(primary, grid["runs"])
    save_bundle(producer / "outcomes", original_plan, original_evidence, tables)
    admitted, _ = admitted_fixture(contract)
    admitted["complete_runs"] = runs
    common = {
        key: value
        for key, value in admitted.items()
        if key not in {"states", "reference", "complete_runs"}
    }
    states = []
    for state in planned_states(primary):
        if state["cell"] == "pretrained":
            model = {"kind": "immutable_pretrained", "reference": admitted["reference"]}
        else:
            run = runs[state["meta"]["run_id"]]
            model = {
                "kind": "complete_primary_checkpoint",
                "run_identity_sha256": run["run_identity_sha256"],
                "complete_run_sha256": digest(run),
                "checkpoint": run["checkpoints"][state["stage"] - 1],
            }
        states.append({**common, "state": state, "model": model})
    admitted["states"] = states
    (producer / "vectors").mkdir()
    write_new(producer / "vectors/admission.json", admitted)
    # Only the outcome branch is exercised. Raw vector/geometry/full inference authoring is simulated.
    evidence = {
        "upstream_primary_admission_simulated": True,
        "original_bridge_evidence": {"outcome_evidence": original_evidence},
        "producer_path_must_not_be_read": str(producer),
        "vector_geometry_and_inference_payloads_simulated": True,
        "scientific_completion": False,
    }
    plan = {
        "scope": INFERENCE_SCOPE,
        "primary_protocol_sha256": primary.sha256,
        "inference_protocol_sha256": contract.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "manuscript_installation_authorized": False,
    }
    (producer / "inference").mkdir()
    write_new(producer / "inference/evidence.json", evidence)
    write_new(producer / "inference/manifest.json", {"plan": plan})
    selected = source_selection(contract, {})
    parent = require_parent(contract)
    files.select(selected, "source", ACCEPTANCE[0], parent, file_identity(parent))
    for role in ("validation", "beir", "outcomes", "vectors", "inference"):
        for name in files.inventory(producer / role):
            path = producer / role / name
            files.select(selected, role, name, path, file_identity(path))
    metadata = {
        "scope": "dense_primary_v3_checkpoint_backed_reconstruction_selection",
        "primary_protocol_sha256": primary.sha256,
        "inference_protocol_sha256": contract.sha256,
        "inference_acceptance_sha256": ACCEPTANCE[1],
        "dimension_protocol_sha256": contract.dimensions.sha256,
        "trusted_vector_manifest_sha256": "0" * 64,
        "authoring_plan": plan,
        "authoring_evidence_sha256": digest(evidence),
        "validation_identity": data,
        "original_location_fields_preserved": True,
        "boundary": BOUNDARY,
        "authoring_runtime": runtime_snapshot(
            [
                "torch",
                "numpy",
                "scipy",
                "sympy",
                "sentence-transformers",
                "transformers",
                "accelerate",
                "mteb",
            ]
        ),
        "full_raw_vector_reconstruction_repeated_by_transport": False,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    result = files.write(root / "archive", selected, metadata)
    producer.rename(root / "preserved-producer-unavailable-at-original-location")
    return root / "archive", result["manifest_sha256"]
