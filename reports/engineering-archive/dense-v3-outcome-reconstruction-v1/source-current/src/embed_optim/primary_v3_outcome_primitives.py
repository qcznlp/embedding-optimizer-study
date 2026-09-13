"""Rebuild original outcome inputs from local roles, not stored aggregate values.

Previously authenticated checkpoint/data metadata is checked for consistency;
no checkpoint payload, encoder, corpus retrieval or raw text is revalidated.
Original location strings remain provenance only and are never opened.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from . import reconstruction_files as files
from .primary_completion import INFERENCE_FILES, task_score
from .primary_contract import RUN_RECEIPT, SEAL, digest, read_json, relative_path, require_same
from .primary_v3_contract import SCOPE as PRIMARY_SCOPE
from .primary_v3_outcomes import system_rows
from .primary_v3_validation import METRICS, SCOPE, SETTINGS, select_recipes
from .primary_v3_validation_io import inspect_saved

GRID_BOUNDARY = "Structural/deep-state/task-provenance acceptance only. Validation selection, scientific outcomes, mechanism analyses and paper release remain separate required consumers."
SELECTION_BOUNDARY = (
    "Held-out selection only. No BEIR input; not retrieval inference or paper release."
)


def producer_path(value):
    """Pure lexical parsing only: never resolve/stat/open an archived producer path."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Invalid original provenance path")
    path = PurePosixPath(value)
    if path.as_posix() != value or ".." in path.parts:
        raise ValueError("Noncanonical original provenance path")
    return path


def recorded_checkpoint(primary, run_id, step, value):
    """Validate the exact sealed-reader metadata schema, never its absent tensor bytes."""
    if set(value) != {"step", "run_identity_sha256", "checkpoint_seal", "files"}:
        raise ValueError("Archived checkpoint lacks complete sealed-reader metadata")
    if type(value["step"]) is not int or value["step"] != step:
        raise ValueError("Archived checkpoint stage differs")
    require_same(value["run_identity_sha256"], digest(primary.expected_identity(run_id)))
    if set(value["checkpoint_seal"]) != {"bytes", "sha256"}:
        raise ValueError("Invalid archived checkpoint seal identity")
    files.identity(value["checkpoint_seal"])
    rows = value["files"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Empty archived checkpoint inventory")
    names = []
    for row in rows:
        if set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid archived checkpoint file identity")
        names.append(relative_path(row["path"]).as_posix())
        files.identity(row)
    expected = primary.expected_identity(run_id)
    world = expected["execution"]["world_size"]
    rng = {"rng_state.pth"} if world == 1 else {f"rng_state_{r}.pth" for r in range(world)}
    required = {
        RUN_RECEIPT,
        "dense_numerical_contract.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        *INFERENCE_FILES,
        *rng,
    }
    if (
        names[-1] != SEAL
        or names[:-1] != sorted(set(names[:-1]))
        or SEAL in names[:-1]
        or not required.issubset(names)
    ):
        raise ValueError("Incomplete, duplicated or reordered archived checkpoint inventory")
    require_same(files.identity(rows[-1]), value["checkpoint_seal"])
    return value


def recorded_runs(primary, admitted):
    """The shared vector admission supplies the same complete twelve-run population."""
    runs = admitted["complete_runs"]
    names = [row["run_id"] for row in primary.inputs["runs"]]
    require_same(sorted(runs), sorted(names))
    steps = primary.payload["checkpoint_steps"]
    for run_id in names:
        run = runs[run_id]
        for key, expected in {
            "scope": PRIMARY_SCOPE,
            "protocol_sha256": primary.sha256,
            "run_id": run_id,
            "run_identity_sha256": digest(primary.expected_identity(run_id)),
            "whole_run_artifacts_verified": True,
            "scientific_completion": False,
            "dataset_fingerprint": primary.inputs["data_linkage_audit"][
                "training_view_fingerprint"
            ],
            "steps": steps,
        }.items():
            require_same(run[key], expected)
        if len(run["checkpoints"]) != len(steps):
            raise ValueError("Incomplete archived primary checkpoint population")
        for step, value in zip(steps, run["checkpoints"], strict=True):
            recorded_checkpoint(primary, run_id, step, value)
        for key in ("checkpoint_bytes", "optimizer_state_bytes"):
            wanted = {
                f"checkpoint-{row['step']}": sum(
                    file["bytes"]
                    for file in row["files"]
                    if key == "checkpoint_bytes" or file["path"] == "optimizer.pt"
                )
                for row in run["checkpoints"]
            }
            require_same(run["system_metrics"][key], wanted)
    # Reuse every existing timing, memory, hardware and complete-run systems constraint.
    system_rows(primary, [runs[name] for name in names])
    return runs


def validation_identity(primary, value):
    if set(value) != {"content", "row_identities", "row_identities_sha256"}:
        raise ValueError("Incomplete archived validation identity")
    expected = primary.payload["datasets"]["validation"]
    require_same(
        value["content"],
        {"partition": "validation", "content_sha256": digest(expected), **expected},
    )
    rows = value["row_identities"]
    if not isinstance(rows, list) or len(rows) != SETTINGS["rows"]:
        raise ValueError("Validation identity lacks all 4096 rows")
    require_same(value["row_identities_sha256"], digest(rows))
    for index, row in enumerate(rows):
        if set(row) != {
            "position",
            "sample_id",
            "source",
            "query_id",
            "positive_id",
            "negative_ids",
            "row_sha256",
        }:
            raise ValueError("Unexpected validation row identity schema")
        if (
            any(
                type(row[k]) is not int or row[k] < 0
                for k in ("position", "sample_id", "query_id", "positive_id")
            )
            or row["position"] != index
            or row["sample_id"] != index
            or not isinstance(row["source"], str)
            or not row["source"]
            or not isinstance(row["negative_ids"], list)
            or len(row["negative_ids"]) != 7
            or any(type(n) is not int or n < 0 for n in row["negative_ids"])
        ):
            raise ValueError("Invalid ordered validation query/document identities")
        files.identity({"bytes": 0, "sha256": row["row_sha256"]})
    if len({(row["source"], row["query_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate source-qualified validation query")
    return rows


def require_used_role(root, role, used):
    actual = {role + "/" + name for name in files.inventory(root / role)}
    require_same(sorted(actual), sorted(used))


def selection_from_records(inputs, runs, original):
    """No BEIR values are passed to this selection calculation."""
    primary = inputs.contract.primary
    contract = inputs.contract.bridge.outcomes.validation
    data = inputs.manifest["metadata"]["validation_identity"]
    identities = validation_identity(primary, data)
    old = original["source_receipts"]
    require_same(
        [row["plan"]["run_id"] for row in old], [row["run_id"] for row in primary.inputs["runs"]]
    )
    metrics, receipts, addresses, used = [], [], [], set()
    for recipe, stored in zip(primary.inputs["runs"], old, strict=True):
        run_id = recipe["run_id"]
        plan = {
            "scope": SCOPE,
            "validation_protocol_sha256": contract.sha256,
            "primary_protocol_sha256": primary.sha256,
            "run_id": run_id,
            "run_identity_sha256": runs[run_id]["run_identity_sha256"],
            "checkpoint": runs[run_id]["checkpoints"][-1],
            "dataset_content_sha256": data["content"]["content_sha256"],
            "row_identities_sha256": data["row_identities_sha256"],
            "settings": SETTINGS,
            "scientific_completion": False,
        }
        key = digest(plan)
        original_manifest = producer_path(stored["manifest"]["path"])
        if original_manifest.name != "manifest.json" or original_manifest.parent.name != key:
            raise ValueError("Original validation location differs from its plan identity")
        relative = "validation/" + key
        result = inspect_saved(inputs.root / relative, plan, identities)
        require_same(files.identity(result["manifest"]), files.identity(stored["manifest"]))
        # Only the location string is restored, after the same locally parsed metadata matches.
        result["manifest"]["path"] = stored["manifest"]["path"]
        require_same(result, stored)
        overall = [row for row in result["summary"]["groups"] if row["group"] == "__all__"]
        if len(overall) != 1 or overall[0]["samples"] != SETTINGS["rows"]:
            raise ValueError("Missing complete validation aggregate")
        optimizer = primary.expected_identity(run_id)["recipe"]["optimizer"]
        metrics.append(
            {
                "run_id": run_id,
                "optimizer": optimizer["name"],
                "learning_rate": optimizer["lr"],
                **{k: overall[0][k] for k in METRICS},
            }
        )
        receipts.append(result)
        for name in ("admission.json", "manifest.json", "sample_scores.jsonl", "summary.json"):
            local = relative + "/" + name
            used.add(local)
            addresses.append({"original": str(original_manifest.parent / name), "local": local})
    require_used_role(inputs.root, "validation", used)
    selection = {
        "scope": "dense_primary_v3_validation_selection",
        "primary_protocol_sha256": primary.sha256,
        "validation_protocol_sha256": contract.sha256,
        "selected": select_recipes(metrics, primary.inputs["runs"]),
        "run_metrics": metrics,
        "source_receipts": receipts,
        "scientific_completion": False,
        "boundary": SELECTION_BOUNDARY,
    }
    require_same(selection, original)
    return selection, addresses


def grid_from_tasks(inputs, runs, original):
    """Reparse all 840 task files only after the independent validation selection."""
    primary = inputs.contract.primary
    versions = read_json(primary.repository / "configs/formal_runtime.json")["packages"]
    population = [
        (row["run_id"], stage, step)
        for row in primary.inputs["runs"]
        for stage, step in enumerate(primary.payload["checkpoint_steps"], 1)
    ]
    require_same(
        [[row["run_id"], row["stage"], row["step"]] for row in original["evaluations"]],
        [list(cell) for cell in population],
    )
    evaluations, scores, addresses, used = [], [], [], set()
    for (run_id, stage, step), stored in zip(population, original["evaluations"], strict=True):
        checkpoint = {
            "scope": PRIMARY_SCOPE,
            "protocol_sha256": primary.sha256,
            "run_id": run_id,
            **runs[run_id]["checkpoints"][stage - 1],
        }
        key = digest({"protocol": primary.sha256, "checkpoint": checkpoint})
        source_root = producer_path(stored["plan"]["results_root"])
        if source_root.name != key:
            raise ValueError("Original BEIR location differs from its checkpoint cache identity")
        plan = {
            "scope": "dense_primary_v3_beir_admission",
            "protocol_sha256": primary.sha256,
            "checkpoint": checkpoint,
            "cache_key": key,
            "results_root": stored["plan"]["results_root"],
            "tasks": primary.payload["evaluation"]["tasks"],
            "scientific_completion": False,
        }
        require_same(stored["plan"], plan)
        admission = "beir/" + key + "/primary_admission.json"
        require_same(read_json(inputs.root / admission), plan)
        used.add(admission)
        addresses.append(
            {"original": str(source_root / "primary_admission.json"), "local": admission}
        )
        require_same([row["task"] for row in stored["tasks"]], plan["tasks"])
        expected = primary.expected_identity(run_id)
        tasks = []
        for task, old in zip(plan["tasks"], stored["tasks"], strict=True):
            if len(old["files"]) != 3:
                raise ValueError("Incomplete original task provenance")
            original_task = producer_path(old["files"][0]["path"])
            relative = relative_path(original_task.relative_to(source_root).as_posix())
            if relative.parts[0] != "dense" or relative.name != task + "Decontaminated.json":
                raise ValueError("Task file does not belong to its declared local BEIR job")
            original_files = [
                original_task,
                original_task.parent / "model_meta.json",
                original_task.parent / "run_settings.jsonl",
            ]
            require_same([row["path"] for row in old["files"]], [str(p) for p in original_files])
            local = "beir/" + key + "/" + relative.as_posix()
            result = task_score(
                inputs.root / local,
                task,
                run_id,
                step,
                expected["recipe"]["max_length"],
                primary.payload["beir_task_revisions"][task]["revision"],
                versions,
            )
            for checked_file, original_file in zip(result["files"], old["files"], strict=True):
                require_same(files.identity(checked_file), files.identity(original_file))
                local_name = str(PurePosixPath(checked_file["path"]).relative_to(inputs.root))
                files.safe_name(local_name)
                used.add(local_name)
                addresses.append({"original": original_file["path"], "local": local_name})
                checked_file["path"] = original_file["path"]
            require_same(result, old)
            tasks.append(result)
            optimizer = expected["recipe"]["optimizer"]
            scores.append(
                {
                    "run_id": run_id,
                    "model_family": "dense",
                    "optimizer": optimizer["name"],
                    "learning_rate": optimizer["lr"],
                    "stage": stage,
                    "step": step,
                    "fraction": expected["recipe"]["checkpoint_fractions"][stage - 1],
                    "task": task,
                    "ndcg_at_10": result["ndcg_at_10"],
                }
            )
        current = {
            "run_id": run_id,
            "stage": stage,
            "step": step,
            "complete_task_files_verified": True,
            "scientific_completion": False,
            "plan": plan,
            "tasks": tasks,
        }
        require_same(current, stored)
        evaluations.append(current)
    require_used_role(inputs.root, "beir", used)
    grid = {
        "scope": "dense_primary_v3_complete_grid",
        "artifact_grid_verified": True,
        "scientific_completion": False,
        "protocol_sha256": primary.sha256,
        "runs": [runs[row["run_id"]] for row in primary.inputs["runs"]],
        "evaluations": evaluations,
        "score_rows": scores,
        "boundary": GRID_BOUNDARY,
    }
    if len(evaluations) != 60 or len(scores) != 840:
        raise ValueError("Incomplete original outcome checkpoint/task grid")
    require_same(grid, original)
    return grid, addresses
