"""Whole-run and full-task acceptance for the revised dense primary experiment.

These readers do not train, score, upload, retrofit artifacts, or render the paper.
Generic helpers support explicitly diagnostic receipts; only the primary wrapper
can attach a primary protocol identity, and a draft never gains release status.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .primary_contract import (
    PrimaryContract,
    digest,
    file_identity,
    inspect_sealed_checkpoint,
    read_json,
    require_same,
    verify_file,
)
from .primary_io import evaluation_plan

DENSE_PARTITION = {
    "hidden": {"tensors": 88, "parameters": 110297088},
    "aux_decay": {"tensors": 1, "parameters": 38682624},
    "aux_no_decay": {"tensors": 45, "parameters": 34560},
}
INFERENCE_FILES = (
    "model.safetensors",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
)
RUNTIME_PACKAGES = ("mteb", "torch", "sentence-transformers", "flash-attn", "transformers")


def integer(value, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Require an integer in the declared range")
    return value


def number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Require a finite non-boolean number")
    if positive and value <= 0:
        raise ValueError("Require a positive number")
    return value


def close(actual, expected):
    if not math.isclose(number(actual), expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Optimizer/scheduler numerical setting differs")


def timing_summary(payload, steps):
    if payload.get("schema_version") != 1 or not isinstance(payload.get("segments"), list):
        raise ValueError("Invalid whole-run timing ledger")
    previous, elapsed, previous_time = 0, 0.0, None
    segments = payload["segments"]
    if len(segments) != len(steps):
        raise ValueError("Timing does not cover every scheduled checkpoint")
    for segment, expected_end in zip(segments, steps, strict=True):
        if (
            integer(segment["start_step_exclusive"]) != previous
            or integer(segment["end_step_inclusive"]) != expected_end
        ):
            raise ValueError("Timing has a missing/overlapping prefix or wrong endpoint")
        started, ended = (
            datetime.fromisoformat(segment[k].replace("Z", "+00:00"))
            for k in ("started_at_utc", "checkpoint_completed_at_utc")
        )
        if any(t.tzinfo is None or t.utcoffset().total_seconds() != 0 for t in (started, ended)):
            raise ValueError("Timing requires actual UTC timestamps")
        if ended <= started or (previous_time is not None and started < previous_time):
            raise ValueError("Timing timestamps overlap or reverse")
        previous, previous_time = expected_end, ended
        elapsed += number(segment["wall_time_seconds_max_rank"], positive=True)
    if not math.isclose(
        number(payload["total_wall_time_seconds_max_rank"]), elapsed, rel_tol=1e-9, abs_tol=1e-6
    ):
        raise ValueError("Whole-run timing total differs from segments")
    return {
        "schema_version": 1,
        "segments": len(segments),
        "total_wall_time_seconds_max_rank": elapsed,
    }


def group_contract(expected):
    recipe, numerical = expected["recipe"], expected["numerical_contract"]
    opt = recipe["optimizer"]
    name = opt["name"]
    if name not in {"adamw", "muon", "normuon"}:
        raise ValueError("Unsupported primary optimizer")

    def adam(lr, wd, auxiliary=False):
        prefix = "aux_" if auxiliary else ""
        return {
            "algorithm": "adamw",
            "initial_lr": lr,
            "weight_decay": wd,
            "betas": (opt[f"{prefix}beta1"], opt[f"{prefix}beta2"]),
            "eps": opt[f"{prefix}eps"],
        }

    if name == "adamw":
        return [adam(opt["lr"], opt["weight_decay"]), adam(opt["lr"], 0.0)]
    return [
        {
            "algorithm": name,
            "initial_lr": opt["lr"],
            "weight_decay": opt["weight_decay"],
            "momentum": opt["momentum"],
            "beta2": opt["normuon_beta2"],
            "ns_steps": opt["ns_steps"],
            "ns_implementation": numerical["ns_implementation"],
            "adjust_lr_fn": opt["adjust_lr_fn"],
        },
        adam(opt["aux_lr"], opt["weight_decay"], True),
        adam(opt["aux_lr"], 0.0, True),
    ]


def inspect_optimizer_state(optimizer, scheduler, expected, step, horizon, model_shapes):
    """Validate actual tensor state and the new NorMuon implementation identity."""
    import torch

    groups = group_contract(expected)
    actual_groups = optimizer.get("param_groups")
    if not isinstance(actual_groups, list) or len(actual_groups) != len(groups):
        raise ValueError("Wrong optimizer group topology")
    warmup = math.ceil(horizon * expected["recipe"]["warmup_ratio"])
    factor = (
        step / max(1, warmup)
        if step < warmup
        else max(0.0, (horizon - step) / max(1, horizon - warmup))
    )
    state = optimizer.get("state")
    if not isinstance(state, dict):
        raise ValueError("Missing optimizer tensor state")
    seen, shape_counts = [], Counter()
    for observed, contract in zip(actual_groups, groups, strict=True):
        for key, value in contract.items():
            if isinstance(value, (float, int)):
                close(observed.get(key), value)
            elif observed.get(key) != value:
                raise ValueError(f"Optimizer {key} differs from its sealed recipe")
        close(observed.get("lr"), contract["initial_lr"] * factor)
        params = observed.get("params")
        if not isinstance(params, list) or not params:
            raise ValueError("Empty optimizer group")
        algorithm = contract["algorithm"]
        fields = {
            "adamw": {"step", "exp_avg", "exp_avg_sq"},
            "muon": {"momentum_buffer"},
            "normuon": {"momentum_buffer", "second_moment"},
        }[algorithm]
        for parameter_id in params:
            integer(parameter_id)
            seen.append(parameter_id)
            values = state.get(parameter_id)
            if not isinstance(values, dict) or set(values) != fields:
                raise ValueError("Missing or wrong optimizer state fields")
            for value in values.values():
                if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
                    raise ValueError("Optimizer state is not a finite tensor")
            if algorithm == "adamw":
                if values["step"].numel() != 1 or values["step"].item() != step:
                    raise ValueError("Adam state counter differs from the checkpoint step")
                shape = tuple(values["exp_avg"].shape)
                if tuple(values["exp_avg_sq"].shape) != shape or bool(
                    (values["exp_avg_sq"] < 0).any()
                ):
                    raise ValueError("Invalid Adam second moment")
            else:
                shape = tuple(values["momentum_buffer"].shape)
                if len(shape) != 2:
                    raise ValueError("Muon hidden state is not matrix-shaped")
                if algorithm == "normuon" and (
                    tuple(values["second_moment"].shape) != (shape[0], 1)
                    or bool((values["second_moment"] < 0).any())
                ):
                    raise ValueError("Invalid NorMuon row second moment")
            shape_counts[shape] += 1
    if (
        len(set(seen)) != len(seen)
        or set(seen) != set(state)
        or shape_counts != Counter(model_shapes)
    ):
        raise ValueError("Optimizer state does not cover model tensors exactly once")
    if (
        integer(scheduler.get("last_epoch")) != step
        or integer(scheduler.get("_step_count")) != step + 1
    ):
        raise ValueError("Scheduler counter differs from the checkpoint step")
    for key, target in (
        ("base_lrs", [g["initial_lr"] for g in groups]),
        ("_last_lr", [g["initial_lr"] * factor for g in groups]),
    ):
        values = scheduler.get(key)
        if not isinstance(values, list) or len(values) != len(target):
            raise ValueError("Scheduler group coverage differs")
        for actual, wanted in zip(values, target, strict=True):
            close(actual, wanted)
    if scheduler.get("lr_lambdas") != [{} for _ in groups]:
        raise ValueError("Unexpected scheduler lambda state")
    return {
        "parameter_states": len(seen),
        "groups": len(groups),
        "scheduler_step": step,
        "group_parameter_counts": [len(g["params"]) for g in actual_groups],
    }


def deep_checkpoint(checkpoint, expected, step, horizon):
    # Invoked only AFTER the entire supplied checkpoint payload seal was checked.
    import torch
    from safetensors import safe_open

    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as model:
        shapes = []
        for key in model.keys():
            tensor = model.get_tensor(key)
            if not tensor.numel() or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Invalid/non-finite saved model tensor")
            shapes.append(tuple(tensor.shape))
        if len(shapes) != 134 or sum(math.prod(s) for s in shapes) != 149014272:
            raise ValueError("Saved model is not the full declared DenseOn topology")
    optimizer = torch.load(
        checkpoint / "optimizer.pt", map_location="cpu", weights_only=True, mmap=True
    )
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    checked = inspect_optimizer_state(optimizer, scheduler, expected, step, horizon, shapes)
    require_same(
        checked["group_parameter_counts"],
        [89, 45] if expected["recipe"]["optimizer"]["name"] == "adamw" else [88, 1, 45],
    )
    return checked


def inspect_complete_run(run_root, expected, steps):
    """Read a complete local run; continuation-only directories are not whole runs."""
    root = Path(run_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Require an ordinary retained run directory")
    steps = [integer(s, minimum=1) for s in steps]
    if steps != sorted(set(steps)) or not steps:
        raise ValueError("Invalid full-run stage schedule")
    require_same(read_json(root / "dense_run_contract.json"), expected)
    names = (
        "completed.json",
        "run_config.json",
        "checkpoint_schedule.json",
        "accepted_timing.json",
        "trainer_state_final.json",
        "dense_run_contract.json",
    )
    before = {name: file_identity(root / name) for name in names}
    if (root / "diagnostic_completed.json").exists():
        raise ValueError("Diagnostic stop is not terminal primary execution")
    completed, schedule, state, timing = (
        read_json(root / name)
        for name in (
            "completed.json",
            "checkpoint_schedule.json",
            "trainer_state_final.json",
            "accepted_timing.json",
        )
    )
    recipe = dict(read_json(root / "run_config.json"))
    for key in ("dataset_path", "output_root", "wandb_entity", "wandb_project"):
        recipe.pop(key, None)
    if isinstance(expected["recipe"]["model_name"], dict):
        recipe["model_name"] = {"kind": "local_content_bound_model"}
    require_same(recipe, expected["recipe"])
    for value in (completed, state):
        if integer(value["global_step"]) != steps[-1]:
            raise ValueError("Whole-run terminal step differs")
    if (
        integer(state["max_steps"]) != steps[-1]
        or number(state["epoch"]) != expected["recipe"]["epochs"]
    ):
        raise ValueError("Actual Trainer horizon or completed epoch differs")
    if (
        completed.get("run_id") != expected["recipe"]["run_id"]
        or completed.get("model_family") != "dense"
        or completed.get("run_identity_sha256") != digest(expected)
    ):
        raise ValueError("Completion identity differs from the primary recipe")
    if integer(completed["dataset_rows"]) != expected["data"]["rows"]:
        raise ValueError("Completion dataset row count differs")
    require_same(completed["numerical_contract"], expected["numerical_contract"])
    require_same(completed["optimizer_partition"], DENSE_PARTITION)
    require_same(completed["checkpoints"], steps)
    require_same(
        schedule,
        {
            "max_steps": steps[-1],
            "fractions": expected["recipe"]["checkpoint_fractions"],
            "steps": steps,
        },
    )
    if sorted(p.name for p in root.glob("checkpoint-*")) != sorted(
        f"checkpoint-{s}" for s in steps
    ):
        raise ValueError("Whole-run checkpoint prefix is incomplete or contains extra stages")
    require_same(completed["accepted_timing"], timing_summary(timing, steps))
    for record in state["log_history"]:
        integer(record["step"])
        for key, value in record.items():
            if key != "step":
                number(value)
    systems = completed["system_metrics"]
    if (
        integer(systems["world_size"]) != expected["execution"]["world_size"]
        or not isinstance(systems["gpu_name"], str)
        or not systems["gpu_name"].strip()
    ):
        raise ValueError("Invalid actual device/world-size metadata")
    for key in (
        "wall_time_seconds_max_rank",
        "peak_allocated_bytes_max_rank",
        "peak_reserved_bytes_max_rank",
    ):
        number(systems[key], positive=True)
    for key in (
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_loss",
        "epoch",
    ):
        number(systems["trainer"][key], positive=key != "train_loss")
    if systems["trainer"]["epoch"] != expected["recipe"]["epochs"]:
        raise ValueError("Trainer metrics do not reach the declared epoch")
    for name in ("torch", "sentence-transformers", "transformers"):
        if completed["versions"].get(name) != expected["source"]["packages"][name]:
            raise ValueError("Training package version differs")
    checks = [inspect_sealed_checkpoint(root / f"checkpoint-{s}", expected, s) for s in steps]
    for field in ("checkpoint_bytes", "optimizer_state_bytes"):
        require_same(
            systems[field],
            {
                f"checkpoint-{row['step']}": sum(
                    p["bytes"]
                    for p in row["files"]
                    if field == "checkpoint_bytes" or p["path"] == "optimizer.pt"
                )
                for row in checks
            },
        )
    final = root / "final"
    require_same(read_json(final / "dense_run_contract.json"), expected)
    last = {row["path"]: row for row in checks[-1]["files"]}
    final_files = []
    for name in INFERENCE_FILES:
        final_files.append({"path": name, **verify_file(final / name, last[name])})
    if "materialization_manifest" in expected["data"]:
        verify_file(root / "dataset_manifest.json", expected["data"]["materialization_manifest"])
    # Deep model/optimizer inspection follows ALL metadata and payload checks, and
    # never imports or initializes a trainable model or a GPU context.
    deep = [deep_checkpoint(root / f"checkpoint-{s}", expected, s, steps[-1]) for s in steps]
    for checked in checks:
        for row in checked["files"]:
            verify_file(root / f"checkpoint-{checked['step']}" / row["path"], row)
    for row in final_files:
        verify_file(final / row["path"], row)
    for name, identity in before.items():
        verify_file(root / name, identity)
    return {
        "whole_run_artifacts_verified": True,
        "scientific_completion": False,
        "run_identity_sha256": digest(expected),
        "dataset_fingerprint": completed["dataset_fingerprint"],
        "steps": steps,
        "metadata": before,
        "checkpoints": checks,
        "final_inference_files": final_files,
        "deep_checkpoint_checks": deep,
        "accepted_timing": completed["accepted_timing"],
        "system_metrics": systems,
    }


def unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate result/settings JSON key")
        result[key] = value
    return result


def read_mteb_result(path):
    """Keep undefined auxiliary nAUC explicit; the predeclared nDCG must be finite.

    MTEB 2.18 deliberately emits NaN for nAUC when its oracle-minus-flat
    denominator is zero. These unused auxiliary fields never enter the study's
    nDCG estimand. No raw file is rewritten and no missing metric is imputed.
    """
    path = Path(path)
    before = file_identity(path)
    value = json.loads(path.read_text(), object_pairs_hook=unique_json_keys)
    undefined = []
    allowed = re.compile(
        r"nauc_(ndcg|map|recall|precision|mrr)_at_(1|3|5|10|20|100|1000)_(max|std|diff1)"
    )
    for split, rows in value.get("scores", {}).items():
        for index, row in enumerate(rows):
            for name, score in row.items():
                if allowed.fullmatch(name) and isinstance(score, float) and math.isnan(score):
                    undefined.append({"split": split, "subset_row": index, "metric": name})
                    row[name] = None  # In-memory only; disclosed separately, never scored.
    digest(value)  # Reject every other non-finite number, including infinities.
    verify_file(path, before)
    return value, undefined


def task_score(path, task, run_id, step, model_length, revision, versions):
    """Require exact task/revision/split/subset/runtime/model metadata and bounded nDCG."""
    path = Path(path)
    value, undefined = read_mteb_result(path)
    split, name = ("dev" if task == "MSMARCO" else "test"), f"{task}Decontaminated"
    if (
        value["task_name"] != name
        or value["dataset_revision"] != revision
        or value["mteb_version"] != versions["mteb"]
    ):
        raise ValueError("MTEB task/revision/runtime identity differs")
    number(value["evaluation_time"], positive=True)
    scores = value["scores"]
    if set(scores) != {split} or len(scores[split]) != 1:
        raise ValueError("Incomplete or ambiguous task split/subset coverage")
    score = scores[split][0]
    if score["hf_subset"] != "default" or score["mteb_version"] != versions["mteb"]:
        raise ValueError("Subset identity/version differs")
    ndcg = number(score["ndcg_at_10"])
    if not 0 <= ndcg <= 1 or not math.isclose(
        ndcg, number(score["main_score"]), rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("Invalid or inconsistent normalized nDCG@10")
    meta = read_json(path.parent / "model_meta.json")
    for key, expected in {
        "name": f"{run_id}/checkpoint-{step}",
        "revision": "local",
        "max_tokens": model_length,
        "embed_dim": 768,
        "similarity_fn_name": "cosine",
    }.items():
        require_same(meta[key], expected)
    if "Sentence Transformers" not in meta["framework"]:
        raise ValueError("Wrong dense evaluation framework")
    settings_path = path.parent / "run_settings.jsonl"

    settings = [
        json.loads(line, object_pairs_hook=unique_json_keys)
        for line in settings_path.read_text().splitlines()
        if line.strip()
    ]
    digest(settings)  # Reject non-finite JSON, including unrelated settings fields.
    matching = [row for row in settings if row.get("task") == name]
    if len(matching) != 1:
        raise ValueError("Missing or ambiguous task run settings")
    record = matching[0]
    if (
        record.get("split") != split
        or record.get("subset") != "default"
        or "splits" in record
        or "subsets" in record
    ):
        raise ValueError("Wrong pinned MTEB 2.18 split/subset settings")
    require_same(record["version"], {k: versions[k] for k in RUNTIME_PACKAGES})
    return {
        "task": task,
        "ndcg_at_10": ndcg,
        "undefined_auxiliary_metrics_not_used": undefined,
        "files": [
            {"path": str(p), **file_identity(p)}
            for p in (path, path.parent / "model_meta.json", settings_path)
        ],
    }


def inspect_evaluation(contract, checkpoint, run_id, step, results_root):
    plan = evaluation_plan(contract, checkpoint, run_id, step, results_root)
    output = Path(plan["results_root"])
    require_same(read_json(output / "primary_admission.json"), plan)
    versions = read_json(contract.repository / "configs/formal_runtime.json")["packages"]
    names = [f"{task}Decontaminated.json" for task in plan["tasks"]]
    files = sorted((output / "dense").rglob("*Decontaminated.json"))
    if Counter(p.name for p in files) != Counter(names):
        raise ValueError("Evaluation lacks the exact fourteen distinct complete task files")
    expected = contract.expected_identity(run_id)
    rows = []
    for task in plan["tasks"]:
        path = next(p for p in files if p.name == f"{task}Decontaminated.json")
        rows.append(
            task_score(
                path,
                task,
                run_id,
                step,
                expected["recipe"]["max_length"],
                contract.payload["beir_task_revisions"][task]["revision"],
                versions,
            )
        )
    require_same(evaluation_plan(contract, checkpoint, run_id, step, results_root), plan)
    return {
        "complete_task_files_verified": True,
        "scientific_completion": False,
        "plan": plan,
        "tasks": rows,
    }


def inspect_matrix(contract, experiment_root, results_root):
    """Require all 12 runs, 60 stages and 840 task cells before downstream use."""
    runs, evaluations, score_rows = [], [], []
    for recipe in contract.inputs["runs"]:
        run_id = recipe["run_id"]
        expected = contract.expected_identity(run_id)
        directory = Path(experiment_root) / contract.payload["output_root"] / "dense" / run_id
        run = inspect_complete_run(directory, expected, contract.payload["checkpoint_steps"])
        if (
            run["dataset_fingerprint"]
            != contract.inputs["data_linkage_audit"]["training_view_fingerprint"]
        ):
            raise ValueError("Run consumed a different materialized training view")
        runs.append({"run_id": run_id, **run})
    # Do not collect outcome rows from a partly complete optimizer family.
    for recipe in contract.inputs["runs"]:
        run_id = recipe["run_id"]
        expected = contract.expected_identity(run_id)
        directory = Path(experiment_root) / contract.payload["output_root"] / "dense" / run_id
        for stage, step in enumerate(contract.payload["checkpoint_steps"], 1):
            evaluated = inspect_evaluation(
                contract, directory / f"checkpoint-{step}", run_id, step, results_root
            )
            evaluations.append({"run_id": run_id, "stage": stage, "step": step, **evaluated})
            for row in evaluated["tasks"]:
                score_rows.append(
                    {
                        "run_id": run_id,
                        "model_family": "dense",
                        "optimizer": expected["recipe"]["optimizer"]["name"],
                        "learning_rate": expected["recipe"]["optimizer"]["lr"],
                        "stage": stage,
                        "step": step,
                        "fraction": expected["recipe"]["checkpoint_fractions"][stage - 1],
                        "task": row["task"],
                        "ndcg_at_10": row["ndcg_at_10"],
                    }
                )
    cells = {(row["run_id"], row["stage"], row["task"]) for row in score_rows}
    wanted = {
        (row["run_id"], stage, task)
        for row in contract.inputs["runs"]
        for stage in range(1, 6)
        for task in contract.payload["evaluation"]["tasks"]
    }
    if len(runs) != 12 or len(evaluations) != 60 or len(score_rows) != 840 or cells != wanted:
        raise ValueError("Full primary 12-by-5-by-14 evidence grid is incomplete")
    return {
        "scope": "prepared_dense_primary_complete_grid",
        "artifact_grid_verified": True,
        "scientific_completion": False,
        "protocol_sha256": contract.sha256,
        "runs": runs,
        "evaluations": evaluations,
        "score_rows": score_rows,
        "boundary": "Structural/deep-state/task-provenance acceptance only. Validation selection, scientific outcomes, mechanism analyses and paper release remain separate required consumers.",
    }


def check_completion_lock(path, contract):
    lock = read_json(path)
    if (
        lock.get("scope") != "prepared_dense_primary_completion_lock"
        or lock.get("schema_version") != 1
    ):
        raise ValueError("Not the new source-bound completion reader lock")
    if (
        lock.get("status") != "prepared_not_execution_authorized"
        or lock.get("scientific_completion") is not False
    ):
        raise ValueError("This completion proposal cannot claim release or scientific completion")
    if lock["parent_protocol"]["sha256"] != contract.sha256:
        raise ValueError("Completion lock points to another primary revision")
    if (
        lock["parent_protocol"]["path"] != contract.path.relative_to(contract.repository).as_posix()
        or lock["reader_source"]["path"] != "src/embed_optim/primary_completion.py"
    ):
        raise ValueError("Completion source/parent path labels differ")
    verify_file(contract.path, lock["parent_protocol"])
    verify_file(Path(__file__).resolve(), lock["reader_source"])
    require_same(lock["required_grid"], {"runs": 12, "stages": 5, "tasks": 14, "cells": 840})
    return file_identity(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "evaluation", "matrix"))
    parser.add_argument("--completion-lock", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args(argv)
    contract = PrimaryContract.load(args.protocol, args.repository, args.training_root)
    lock = check_completion_lock(args.completion_lock, contract)
    if args.mode == "matrix":
        if args.experiment_root is None or args.results_root is None:
            parser.error("matrix requires --experiment-root and --results-root")
        result = inspect_matrix(contract, args.experiment_root, args.results_root)
    elif args.run_id is None or args.run_root is None:
        parser.error("run/evaluation require --run-id and --run-root")
    elif args.mode == "run":
        result = inspect_complete_run(
            args.run_root,
            contract.expected_identity(args.run_id),
            contract.payload["checkpoint_steps"],
        )
    else:
        if args.step is None or args.results_root is None:
            parser.error("evaluation requires --step and --results-root")
        result = inspect_evaluation(
            contract,
            args.run_root / f"checkpoint-{args.step}",
            args.run_id,
            args.step,
            args.results_root,
        )
    print(
        json.dumps(
            {
                "scope": "prepared_dense_primary_completion_inspection",
                "protocol_sha256": contract.sha256,
                "completion_lock": lock,
                "scientific_completion": False,
                **result,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
