"""V3 whole-run and complete BEIR grid admission using the retained deep readers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .primary_completion import task_score
from .primary_contract import read_json, require_same
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_io import evaluation_plan


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
        run = contract.complete_run(directory, run_id)
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
        "scope": "dense_primary_v3_complete_grid",
        "artifact_grid_verified": True,
        "scientific_completion": False,
        "protocol_sha256": contract.sha256,
        "runs": runs,
        "evaluations": evaluations,
        "score_rows": score_rows,
        "boundary": "Structural/deep-state/task-provenance acceptance only. Validation selection, scientific outcomes, mechanism analyses and paper release remain separate required consumers.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "evaluation", "matrix"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args(argv)
    contract = PrimaryV3Contract.load(args.protocol, args.repository, args.training_root)
    if args.mode == "matrix":
        if args.experiment_root is None or args.results_root is None:
            parser.error("matrix requires --experiment-root and --results-root")
        result = inspect_matrix(contract, args.experiment_root, args.results_root)
    elif args.run_id is None or args.run_root is None:
        parser.error("run/evaluation require --run-id and --run-root")
    elif args.mode == "run":
        result = contract.complete_run(args.run_root, args.run_id)
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
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
