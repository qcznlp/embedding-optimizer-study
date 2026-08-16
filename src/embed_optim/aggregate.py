"""Aggregate checkpoint-level MTEB results and training histories."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .config import RunConfig, load_matrix
from .decontamination import DECONTAMINATED_TASK_NAMES

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


def _run_for_result(path: Path, configs: list[RunConfig]) -> RunConfig | None:
    matches = [
        config
        for config in configs
        if config.run_id in str(path) and config.model_family in path.parts
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def collect_evaluations(results_root: Path, configs: list[RunConfig]) -> list[dict]:
    rows: list[dict] = []
    for path in results_root.rglob("*Decontaminated.json"):
        config = _run_for_result(path, configs)
        match = CHECKPOINT_PATTERN.search(str(path))
        if config is None or match is None:
            continue
        step = int(match.group(1))
        schedule_path = config.output_dir / "checkpoint_schedule.json"
        steps = sorted(json.loads(schedule_path.read_text())["steps"])
        if step not in steps:
            continue
        payload = json.loads(path.read_text())
        split_rows = [item for values in payload["scores"].values() for item in values]
        scores = [float(item["main_score"]) for item in split_rows]
        rows.append(
            {
                "model_family": config.model_family,
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "aux_learning_rate": config.optimizer.aux_lr,
                "run_id": config.run_id,
                "stage": steps.index(step) + 1,
                "fraction": (steps.index(step) + 1) / 5,
                "checkpoint_step": step,
                "task": payload["task_name"],
                "main_score": sum(scores) / len(scores),
                "subsets": len(scores),
                "result_path": str(path),
            }
        )
    return rows


def collect_training_history(configs: list[RunConfig]) -> list[dict]:
    rows: list[dict] = []
    for config in configs:
        schedule_path = config.output_dir / "checkpoint_schedule.json"
        if not schedule_path.is_file():
            continue
        steps = sorted(json.loads(schedule_path.read_text())["steps"])
        state_path = config.output_dir / "trainer_state_final.json"
        if not state_path.is_file():
            state_path = config.output_dir / f"checkpoint-{steps[-1]}" / "trainer_state.json"
        if not state_path.is_file():
            continue
        for item in json.loads(state_path.read_text()).get("log_history", []):
            if "loss" not in item:
                continue
            rows.append(
                {
                    "model_family": config.model_family,
                    "optimizer": config.optimizer.name,
                    "learning_rate_config": config.optimizer.lr,
                    "run_id": config.run_id,
                    **item,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["model_family"],
            row["optimizer"],
            row["learning_rate"],
            row["run_id"],
            row["stage"],
            row["fraction"],
            row["checkpoint_step"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        output.append(
            {
                "model_family": key[0],
                "optimizer": key[1],
                "learning_rate": key[2],
                "run_id": key[3],
                "stage": key[4],
                "fraction": key[5],
                "checkpoint_step": key[6],
                "mean_ndcg_at_10": sum(row["main_score"] for row in values) / len(values),
                "tasks_completed": len(values),
            }
        )
    return output


def _plot(summary: list[dict], output_dir: Path) -> None:
    if not summary:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(summary)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for family in sorted(frame.model_family.unique()):
        subset = frame[frame.model_family == family]
        fig, axis = plt.subplots(figsize=(8, 5))
        for optimizer, values in subset.groupby("optimizer"):
            grouped = values.groupby("fraction").mean(numeric_only=True)
            std = values.groupby("fraction")["mean_ndcg_at_10"].std().fillna(0)
            axis.plot(grouped.index, grouped.mean_ndcg_at_10, marker="o", label=optimizer)
            axis.fill_between(
                grouped.index,
                grouped.mean_ndcg_at_10 - std,
                grouped.mean_ndcg_at_10 + std,
                alpha=0.15,
            )
        axis.set(xlabel="Training fraction", ylabel="Mean decontaminated BEIR nDCG@10")
        axis.set_title(f"{family.capitalize()} training dynamics (mean ± LR-config std)")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{family}-training-dynamics.png", dpi=180)
        plt.close(fig)


def aggregate(args: argparse.Namespace) -> None:
    configs = load_matrix(args.matrix)
    rows = collect_evaluations(Path(args.results_root), configs)
    summary = _summaries(rows)
    output = Path(args.output_dir)
    _write_csv(output / "evaluation_long.csv", rows)
    _write_csv(output / "checkpoint_summary.csv", summary)
    _write_csv(output / "training_history.csv", collect_training_history(configs))
    coverage = {
        "observed_results": len(rows),
        "expected_results": len(configs) * 5 * len(DECONTAMINATED_TASK_NAMES),
        "observed_checkpoint_summaries": len(summary),
        "expected_checkpoint_summaries": len(configs) * 5,
    }
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    _plot(summary, output)
    print(json.dumps(coverage, indent=2))
    if args.strict and coverage["observed_results"] != coverage["expected_results"]:
        raise RuntimeError("Evaluation matrix is incomplete; see coverage.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--results-root", default="results/decontaminated-beir")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    aggregate(parse_args(argv))


if __name__ == "__main__":
    main()
