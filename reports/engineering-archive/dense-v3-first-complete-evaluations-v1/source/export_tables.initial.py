"""Independently reconstruct the first six complete checkpoint tables from raw MTEB JSON."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_snapshot(item):
    raw = item["text"].encode()
    require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"],
            "Embedded raw snapshot changed")
    require(Path(item["path"]).read_bytes() == raw, "Original raw input changed")
    return json.loads(item["text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    bundle = json.loads(raw)
    require(bundle["scope"] == "descriptive_complete_fourteen_task_checkpoint_readback" and
            bundle["scientific_completion"] is False and
            bundle["full_primary_grid_complete"] is False and
            len(bundle["checkpoints"]) == 6, "Wrong bounded input")
    summaries, tasks, keys = [], [], set()
    for checkpoint in bundle["checkpoints"]:
        run, step = checkpoint["run_id"], checkpoint["step"]
        match = re.fullmatch(r"verified-v3-(adamw|muon|normuon)-([13]e-[3456])", run)
        require(match is not None and step == 3907 and (run, step) not in keys, "Wrong state set")
        keys.add((run, step))
        optimizer, rate = match.groups()
        native = checkpoint["native_complete_reread"]
        require(verify_snapshot(checkpoint["original_complete_receipt"]) == native,
                "Native complete receipt changed")
        verify_snapshot(checkpoint["original_operational_receipt"])
        expected = native["plan"]["tasks"]
        require(len(expected) == len(set(expected)) == 14, "Incomplete task denominator")
        actual = {}
        for item in checkpoint["raw_score_and_metadata_snapshots"]:
            value = verify_snapshot(item)
            if not Path(item["path"]).name.endswith("Decontaminated.json"):
                continue
            task = value["task_name"].removesuffix("Decontaminated")
            split = "dev" if task == "MSMARCO" else "test"
            require(task in expected and task not in actual and
                    set(value["scores"]) == {split} and len(value["scores"][split]) == 1,
                    "Wrong raw task coverage")
            row = value["scores"][split][0]
            score = row["ndcg_at_10"]
            require(row["hf_subset"] == "default" and math.isfinite(score) and
                    0 <= score <= 1 and abs(score - row["main_score"]) <= 1e-12,
                    "Wrong raw scored metric")
            actual[task] = score
        require(set(actual) == set(expected) and len(actual) == 14 and
                actual == {row["task"]: row["ndcg_at_10"] for row in native["tasks"]},
                "Raw scores differ from native task reader")
        for job in checkpoint["original_workers"]:
            started, exited = verify_snapshot(job["started"]), verify_snapshot(job["exited"])
            require(started["pid"] == exited["pid"] and exited["exit_code"] == 0 and
                    started["job"] == exited["job"], "Original worker proof differs")
        exact = sum((Fraction.from_float(actual[task]) for task in expected), Fraction()) / 14
        require(checkpoint["exact_binary64_input_mean"] ==
                {"numerator": exact.numerator, "denominator": exact.denominator} and
                checkpoint["macro_ndcg_at_10"] == float(exact) and
                checkpoint["macro_score_0_to_100"] == float(100 * exact),
                "Independent raw-score mean differs")
        summaries.append({"run_id": run, "optimizer": optimizer, "learning_rate": rate,
                          "step": step, "tasks": 14, "macro_ndcg_at_10": float(exact),
                          "macro_score_0_to_100": float(100 * exact)})
        tasks.extend({"run_id": run, "optimizer": optimizer, "learning_rate": rate,
                      "step": step, "task": task, "ndcg_at_10": actual[task]} for task in expected)
    order = {"adamw": 0, "muon": 1, "normuon": 2}
    summaries.sort(key=lambda row: (order[row["optimizer"]], float(row["learning_rate"])))
    tasks.sort(key=lambda row: (order[row["optimizer"]], float(row["learning_rate"]), row["task"]))
    require(len(tasks) == 84, "Incomplete six-checkpoint raw table")
    args.output.mkdir(exist_ok=False)
    for name, rows in (("summary.csv", summaries), ("task-scores.csv", tasks)):
        with (args.output / name).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = [
        "# First six complete v3 checkpoint evaluations", "",
        f"Native full-task readback: {bundle['observed_at_utc']}.",
        "Each row is a final checkpoint (step 3907), averaged equally over all fourteen pinned decontaminated BEIR tasks.",
        "Scores below are nDCG@10 multiplied by 100. Rows follow optimizer/rate order, not performance order.", "",
        "| Optimizer | Learning rate | Complete tasks | Macro score |",
        "| --- | ---: | ---: | ---: |",
    ]
    text.extend(f"| {row['optimizer']} | {row['learning_rate']} | 14 / 14 | {row['macro_score_0_to_100']:.4f} |"
                for row in summaries)
    text += ["", "This is the complete first pool-B final-checkpoint cohort, not the complete primary experiment.",
             "The other six final checkpoints and all forty-eight intermediate checkpoints remain required.",
             "No available-task average, optimizer-wide ranking, validation-based selection, significance test, or mechanism conclusion is produced.",
             "The 84 original task values are in [task-scores.csv](task-scores.csv).",
             "Original source-release and scientific-completion flags remain false.", ""]
    with (args.output / "summary.md").open("x") as handle:
        handle.write("\n".join(text))
    outputs = {}
    for path in sorted(args.output.iterdir()):
        payload = path.read_bytes()
        outputs[path.name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    source = Path(__file__).read_bytes()
    manifest = {"input": {"path": str(args.input), "bytes": len(raw),
                           "sha256": hashlib.sha256(raw).hexdigest()},
                "source": {"path": __file__, "bytes": len(source),
                           "sha256": hashlib.sha256(source).hexdigest()},
                "outputs": outputs, "complete_checkpoints": 6, "task_rows": 84,
                "independent_raw_score_reconstruction_passed": True,
                "scientific_completion": False, "optimizer_ranking_or_selection": False}
    with (args.output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

