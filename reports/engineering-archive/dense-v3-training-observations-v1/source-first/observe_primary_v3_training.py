"""Complete native primary training telemetry, without model/GPU execution.

This descriptive artifact uses separately accepted original run metadata. It is
not the full primary-outcome consumer, a source release or a scientific verdict.
The fixed historical reader/admission flags remain unchanged in the input copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
from pathlib import Path

SCOPE = "dense-primary-v3-descriptive-training-observations-v1"
INPUT_SHA = "be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067"
PROTOCOL_SHA = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
STEPS = (782, 1563, 2345, 3126, 3907)
RATES = {
    "adamw": ("1e-6", "3e-6", "1e-5", "3e-5"),
    "muon": ("1e-4", "3e-4", "1e-3", "3e-3"),
    "normuon": ("1e-4", "3e-4", "1e-3", "3e-3"),
}
METADATA = {
    "accepted_timing.json",
    "checkpoint_schedule.json",
    "completed.json",
    "dense_run_contract.json",
    "run_config.json",
    "trainer_state_final.json",
}
LOG_STEPS = [1, *range(10, 3901, 10)]
BOUNDARY = (
    "Complete descriptive native training telemetry, not retrieval inference or a mechanism. "
    "Learning-rate points are not independent training seeds. Timing includes training and "
    "checkpoint-save work, not isolated optimizer kernels. Gradient norms are sampled logs. "
    "No step-3907 logged loss or per-step timestamp is imputed. Scientific release remains pending."
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def ordinary(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError("Use explicit absolute non-symlink paths")
    return path


def identity(path):
    path = ordinary(path)
    if not path.is_file():
        raise ValueError(f"Missing ordinary input: {path.name}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Input changed while being read")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read(path, expected=None):
    if expected is not None and identity(path) != {k: expected[k] for k in ("bytes", "sha256")}:
        raise ValueError(f"Original input binding differs: {Path(path).name}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    canonical(value)
    return value


def same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Native training identity or observation differs")


def finite(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("Expected finite numeric training telemetry")
    return value


def load_parent(path):
    bound = identity(path)
    if bound["sha256"] != INPUT_SHA:
        raise ValueError("Require the exact existing all-twelve genuine admission")
    parent = read(path, bound)
    same(parent["primary_protocol_sha256"], PROTOCOL_SHA)
    if (
        parent["scientific_completion"] is not False
        or parent["committed_source_release"] is not False
    ):
        raise ValueError("Admission scope changed")
    expected_ids = {f"verified-v3-{op}-{rate}" for op, rates in RATES.items() for rate in rates}
    rows = parent["actual_completion"]["rows"]
    admitted = parent["admitted"]["complete_runs"]
    if (
        len(rows) != 12
        or {r["run_id"] for r in rows} != expected_ids
        or set(admitted) != expected_ids
    ):
        raise ValueError("Require the complete twelve-run population")
    for row in rows:
        proof, expected = row["proof"], row["expected"]
        same(proof, admitted[row["run_id"]])
        same(proof["run_identity_sha256"], digest(expected))
        same(proof["steps"], list(STEPS))
        if (
            proof["whole_run_artifacts_verified"] is not True
            or proof["scientific_completion"] is not False
            or proof["protocol_sha256"] != PROTOCOL_SHA
            or set(proof["metadata"]) != METADATA
        ):
            raise ValueError("Incomplete genuine training admission")
    return parent


def declared_files(parent):
    result = {}
    for run_id, run in sorted(parent["admitted"]["complete_runs"].items()):
        for name, binding in run["metadata"].items():
            result[f"{run_id}/{name}"] = dict(binding)
        if len(run["checkpoints"]) != 5:
            raise ValueError("Missing saved stages")
        for step, checkpoint in zip(STEPS, run["checkpoints"], strict=True):
            same(checkpoint["step"], step)
            matches = [r for r in checkpoint["files"] if r["path"] == "trainer_state.json"]
            if len(matches) != 1:
                raise ValueError("Missing or duplicate native Trainer state")
            result[f"{run_id}/checkpoint-{step}/trainer_state.json"] = {
                k: matches[0][k] for k in ("bytes", "sha256")
            }
    if len(result) != 132:
        raise ValueError("Incomplete native JSON population")
    return result


def inspect_history(state, expected_rate):
    if state["global_step"] != 3907 or state["max_steps"] != 3907 or state["epoch"] != 1.0:
        raise ValueError("Incomplete Trainer horizon")
    history = state["log_history"]
    logs = [r for r in history if "loss" in r]
    if len(history) != 392 or len(logs) != 391 or [r["step"] for r in logs] != LOG_STEPS:
        raise ValueError("Missing, duplicated or reordered native training observations")
    for row in logs:
        if set(row) != {"epoch", "grad_norm", "learning_rate", "loss", "step"}:
            raise ValueError("Unexpected training observation fields")
        step = row["step"]
        if type(step) is not int:
            raise ValueError("Training step must be an integer")
        for name in ("epoch", "grad_norm", "learning_rate", "loss"):
            finite(row[name])
        if row["loss"] < 0 or row["grad_norm"] < 0:
            raise ValueError("Negative loss or gradient norm")
        expected_lr = expected_rate * (
            (step - 1) / 391 if step - 1 < 391 else (3907 - (step - 1)) / (3907 - 391)
        )
        if not math.isclose(row["learning_rate"], expected_lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Logged learning rate differs from the actual pre-update schedule")
        if not math.isclose(row["epoch"], step * 128 / 500000, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Logged progress differs from the fixed query-group schedule")
    if history[-1]["step"] != 3907 or "train_loss" not in history[-1] or "loss" in history[-1]:
        raise ValueError("Do not invent a terminal logged minibatch loss")
    return logs


def systems_record(run_id, expected, proof):
    metrics = proof["system_metrics"]
    elapsed = finite(proof["accepted_timing"]["total_wall_time_seconds_max_rank"], positive=True)
    opt = expected["recipe"]["optimizer"]
    return {
        "run_id": run_id,
        "optimizer": opt["name"],
        "learning_rate": opt["lr"],
        "wall_time_seconds": elapsed,
        "samples_per_second": expected["data"]["rows"] / elapsed,
        "steps_per_second": 3907 / elapsed,
        "trainer_reported_samples_per_second": metrics["trainer"]["train_samples_per_second"],
        "trainer_reported_steps_per_second": metrics["trainer"]["train_steps_per_second"],
        "peak_allocated_gib": metrics["peak_allocated_bytes_max_rank"] / 2**30,
        "peak_reserved_gib": metrics["peak_reserved_bytes_max_rank"] / 2**30,
        "maximum_checkpoint_gib": max(metrics["checkpoint_bytes"].values()) / 2**30,
        "maximum_optimizer_state_gib": max(metrics["optimizer_state_bytes"].values()) / 2**30,
        "gpu_name": metrics["gpu_name"],
        "world_size": metrics["world_size"],
    }


def summarize(parent, payloads):
    traces, stages, timings, systems, summaries = [], [], [], [], []
    for row in parent["actual_completion"]["rows"]:
        run_id, proof, expected = row["run_id"], row["proof"], row["expected"]
        raw = {name: payloads[f"{run_id}/{name}"] for name in METADATA}
        same(raw["dense_run_contract.json"], expected)
        completed, state = raw["completed.json"], raw["trainer_state_final.json"]
        same(completed["run_identity_sha256"], digest(expected))
        same(completed["system_metrics"], proof["system_metrics"])
        same(completed["accepted_timing"], proof["accepted_timing"])
        same(
            raw["checkpoint_schedule.json"],
            {"max_steps": 3907, "fractions": [0.2, 0.4, 0.6, 0.8, 1.0], "steps": list(STEPS)},
        )
        opt = expected["recipe"]["optimizer"]
        label = {"run_id": run_id, "optimizer": opt["name"], "configured_learning_rate": opt["lr"]}
        logs = inspect_history(state, opt["lr"])
        for record in logs:
            traces.append(
                {**label, **record, "normalized_optimizer_progress": record["step"] / 3907}
            )
        cumulative, previous = 0.0, 0
        ledger = raw["accepted_timing.json"]
        if len(ledger["segments"]) != 5:
            raise ValueError("Incomplete original timing segments")
        for stage, step in enumerate(STEPS, 1):
            saved = payloads[f"{run_id}/checkpoint-{step}/trainer_state.json"]
            same(saved["global_step"], step)
            same(saved["max_steps"], 3907)
            same(
                [r for r in saved["log_history"] if "loss" in r],
                [r for r in logs if r["step"] <= step],
            )
            segment = ledger["segments"][stage - 1]
            same((segment["start_step_exclusive"], segment["end_step_inclusive"]), (previous, step))
            elapsed = finite(segment["wall_time_seconds_max_rank"], positive=True)
            cumulative += elapsed
            groups = min(step * 128, 500000) - min(previous * 128, 500000)
            timings.append(
                {
                    **label,
                    "stage": stage,
                    **segment,
                    "optimizer_steps": step - previous,
                    "query_groups": groups,
                    "samples_per_second": groups / elapsed,
                    "cumulative_accepted_seconds": cumulative,
                }
            )
            window = [r for r in logs if r["step"] <= step][-10:]
            if len(window) != 10:
                raise ValueError("Incomplete trailing logging window")
            stages.append(
                {
                    **label,
                    "stage": stage,
                    "fraction": stage / 5,
                    "checkpoint_step": step,
                    "observed_step": window[-1]["step"],
                    "window_start_step": window[0]["step"],
                    "window_observations": 10,
                    "mean_loss": statistics.mean(r["loss"] for r in window),
                    "loss_standard_deviation": statistics.stdev(r["loss"] for r in window),
                    "median_grad_norm": statistics.median(r["grad_norm"] for r in window),
                    "end_learning_rate": window[-1]["learning_rate"],
                    "end_epoch": window[-1]["epoch"],
                    "checkpoint_accepted_seconds": cumulative,
                }
            )
            previous = step
        same(cumulative, ledger["total_wall_time_seconds_max_rank"])
        same(cumulative, proof["accepted_timing"]["total_wall_time_seconds_max_rank"])
        system = systems_record(run_id, expected, proof)
        systems.append(system)
        same(
            state["log_history"][-1]["train_loss"],
            completed["system_metrics"]["trainer"]["train_loss"],
        )
        summaries.append(
            {
                **label,
                "native_full_run_train_loss": state["log_history"][-1]["train_loss"],
                "initial_logged_loss": logs[0]["loss"],
                "final_observed_step": logs[-1]["step"],
                "last_logged_loss": logs[-1]["loss"],
                "final_trailing_mean_loss": stages[-1]["mean_loss"],
                "final_trailing_median_grad_norm": stages[-1]["median_grad_norm"],
                "logged_grad_norm_gt_clip_count": sum(r["grad_norm"] > 1 for r in logs),
                "logged_grad_norm_observations": len(logs),
                "accepted_wall_hours": cumulative / 3600,
                "os_exit_code_observed": row["binding"]["exit_code_observed"],
                "os_exit_code": row["binding"]["exit_code"],
            }
        )
    tables = {
        "logged_training": traces,
        "stage_training": stages,
        "timing_segments": timings,
        "systems": systems,
        "run_summary": summaries,
    }
    for name, count in {
        "logged_training": 4692,
        "stage_training": 60,
        "timing_segments": 60,
        "systems": 12,
        "run_summary": 12,
    }.items():
        if len(tables[name]) != count:
            raise ValueError("Incomplete descriptive training population")
    return tables


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def figure(tables, output):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {"svg.hashsalt": SCOPE, "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10}
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), sharex=True, sharey="row")
    colors = ("#38588C", "#2A9D8F", "#DB8A29", "#B8475E")
    for column, (op, rates) in enumerate(RATES.items()):
        for color, rate in zip(colors, rates, strict=True):
            run_id = f"verified-v3-{op}-{rate}"
            rows = [r for r in tables["logged_training"] if r["run_id"] == run_id]
            x = [r["normalized_optimizer_progress"] for r in rows]
            for index, metric in enumerate(("loss", "grad_norm")):
                axes[index, column].plot(
                    x, [r[metric] for r in rows], color=color, linewidth=1, label=rate
                )
        axes[0, column].set_title({"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}[op])
        axes[0, column].legend(title="Configured LR", frameon=False, fontsize=9)
        axes[1, column].set_yscale("symlog", linthresh=1e-3)
        axes[1, column].axhline(1.0, color="#666666", linestyle="--", linewidth=0.8)
        axes[1, column].set_xlabel("Normalized optimizer-step progress")
        for axis in axes[:, column]:
            axis.set_xlim(0, 1)
            axis.grid(alpha=0.18)
    axes[0, 0].set_ylabel("Logged training loss")
    axes[1, 0].set_ylabel("Logged gradient norm (symlog)")
    fig.suptitle("DenseOn: all twelve complete v3 training trajectories", fontsize=14)
    fig.text(
        0.5,
        0.013,
        "Raw logged observations; no smoothing. Gradient norms are sampled, not an every-update clipping count.\nLogging ends at step 3900; all runs finish at step 3907. Training loss is not retrieval quality.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    paths = []
    for extension in ("svg", "pdf", "png"):
        path = output / f"training-trajectories.{extension}"
        metadata = {"Creator": SCOPE}
        if extension == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        elif extension == "svg":
            metadata["Date"] = None
        fig.savefig(path, metadata=metadata, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths, matplotlib.__version__


def produce(parent_path, native_root, output, *, copy_inputs):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This CPU-only observation requires hidden CUDA")
    parent_path, native_root, output = map(ordinary, (parent_path, native_root, output))
    if output.exists() or native_root.is_relative_to(output) or parent_path.is_relative_to(output):
        raise ValueError("Use a fresh output namespace separate from original inputs")
    parent = load_parent(parent_path)
    files = declared_files(parent)
    payloads = {name: read(native_root / name, binding) for name, binding in files.items()}
    tables = summarize(parent, payloads)
    output.mkdir(parents=True, exist_ok=False)
    if copy_inputs:
        (output / "inputs").mkdir()
        shutil.copyfile(parent_path, output / "inputs/admission.json")
        for name, binding in files.items():
            path = output / "inputs/native" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(native_root / name, path)
            same(identity(path), binding)
        same(identity(output / "inputs/admission.json"), identity(parent_path))
    outputs = {}
    for name, rows in tables.items():
        path = output / f"{name}.csv"
        columns = sorted(rows[0])
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        outputs[path.name] = {**identity(path), "rows": len(rows)}
    write_json(output / "tables.json", tables)
    outputs["tables.json"] = identity(output / "tables.json")
    figures, version = figure(tables, output)
    for path in figures:
        outputs[path.name] = identity(path)
    for name, binding in files.items():
        same(identity(native_root / name), binding)
    manifest = {
        "scope": SCOPE,
        "complete_descriptive_population": True,
        "primary_protocol_sha256": PROTOCOL_SHA,
        "input_parent": identity(parent_path),
        "native_json_files": files,
        "producer": identity(Path(__file__).resolve()),
        "outputs": outputs,
        "matplotlib_version": version,
        "coverage": {name: len(rows) for name, rows in tables.items()},
        "input_copy_included": copy_inputs,
        "all_original_metadata_rechecked": True,
        "new_model_or_optimizer_tensor_read": False,
        "gpu_execution": False,
        "scientific_completion": False,
        "source_publication": False,
        "boundary": BOUNDARY,
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "scope": SCOPE,
                "output": str(output),
                "manifest": identity(output / "manifest.json"),
                "coverage": manifest["coverage"],
                "scientific_completion": False,
            },
            sort_keys=True,
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-inputs", action="store_true")
    args = parser.parse_args()
    produce(args.parent, args.native_root, args.output, copy_inputs=args.copy_inputs)


if __name__ == "__main__":
    main()
