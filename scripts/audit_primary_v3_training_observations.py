"""Independent native-metadata replay of the complete descriptive training tables.

This never imports the observation producer, Torch, or a full primary consumer.
Only three hash-authenticated original pure helper groups are executed. That
limited reference calculation does not waive their surrounding release guards.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

PARENT_SHA = "be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067"
MANIFEST_SHA = "98e3a7cae491d4bb06ae949a909f9466690e8f5d40edb600a3e5ce1e2ba17205"
PRODUCER_SHA = "5424027d6363b1edbccb5c7916f0211bd9f67d355d39aaf1c4a908fe3e8127de"
REFERENCES = {
    "primary_contract.py": (
        "e49f55d5d01408de001b8e1d56244fbcaf189de45c0209caffa0b4e34f828e19",
        ("canonical", "digest", "require_same"),
    ),
    "primary_completion.py": (
        "a1aa61058654bd46afb41710ed606992355dab95d7c8b171a7485a8ac24d0cd5",
        ("integer", "number", "timing_summary"),
    ),
    "primary_v3_outcomes.py": (
        "b97aaf812cd76b4ab4e852b7a13b7d5e66a680c638bf42c90b89108d66ace837",
        ("system_rows",),
    ),
}
STEPS = (782, 1563, 2345, 3126, 3907)
COUNTS = {
    "logged_training": 4692,
    "stage_training": 60,
    "timing_segments": 60,
    "systems": 12,
    "run_summary": 12,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    require(path.is_file() and not path.is_symlink(), "Missing ordinary audit input")
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def authenticate(path, expected):
    actual = binding(path)
    require(actual == {k: expected[k] for k in ("bytes", "sha256")}, "Input hash differs")
    return actual


def pure_references(root):
    namespace = {
        "json": json,
        "hashlib": hashlib,
        "math": math,
        "datetime": datetime,
        "Counter": Counter,
    }
    sources = {}
    for name, (sha, functions) in REFERENCES.items():
        path = root / name
        sources[name] = binding(path)
        require(sources[name]["sha256"] == sha, "Original reference source changed")
        nodes = ast.parse(path.read_text()).body
        selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in functions]
        require(tuple(n.name for n in selected) == functions, "Reference extraction differs")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, sources


def scalar_window(rows):
    losses = [Fraction(r["loss"]) for r in rows]
    average = sum(losses, Fraction()) / len(losses)
    variance = sum(((v - average) ** 2 for v in losses), Fraction()) / (len(losses) - 1)
    grads = sorted(Fraction(r["grad_norm"]) for r in rows)
    require(len(rows) == 10, "Require the fixed ten-observation window")
    return float(average), math.sqrt(float(variance)), float((grads[4] + grads[5]) / 2)


def compare_table(actual, expected, *, tolerant_fields=()):
    require(len(actual) == len(expected), "Numerical table coverage differs")
    comparisons = 0
    maximum_tolerated_error = 0.0
    for observed, reference in zip(actual, expected, strict=True):
        require(set(observed) == set(reference), "Numerical table fields differ")
        for key, value in reference.items():
            if key in tolerant_fields:
                error = abs(observed[key] - value)
                maximum_tolerated_error = max(maximum_tolerated_error, error)
                require(math.isclose(observed[key], value, rel_tol=2e-15, abs_tol=1e-15), key)
            else:
                require(type(observed[key]) is type(value) and observed[key] == value, key)
            comparisons += 1
    return {"scalar_fields": comparisons, "maximum_rounding_difference": maximum_tolerated_error}


def audit(artifact, parent_path, native_root, references):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Require CPU-only audit")
    manifest_path = artifact / "manifest.json"
    require(binding(manifest_path)["sha256"] == MANIFEST_SHA, "Require original first artifact")
    require(binding(parent_path)["sha256"] == PARENT_SHA, "Require original genuine admission")
    manifest, parent = read(manifest_path), read(parent_path)
    require(manifest["producer"]["sha256"] == PRODUCER_SHA, "Observation producer differs")
    require(manifest["coverage"] == COUNTS, "Manifest population differs")
    require(manifest["scientific_completion"] is False, "Descriptive scope changed")
    refs, source_bindings = pure_references(references)
    rows = parent["actual_completion"]["rows"]
    expectations = {row["run_id"]: row["expected"] for row in rows}
    require(len(expectations) == len(rows) == 12, "Require all twelve genuine runs")
    facade = SimpleNamespace(
        inputs={"runs": [{"run_id": key} for key in expectations]},
        payload={"checkpoint_steps": list(STEPS)},
        expected_identity=expectations.__getitem__,
    )
    systems = refs["system_rows"](facade, [r["proof"] for r in rows])
    native = {}
    for relative, declared in manifest["native_json_files"].items():
        authenticate(native_root / relative, declared)
        authenticate(artifact / "inputs/native" / relative, declared)
        native[relative] = read(native_root / relative)
    require(len(native) == 132, "Require all native JSON inputs")
    authenticate(artifact / "inputs/admission.json", binding(parent_path))
    output_bindings = {}
    for name, declared in manifest["outputs"].items():
        output_bindings[name] = authenticate(artifact / name, declared)
    tables = read(artifact / "tables.json")
    require(set(tables) == set(COUNTS), "Changed numerical table population")
    checks = {
        "systems": compare_table(sorted(tables["systems"], key=lambda r: r["run_id"]), systems)
    }
    traces, stages, timings, summaries = [], [], [], []
    for row in rows:
        run_id, expected, proof = row["run_id"], row["expected"], row["proof"]
        state = native[f"{run_id}/trainer_state_final.json"]
        logs = [r for r in state["log_history"] if "loss" in r]
        require([r["step"] for r in logs] == [1, *range(10, 3901, 10)], "Raw log steps differ")
        require(state["global_step"] == state["max_steps"] == 3907, "Incomplete horizon")
        ledger = native[f"{run_id}/accepted_timing.json"]
        timing = refs["timing_summary"](ledger, list(STEPS))
        require(timing == proof["accepted_timing"], "Original timing summary differs")
        opt = expected["recipe"]["optimizer"]
        label = {"run_id": run_id, "optimizer": opt["name"], "configured_learning_rate": opt["lr"]}
        for record in logs:
            traces.append(
                {**label, **record, "normalized_optimizer_progress": record["step"] / 3907}
            )
        cumulative, previous = 0.0, 0
        for stage, step in enumerate(STEPS, 1):
            saved = native[f"{run_id}/checkpoint-{step}/trainer_state.json"]
            prefix = [r for r in logs if r["step"] <= step]
            require(saved["global_step"] == step, "Saved step differs")
            require(
                [r for r in saved["log_history"] if "loss" in r] == prefix,
                "Saved log prefix differs",
            )
            window = prefix[-10:]
            mean, deviation, median = scalar_window(window)
            segment = ledger["segments"][stage - 1]
            cumulative += segment["wall_time_seconds_max_rank"]
            groups = min(step * 128, 500000) - min(previous * 128, 500000)
            timings.append(
                {
                    **label,
                    "stage": stage,
                    **segment,
                    "optimizer_steps": step - previous,
                    "query_groups": groups,
                    "samples_per_second": groups / segment["wall_time_seconds_max_rank"],
                    "cumulative_accepted_seconds": cumulative,
                }
            )
            stages.append(
                {
                    **label,
                    "stage": stage,
                    "fraction": stage / 5,
                    "checkpoint_step": step,
                    "observed_step": window[-1]["step"],
                    "window_start_step": window[0]["step"],
                    "window_observations": 10,
                    "mean_loss": mean,
                    "loss_standard_deviation": deviation,
                    "median_grad_norm": median,
                    "end_learning_rate": window[-1]["learning_rate"],
                    "end_epoch": window[-1]["epoch"],
                    "checkpoint_accepted_seconds": cumulative,
                }
            )
            previous = step
        terminal = state["log_history"][-1]
        require(terminal["step"] == 3907 and "loss" not in terminal, "Imputed terminal loss")
        summaries.append(
            {
                **label,
                "native_full_run_train_loss": terminal["train_loss"],
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
    for name, expected in (
        ("logged_training", traces),
        ("stage_training", stages),
        ("timing_segments", timings),
        ("run_summary", summaries),
    ):
        require(len(expected) == COUNTS[name], "Incomplete independent reconstruction")
        checks[name] = compare_table(
            tables[name], expected, tolerant_fields=("loss_standard_deviation",)
        )
    for name, table in tables.items():
        with (artifact / f"{name}.csv").open(newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        text_rows = [{k: "" if v is None else str(v) for k, v in r.items()} for r in table]
        require(csv_rows == text_rows, "CSV is not a complete transcription of JSON values")
    for relative, declared in manifest["native_json_files"].items():
        authenticate(native_root / relative, declared)
    require(binding(parent_path)["sha256"] == PARENT_SHA, "Parent changed during audit")
    return {
        "scope": "independent-complete-primary-training-telemetry-replay-v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "parent": binding(parent_path),
        "manifest": binding(manifest_path),
        "auditor": binding(Path(__file__).resolve()),
        "reference_sources": source_bindings,
        "output_files": output_bindings,
        "coverage": COUNTS,
        "checks": checks,
        "native_files_verified": 132,
        "saved_log_prefixes_verified": 60,
        "original_timing_summary_runs": 12,
        "original_system_rows_runs": 12,
        "mean_variance_median_reference": "Exact binary-input Fraction arithmetic; square root rounded to float",
        "independent_of_producer_import": True,
        "scientific_completion": False,
        "gpu_execution": False,
        "new_model_or_optimizer_tensor_read": False,
        "boundary": "Actual metadata/tables, not retrieval inference, full source admission or cross-host training replay.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "parent", "native-root", "references", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for path in vars(args).values():
        require(
            path.is_absolute()
            and ".." not in path.parts
            and not any(p.is_symlink() for p in (path, *path.parents)),
            "Require absolute ordinary paths",
        )
    require(not args.output.exists() and args.output.parent.is_dir(), "Require fresh receipt path")
    result = audit(args.artifact, args.parent, args.native_root, args.references)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "receipt": binding(args.output), "coverage": COUNTS}))


if __name__ == "__main__":
    main()
