from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .aggregate import (
    _deep_checkpoint_problems,
    _linear_schedule_multiplier,
    _scheduler_contract_problem,
    _training_arguments_problem,
    audit_dataset_artifacts,
    audit_training_artifacts,
    collect_evaluations,
    collect_system_metrics,
)
from .config import RunConfig, load_matrix
from .decontamination import DECONTAMINATED_TASK_NAMES
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .scope import ALL_FAMILIES, normalize_families, resolve_scope

FAMILIES = ("dense", "late")
LEARNING_RATES = (1e-6, 3e-6, 1e-5, 3e-5)
EXPECTED_HYBRID_RUNS = 8
EXPECTED_HYBRID_CHECKPOINTS = 40
EXPECTED_HYBRID_EVALUATIONS = 112


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty strict table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_protocol(path: Path, *, require_external: bool = True) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload.get("selection") or {}
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "prospective_completion_lock"
        or selection.get("families") != list(FAMILIES)
        or selection.get("optimizer") != "hybrid_adamw"
        or selection.get("hidden_learning_rates") != list(LEARNING_RATES)
        or selection.get("aux_learning_rate") != 3e-6
        or selection.get("expected_runs") != EXPECTED_HYBRID_RUNS
        or selection.get("checkpoint_fractions") != [0.2, 0.4, 0.6, 0.8, 1.0]
        or selection.get("formal_beir_stages") != [5]
        or selection.get("formal_beir_tasks") != len(DECONTAMINATED_TASK_NAMES)
        or selection.get("expected_beir_units") != EXPECTED_HYBRID_EVALUATIONS
    ):
        raise ValueError("Hybrid AdamW protocol differs from its frozen selection contract")
    repository = path.parent.parent
    sources = payload.get("sources") or {}
    if set(sources) != {
        "discovery_matrix",
        "control_matrix",
        "training_data_manifest",
        "visible_weight_summary",
    }:
        raise ValueError("Hybrid AdamW protocol has an incomplete source ledger")
    for label, source in sources.items():
        source_path = (repository / source.get("path", "")).resolve()
        if not require_external and label == "training_data_manifest" and not source_path.exists():
            continue
        if not source_path.is_file() or _sha256(source_path) != source.get("sha256"):
            raise ValueError(f"Hybrid AdamW frozen source differs: {label} ({source_path})")
    return payload


def _hybrid_optimizer_contract_problem(
    optimizer: object,
    config: RunConfig,
    expected_step: int,
    final_step: int,
) -> str | None:
    if config.optimizer.name != "hybrid_adamw":
        return f"expected hybrid_adamw config, got {config.optimizer.name!r}"
    if not isinstance(optimizer, dict):
        return "optimizer state has an invalid structure"
    state = optimizer.get("state")
    groups = optimizer.get("param_groups")
    if not isinstance(state, dict) or not isinstance(groups, list):
        return "optimizer state has an invalid structure"
    expected = [
        (
            config.optimizer.lr,
            config.optimizer.weight_decay,
            config.optimizer.beta1,
            config.optimizer.beta2,
            config.optimizer.eps,
        ),
        (
            config.optimizer.aux_lr,
            config.optimizer.weight_decay,
            config.optimizer.aux_beta1,
            config.optimizer.aux_beta2,
            config.optimizer.aux_eps,
        ),
        (
            config.optimizer.aux_lr,
            0.0,
            config.optimizer.aux_beta1,
            config.optimizer.aux_beta2,
            config.optimizer.aux_eps,
        ),
    ]
    if len(groups) != len(expected):
        return f"optimizer has {len(groups)} parameter groups, expected {len(expected)}"
    multiplier = _linear_schedule_multiplier(expected_step, final_step, config.warmup_ratio)
    grouped_ids: set[object] = set()
    for index, (group, values) in enumerate(zip(groups, expected)):
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, list) or not parameter_ids:
            return f"optimizer parameter group {index} is empty or invalid"
        if group.get("algorithm") != "adamw":
            return f"optimizer parameter group {index} algorithm is not AdamW"
        expected_lr, weight_decay, beta1, beta2, eps = values
        numeric = {
            "lr": expected_lr * multiplier,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        for field, expected_value in numeric.items():
            try:
                matches = math.isclose(
                    float(group.get(field)), expected_value, rel_tol=1e-12, abs_tol=1e-15
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                return (
                    f"optimizer parameter group {index} {field} is {group.get(field)!r}, "
                    f"expected {expected_value!r}"
                )
        if tuple(group.get("betas") or ()) != (beta1, beta2):
            return f"optimizer parameter group {index} betas differ from the frozen recipe"
        for parameter_id in parameter_ids:
            if parameter_id in grouped_ids:
                return f"optimizer parameter {parameter_id!r} appears in multiple groups"
            grouped_ids.add(parameter_id)
            parameter_state = state.get(parameter_id)
            if not isinstance(parameter_state, dict):
                return f"optimizer state is missing parameter {parameter_id!r}"
            if set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}:
                return f"optimizer state fields differ for parameter {parameter_id!r}"
            try:
                state_step = float(parameter_state["step"])
            except (TypeError, ValueError):
                return f"optimizer state step is invalid for parameter {parameter_id!r}"
            if state_step != expected_step:
                return (
                    f"optimizer state step for parameter {parameter_id!r} is {state_step}, "
                    f"expected {expected_step}"
                )
            first_shape = getattr(parameter_state["exp_avg"], "shape", None)
            second_shape = getattr(parameter_state["exp_avg_sq"], "shape", None)
            if first_shape is None or first_shape != second_shape:
                return f"optimizer moments have incompatible shapes for parameter {parameter_id!r}"
    if set(state) != grouped_ids:
        return "optimizer state does not cover every grouped parameter"
    return None


def audit_hybrid_training(
    configs: list[RunConfig], families: tuple[str, ...] = ALL_FAMILIES
) -> dict[str, Any]:
    families = normalize_families(families)
    expected_runs = len(families) * len(LEARNING_RATES)
    expected_checkpoints = expected_runs * 5
    if (
        len(configs) != expected_runs
        or {config.model_family for config in configs} != set(families)
        or {config.optimizer.name for config in configs} != {"hybrid_adamw"}
        or any(
            {config.optimizer.lr for config in configs if config.model_family == family}
            != set(LEARNING_RATES)
            for family in families
        )
    ):
        raise ValueError(f"Hybrid training matrix is not the active {len(families)}×4 control")
    dataset_audit = audit_dataset_artifacts(configs)
    if not dataset_audit.get("complete"):
        return {
            "complete": False,
            "verified_runs": 0,
            "verified_checkpoints": 0,
            "errors": list(dataset_audit.get("errors") or ["dataset audit failed"]),
        }
    shallow = audit_training_artifacts(
        configs,
        deep=False,
        expected_dataset_fingerprint=dataset_audit.get("training_view_fingerprint"),
    )
    errors = list(shallow.get("errors") or [])
    verified_checkpoints = 0
    for config in configs:
        schedule_path = config.output_dir / "checkpoint_schedule.json"
        try:
            steps = [int(step) for step in json.loads(schedule_path.read_text())["steps"]]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if len(steps) != 5 or steps != sorted(set(steps)):
            continue
        final_step = steps[-1]
        for step in steps:
            checkpoint = config.output_dir / f"checkpoint-{step}"
            problems = _deep_checkpoint_problems(checkpoint, step, world_size=4)
            optimizer = None
            try:
                optimizer = torch.load(
                    checkpoint / "optimizer.pt", map_location="cpu", weights_only=True, mmap=True
                )
                if problem := _hybrid_optimizer_contract_problem(
                    optimizer, config, step, final_step
                ):
                    problems.append(problem)
            except Exception as error:  # noqa: BLE001
                problems.append(
                    f"hybrid optimizer contract load failed ({type(error).__name__}: {error})"
                )
            finally:
                del optimizer
                gc.collect()
            try:
                scheduler = torch.load(
                    checkpoint / "scheduler.pt", map_location="cpu", weights_only=True
                )
                if problem := _scheduler_contract_problem(scheduler, config, step, final_step):
                    problems.append(problem)
            except Exception as error:  # noqa: BLE001
                problems.append(f"scheduler contract load failed ({type(error).__name__}: {error})")
            if problem := _training_arguments_problem(
                checkpoint / "training_args.bin", config, world_size=4, final_step=final_step
            ):
                problems.append(problem)
            label = f"{config.model_family}/{config.run_id}/checkpoint-{step}"
            if problems:
                errors.extend(f"{label}: {problem}" for problem in problems)
            else:
                verified_checkpoints += 1
    return {
        "complete": not errors
        and shallow.get("verified_runs") == expected_runs
        and verified_checkpoints == expected_checkpoints,
        "verified_runs": shallow.get("verified_runs", 0),
        "verified_checkpoints": verified_checkpoints,
        "expected_runs": expected_runs,
        "expected_checkpoints": expected_checkpoints,
        "training_view_fingerprint": dataset_audit.get("training_view_fingerprint"),
        "errors": errors,
    }


def summarize_final_evaluations(
    native_rows: list[dict[str, Any]],
    hybrid_rows: list[dict[str, Any]],
    families: tuple[str, ...] = ALL_FAMILIES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = normalize_families(families)
    tasks = set(DECONTAMINATED_TASK_NAMES)

    def index_rows(
        rows: list[dict[str, Any]], optimizer: str
    ) -> dict[tuple[str, float, str], dict[str, Any]]:
        indexed: dict[tuple[str, float, str], dict[str, Any]] = {}
        for row in rows:
            if row.get("optimizer") != optimizer or int(row.get("stage", -1)) != 5:
                raise ValueError(f"Unexpected {optimizer} formal evaluation row: {row}")
            identity = (
                str(row.get("model_family")),
                float(row.get("learning_rate")),
                str(row.get("task")),
            )
            if identity in indexed:
                raise ValueError(f"Duplicate {optimizer} formal evaluation: {identity}")
            indexed[identity] = row
        return indexed

    native = index_rows(native_rows, "adamw")
    hybrid = index_rows(hybrid_rows, "hybrid_adamw")
    expected = {
        (family, learning_rate, task)
        for family in families
        for learning_rate in LEARNING_RATES
        for task in tasks
    }
    if set(native) != expected or set(hybrid) != expected:
        raise ValueError(
            "Final hybrid/native evaluation coverage differs from the active family×4×14 design"
        )
    contrasts: list[dict[str, Any]] = []
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for identity in sorted(expected):
        native_score = float(native[identity]["ndcg_at_10"])
        hybrid_score = float(hybrid[identity]["ndcg_at_10"])
        if not math.isfinite(native_score) or not math.isfinite(hybrid_score):
            raise ValueError(f"Non-finite hybrid control score: {identity}")
        row = {
            "model_family": identity[0],
            "learning_rate": identity[1],
            "task": identity[2],
            "adamw_ndcg_at_10": native_score,
            "hybrid_adamw_ndcg_at_10": hybrid_score,
            "hybrid_minus_adamw": hybrid_score - native_score,
            "adamw_result_path": native[identity]["result_path"],
            "hybrid_result_path": hybrid[identity]["result_path"],
        }
        contrasts.append(row)
        grouped[(identity[0], identity[1])].append(row)
    summaries: list[dict[str, Any]] = []
    for (family, learning_rate), rows in sorted(grouped.items()):
        deltas = [row["hybrid_minus_adamw"] for row in rows]
        summaries.append(
            {
                "model_family": family,
                "learning_rate": learning_rate,
                "tasks": len(rows),
                "adamw_mean_ndcg_at_10": statistics.mean(row["adamw_ndcg_at_10"] for row in rows),
                "hybrid_adamw_mean_ndcg_at_10": statistics.mean(
                    row["hybrid_adamw_ndcg_at_10"] for row in rows
                ),
                "hybrid_minus_adamw_mean": statistics.mean(deltas),
                "hybrid_task_wins": sum(delta > 0 for delta in deltas),
                "task_ties": sum(delta == 0 for delta in deltas),
                "hybrid_task_losses": sum(delta < 0 for delta in deltas),
            }
        )
    expected_evaluations = len(families) * len(LEARNING_RATES) * len(tasks)
    expected_summaries = len(families) * len(LEARNING_RATES)
    if len(contrasts) != expected_evaluations or len(summaries) != expected_summaries:
        raise AssertionError("Hybrid control summary cardinality invariant failed")
    return contrasts, summaries


def _system_contrasts(
    native_rows: list[dict[str, Any]],
    hybrid_rows: list[dict[str, Any]],
    families: tuple[str, ...] = ALL_FAMILIES,
) -> list[dict[str, Any]]:
    families = normalize_families(families)
    expected_rows = len(families) * len(LEARNING_RATES)

    def index(
        rows: list[dict[str, Any]], optimizer: str
    ) -> dict[tuple[str, float], dict[str, Any]]:
        result = {
            (str(row["model_family"]), float(row["learning_rate"])): row
            for row in rows
            if row.get("optimizer") == optimizer
        }
        if len(result) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} {optimizer} system rows, found {len(result)}"
            )
        return result

    native = index(native_rows, "adamw")
    hybrid = index(hybrid_rows, "hybrid_adamw")
    if set(native) != set(hybrid):
        raise ValueError("Native and hybrid system rows do not share the frozen 2×4 identities")
    fields = (
        "wall_time_hours",
        "samples_per_second",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "optimizer_state_gib",
    )
    output = []
    for identity in sorted(native):
        row = {"model_family": identity[0], "learning_rate": identity[1]}
        for field in fields:
            native_value = float(native[identity][field])
            hybrid_value = float(hybrid[identity][field])
            if not math.isfinite(native_value) or not math.isfinite(hybrid_value):
                raise ValueError(f"Non-finite system metric for {identity}/{field}")
            row[f"adamw_{field}"] = native_value
            row[f"hybrid_adamw_{field}"] = hybrid_value
            row[f"hybrid_minus_adamw_{field}"] = hybrid_value - native_value
        output.append(row)
    return output


def build_hybrid_report(
    discovery_matrix: Path,
    control_matrix: Path,
    protocol_path: Path,
    discovery_results: Path,
    control_results: Path,
    output_dir: Path,
    families: tuple[str, ...] = ALL_FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    families, scope = resolve_scope(families, scope_amendment)
    protocol = _validate_protocol(protocol_path)
    discovery = [
        config
        for config in load_matrix(discovery_matrix)
        if config.optimizer.name == "adamw" and config.model_family in families
    ]
    control = [config for config in load_matrix(control_matrix) if config.model_family in families]
    expected_runs = len(families) * len(LEARNING_RATES)
    if len(discovery) != expected_runs:
        raise ValueError(
            f"Discovery matrix does not contain {expected_runs} scoped native AdamW runs"
        )
    native_dataset = audit_dataset_artifacts(discovery)
    native_training = audit_training_artifacts(
        discovery,
        deep=True,
        expected_dataset_fingerprint=native_dataset.get("training_view_fingerprint"),
    )
    hybrid_training = audit_hybrid_training(control, families)
    if not native_dataset.get("complete") or not native_training.get("complete"):
        raise ValueError("Native AdamW training sources do not pass their strict audit")
    if not hybrid_training.get("complete"):
        raise ValueError("Hybrid AdamW training sources do not pass their strict audit")
    native_all = collect_evaluations(discovery_results, discovery)
    if len(native_all) != expected_runs * 5 * len(DECONTAMINATED_TASK_NAMES):
        raise ValueError("Native AdamW evaluation source is not the complete five-stage matrix")
    native_final = [row for row in native_all if row["stage"] == 5]
    hybrid_all = collect_evaluations(control_results, control)
    expected_hybrid_evaluations = expected_runs * len(DECONTAMINATED_TASK_NAMES)
    if len(hybrid_all) != expected_hybrid_evaluations or any(
        row["stage"] != 5 for row in hybrid_all
    ):
        raise ValueError("Hybrid evaluation source is not exactly the frozen final-stage matrix")
    contrasts, summaries = summarize_final_evaluations(native_final, hybrid_all, families)
    system = _system_contrasts(
        collect_system_metrics(discovery), collect_system_metrics(control), families
    )
    output_dir = output_dir.resolve()
    tables = {
        "paired_task_contrasts": (output_dir / "paired_task_contrasts.csv", contrasts),
        "final_summary": (output_dir / "final_summary.csv", summaries),
        "system_contrasts": (output_dir / "system_contrasts.csv", system),
    }
    for path, rows in tables.values():
        _atomic_csv(path, rows)
    sources = sorted({Path(row["result_path"]).resolve() for row in [*native_final, *hybrid_all]})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "families": list(families),
        "scope_amendment": scope,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": _sha256(protocol_path.resolve()),
            "status": protocol["status"],
        },
        "matrices": {
            "discovery": {
                "path": str(discovery_matrix.resolve()),
                "sha256": _sha256(discovery_matrix.resolve()),
            },
            "control": {
                "path": str(control_matrix.resolve()),
                "sha256": _sha256(control_matrix.resolve()),
            },
        },
        "training_audits": {
            "native_adamw": native_training,
            "hybrid_adamw": hybrid_training,
        },
        "evaluations": {
            "native_five_stage_units": len(native_all),
            "native_final_units": len(native_final),
            "hybrid_final_units": len(hybrid_all),
            "tasks": len(DECONTAMINATED_TASK_NAMES),
            "result_sources": [{"path": str(path), "sha256": _sha256(path)} for path in sources],
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, (path, rows) in tables.items()
        },
        "interpretation": (
            "Paired final-checkpoint differences isolate hidden/auxiliary AdamW routing; they do "
            "not by themselves identify Muon's matrix transform or select a headline recipe."
        ),
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly audit and summarize the frozen hybrid AdamW routing control"
    )
    parser.add_argument("--discovery-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--control-matrix", type=Path, default=Path("configs/hybrid_adamw.yaml"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/hybrid_adamw_control.json"))
    parser.add_argument(
        "--discovery-results", type=Path, default=Path("results/decontaminated-beir")
    )
    parser.add_argument("--control-results", type=Path, default=Path("results/hybrid-adamw-beir"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hybrid-adamw"))
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_hybrid_report(
        args.discovery_matrix,
        args.control_matrix,
        args.protocol,
        args.discovery_results,
        args.control_results,
        args.output_dir,
        tuple(args.families),
        args.scope_amendment,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
