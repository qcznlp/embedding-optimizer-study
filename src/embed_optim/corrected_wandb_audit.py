"""Read-only provenance audit for corrected Dense source runs on W&B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import RunConfig, load_matrix, resolve_matrix_path, source_wandb_run_id
from .matrix import _run_is_complete


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path, repository: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        label = str(resolved.relative_to(repository))
    except ValueError:
        label = str(resolved)
    return {"path": label, "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _validate_inputs(
    matrix_path: Path,
    protocol_path: Path,
    repository: Path,
) -> tuple[list[RunConfig], dict[str, Any]]:
    configs = load_matrix(matrix_path)
    if (
        len(configs) != 12
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
        or {config.optimizer.name for config in configs} != {"adamw", "muon", "normuon"}
        or len({config.run_id for config in configs}) != 12
        or any(len(config.checkpoint_fractions) != 5 for config in configs)
    ):
        raise ValueError("Corrected W&B audit requires the frozen 12-run padded Dense matrix")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    matrix_binding = protocol.get("source_bindings", {}).get("formal_matrix", {})
    if (
        protocol.get("status") != "prospective_corrective_execution_lock"
        or matrix_binding.get("sha256") != _sha256(matrix_path)
        or int(protocol.get("training", {}).get("expected_optimizer_steps", -1)) != 3907
        or int(protocol.get("training", {}).get("world_size", -1)) != 4
    ):
        raise ValueError("Corrected W&B inputs differ from the frozen execution protocol")
    entities = {(config.wandb_entity, config.wandb_project) for config in configs}
    if len(entities) != 1 or not all(next(iter(entities))):
        raise ValueError("Corrected matrix must declare one non-empty W&B entity/project")
    if not protocol_path.resolve().is_relative_to(repository):
        raise ValueError("Corrected W&B protocol must be under the repository")
    return configs, protocol


def _expected_remote_config(config: RunConfig, *, world_size: int) -> dict[str, Any]:
    expected = _normalized(config.as_dict())
    denominator = config.micro_batch_size * world_size
    if config.global_batch_size % denominator:
        raise ValueError(f"Invalid global batch for {config.run_id}")
    expected.update(
        {
            "gradient_accumulation_steps": config.global_batch_size // denominator,
            "num_train_epochs": config.epochs,
            "per_device_train_batch_size": config.micro_batch_size,
            "run_name": f"{config.model_family}-{config.run_id}",
        }
    )
    return expected


def _summary_value(summary: Any, key: str) -> Any:
    try:
        return summary.get(key)
    except AttributeError:
        return None


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def audit_run(
    config: RunConfig,
    remote: Any | None,
    *,
    local_complete: bool,
    expected_steps: int,
    world_size: int,
) -> dict[str, Any]:
    expected_id = source_wandb_run_id(config)
    record: dict[str, Any] = {
        "run_id": config.run_id,
        "optimizer": config.optimizer.name,
        "learning_rate": config.optimizer.lr,
        "source_wandb_run_id": expected_id,
        "local_complete": local_complete,
        "remote_visible": remote is not None,
        "problems": [],
    }
    if remote is None:
        record["status"] = "missing_after_local_completion" if local_complete else "not_started"
        if local_complete:
            record["problems"].append("locally complete run is missing from W&B")
        return record

    observed_config = _normalized(dict(remote.config or {}))
    expected_config = _expected_remote_config(config, world_size=world_size)
    mismatches = sorted(
        key for key, value in expected_config.items() if observed_config.get(key) != value
    )
    observed_tags = sorted(str(tag) for tag in (remote.tags or []))
    expected_tags = sorted((config.model_family, config.optimizer.name, f"seed-{config.seed}"))
    remote_state = str(remote.state)
    summary_step = _summary_value(remote.summary, "train/global_step")
    summary_epoch = _summary_value(remote.summary, "train/epoch")
    record.update(
        {
            "url": getattr(remote, "url", None),
            "remote_state": remote_state,
            "remote_name": str(remote.name),
            "remote_group": str(remote.group),
            "remote_tags": observed_tags,
            "summary_global_step": summary_step,
            "summary_epoch": summary_epoch,
            "config_mismatches": mismatches,
        }
    )
    if str(remote.id) != expected_id:
        record["problems"].append("remote ID differs from the deterministic source run ID")
    if str(remote.name) != f"{config.model_family}-{config.run_id}":
        record["problems"].append("remote name differs from the matrix run name")
    if str(remote.group) != config.model_family:
        record["problems"].append("remote group differs from the model family")
    if observed_tags != expected_tags:
        record["problems"].append("remote tags differ from the matrix tags")
    if mismatches:
        record["problems"].append("remote config differs from the resolved matrix config")

    if local_complete:
        if remote_state != "finished":
            record["problems"].append("locally complete run is not finished on W&B")
        if not _finite_number(summary_step) or int(summary_step) != expected_steps:
            record["problems"].append("finished W&B summary has the wrong global step")
        if not _finite_number(summary_epoch) or not math.isclose(
            float(summary_epoch), config.epochs, rel_tol=0.0, abs_tol=1e-9
        ):
            record["problems"].append("finished W&B summary has the wrong epoch")
    elif remote_state != "running":
        record["problems"].append("incomplete local run has an unexpected terminal W&B state")

    record["status"] = "valid" if not record["problems"] else "invalid"
    return record


def build_audit(
    configs: list[RunConfig],
    remote_by_id: dict[str, Any],
    *,
    expected_steps: int,
    world_size: int,
    allow_partial: bool,
    matrix_identity: dict[str, Any],
    protocol_identity: dict[str, Any],
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    records = [
        audit_run(
            config,
            remote_by_id.get(source_wandb_run_id(config)),
            local_complete=_run_is_complete(config),
            expected_steps=expected_steps,
            world_size=world_size,
        )
        for config in configs
    ]
    problems = [
        f"{record['run_id']}: {problem}" for record in records for problem in record["problems"]
    ]
    local_complete = sum(bool(record["local_complete"]) for record in records)
    remote_visible = sum(bool(record["remote_visible"]) for record in records)
    remote_finished = sum(record.get("remote_state") == "finished" for record in records)
    valid = sum(record["status"] == "valid" for record in records)
    full_complete = local_complete == remote_finished == valid == len(configs)
    if problems or (not allow_partial and not full_complete):
        status = "invalid" if problems else "incomplete"
    else:
        status = "complete" if full_complete else "partial"
    entity, project = configs[0].wandb_entity, configs[0].wandb_project
    return {
        "schema_version": 1,
        "status": status,
        "complete": full_complete and not problems,
        "classification": "post-output read-only operational provenance audit",
        "observed_at_utc": _timestamp(),
        "entity": entity,
        "project": project,
        "matrix": matrix_identity,
        "execution_protocol": protocol_identity,
        "audit_source": source_identity,
        "expected_runs": len(configs),
        "local_complete_runs": local_complete,
        "remote_visible_runs": remote_visible,
        "remote_finished_runs": remote_finished,
        "valid_remote_runs": valid,
        "problems": problems,
        "runs": records,
        "claim_boundary": (
            "This receipt verifies W&B source-run identity, configuration, and terminal metadata. "
            "It does not validate retrieval results or add a scientific endpoint."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.workdir.resolve()
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    protocol_path = args.protocol.resolve()
    configs, protocol = _validate_inputs(matrix_path, protocol_path, repository)
    import wandb

    entity, project = configs[0].wandb_entity, configs[0].wandb_project
    expected_ids = {source_wandb_run_id(config) for config in configs}
    remote_by_id = {
        str(remote.id): remote
        for remote in wandb.Api(timeout=args.timeout).runs(f"{entity}/{project}")
        if str(remote.id) in expected_ids
    }
    audit = build_audit(
        configs,
        remote_by_id,
        expected_steps=int(protocol["training"]["expected_optimizer_steps"]),
        world_size=int(protocol["training"]["world_size"]),
        allow_partial=args.allow_partial,
        matrix_identity=_file_identity(matrix_path, repository),
        protocol_identity=_file_identity(protocol_path, repository),
        source_identity=_file_identity(Path(__file__), repository),
    )
    _atomic_json(args.output.resolve(), audit)
    if audit["status"] in {"invalid", "incomplete"}:
        raise RuntimeError(
            f"Corrected W&B audit {audit['status']}: " + "; ".join(audit["problems"][:5])
        )
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/dense_no_packing_execution_protocol.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/dense-no-packing/wandb-audit.json")
    )
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
