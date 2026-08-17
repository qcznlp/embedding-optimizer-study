from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import RunConfig, load_matrix, matrix_runtime_spec


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_formal_runtime(matrix: str | Path) -> dict[str, Any] | None:
    runtime_spec = matrix_runtime_spec(matrix)
    if runtime_spec is None:
        return None
    from .runtime import verify_runtime_spec

    runtime = verify_runtime_spec(runtime_spec)
    print(
        f"formal runtime verified: {runtime['python_executable']} | "
        f"torch={runtime['packages']['torch']} cuda={runtime['torch_cuda']}",
        flush=True,
    )
    return runtime


def _read_schedule(config: RunConfig) -> list[int] | None:
    path = config.output_dir / "checkpoint_schedule.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        steps = [int(step) for step in payload["steps"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"Invalid checkpoint schedule {path}: {error}") from error
    if len(steps) != 5 or len(set(steps)) != 5 or steps != sorted(steps) or steps[0] <= 0:
        raise RuntimeError(f"Invalid checkpoint schedule {path}: expected five increasing steps")
    return steps


def _payload_signature(path: Path) -> str:
    """Fingerprint checkpoint file metadata so a changed payload is audited again."""

    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = item.stat()
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _audit_contract_signature(config: RunConfig, world_size: int) -> str:
    if callable(getattr(config, "as_dict", None)):
        config_payload = config.as_dict()
    else:
        config_payload = {
            "model_family": config.model_family,
            "run_id": config.run_id,
            "output_dir": str(config.output_dir),
        }
    payload = {"config": config_payload, "world_size": world_size}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _new_state(configs: list[RunConfig]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at_utc": _utc_now(),
        "audit_complete": False,
        "training_complete": False,
        "audited_checkpoints": 0,
        "expected_checkpoints": len(configs) * 5,
        "runs": {},
    }


def _load_state(path: Path, configs: list[RunConfig]) -> dict[str, Any]:
    if not path.is_file():
        return _new_state(configs)
    try:
        state = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"Invalid checkpoint audit state {path}: {error}") from error
    if state.get("schema_version") != 1 or not isinstance(state.get("runs"), dict):
        raise RuntimeError(f"Invalid checkpoint audit state {path}: unsupported schema")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def audit_once(
    configs: list[RunConfig],
    state_path: Path,
    *,
    world_size: int = 4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deep-audit every newly complete or changed scheduled checkpoint once."""

    from .aggregate import _deep_checkpoint_problems, _safetensors_digest
    from .matrix import _checkpoint_is_resumable

    state = _load_state(state_path, configs)
    expected_labels = {f"{config.model_family}/{config.run_id}" for config in configs}
    state["runs"] = {
        label: value for label, value in state["runs"].items() if label in expected_labels
    }
    events: list[dict[str, Any]] = []

    for config in configs:
        label = f"{config.model_family}/{config.run_id}"
        run_state = state["runs"].setdefault(label, {"checkpoints": {}})
        contract_signature = _audit_contract_signature(config, world_size)
        if run_state.get("audit_contract_signature") not in {None, contract_signature}:
            run_state["checkpoints"] = {}
        run_state["audit_contract_signature"] = contract_signature
        schedule = _read_schedule(config)
        run_state["schedule_ready"] = schedule is not None
        run_state["training_complete"] = (config.output_dir / "completed.json").is_file()
        if schedule is None:
            run_state["expected_steps"] = []
            continue

        run_state["expected_steps"] = schedule
        final_step = schedule[-1]
        checkpoints = run_state.setdefault("checkpoints", {})
        for stale_step in set(checkpoints) - {str(step) for step in schedule}:
            del checkpoints[stale_step]

        previous_digest: str | None = None
        for step in schedule:
            checkpoint = config.output_dir / f"checkpoint-{step}"
            prior = checkpoints.get(str(step), {})
            if not _checkpoint_is_resumable(checkpoint, world_size=world_size):
                if prior and prior.get("status") != "missing":
                    missing = {
                        **prior,
                        "status": "missing",
                        "audited_at_utc": _utc_now(),
                        "problems": ["checkpoint is no longer atomically resumable"],
                    }
                    checkpoints[str(step)] = missing
                    events.append({"run": label, "step": step, **missing})
                continue
            signature = _payload_signature(checkpoint)
            if prior.get("payload_signature") == signature and prior.get("status") != "missing":
                if prior.get("status") == "passed":
                    previous_digest = prior.get("model_digest")
                continue

            problems = _deep_checkpoint_problems(
                checkpoint,
                step,
                world_size,
                config=config,
                final_step=final_step,
            )
            model_digest = _safetensors_digest(checkpoint)
            if previous_digest is not None and model_digest == previous_digest:
                problems.append("model payload is unchanged from the previous checkpoint")
            audited = {
                "status": "failed" if problems else "passed",
                "audited_at_utc": _utc_now(),
                "payload_signature": signature,
                "model_digest": model_digest,
                "problems": problems,
            }
            checkpoints[str(step)] = audited
            event = {"run": label, "step": step, **audited}
            events.append(event)
            if not problems:
                previous_digest = model_digest

    passed = 0
    audit_complete = True
    training_complete = True
    for config in configs:
        label = f"{config.model_family}/{config.run_id}"
        run_state = state["runs"].get(label, {})
        schedule = run_state.get("expected_steps", [])
        checkpoints = run_state.get("checkpoints", {})
        passed += sum(item.get("status") == "passed" for item in checkpoints.values())
        audit_complete &= len(schedule) == 5 and all(
            checkpoints.get(str(step), {}).get("status") == "passed" for step in schedule
        )
        training_complete &= bool(run_state.get("training_complete"))

    state.update(
        {
            "updated_at_utc": _utc_now(),
            "audit_complete": audit_complete,
            "training_complete": training_complete,
            "audited_checkpoints": passed,
            "expected_checkpoints": len(configs) * 5,
        }
    )
    _write_state(state_path, state)
    return state, events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously deep-audit scheduled training checkpoints after atomic writes"
    )
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--families", nargs="+", choices=("dense", "late"))
    parser.add_argument("--run-ids", nargs="+")
    parser.add_argument("--state", default="logs/checkpoint-audit.json")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--fail-on-problem", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive")
    if args.watch and args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive when --watch is set")

    _validate_formal_runtime(args.matrix)
    configs = load_matrix(args.matrix)
    if args.families:
        configs = [config for config in configs if config.model_family in args.families]
    if args.run_ids:
        requested = set(args.run_ids)
        configs = [config for config in configs if config.run_id in requested]
    if not configs:
        raise ValueError("No experiment configurations matched the requested filters")

    state_path = Path(args.state)
    while True:
        state, events = audit_once(configs, state_path, world_size=args.world_size)
        for event in events:
            print(json.dumps(event, sort_keys=True), flush=True)
        print(
            json.dumps(
                {
                    "audited_checkpoints": state["audited_checkpoints"],
                    "expected_checkpoints": state["expected_checkpoints"],
                    "audit_complete": state["audit_complete"],
                    "training_complete": state["training_complete"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        has_problem = any(
            checkpoint.get("status") in {"failed", "missing"}
            for run in state["runs"].values()
            for checkpoint in run.get("checkpoints", {}).values()
        )
        if args.fail_on_problem and has_problem:
            raise SystemExit(1)
        if not args.watch or (state["audit_complete"] and state["training_complete"]):
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
