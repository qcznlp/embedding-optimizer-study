"""Resume-safe controller for the prospective state-by-operator follow-up.

The controller waits for the entire corrected 12-run publication pipeline to
finish before requesting any GPU lease.  It then executes the already frozen
factorial commands, backs up every short training wave, renders the mechanism
result into the paper, and reruns the release gates.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import load_matrix
from .geometry import _sha256
from .incremental_checkpoint_backup import (
    inventory_digest,
    local_checkpoint_inventory,
    validate_sealed_checkpoint,
)

COMPLETION_PROTOCOL = Path(
    "configs/dense_no_packing_state_operator_factorial_completion_protocol.json"
)
MAIN_LEDGER = Path("logs/dense-no-packing-finalization/pipeline-ledger.json")
LOG_ROOT = Path("logs/state-operator-factorial/completion")
SOURCE_RECEIPT_ROOT = Path("reports/dense-no-packing/incremental-checkpoint-backup")


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]
    parallel_group: str | None = None


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _identity(path: Path, repository: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(repository.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _binding_valid(binding: Any, repository: Path) -> bool:
    if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
        return False
    path = repository / binding["path"]
    try:
        return (
            path.is_file()
            and path.stat().st_size == int(binding["bytes"])
            and _sha256(path) == binding["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def load_completion_protocol(path: Path, repository: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "prospective_state_operator_completion_lock":
        raise ValueError("State-operator completion protocol status differs")
    bindings = [
        *protocol.get("parent_bindings", {}).values(),
        *protocol.get("source_bindings", {}).values(),
        *protocol.get("configuration_bindings", {}).values(),
    ]
    if not bindings or any(not _binding_valid(binding, repository) for binding in bindings):
        raise ValueError("State-operator completion source contract changed")
    return protocol


def _split(command: str) -> tuple[str, ...]:
    values = tuple(shlex.split(command))
    if not values:
        raise ValueError("State-operator completion command is empty")
    return values


def pipeline_steps(
    args: argparse.Namespace,
    repository: Path,
    protocol: dict[str, Any],
) -> list[PipelineStep]:
    implementation_path = (
        repository / protocol["parent_bindings"]["implementation_protocol"]["path"]
    )
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    commands = implementation["commands"]
    steps = [PipelineStep("protocol-audit", _split(commands["protocol_audit"]))]
    source_states = protocol["source_checkpoint_gate"]["states"]
    source_backup = (
        "uv",
        "run",
        "embed-optim-backup-sealed-checkpoint",
        "--run-ids",
        *(item["run_id"] for item in source_states),
        "--steps",
        str(protocol["source_checkpoint_gate"]["checkpoint_step"]),
    )
    steps.append(PipelineStep("source-checkpoint-durability", source_backup))
    steps.extend(
        PipelineStep(f"calibration-{index:02d}", _split(command))
        for index, command in enumerate(commands["calibration"], start=1)
    )
    steps.extend(
        (
            PipelineStep("matrix-generate", _split(commands["matrix_generation"])),
            PipelineStep(
                "matrix-audit",
                ("uv", "run", "embed-optim-state-operator-factorial", "audit"),
            ),
        )
    )
    for label in commands["training_order"]:
        state, _, seed_label = label.partition("-seed")
        seed = int(seed_label)
        train = commands["training_template"].format(state=state, seed=seed)
        steps.append(PipelineStep(f"training-{label}", _split(train)))
        steps.append(
            PipelineStep(
                f"checkpoint-backup-{label}",
                (
                    "uv",
                    "run",
                    "embed-optim-backup-state-operator-factorial",
                    "--state",
                    state,
                    "--seed",
                    str(seed),
                    "--repo-id",
                    args.checkpoint_repo,
                    "--remote-prefix",
                    args.checkpoint_prefix,
                ),
            )
        )
    steps.append(PipelineStep("probe", _split(commands["probe"])))
    pools = tuple(args.beir_gpu_pools.split(";"))
    if len(pools) != 2:
        raise ValueError("Factorial completion requires exactly two BEIR GPU pools")
    for pair_index, pair in enumerate(commands["full_beir_parallel_pairs"], start=1):
        group = f"full-beir-pair-{pair_index}"
        for cell, pool in zip(pair, pools, strict=True):
            state, _, seed_label = cell.partition("-seed")
            command = commands["full_beir_template"].format(
                state=state,
                seed=int(seed_label),
                four_gpu_pool=pool,
            )
            steps.append(PipelineStep(f"full-beir-{cell}", _split(command), group))
    python = str(args.python)
    steps.extend(
        (
            PipelineStep("summary", _split(commands["summary"])),
            PipelineStep(
                "publication-render",
                (python, "-m", "embed_optim.state_operator_factorial_publication"),
            ),
            PipelineStep(
                "paper-release",
                ("make", "-C", str(repository / "paper"), "release", f"PYTHON={python}"),
            ),
            PipelineStep(
                "portable-evidence-refresh",
                (python, str(repository / "scripts/portable_evidence.py")),
            ),
            PipelineStep(
                "paper-audit",
                (
                    python,
                    "-m",
                    "embed_optim.paper_audit",
                    "--strict",
                    "--families",
                    "dense",
                    "--scope-amendment",
                    str(repository / "configs/dense_scope_amendment.json"),
                ),
            ),
            PipelineStep(
                "portable-evidence-audit",
                (python, str(repository / "scripts/portable_evidence.py"), "--audit-only"),
            ),
            PipelineStep("tests", (python, "-m", "pytest", "-q")),
            PipelineStep("ruff-check", (python, "-m", "ruff", "check", "src", "tests", "scripts")),
            PipelineStep(
                "ruff-format-check",
                (python, "-m", "ruff", "format", "--check", "src", "tests", "scripts"),
            ),
        )
    )
    return steps


def _contract(
    args: argparse.Namespace,
    repository: Path,
    protocol_path: Path,
    steps: list[PipelineStep],
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "protocol": _identity(protocol_path, repository),
        "steps": [
            {
                "index": index,
                "name": step.name,
                "command": list(step.command),
                "parallel_group": step.parallel_group,
            }
            for index, step in enumerate(steps, start=1)
        ],
        "arguments": {
            "python": str(args.python),
            "beir_gpu_pools": args.beir_gpu_pools,
            "checkpoint_repo": args.checkpoint_repo,
            "checkpoint_prefix": args.checkpoint_prefix,
            "main_ledger": str(args.main_ledger.resolve()),
        },
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {**body, "sha256": hashlib.sha256(encoded).hexdigest()}


@contextmanager
def _exclusive_lease(path: Path) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "A state-operator completion controller is already active"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at={_timestamp()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield handle.fileno()


def _main_complete(ledger_path: Path, protocol: dict[str, Any]) -> bool:
    if not ledger_path.is_file():
        return False
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    gate = protocol["main_completion_gate"]
    steps = ledger.get("steps")
    return bool(
        ledger.get("scope") == "corrected_dense_no_packing_completion"
        and ledger.get("status") == "complete"
        and ledger.get("complete") is True
        and ledger.get("training_runs_complete") == 12
        and ledger.get("training_runs_expected") == 12
        and ledger.get("contract", {}).get("sha256") == gate["contract_sha256"]
        and isinstance(steps, list)
        and [step.get("name") for step in steps] == gate["required_steps"]
        and all(step.get("complete") is True for step in steps)
        and set(ledger.get("backups", {})) == set(gate["required_run_ids"])
        and all(item.get("complete") is True for item in ledger["backups"].values())
    )


def _source_readiness(repository: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    gate = protocol["source_checkpoint_gate"]
    matrix_path = repository / gate["matrix"]
    configs = {config.run_id: config for config in load_matrix(matrix_path)}
    records = []
    for state in gate["states"]:
        config = configs[state["run_id"]]
        output_root = Path(config.output_root)
        if not output_root.is_absolute():
            config = replace(config, output_root=str(repository / output_root))
        checkpoint = repository / state["checkpoint"]
        validate_sealed_checkpoint(config, gate["checkpoint_step"])
        if (
            checkpoint.resolve()
            != (config.output_dir / f"checkpoint-{gate['checkpoint_step']}").resolve()
        ):
            raise ValueError("Factorial source checkpoint path differs from its run configuration")
        receipt_path = (
            repository
            / SOURCE_RECEIPT_ROOT
            / (f"{state['run_id']}-checkpoint-{gate['checkpoint_step']}.json")
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        declared_local_root = Path(str(receipt.get("local_root")))
        if not declared_local_root.is_absolute():
            declared_local_root = repository / declared_local_root
        inventory = local_checkpoint_inventory(checkpoint)
        audit = receipt.get("inventory", {})
        if (
            receipt.get("status") != "complete"
            or receipt.get("scientific_completion") is not False
            or receipt.get("run_id") != state["run_id"]
            or receipt.get("checkpoint_step") != gate["checkpoint_step"]
            or declared_local_root.resolve() != checkpoint.resolve()
            or receipt.get("matrix", {}).get("sha256") != _sha256(matrix_path)
            or receipt.get("inventory_sha256") != inventory_digest(inventory)
            or audit.get("complete") is not True
            or any(
                audit.get(name) for name in ("missing", "extra", "size_mismatch", "digest_mismatch")
            )
        ):
            raise ValueError(f"Factorial source durability receipt differs: {receipt_path}")
        records.append(_identity(receipt_path, repository))
    return {"status": "ready", "receipts": records}


def _new_ledger(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "corrected_state_operator_factorial_completion",
        "status": "waiting_for_main_completion",
        "complete": False,
        "started_at_utc": _timestamp(),
        "observed_at_utc": _timestamp(),
        "contract": contract,
        "steps": [],
    }


def _load_ledger(path: Path, contract: dict[str, Any], resume: bool) -> dict[str, Any]:
    if not path.is_file():
        return _new_ledger(contract)
    if not resume:
        raise FileExistsError(f"State-operator completion ledger already exists: {path}")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("contract") != contract:
        raise RuntimeError("State-operator completion contract differs from its ledger")
    return ledger


def _attempt(
    command: tuple[str, ...],
    *,
    repository: Path,
    log_path: Path,
    run_command: Callable[..., subprocess.CompletedProcess[Any]],
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _timestamp()
    error = None
    return_code = None
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            result = run_command(
                command,
                cwd=repository,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            return_code = result.returncode
        except OSError as caught:
            error = f"{type(caught).__name__}: {caught}"
            handle.write(f"controller execution error: {error}\n")
    result = {
        "started_at_utc": started,
        "finished_at_utc": _timestamp(),
        "return_code": return_code,
        "log": _identity(log_path, repository),
    }
    if error:
        result["execution_error"] = error
    return result


def _step_record(ledger: dict[str, Any], index: int, step: PipelineStep) -> dict[str, Any]:
    existing = {item.get("name"): item for item in ledger["steps"]}.get(step.name)
    expected = {
        "index": index,
        "name": step.name,
        "command": list(step.command),
        "parallel_group": step.parallel_group,
    }
    if existing is not None:
        if any(existing.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"State-operator step record changed: {step.name}")
        return existing
    record = {**expected, "attempts": [], "complete": False}
    ledger["steps"].append(record)
    return record


def _record_attempt(
    record: dict[str, Any],
    attempt: dict[str, Any],
    ledger: dict[str, Any],
) -> bool:
    attempt = {"attempt": len(record["attempts"]) + 1, **attempt}
    record["attempts"].append(attempt)
    success = attempt["return_code"] == 0
    if success:
        record["complete"] = True
        record["finished_at_utc"] = _timestamp()
        ledger.pop("failed_step", None)
    else:
        ledger["failed_step"] = record["name"]
    ledger["observed_at_utc"] = _timestamp()
    return success


def _run_single(
    step: PipelineStep,
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
    repository: Path,
    ledger: dict[str, Any],
    ledger_path: Path,
    run_command: Callable[..., subprocess.CompletedProcess[Any]],
    sleeper: Callable[[float], None],
    contract_check: Callable[[], None],
) -> None:
    while record.get("complete") is not True:
        contract_check()
        number = len(record["attempts"]) + 1
        ledger["status"] = "running"
        ledger["active_step"] = step.name
        _atomic_json(ledger_path, ledger)
        attempt = _attempt(
            step.command,
            repository=repository,
            log_path=ledger_path.parent / f"{record['index']:02d}-{step.name}.attempt-{number}.log",
            run_command=run_command,
        )
        if _record_attempt(record, attempt, ledger):
            _atomic_json(ledger_path, ledger)
            return
        ledger["status"] = "retry_wait"
        _atomic_json(ledger_path, ledger)
        if args.max_attempts and len(record["attempts"]) >= args.max_attempts:
            raise RuntimeError(f"State-operator step exhausted retries: {step.name}")
        sleeper(args.retry_delay)


def _run_parallel(
    group: list[tuple[PipelineStep, dict[str, Any]]],
    *,
    args: argparse.Namespace,
    repository: Path,
    ledger: dict[str, Any],
    ledger_path: Path,
    run_command: Callable[..., subprocess.CompletedProcess[Any]],
    sleeper: Callable[[float], None],
    contract_check: Callable[[], None],
) -> None:
    while any(record.get("complete") is not True for _, record in group):
        contract_check()
        pending = [(step, record) for step, record in group if record.get("complete") is not True]
        ledger["status"] = "running_parallel"
        ledger["active_parallel_group"] = group[0][0].parallel_group
        ledger["active_steps"] = [step.name for step, _ in pending]
        _atomic_json(ledger_path, ledger)

        def execute(
            item: tuple[PipelineStep, dict[str, Any]],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            step, record = item
            number = len(record["attempts"]) + 1
            attempt = _attempt(
                step.command,
                repository=repository,
                log_path=ledger_path.parent
                / f"{record['index']:02d}-{step.name}.attempt-{number}.log",
                run_command=run_command,
            )
            return record, attempt

        with ThreadPoolExecutor(max_workers=len(pending)) as executor:
            results = list(executor.map(execute, pending))
        failures = []
        for record, attempt in results:
            if not _record_attempt(record, attempt, ledger):
                failures.append(record)
        _atomic_json(ledger_path, ledger)
        if failures:
            if args.max_attempts and any(
                len(record["attempts"]) >= args.max_attempts for record in failures
            ):
                names = [record["name"] for record in failures]
                raise RuntimeError(f"Parallel state-operator steps exhausted retries: {names}")
            ledger["status"] = "retry_wait"
            _atomic_json(ledger_path, ledger)
            sleeper(args.retry_delay)
    ledger.pop("active_parallel_group", None)
    ledger.pop("active_steps", None)


def run_pipeline(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    repository = args.workdir.resolve()
    protocol_path = (repository / args.protocol).resolve()
    protocol = load_completion_protocol(protocol_path, repository)
    steps = pipeline_steps(args, repository, protocol)
    contract = _contract(args, repository, protocol_path, steps)
    log_dir = (repository / args.log_dir).resolve()
    ledger_path = log_dir / "pipeline-ledger.json"
    with _exclusive_lease(log_dir / "controller.lease"):
        ledger = _load_ledger(ledger_path, contract, args.resume)
        if ledger.get("complete") is True:
            return 0
        _atomic_json(ledger_path, ledger)
        while not _main_complete(args.main_ledger.resolve(), protocol):
            ledger["status"] = "waiting_for_main_completion"
            ledger["observed_at_utc"] = _timestamp()
            _atomic_json(ledger_path, ledger)
            sleeper(args.poll_seconds)
        ledger["main_completion_ledger"] = _identity(args.main_ledger.resolve(), repository)
        ledger["status"] = "running"
        _atomic_json(ledger_path, ledger)

        indexed = [
            (step, _step_record(ledger, index, step)) for index, step in enumerate(steps, start=1)
        ]

        def assert_contract_unchanged() -> None:
            current_protocol = load_completion_protocol(protocol_path, repository)
            current_steps = pipeline_steps(args, repository, current_protocol)
            if (
                current_protocol != protocol
                or _contract(args, repository, protocol_path, current_steps) != contract
            ):
                raise RuntimeError("State-operator completion contract changed while running")

        cursor = 0
        while cursor < len(indexed):
            step, record = indexed[cursor]
            if step.parallel_group is None:
                _run_single(
                    step,
                    record,
                    args=args,
                    repository=repository,
                    ledger=ledger,
                    ledger_path=ledger_path,
                    run_command=run_command,
                    sleeper=sleeper,
                    contract_check=assert_contract_unchanged,
                )
                if step.name == "source-checkpoint-durability":
                    ledger["source_checkpoint_gate"] = _source_readiness(repository, protocol)
                    _atomic_json(ledger_path, ledger)
                cursor += 1
                continue
            group_name = step.parallel_group
            group = []
            while cursor < len(indexed) and indexed[cursor][0].parallel_group == group_name:
                group.append(indexed[cursor])
                cursor += 1
            _run_parallel(
                group,
                args=args,
                repository=repository,
                ledger=ledger,
                ledger_path=ledger_path,
                run_command=run_command,
                sleeper=sleeper,
                contract_check=assert_contract_unchanged,
            )

        ledger.pop("active_step", None)
        ledger.pop("failed_step", None)
        ledger["status"] = "complete"
        ledger["complete"] = True
        ledger["finished_at_utc"] = _timestamp()
        ledger["observed_at_utc"] = _timestamp()
        _atomic_json(ledger_path, ledger)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=COMPLETION_PROTOCOL)
    parser.add_argument("--main-ledger", type=Path, default=MAIN_LEDGER)
    parser.add_argument("--log-dir", type=Path, default=LOG_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--beir-gpu-pools", default="0,1,2,3;4,5,6,7")
    parser.add_argument("--checkpoint-repo", default="qcz/embedding-optimizer-study-checkpoints")
    parser.add_argument("--checkpoint-prefix", default="state-operator-factorial-v1/dense")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--retry-delay", type=float, default=300.0)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.main_ledger.is_absolute():
        args.main_ledger = args.workdir / args.main_ledger
    pools = [pool for pool in args.beir_gpu_pools.split(";") if pool]
    tokens = [token.strip() for pool in pools for token in pool.split(",") if token.strip()]
    if len(pools) != 2 or len(tokens) != 8 or len(tokens) != len(set(tokens)):
        parser.error("--beir-gpu-pools must declare two disjoint four-GPU pools")
    if args.poll_seconds <= 0 or args.retry_delay < 0 or args.max_attempts < 0:
        parser.error("Polling must be positive and retry controls non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run_pipeline(parse_args(argv)))


if __name__ == "__main__":
    main()
