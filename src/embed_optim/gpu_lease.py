"""Cooperative, process-scoped GPU leases for evaluation coordinators."""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_gpu_tokens(value: str, *, expected_count: int | None = None) -> tuple[str, ...]:
    """Parse a canonical comma-separated physical-GPU token list.

    Canonical spelling is intentional: provenance comparisons must not treat
    whitespace, duplicate devices, or aliases such as ``00`` as equivalent.
    """

    if not isinstance(value, str) or not value:
        raise ValueError("GPU tokens must be a non-empty comma-separated string")
    tokens = value.split(",")
    if any(not re.fullmatch(r"0|[1-9][0-9]*", token) for token in tokens):
        raise ValueError(f"GPU tokens are not canonically encoded: {value!r}")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"GPU tokens contain duplicates: {value!r}")
    if expected_count is not None and len(tokens) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} GPU tokens, got {len(tokens)}")
    return tuple(tokens)


def validate_disjoint_gpu_pools(pool_a: str, pool_b: str) -> dict[str, tuple[str, ...]]:
    """Validate the frozen two-pool topology used by the Dense training queue."""

    tokens_a = parse_gpu_tokens(pool_a, expected_count=4)
    tokens_b = parse_gpu_tokens(pool_b, expected_count=4)
    if set(tokens_a) & set(tokens_b):
        raise ValueError("Dense training GPU pools must be disjoint")
    if len(set(tokens_a) | set(tokens_b)) != 8:
        raise ValueError("Dense training GPU pools must cover eight unique devices")
    return {"a": tokens_a, "b": tokens_b}


def evaluation_gpu_tokens(
    *,
    has_dense: bool,
    has_late: bool,
    gpus_a: str,
    gpus_b: str,
) -> tuple[str, ...]:
    """Return the physical devices an evaluation coordinator can launch on."""

    if not has_dense and not has_late:
        raise ValueError("Evaluation has no GPU-using model family")
    tokens_a = parse_gpu_tokens(gpus_a)
    tokens_b = parse_gpu_tokens(gpus_b)
    requested = set(tokens_a)
    if has_late:
        if requested & set(tokens_b):
            raise ValueError("Late evaluation GPU pools must be disjoint")
        requested.update(tokens_b)
    return tuple(sorted(requested, key=int))


def _lease_payload(
    *, tokens: Sequence[str], purpose: str, timeout_seconds: float, status: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "pid": os.getpid(),
        "gpu_tokens": list(tokens),
        "purpose": purpose,
        "timeout_seconds": timeout_seconds,
        "started_at": _timestamp(),
    }


@contextmanager
def acquire_gpu_lease(
    tokens: Sequence[str],
    *,
    lock_dir: str | Path,
    timeout_seconds: float,
    purpose: str,
    ledger_path: str | Path | None = None,
    poll_seconds: float = 0.25,
) -> Iterator[None]:
    """Acquire every requested GPU token or time out without holding a subset.

    All cooperating evaluators acquire token files in numeric order.  Training
    predates this lease protocol, so callers must independently prove that a
    training pool is complete before requesting its tokens.
    """

    canonical = tuple(sorted(set(tokens), key=int))
    if not canonical or canonical != tuple(sorted(tokens, key=int)):
        raise ValueError("GPU lease tokens must be unique and numerically sorted")
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("GPU lease timeout and poll interval must be positive")
    directory = Path(lock_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ledger = Path(ledger_path).resolve() if ledger_path is not None else None
    payload = _lease_payload(
        tokens=canonical,
        purpose=purpose,
        timeout_seconds=timeout_seconds,
        status="waiting",
    )
    if ledger is not None:
        _atomic_json(ledger, payload)

    deadline = time.monotonic() + timeout_seconds
    handles: list[Any] = []
    try:
        while True:
            handles = []
            acquired = True
            for token in canonical:
                handle = (directory / f"gpu-{token}.lock").open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    acquired = False
                    break
                handles.append(handle)
            if acquired:
                break
            for handle in reversed(handles):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            handles = []
            if time.monotonic() >= deadline:
                payload.update(status="timeout", finished_at=_timestamp())
                if ledger is not None:
                    _atomic_json(ledger, payload)
                raise TimeoutError(
                    f"Timed out waiting {timeout_seconds}s for GPU lease {canonical}"
                )
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

        holder = {
            "schema_version": 1,
            "pid": os.getpid(),
            "gpu_tokens": list(canonical),
            "purpose": purpose,
            "acquired_at": _timestamp(),
        }
        encoded = json.dumps(holder, sort_keys=True) + "\n"
        for handle in handles:
            handle.seek(0)
            handle.truncate()
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        payload.update(status="acquired", acquired_at=holder["acquired_at"])
        if ledger is not None:
            _atomic_json(ledger, payload)
        yield
    except BaseException as error:
        if payload.get("status") != "timeout":
            payload.update(
                status="error",
                finished_at=_timestamp(),
                error=f"{type(error).__name__}: {error}",
            )
            if ledger is not None:
                _atomic_json(ledger, payload)
        raise
    else:
        payload.update(status="released", finished_at=_timestamp())
        if ledger is not None:
            _atomic_json(ledger, payload)
    finally:
        for handle in reversed(handles):
            try:
                handle.seek(0)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
