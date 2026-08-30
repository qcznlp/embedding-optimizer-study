from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .config import ModelFamily, RunConfig, resolve_matrix_path

ALL_FAMILIES: tuple[ModelFamily, ...] = ("dense", "late")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_families(families: Iterable[str]) -> tuple[ModelFamily, ...]:
    requested = set(families)
    unknown = requested - set(ALL_FAMILIES)
    if unknown or not requested:
        raise ValueError(f"Invalid model-family scope: {sorted(requested)}")
    return tuple(family for family in ALL_FAMILIES if family in requested)


def load_scope_amendment(
    path: str | Path,
    *,
    families: Iterable[str],
) -> tuple[Path, dict[str, Any]]:
    requested = normalize_families(families)
    resolved = resolve_matrix_path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    active = tuple(payload.get("active_scope", {}).get("families") or ())
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "user_directed_post_hoc_scope_amendment"
        or active != requested
        or not payload.get("claim_boundary")
    ):
        raise ValueError("Scope amendment does not authorize the requested model families")
    repository = resolved.parent.parent
    if not (repository / "pyproject.toml").is_file():
        raise ValueError(f"Scope amendment is outside a repository configs directory: {resolved}")
    bindings = payload.get("source_bindings") or []
    for binding in bindings:
        source = (repository / str(binding.get("path", ""))).resolve()
        if not source.is_file() or _sha256(source) != binding.get("sha256"):
            raise ValueError(f"Scope-amendment source differs: {source}")
    if len(bindings) != 6:
        raise ValueError("Scope amendment source ledger is incomplete")
    return resolved, payload


def resolve_scope(
    families: Iterable[str],
    amendment: str | Path | None,
) -> tuple[tuple[ModelFamily, ...], dict[str, Any] | None]:
    requested = normalize_families(families)
    if requested == ALL_FAMILIES:
        if amendment is not None:
            raise ValueError("A scope amendment cannot be applied to the original two-family scope")
        return requested, None
    if amendment is None:
        raise ValueError("A reduced family scope requires --scope-amendment")
    path, payload = load_scope_amendment(amendment, families=requested)
    return requested, {
        "path": str(path),
        "sha256": _sha256(path),
        "status": payload["status"],
        "amended_at_utc": payload["amended_at_utc"],
        "claim_boundary": payload["claim_boundary"],
    }


def select_family_configs(configs: Iterable[RunConfig], families: Iterable[str]) -> list[RunConfig]:
    requested = set(normalize_families(families))
    selected = [config for config in configs if config.model_family in requested]
    if not selected or {config.model_family for config in selected} != requested:
        raise ValueError("Training matrix does not cover every requested model family")
    return selected
