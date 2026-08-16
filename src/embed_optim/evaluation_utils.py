from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

FAST_PLAID_INDEX_KWARGS = {
    "override": True,
    "nbits": 4,
    "n_ivf_probe": 8,
    "n_full_scores": 8192,
    "seed": 42,
}


def find_result_json(search_root: str | Path, task_name: str) -> Path | None:
    """Return the deterministic last matching MTEB task result, if one exists."""

    root = Path(search_root)
    matches = sorted(root.rglob(f"{task_name}.json")) if root.is_dir() else []
    return matches[-1] if matches else None


def task_result_remaining(
    search_root: str | Path,
    task_name: str,
    subsets: list[str],
    splits: list[str],
) -> bool:
    """Return whether any expected MTEB split/subset result is absent."""

    match = find_result_json(search_root, task_name)
    if match is None:
        return True
    try:
        scores = json.loads(match.read_text()).get("scores", {})
    except (json.JSONDecodeError, OSError):
        return True
    needed = set(subsets)
    return any(
        not needed.issubset(
            {row.get("hf_subset") for row in scores.get(split, []) if isinstance(row, dict)}
        )
        for split in splits
    )


def late_ipc_result_path(
    model_name: str,
    task_name: str,
    hf_subset: str,
    hf_split: str,
) -> str:
    """Return a rank-shared IPC path unique to one model/task evaluation."""

    identity = json.dumps(
        [model_name, task_name, hf_subset, hf_split],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return os.path.join(tempfile.gettempdir(), f"mteb_plaid_results_{digest}.pkl")
