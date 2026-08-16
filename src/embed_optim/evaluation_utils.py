from __future__ import annotations

import hashlib
import json
import os
import tempfile


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
