from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any

FAST_PLAID_INDEX_KWARGS = {
    "override": True,
    "nbits": 4,
    "n_ivf_probe": 8,
    "n_full_scores": 8192,
    "seed": 42,
}


def configure_atomic_mteb_results() -> None:
    """Make MTEB result and sidecar writes atomic across evaluator workers."""

    import fcntl

    import mteb.cache.result_cache as result_cache_module
    from mteb.cache import ResultCache
    from mteb.models.model_meta import ModelMeta
    from mteb.results.task_result import TaskResult

    if not getattr(TaskResult, "_embed_optim_atomic_to_disk", False):
        original_to_disk = TaskResult.to_disk

        @wraps(original_to_disk)
        def atomic_to_disk(self, path: Path) -> None:
            path = Path(path)
            temporary = path.with_name(f".{path.stem}.{os.getpid()}.{id(self)}.tmp{path.suffix}")
            try:
                original_to_disk(self, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

        TaskResult.to_disk = atomic_to_disk
        TaskResult._embed_optim_atomic_to_disk = True

    if not getattr(result_cache_module, "_embed_optim_atomic_run_settings", False):
        original_merge = result_cache_module._write_and_merge_keyed_json

        @wraps(original_merge)
        def locked_atomic_merge(path: Path, entries: list[dict[str, Any]]) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = path.with_name(f".{path.name}.lock")
            with lock_path.open("a+", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                temporary = path.with_name(
                    f".{path.stem}.{os.getpid()}.{id(entries)}.tmp{path.suffix}"
                )
                try:
                    if path.is_file():
                        shutil.copyfile(path, temporary)
                    original_merge(temporary, entries)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        result_cache_module._write_and_merge_keyed_json = locked_atomic_merge
        result_cache_module._embed_optim_atomic_run_settings = True

    if not getattr(ResultCache, "_embed_optim_atomic_model_meta", False):
        original_save = ResultCache.save_to_cache

        @wraps(original_save)
        def atomic_sidecar_save(
            self,
            task_result,
            model_name,
            model_revision=None,
            *,
            encode_kwargs=None,
        ) -> None:
            if not isinstance(model_name, ModelMeta) or model_name.experiment_name is not None:
                return original_save(
                    self,
                    task_result,
                    model_name,
                    model_revision,
                    encode_kwargs=encode_kwargs,
                )

            result_path = self.get_task_result_path(
                model_name=model_name,
                model_revision=model_revision,
                task_name=task_result.task_name,
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path = result_path.parent / "model_meta.json"
            temporary = meta_path.with_name(
                f".{meta_path.stem}.{os.getpid()}.{id(model_name)}.tmp{meta_path.suffix}"
            )
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(model_name.to_dict(), handle, default=str, indent=4)
                temporary.replace(meta_path)
            finally:
                temporary.unlink(missing_ok=True)

            # Passing the equivalent string identity keeps MTEB from performing
            # its own non-atomic model_meta.json write. Our study models never use
            # experiment_kwargs, so both path calculations are identical.
            return original_save(
                self,
                task_result,
                model_name.name,
                model_name.revision,
                encode_kwargs=encode_kwargs,
            )

        ResultCache.save_to_cache = atomic_sidecar_save
        ResultCache._embed_optim_atomic_model_meta = True


def disable_mteb_cache_writes(cache: Any) -> None:
    """Make a ResultCache read-only on non-main distributed evaluator ranks."""

    def no_op_save(*args, **kwargs) -> None:
        return None

    cache.save_to_cache = no_op_save


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
