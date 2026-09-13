"""Undeployed full-recipe resume identity, for review before a new execution lock.

This module does not load a model, launch training, edit a checkpoint or make a
trust decision about user-supplied metadata. Its caller must authenticate the
source/configuration receipts and obtain data identities from actual files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

RELOCATABLE = ("output_root", "dataset_path", "wandb_project", "wandb_entity")
REQUIRED_RECIPE = {
    "run_id",
    "model_family",
    "model_name",
    "model_revision",
    "optimizer",
    "seed",
    "epochs",
    "global_batch_size",
    "micro_batch_size",
    "temperature",
    "max_length",
    "warmup_ratio",
    "max_grad_norm",
    "checkpoint_fractions",
    "numerical_policy",
}


def canonical(value):
    # Also rejects NaN/Inf and distinguishes true from 1, unlike ordinary dict equality.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def dataset_files(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Require an ordinary dataset directory")
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Dataset identity may not follow symlinks")
        if not path.is_file():
            continue
        before = path.stat()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = path.stat()
        if any(
            getattr(before, key) != getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        ):
            raise ValueError("Dataset changed while hashing")
        result.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": after.st_size,
                "sha256": digest,
            }
        )
    if not result:
        raise ValueError("Empty dataset identity")
    return result


def build_identity(run_config, data_identity, execution, numerical_contract, source_identity):
    recipe = json.loads(canonical(run_config))
    if not REQUIRED_RECIPE.issubset(recipe) or recipe["model_family"] != "dense":
        raise ValueError("Require the complete explicit Dense recipe")
    if recipe["numerical_policy"] != numerical_contract.get("numerical_policy"):
        raise ValueError("Numerical policy disagrees with the recipe")
    for key in RELOCATABLE:
        recipe.pop(key, None)
    if type(data_identity.get("rows")) is not int or data_identity["rows"] < 1:
        raise ValueError("Require a positive observed data row count")
    if not data_identity.get("selected_columns") or not data_identity.get("files"):
        raise ValueError("Require actual selected columns and content identities")
    names = set()
    for record in data_identity["files"]:
        name = PurePosixPath(record["path"])
        if name.is_absolute() or ".." in name.parts or name.as_posix() in names:
            raise ValueError("Invalid dataset-relative identity")
        names.add(name.as_posix())
        if (
            type(record.get("bytes")) is not int
            or record["bytes"] < 0
            or len(record.get("sha256", "")) != 64
        ):
            raise ValueError("Incomplete dataset file identity")
    if not execution or not source_identity:
        raise ValueError("Execution and source identity must be explicit")
    return json.loads(
        canonical(
            {
                "schema_version": 1,
                "scope": "prospective_dense_full_run_identity_proposal",
                "recipe": recipe,
                "data": data_identity,
                "execution": execution,
                "numerical_contract": numerical_contract,
                "source": source_identity,
            }
        )
    )


def require_identity(saved, requested):
    if canonical(saved) != canonical(requested):
        raise ValueError("Full run/data/execution/source identity differs; not a continuation")
