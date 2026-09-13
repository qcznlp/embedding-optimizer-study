"""Content-bound run identity and checkpoint admission for corrected Dense runs.

These digests detect configuration drift and accidental payload substitution.
They are not signatures: untrusted remote downloads still require the independent
trusted-digest backup audit before this local producer/consumer contract is used.
All admission reads precede Trainer/device setup, seeding and output writes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import stat
from pathlib import Path

from datasets import Dataset
from huggingface_hub import snapshot_download

from . import dense_numerical_contract as numerical
from .collators import TEXT_COLUMNS

RECEIPT_NAME = "dense_run_contract.json"
SEAL_NAME = "dense_checkpoint_seal.json"
SCOPE = "content_bound_corrected_dense_run"
SOURCE_FILES = (
    "__init__.py",
    "train.py",
    "config.py",
    "losses.py",
    "collators.py",
    "optimizers.py",
    "callbacks.py",
    "dense_numerical_contract.py",
    "dense_run_contract.py",
)
PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "sentence-transformers",
    "datasets",
    "numpy",
    "safetensors",
    "flash-attn",
    "huggingface-hub",
    "tokenizers",
)
RELOCATABLE_CONFIG = {"output_root", "dataset_path", "wandb_project", "wandb_entity"}
NONSCIENTIFIC_ARGUMENTS = {"output_dir", "project", "run_name", "report_to"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing ordinary contract file: {path.name}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate contract JSON key")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    canonical(value)  # No NaN/Inf, even when the parser accepts them.
    return value


def file_identity(path, *, allow_cache_link=False):
    path = Path(path)
    if path.is_symlink() and not allow_cache_link:
        raise ValueError("Identity may not follow a symlink")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Identity requires a regular file")
    with path.open("rb") as handle:
        value = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    if any(
        getattr(before, k) != getattr(after, k)
        for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("Input changed while hashing")
    return {"bytes": after.st_size, "sha256": value}


def inventory(directory, *, exclude=(), cache_links=False):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Require an ordinary input directory")
    paths = sorted(directory.rglob("*"))
    result = []
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        if relative in exclude:
            continue
        if path.is_symlink():
            if not cache_links or not path.is_file():
                raise ValueError("Input directory may not contain symlinks")
            # HF snapshots use only regular blob links inside the model cache.
            blob_root = directory.parent.parent / "blobs"
            if path.resolve().parent != blob_root.resolve():
                raise ValueError("Model cache link leaves its blob directory")
        elif path.is_dir():
            continue
        result.append({"path": relative, **file_identity(path, allow_cache_link=cache_links)})
    if paths != sorted(directory.rglob("*")):
        raise ValueError("Directory changed while hashing")
    if not result:
        raise ValueError("Empty input inventory")
    return result


def recipe_identity(config):
    recipe = json.loads(canonical(config.as_dict()))
    for key in RELOCATABLE_CONFIG:
        recipe.pop(key, None)
    # A local base can relocate as well; its complete bytes are bound separately.
    if Path(config.model_name).is_dir():
        if config.model_revision is not None:
            raise ValueError("Local model sources cannot claim a remote revision")
        recipe["model_name"] = {"kind": "local_content_bound_model"}
    elif not re.fullmatch(r"[0-9a-f]{40}", config.model_revision or ""):
        raise ValueError("Remote model requires an immutable full commit revision")
    return recipe


def execution_identity(argument_values, world_size):
    values = {k: v for k, v in argument_values.items() if k not in NONSCIENTIFIC_ARGUMENTS}
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Require a positive declared world size")
    return json.loads(
        canonical(
            {
                "training_arguments": values,
                "world_size": world_size,
                "environment": {
                    k: os.environ.get(k)
                    for k in (
                        "CUBLAS_WORKSPACE_CONFIG",
                        "NVIDIA_TF32_OVERRIDE",
                        "FLASH_ATTENTION_DETERMINISTIC",
                    )
                },
            }
        )
    )


def source_identity():
    root = Path(__file__).resolve().parent
    return {
        "local_files": [{"path": name, **file_identity(root / name)} for name in SOURCE_FILES],
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "numerical_upstream": numerical.SOURCE_HASHES,
    }


def require_arguments(arguments, expected):
    observed = {key: getattr(arguments, key) for key in expected["training_arguments"]}
    require_equal(observed, expected["training_arguments"])
    if arguments.world_size != expected["world_size"]:
        raise ValueError("Observed and declared world sizes differ")


def read_identity(directory):
    value = load_json(Path(directory) / RECEIPT_NAME)
    if (
        not isinstance(value, dict)
        or value.get("scope") != SCOPE
        or value.get("schema_version") != 1
    ):
        raise ValueError("Checkpoint lacks this complete run identity; do not retrofit old state")
    if set(value) != {
        "schema_version",
        "scope",
        "recipe",
        "execution",
        "source",
        "data",
        "model",
        "numerical_contract",
    }:
        raise ValueError("Incomplete full run identity")
    return value


def require_equal(saved, expected):
    if canonical(saved) != canonical(expected):
        raise ValueError("Run/data/model/execution/source identity differs; not a continuation")


def prepare(config, argument_values, world_size, resume_from_checkpoint=None):
    """Read actual inputs and check admission without setup, writes or model loading."""
    expected = {
        "schema_version": 1,
        "scope": SCOPE,
        "recipe": recipe_identity(config),
        "execution": execution_identity(argument_values, world_size),
        "source": source_identity(),
        "numerical_contract": numerical.receipt(
            config.optimizer, argument_values["gradient_accumulation_steps"]
        ),
    }
    saved = read_identity(resume_from_checkpoint) if resume_from_checkpoint else None
    if saved is not None:
        for key, value in expected.items():
            require_equal(saved[key], value)

    data_root = Path(config.dataset_path)
    data_path = data_root / "dataset" if (data_root / "dataset").is_dir() else data_root
    files = inventory(data_path)
    dataset = Dataset.load_from_disk(str(data_path))
    columns = [*TEXT_COLUMNS, "length"]
    missing = [c for c in columns if c not in dataset.column_names]
    if missing or len(dataset) < 1:
        raise ValueError(f"Dataset is empty or missing required columns: {missing}")
    dataset = dataset.select_columns(columns)
    expected["data"] = {"files": files, "rows": len(dataset), "selected_columns": columns}
    if data_path != data_root:
        manifest = data_root / "manifest.json"
        if manifest.is_file() or manifest.is_symlink():
            expected["data"]["materialization_manifest"] = file_identity(manifest)
    # The immutable initial model must already be present for read-only admission.
    # On a new host, populate the pinned cache as a separate verified download step.
    local_model = Path(config.model_name)
    remote = not local_model.is_dir()
    if remote:
        local_model = Path(
            snapshot_download(
                config.model_name, revision=config.model_revision, local_files_only=True
            )
        )
        if local_model.name != config.model_revision:
            raise ValueError("Cached model does not resolve to the requested immutable revision")
    expected["model"] = {"files": inventory(local_model, cache_links=remote)}
    expected = json.loads(canonical(expected))
    if saved is not None:
        require_equal(saved, expected)
        require_checkpoint(resume_from_checkpoint, expected)
    require_output(config.output_dir, expected, resume_from_checkpoint)
    # Check the mapped data still denotes the exact files just observed.
    if inventory(data_path) != files:
        raise ValueError("Dataset changed during admission")
    return expected, dataset


def require_output(directory, expected, checkpoint=None):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Output cannot be symlinked")
    if not directory.exists() or not any(directory.iterdir()):
        return
    if checkpoint is None:
        raise ValueError("Fresh run cannot overwrite existing output")
    require_equal(read_identity(directory), expected)
    if (directory / "completed.json").exists() or (directory / "final").exists():
        raise ValueError("Completed/final output must not be overwritten by continuation")
    step = load_json(Path(checkpoint) / SEAL_NAME)["step"]
    for child in directory.glob("checkpoint-*"):
        if not re.fullmatch(r"checkpoint-[0-9]+", child.name) or int(child.name[11:]) > step:
            raise ValueError("Continuation would overwrite later or ambiguous checkpoint evidence")
        require_checkpoint(child, expected)


def write_identity(directory, expected):
    path = Path(directory) / RECEIPT_NAME
    if path.exists() or path.is_symlink():
        require_equal(read_identity(directory), expected)
        return
    with path.open("x") as handle:
        handle.write(json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n")


def seal_checkpoint(directory, expected, step):
    directory = Path(directory)
    if type(step) is not int or step < 1:
        raise ValueError("Require an actual positive checkpoint step")
    write_identity(directory, expected)
    rows = inventory(directory, exclude=(SEAL_NAME,))
    _require_payload_names(rows, expected)
    value = {
        "schema_version": 1,
        "step": step,
        "run_identity_sha256": digest(expected),
        "files": rows,
    }
    with (directory / SEAL_NAME).open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _require_payload_names(rows, expected):
    world = expected["execution"]["world_size"]
    rng = {"rng_state.pth"} if world == 1 else {f"rng_state_{rank}.pth" for rank in range(world)}
    required = {
        RECEIPT_NAME,
        numerical.RECEIPT_NAME,
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        *rng,
    }
    if not required.issubset({x["path"] for x in rows}):
        raise ValueError("Checkpoint is incomplete, including per-rank state")


def require_checkpoint(directory, expected):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Require an ordinary checkpoint directory")
    require_equal(read_identity(directory), expected)
    numerical.require_resume_receipt(directory, expected["numerical_contract"])
    seal = load_json(directory / SEAL_NAME)
    if (
        set(seal) != {"schema_version", "step", "run_identity_sha256", "files"}
        or seal["schema_version"] != 1
    ):
        raise ValueError("Incomplete checkpoint seal")
    if (
        type(seal["step"]) is not int
        or seal["step"] < 1
        or seal["run_identity_sha256"] != digest(expected)
    ):
        raise ValueError("Checkpoint seal belongs to a different run")
    if directory.name != f"checkpoint-{seal['step']}":
        raise ValueError("Checkpoint directory and sealed step differ")
    rows = inventory(directory, exclude=(SEAL_NAME,))
    _require_payload_names(rows, expected)
    require_equal(seal["files"], rows)
    state = load_json(directory / "trainer_state.json")
    if state.get("global_step") != seal["step"]:
        raise ValueError("Trainer state and checkpoint seal step differ")
    return seal
