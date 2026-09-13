"""Local numerical-component checkpoint integrity, NOT scientific run admission.

The externally supplied digest authenticates one complete local component save.
It cannot substitute for v3 source-state, calibration, data, runtime, remote-copy
or whole-experiment admission. No legacy/primary checkpoint is relabeled here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

NAME = "factorial_trainer_component.json"
SCOPE = "dense-v3-factorial-trainer-component-checkpoint-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Require an ordinary component checkpoint directory")
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Component checkpoint contains a link or special file")
        if path.is_file() and path.relative_to(directory).as_posix() != NAME:
            records.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_digest(path),
                }
            )
    return records


def require_state(directory, step, world_size):
    if type(step) is not int or not 1 <= step <= 391 or world_size != 4:
        raise ValueError("Invalid four-rank component checkpoint step")
    directory = Path(directory)
    state = json.loads((directory / "trainer_state.json").read_text())
    if (
        type(state.get("global_step")) is not int
        or state["global_step"] != step
        or state.get("max_steps") != 391
        or state.get("num_train_epochs") != 1
        or state.get("train_batch_size") != 8
    ):
        raise ValueError("Component Trainer counters differ from its fixed horizon")
    required = {
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        "modules.json",
        "trainer_state.json",
        *(f"rng_state_{rank}.pth" for rank in range(4)),
    }
    paths = {row["path"] for row in inventory(directory)}
    if not required <= paths or not any(p.endswith(".safetensors") for p in paths):
        raise ValueError("Incomplete component model/optimizer/scheduler/four-rank RNG payload")


def seal(directory, identity, step):
    directory = Path(directory)
    require_state(directory, step, 4)
    payload = {
        "schema_version": 1,
        "scope": SCOPE,
        "scientific_admission": False,
        "identity": identity,
        "step": step,
        "files": inventory(directory),
    }
    with (directory / NAME).open("x") as stream:
        stream.write(canonical(payload) + "\n")
    return {"path": str(directory.resolve()), "sha256": file_digest(directory / NAME)}


def read(binding, expected_identity):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Require an explicit external component path and digest")
    directory = Path(binding["path"])
    manifest = directory / NAME
    if directory.is_symlink() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Missing ordinary numerical-component checkpoint receipt")
    if file_digest(manifest) != binding["sha256"]:
        raise ValueError("Component checkpoint external digest differs")
    payload = json.loads(manifest.read_text())
    if (
        set(payload)
        != {"schema_version", "scope", "scientific_admission", "identity", "step", "files"}
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["scope"] != SCOPE
        or payload["scientific_admission"] is not False
        or canonical(payload["identity"]) != canonical(expected_identity)
        or payload["files"] != inventory(directory)
    ):
        raise ValueError("Component checkpoint identity or complete payload changed")
    require_state(directory, payload["step"], 4)
    # Detect replacement of the receipt while its payload was being checked.
    if file_digest(manifest) != binding["sha256"]:
        raise ValueError("Component receipt changed during verification")
    return payload
