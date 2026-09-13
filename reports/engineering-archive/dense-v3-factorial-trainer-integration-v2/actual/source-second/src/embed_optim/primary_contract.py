"""Shared strict input boundary for the revised primary train/backup/eval chain.

Draft contracts support read-only inspection only. State-changing consumers must
require a released contract and exact committed source; none of the older locks
or scientific artifacts is modified or automatically adopted by this module.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCOPE = "dense_primary_correctness_v2"
INPUT_SHA = "02561536a7ac691290f127dc07ddef41cede4b15e9bfbabb97eee0b569a0fa59"
DRAFT = "prepared_not_execution_authorized"
RELEASED = "reviewed_primary_execution_lock"
RUN_RECEIPT = "dense_run_contract.json"
SEAL = "dense_checkpoint_seal.json"
STEPS = (782, 1563, 2345, 3126, 3907)
RATES = {
    "adamw": (1e-6, 3e-6, 1e-5, 3e-5),
    "muon": (1e-4, 3e-4, 1e-3, 3e-3),
    "normuon": (1e-4, 3e-4, 1e-3, 3e-3),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing ordinary contract file: {path.name}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    canonical(value)
    return value


def file_identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Require an ordinary source/payload file")
    before = path.stat()
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if any(
        getattr(before, k) != getattr(after, k)
        for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("File changed while checking identity")
    return {"bytes": after.st_size, "sha256": sha256}


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Require a nonempty canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError("Path escapes its declared source/payload root")
    return Path(value)


def verify_file(path, expected):
    actual = file_identity(path)
    if actual != {k: expected[k] for k in ("bytes", "sha256")}:
        raise ValueError(f"File content identity differs: {Path(path).name}")
    return actual


def committed_file(root, relative, expected):
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"], capture_output=True, check=False
    )
    if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected["sha256"]:
        raise ValueError("Formal execution requires the exact committed source and contract")


def require_same(actual, expected):
    if canonical(actual) != canonical(expected):
        raise ValueError("Artifact belongs to a different primary identity")


def inspect_sealed_checkpoint(checkpoint, expected, step):
    """Pure local verifier, also usable with explicitly diagnostic expected identities."""
    checkpoint = Path(checkpoint)
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_dir()
        or checkpoint.name != f"checkpoint-{step}"
    ):
        raise ValueError("Checkpoint directory and requested step disagree")
    full = read_json(checkpoint / RUN_RECEIPT)
    require_same(full, expected)
    require_same(
        read_json(checkpoint / "dense_numerical_contract.json"), expected["numerical_contract"]
    )
    seal = read_json(checkpoint / SEAL)
    if set(seal) != {"schema_version", "step", "run_identity_sha256", "files"}:
        raise ValueError("Checkpoint lacks a complete content seal")
    if (
        type(seal["step"]) is not int
        or seal["step"] != step
        or seal["schema_version"] != 1
        or seal["run_identity_sha256"] != digest(expected)
    ):
        raise ValueError("Checkpoint step/identity seal disagrees")
    rows = seal["files"]
    names = [str(relative_path(row["path"])) for row in rows]
    if names != sorted(set(names)) or SEAL in names:
        raise ValueError("Invalid or duplicate checkpoint inventory")
    world = expected["execution"]["world_size"]
    rng = {"rng_state.pth"} if world == 1 else {f"rng_state_{r}.pth" for r in range(world)}
    required = {
        RUN_RECEIPT,
        "dense_numerical_contract.json",
        "config.json",
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        *rng,
    }
    if not required.issubset(names):
        raise ValueError("Incomplete checkpoint, including rank state")
    files = sorted(checkpoint.rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("Checkpoint inventory may not follow symlinks")
    actual_names = sorted(
        p.relative_to(checkpoint).as_posix() for p in files if p.is_file() and p.name != SEAL
    )
    if actual_names != names:
        raise ValueError("Checkpoint contains missing or unsealed files")
    for row in rows:
        verify_file(checkpoint / relative_path(row["path"]), row)
    if files != sorted(checkpoint.rglob("*")):
        raise ValueError("Checkpoint changed during inspection")
    state = read_json(checkpoint / "trainer_state.json")
    if type(state.get("global_step")) is not int or state["global_step"] != step:
        raise ValueError("Actual Trainer state and requested stage differ")
    return {
        "step": step,
        "run_identity_sha256": digest(expected),
        "checkpoint_seal": file_identity(checkpoint / SEAL),
        "files": [*rows, {"path": SEAL, **file_identity(checkpoint / SEAL)}],
    }


@dataclass(frozen=True)
class PrimaryContract:
    path: Path
    repository: Path
    training_root: Path
    payload: dict
    inputs: dict
    sha256: str

    @classmethod
    def load(cls, path, repository, training_root, *, require_released=False):
        path, repository, training_root = (
            Path(path).resolve(),
            Path(repository).resolve(),
            Path(training_root).resolve(),
        )
        payload = read_json(path)
        if (
            payload.get("scope") != SCOPE
            or payload.get("schema_version") != 1
            or payload.get("status") not in {DRAFT, RELEASED}
        ):
            raise ValueError("Not the revised primary execution contract")
        if require_released and payload["status"] != RELEASED:
            raise ValueError("Prepared primary contract is not execution authorized")
        if require_released and repository != training_root:
            raise ValueError("Formal execution requires one assembled committed source checkout")
        binding = payload["input_bindings"]
        if binding.get("sha256") != INPUT_SHA:
            raise ValueError("Primary input receipt is not the independently audited snapshot")
        input_path = repository / relative_path(binding["path"])
        verify_file(input_path, binding)
        inputs = read_json(input_path)
        if (
            inputs.get("scope") != "prepared_dense_primary_actual_input_bindings"
            or inputs.get("preparation_passed") is not True
        ):
            raise ValueError("Require the actual primary input audit, not diagnostic outputs")
        if inputs["common_identity"]["data"]["rows"] != 500000:
            raise ValueError("Primary data must contain the original 500k queries")
        if (
            inputs["initial_model"]["repo"] != "lightonai/DenseOn-unsupervised"
            or inputs["initial_model"]["revision"] != "0edbd55684eb782bce55ee74c95b25c97cbe7f43"
        ):
            raise ValueError("Wrong immutable untrained base")
        if payload["checkpoint_steps"] != list(STEPS) or payload["world_size"] != 4:
            raise ValueError("Primary schedule/world size changed")
        require_same(payload["evaluation"], inputs["original_evaluation"])
        require_same(
            payload["analysis"],
            {k: v for k, v in inputs["original_analysis"].items() if k != "historical_bridge"},
        )
        for field in ("data_path", "output_root", "checkpoint_prefix"):
            relative_path(payload[field])
        if len(inputs["runs"]) != 12 or len({r["run_id"] for r in inputs["runs"]}) != 12:
            raise ValueError("Require all twelve distinct primary recipes")
        expected_rates = {(name, rate) for name, values in RATES.items() for rate in values}
        observed_rates = {
            (
                r["identity_fields"]["recipe"]["optimizer"]["name"],
                r["identity_fields"]["recipe"]["optimizer"]["lr"],
            )
            for r in inputs["runs"]
        }
        if observed_rates != expected_rates or any(
            not r["run_id"].startswith("verified-") for r in inputs["runs"]
        ):
            raise ValueError("Primary optimizer/rate grid or run namespace differs")
        for group, base in (("training_sources", training_root), ("consumer_sources", repository)):
            if not payload[group]:
                raise ValueError("Source closure is missing")
            for relative, expected in payload[group].items():
                source = base / relative_path(relative)
                verify_file(source, expected)
                if require_released:
                    committed_file(base, relative, expected)
        for row in inputs["common_identity"]["source"]["local_files"]:
            relative = f"src/embed_optim/{row['path']}"
            if payload["training_sources"].get(relative) != {
                k: row[k] for k in ("bytes", "sha256")
            }:
                raise ValueError("Training closure omits a run-identity source")
        required_consumers = {
            "src/embed_optim/primary_contract.py",
            "src/embed_optim/primary_io.py",
            "src/embed_optim/primary_training.py",
        }
        if not required_consumers.issubset(payload["consumer_sources"]):
            raise ValueError("Primary consumer closure is incomplete")
        package = Path(__file__).resolve().parent
        for name in ("primary_contract.py", "primary_io.py", "primary_training.py"):
            verify_file(package / name, payload["consumer_sources"][f"src/embed_optim/{name}"])
        if require_released:
            committed_file(repository, path.relative_to(repository).as_posix(), file_identity(path))
            committed_file(repository, input_path.relative_to(repository).as_posix(), binding)
        result = cls(
            path, repository, training_root, payload, inputs, file_identity(path)["sha256"]
        )
        for run in inputs["runs"]:
            result.expected_identity(run["run_id"])
        return result

    def expected_identity(self, run_id):
        rows = [row for row in self.inputs["runs"] if row["run_id"] == run_id]
        if len(rows) != 1:
            raise ValueError("Unknown or ambiguous primary run; old run IDs are ineligible")
        row = rows[0]
        expected = {**self.inputs["common_identity"], **row["identity_fields"]}
        if (
            digest(expected) != row["expected_identity_sha256"]
            or expected["recipe"]["run_id"] != run_id
        ):
            raise ValueError("Primary expected identity is internally inconsistent")
        return json.loads(canonical(expected))

    def checkpoint(self, checkpoint, run_id, step):
        if type(step) is not int or step not in STEPS:
            raise ValueError("Checkpoint is not one of the five primary stages")
        value = inspect_sealed_checkpoint(checkpoint, self.expected_identity(run_id), step)
        return {"scope": SCOPE, "protocol_sha256": self.sha256, "run_id": run_id, **value}

    def require_execution(self):
        # Re-read and re-check immediately before a state-changing consumer acts.
        current = type(self).load(
            self.path, self.repository, self.training_root, require_released=True
        )
        if current.sha256 != self.sha256:
            raise ValueError("Primary contract changed since admission")
        return current
