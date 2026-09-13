"""Versioned primary admission for revised data; the v2 contract remains immutable.

This is a new contract, not a permissive adapter for historical identities. Reading
its draft is allowed. Every state-changing consumer rechecks release and committed
source before importing the training stack or creating an output.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from .primary_contract import (
    DRAFT,
    RELEASED,
    STEPS,
    PrimaryContract,
    committed_file,
    digest,
    file_identity,
    inspect_sealed_checkpoint,
    read_json,
    relative_path,
    require_same,
    verify_file,
)

SCOPE = "dense_primary_correctness_v3"
AMENDMENT_SHA = "5051ec40c7e44ba677980fec422ed4a49ef851ef3e9fff9c29cc0f6fc3d8e4e1"
INPUT_SHA = "40b7d3e63b1c4a17b995507d5d96c042980acca96baa08f6c0c52dda66b1f1f9"
NATURAL_SHA = "1e914d47a0e00ad6e7193445dc5817ab3455963d7261c5de9a0a93ce290e93d5"
PARENTS = {
    "data_amendment": ("configs/dense_primary_v3_data_amendment.json", AMENDMENT_SHA),
    "input_bindings": ("reports/dense-primary-v3/input-bindings.json", INPUT_SHA),
    "natural_acceptance": (
        "reports/engineering-archive/dense-revised-natural-v1/validation.json",
        NATURAL_SHA,
    ),
    "primary_parent": (
        "configs/dense_primary_v2_protocol.json",
        "e21a7226c09740d38d49855738267c080f615f8d4f60c2abf9dbf3ccd874edd2",
    ),
}
ADDITIONAL_CONSUMERS = (
    "src/embed_optim/primary_completion.py",
    "src/embed_optim/primary_v3_contract.py",
    "src/embed_optim/primary_v3_training.py",
    "src/embed_optim/primary_v3_io.py",
    "src/embed_optim/primary_v3_completion.py",
    "scripts/prepare_dense_v3_protocol.py",
)
PATHS = {
    "data_path": "data/denseon-sft-500k-v3",
    "validation_path": "data/validation-4096-v3",
    "output_root": "outputs/dense-correctness-v3",
    "checkpoint_prefix": "corrected-dense-correctness-v3/dense",
}


def expected_evaluation(parent):
    value = copy.deepcopy(parent["evaluation"])
    value["validation_selection"]["dataset"] = PATHS["validation_path"]
    return value


def dataset_inventory(amendment, partition):
    """Relocation-safe content inventory, without relabelling an old parent dataset."""
    value = amendment["datasets"][partition]
    original = Path(value["root"])
    rows = [{"path": p["path"], "bytes": p["bytes"], "sha256": p["sha256"]} for p in value["files"]]
    for row in rows:
        row["path"] = Path(row["path"]).relative_to(original).as_posix()
        relative_path(row["path"])
    return {"rows": value["rows"], "files": sorted(rows, key=lambda r: r["path"])}


@dataclass(frozen=True)
class PrimaryV3Contract(PrimaryContract):
    @classmethod
    def load(cls, path, repository, training_root, *, require_released=False):
        path, repository, training_root = (
            Path(p).resolve() for p in (path, repository, training_root)
        )
        payload = read_json(path)
        if (
            payload.get("scope") != SCOPE
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
            or payload.get("status") not in {DRAFT, RELEASED}
            or payload.get("scientific_completion") is not False
        ):
            raise ValueError("Not the versioned v3 primary contract")
        if require_released and payload["status"] != RELEASED:
            raise ValueError("Prepared v3 primary contract is not execution authorized")
        if require_released and repository != training_root:
            raise ValueError("Formal execution requires one assembled committed source checkout")
        trusted = {}
        for key, (name, sha) in PARENTS.items():
            binding = payload[key]
            if binding.get("path") != name or binding.get("sha256") != sha:
                raise ValueError("A v3 immutable parent binding differs")
            verify_file(repository / name, binding)
            trusted[key] = read_json(repository / name)
            if require_released:
                committed_file(repository, name, binding)
        amendment, inputs, parent = (
            trusted[k] for k in ("data_amendment", "input_bindings", "primary_parent")
        )
        acceptance = trusted["natural_acceptance"]
        if (
            inputs["scope"] != "prepared_dense_revised_primary_actual_inputs_v3"
            or inputs["preparation_passed"] is not True
            or acceptance["natural_data_entrypoint_passed"] is not True
            or acceptance["artifact_validation_passed"] is not True
            or acceptance["formal_replication_ready"] is not False
        ):
            raise ValueError("The required preparation evidence is absent or mis-scoped")
        for key, value in PATHS.items():
            if payload[key] != value:
                raise ValueError("V3 input/output namespaces differ from the declared revision")
        require_same(payload["evaluation"], expected_evaluation(parent))
        require_same(payload["analysis"], parent["analysis"])
        for key in (
            "world_size",
            "checkpoint_steps",
            "beir_task_revisions",
            "gpu_pools",
            "gpu_lease_root",
            "wandb",
            "checkpoint_repository",
            "worker_python",
            "training_sources",
        ):
            require_same(payload[key], parent[key])
        for partition in ("training", "validation"):
            require_same(payload["datasets"][partition], dataset_inventory(amendment, partition))
        wanted_consumers = set(parent["consumer_sources"]) | set(ADDITIONAL_CONSUMERS)
        if set(payload["consumer_sources"]) != wanted_consumers:
            raise ValueError("The v3 consumer closure is incomplete or differs")
        for name, old in parent["consumer_sources"].items():
            require_same(payload["consumer_sources"][name], old)
        for group, base in (("training_sources", training_root), ("consumer_sources", repository)):
            for name, binding in payload[group].items():
                verify_file(base / relative_path(name), binding)
                if require_released:
                    committed_file(base, name, binding)
        package = Path(__file__).resolve().parent
        for name in (
            "primary_v3_contract.py",
            "primary_v3_training.py",
            "primary_v3_io.py",
            "primary_v3_completion.py",
            "primary_contract.py",
            "primary_completion.py",
        ):
            verify_file(package / name, payload["consumer_sources"][f"src/embed_optim/{name}"])
        if require_released:
            committed_file(repository, path.relative_to(repository).as_posix(), file_identity(path))
        result = cls(
            path, repository, training_root, payload, inputs, file_identity(path)["sha256"]
        )
        if len(inputs["runs"]) != 12:
            raise ValueError("Require the twelve revised primary identities")
        for row in inputs["runs"]:
            expected = result.expected_identity(row["run_id"])
            if not row["run_id"].startswith("verified-v3-") or expected["data"]["rows"] != 500000:
                raise ValueError("A v3 primary identity is missing or historical")
        return result

    def checkpoint(self, checkpoint, run_id, step):
        if type(step) is not int or step not in STEPS:
            raise ValueError("Checkpoint is not one of the five v3 primary stages")
        checked = inspect_sealed_checkpoint(checkpoint, self.expected_identity(run_id), step)
        return {"scope": SCOPE, "protocol_sha256": self.sha256, "run_id": run_id, **checked}

    def dataset(self, root, partition):
        """Accept exact byte copies on another host; never alter or regenerate inputs."""
        if partition not in {"training", "validation"}:
            raise ValueError("Unknown v3 data partition")
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Require an ordinary data directory")
        expected = self.payload["datasets"][partition]
        files = sorted(root.rglob("*"))
        if any(p.is_symlink() for p in files):
            raise ValueError("V3 datasets may not follow symlinks")
        names = sorted(p.relative_to(root).as_posix() for p in files if p.is_file())
        require_same(names, [row["path"] for row in expected["files"]])
        for row in expected["files"]:
            verify_file(root / relative_path(row["path"]), row)
        if files != sorted(root.rglob("*")):
            raise ValueError("Dataset changed during admission")
        return {"partition": partition, "content_sha256": digest(expected), **expected}

    def complete_run(self, run_root, run_id):
        from .primary_completion import inspect_complete_run

        checked = inspect_complete_run(run_root, self.expected_identity(run_id), list(STEPS))
        if (
            checked["dataset_fingerprint"]
            != self.inputs["data_linkage_audit"]["training_view_fingerprint"]
        ):
            raise ValueError("Completed v3 run used another training view")
        return {"scope": SCOPE, "protocol_sha256": self.sha256, "run_id": run_id, **checked}
