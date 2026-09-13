"""Actual corrected-state inputs for the fixed crossed continuation.

Read-only input admission, not execution authority or a scientific verdict. The
two independently accepted input audits remain immutable. Explicit location roles
permit content-identical copies without following their recorded absolute paths.
The historical protocols are never rewritten or accepted as v3 runtime locks.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

from .primary_contract import (
    digest,
    file_identity,
    inspect_sealed_checkpoint,
    read_json,
    relative_path,
    require_same,
    verify_file,
)

SCOPE = "dense-v3-factorial-genuine-inputs-v1"
EVIDENCE = "reports/engineering-archive/dense-v3-factorial-inputs-v1"
AUDITS = {
    "data-first.json": "94ea1b27631cf9270f445e6406e79c0c470bd346564a4057cee1f742923d103f",
    "states-first.json": "9958ecc4f23d9165838632c6bd7e24f1e64e3753eb5632206f2228411179e932",
    "verification.json": "9e92896fd14855ea1635f5571e824fc38c9071297e7fa3b4b848a55db08aa526",
}
STATES = {"adamw_state": "verified-v3-adamw-3e-5", "muon_state": "verified-v3-muon-3e-4"}
ORDER_SEEDS = (314159, 271828, 161803)
PRIMARY_SHA = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
CALIBRATION_CONFIG = {
    "family": "dense",
    "gradient_steps": 8,
    "examples_per_gradient": 4,
    "micro_batch_size": 1,
    "seed": 2718,
    "temperature": 0.02,
    "max_grad_norm": 1.0,
    "model_dtype": "float32",
    "forward_dtype": "bfloat16",
    "storage_dtype": "float32",
    "device": "cuda:0",
    "flash_attention": True,
    "model_mode": "train",
    "gradient_checkpointing": True,
    "parameter_partition": "hidden",
    "weights_advanced": False,
}


@dataclass(frozen=True)
class Locations:
    repository: Path
    primary_source: Path
    experiment: Path
    data_store: Path
    evidence: Path

    def locate(self, recorded):
        """Only the declared narrow roles; never fall back to an old host path."""
        original = Path(recorded)
        roots = (
            (Path("/root/embedding-optimizer-story-refactor") / EVIDENCE, self.evidence),
            (Path("/root/embedding-optimizer-story-refactor"), self.repository),
            (Path("/root/embedding-optimizer-primary-v3"), self.primary_source),
            (Path("/root/embedding-optimizer-v3-experiment"), self.experiment),
            (Path("/root/embedding-optimizer-study/data"), self.data_store),
        )
        for old, current in roots:
            if original.is_relative_to(old):
                relative = relative_path(original.relative_to(old).as_posix())
                root = Path(current).absolute()
                result = root / relative
                # Do not traverse symlinks in a supplied role or its descendants.
                if any(p.is_symlink() for p in (root, result, *result.parents)):
                    raise ValueError("Input role cannot follow a symlink")
                return result
        raise ValueError("Recorded input has no supplied local role")


def _fixed_audit(root, name):
    path = Path(root) / name
    if file_identity(path)["sha256"] != AUDITS[name]:
        raise ValueError("The accepted genuine input audit differs")
    return read_json(path)


def _source_binding(locations, row):
    run_id = STATES[row["label"]]
    checkpoint = locations.locate(row["checkpoint"])
    expected_path = (
        Path(locations.experiment)
        / "outputs/dense-correctness-v3/dense"
        / run_id
        / "checkpoint-2345"
    )
    if checkpoint != expected_path.absolute() or row["genuine_v3_run_id"] != run_id:
        raise ValueError("Source state is not the fixed genuine v3 checkpoint")
    identity = read_json(checkpoint / "dense_run_contract.json")
    native = inspect_sealed_checkpoint(checkpoint, identity, 2345)
    require_same(
        row["native_checkpoint"],
        {
            "scope": "dense_primary_correctness_v3",
            "protocol_sha256": PRIMARY_SHA,
            "run_id": run_id,
            **native,
        },
    )
    receipts = {}
    for key in ("original_durability_receipt", "original_remote_audit_receipt"):
        old = row[key]
        path = locations.locate(old["path"])
        verify_file(path, old)
        receipts[key] = {"bytes": old["bytes"], "sha256": old["sha256"]}
    return {
        "label": row["label"],
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "checkpoint_step": 2345,
        "native_checkpoint": copy.deepcopy(row["native_checkpoint"]),
        "whole_run_proof_sha256": digest(row["original_whole_run_proof"]),
        "immutable_remote_commit": row["immutable_remote_commit"],
        "original_durability_proofs": receipts,
        "named_layout": copy.deepcopy(row["named_layout"]),
    }


def load_inputs(locations):
    """Authenticate both actual states and the unchanged branch/calibration data.

    This rereads native checkpoint content seals, not model tensors, gradients or
    remote network state. The genuine earlier full-data/model checks are retained
    as separately scoped evidence, not reinterpreted as a formal run.
    """
    if type(locations) is not Locations:
        raise ValueError("Require explicit factorial input locations")
    data = _fixed_audit(locations.evidence, "data-first.json")
    states = _fixed_audit(locations.evidence, "states-first.json")
    verified = _fixed_audit(locations.evidence, "verification.json")
    if any(v.get("scientific_admission") is not False for v in (data, states, verified)):
        raise ValueError("Original input evidence was mis-scoped")
    for recorded, binding in data["inputs"].items():
        verify_file(locations.locate(recorded), binding)
    for recorded, binding in verified["files"].items():
        verify_file(locations.locate(recorded), binding)
    if (
        data["parent_ledger_rows_checked"] != 500000
        or data["branch"]["rows"] != 50000
        or data["calibration"]["rows"] != 32
        or data["all_selected_field_values_equal_to_revised_primary"] is not True
        or data["historical_gradients_reused"] is not False
        or states["calibration_performed"] is not False
        or [r["label"] for r in states["states"]] != list(STATES)
    ):
        raise ValueError("Genuine data/state scope differs")
    sources = {r["label"]: _source_binding(locations, r) for r in states["states"]}
    if (
        file_identity(Path(locations.repository) / "configs/representation_probe.json")["sha256"]
        != "45e76c8daa5bda0213c346b39200a34ddc1e9295f37198a2bd238276ee6057e2"
    ):
        raise ValueError("Fixed calibration probe specification differs")
    runtime_path = Path(locations.repository) / "configs/formal_runtime.json"
    if (
        file_identity(runtime_path)["sha256"]
        != "b90de16b3796bb8e9d0badcedec688da55a037d46244a0d2a7f5838795cfe77a"
    ):
        raise ValueError("Fixed calibration runtime specification differs")
    branch = locations.locate(
        "/root/embedding-optimizer-study/data/short-branch-50k-seed20260826/manifest.json"
    ).parent
    probe = locations.locate(
        "/root/embedding-optimizer-study/data/probes/training-1024-seed1729/manifest.json"
    ).parent
    return {
        "scope": SCOPE,
        "scientific_admission": False,
        "execution_authorized": False,
        "accepted_audit_digests": copy.deepcopy(AUDITS),
        "branch": {"path": str(branch), **copy.deepcopy(data["branch"])},
        "calibration": {"probe": str(probe), **copy.deepcopy(data["calibration"])},
        "sources": sources,
        "runtime_spec": {"path": str(runtime_path), **file_identity(runtime_path)},
        "common_state_spec": {
            "path": str(Path(locations.repository) / "configs/common_state_probe.json"),
            **file_identity(Path(locations.repository) / "configs/common_state_probe.json"),
        },
        "probe_spec": {
            "path": str(Path(locations.repository) / "configs/representation_probe.json"),
            **file_identity(Path(locations.repository) / "configs/representation_probe.json"),
        },
    }


def calibration_request(inputs, state):
    if (
        inputs.get("scope") != SCOPE
        or inputs.get("scientific_admission") is not False
        or inputs.get("execution_authorized") is not False
        or inputs.get("accepted_audit_digests") != AUDITS
        or state not in STATES
    ):
        raise ValueError("Require actual v3 factorial input admission and a fixed state")
    source = inputs["sources"][state]
    if source["run_id"] != STATES[state] or source["checkpoint_step"] != 2345:
        raise ValueError("Calibration source state differs")
    return {
        "scope": "dense-v3-factorial-calibration-request-v1",
        "input_sha256": digest(inputs),
        "state": state,
        "source": copy.deepcopy(source),
        "probe": inputs["calibration"]["probe"],
        "selection": copy.deepcopy(inputs["calibration"]["selection"]),
        "selection_sha256": inputs["calibration"]["selection_sha256"],
        "common_state_spec": copy.deepcopy(inputs["common_state_spec"]),
        "probe_spec": copy.deepcopy(inputs["probe_spec"]),
        "config": copy.deepcopy(CALIBRATION_CONFIG),
        "target_global_hidden_update_to_weight": 5e-4,
        "weight_decay_included": False,
        "formal_training_authorized": False,
    }


def require_gradient_manifest(manifest, request, directory):
    """Bind all eight newly produced shards to this exact state and row history."""
    require_same(manifest["config"], request["config"])
    expected_inputs = [
        r
        for r in request["source"]["native_checkpoint"]["files"]
        if Path(r["path"]).suffix in {".json", ".safetensors"}
    ]
    expected_inputs.sort(key=lambda r: r["path"])
    require_same(manifest["checkpoint"]["inputs"], expected_inputs)
    require_same(manifest["selection"], request["selection"])
    require_same(manifest["common_state_spec"], request["common_state_spec"])
    if (
        manifest.get("status") != "complete"
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest["checkpoint"]["path"] != request["source"]["checkpoint"]
        or manifest["probe"]["path"] != request["probe"]
        or manifest["selection_sha256"] != request["selection_sha256"]
        or manifest["probe"]["frozen_spec"]["sha256"] != request["probe_spec"]["sha256"]
        or manifest["probe"]["manifest_sha256"]
        != "40953eb60bb5dbfa02d9abde5e634bb3221ee934d7ca654c5e5a7e961b39eed2"
        or manifest["partition_summary"]["hidden"] != {"tensors": 88, "parameters": 110297088}
    ):
        raise ValueError("Gradient history does not belong to this corrected state/probe")
    mapping = manifest["parameter_name_mapping"]
    wanted = request["source"]["named_layout"][0]["members"]
    require_same(
        sorted(mapping, key=lambda r: r["model_name"]),
        sorted(
            [
                {
                    "model_name": r["name"],
                    "checkpoint_name": r["name"].removeprefix("0.model."),
                    "shape": r["shape"],
                }
                for r in wanted
            ],
            key=lambda r: r["model_name"],
        ),
    )
    shards = manifest["gradient_shards"]
    if len(shards) != 8:
        raise ValueError("Calibration requires exactly eight gradient shards")
    for step, shard in enumerate(shards):
        if (
            type(shard.get("step_index")) is not int
            or shard["step_index"] != step
            or shard["path"] != f"gradient-{step:04d}.safetensors"
            or shard["sample_ids"]
            != [r["sample_id"] for r in request["selection"][4 * step : 4 * step + 4]]
        ):
            raise ValueError("Calibration gradient order or four-row grouping differs")
        for key in ("mean_loss", "pre_clip_grad_norm", "clip_coefficient"):
            value = shard[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Calibration shard has an invalid numerical observation")
        if shard["clip_coefficient"] != min(1.0, 1.0 / (shard["pre_clip_grad_norm"] + 1e-6)):
            raise ValueError("Calibration shard has a different clipping rule")
        verify_file(Path(directory) / relative_path(shard["path"]), shard)
    return copy.deepcopy(shards)
