"""Authenticate archived vector inputs without opening producer/model/data paths.

The external archive anchor attests previously completed authoring. This reader
checks its content and internal identities, not the checkpoint payloads again.
It is a reconstruction boundary and never establishes a new primary result.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import reconstruction_files as files
from .primary_contract import digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_dimension_contract import ENCODING, INPUT_EXECUTION, SCOPE, planned_states
from .primary_v3_dimension_inference import SCOPE as INFERENCE_SCOPE
from .primary_v3_dimension_inference import FunctionalInferenceContract
from .primary_v3_dimension_vector_io import MODEL_KEY_PREFIX, validate_arrays
from .primary_v3_reconstruction_authoring import BOUNDARY, require_parent
from .primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from .runtime import runtime_snapshot


def running_sources(root, manifest):
    """Match every currently imported package module, including a -m main module."""
    checked = {}
    for name, module in list(sys.modules.items()):
        if name != "embed_optim" and not name.startswith("embed_optim."):
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        path = files.ordinary(path)
        relative = "source/src/embed_optim/" + path.name
        binding = manifest["files"].get(relative)
        if binding is None:
            raise ValueError("An executing package source is absent from the archive")
        verify_file(path, binding)
        verify_file(root / relative, binding)
        checked[name] = binding
    return checked


def probe_identity(contract):
    """Rebuild content identities from the frozen receipt, not its original locations."""
    accepted = contract.probe_acceptance
    original = next(
        Path(row["path"]).parent
        for row in accepted["inputs"]
        if Path(row["path"]).name == "manifest.json"
    )
    inventory = sorted(
        [
            {"path": str(Path(row["path"]).relative_to(original)), **files.identity(row)}
            for row in accepted["inputs"]
        ],
        key=lambda row: row["path"],
    )
    identities = accepted["row_identities"]
    return {
        "files": inventory,
        "content_sha256": digest(inventory),
        "row_identities": identities,
        "row_identities_sha256": digest(identities),
    }


def vector_admission(contract, admitted):
    """Rebuild all location-free plans from authenticated original authoring metadata."""
    primary = contract.primary
    common = {
        "scope": SCOPE,
        "dimension_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": primary.sha256,
        "probe": probe_identity(contract),
        "encoding": ENCODING,
        "input_execution": INPUT_EXECUTION,
        "scientific_completion": False,
    }
    reference = admitted["reference"]
    if set(reference) != {"files", "all_shapes", "hidden_shapes"}:
        raise ValueError("Incomplete archived reference identity")
    require_same(reference["files"], primary.inputs["common_identity"]["model"]["files"])
    shapes = reference["all_shapes"]
    if len(shapes) != 134 or any(
        not name.startswith("0.") or not shape or any(type(n) is not int or n <= 0 for n in shape)
        for name, shape in shapes.items()
    ):
        raise ValueError("The archived reference tensor-shape population differs")
    from .optimizers import parameter_partition_name

    hidden = {
        name: shape
        for name, shape in shapes.items()
        if parameter_partition_name(name, len(shape)) == "hidden"
    }
    require_same(reference["hidden_shapes"], hidden)
    if len(hidden) != 88 or sum(int(np.prod(s)) for s in hidden.values()) != 110297088:
        raise ValueError("Incomplete archived hidden-matrix population")
    runs = admitted["complete_runs"]
    if set(runs) != {row["run_id"] for row in primary.inputs["runs"]} or len(runs) != 12:
        raise ValueError("Reconstruction requires every original primary run identity")
    steps = primary.payload["checkpoint_steps"]
    for run_id, checked in runs.items():
        if checked["whole_run_artifacts_verified"] is not True or checked[
            "run_identity_sha256"
        ] != digest(primary.expected_identity(run_id)):
            raise ValueError("An archived whole-run authoring identity differs")
        require_same(checked["steps"], steps)
        require_same([row["step"] for row in checked["checkpoints"]], steps)
    jobs = []
    for state in planned_states(primary):
        if state["cell"] == "pretrained":
            model = {"kind": "immutable_pretrained", "reference": reference}
        else:
            checked = runs[state["meta"]["run_id"]]
            model = {
                "kind": "complete_primary_checkpoint",
                "run_identity_sha256": checked["run_identity_sha256"],
                "complete_run_sha256": digest(checked),
                "checkpoint": checked["checkpoints"][state["stage"] - 1],
            }
        jobs.append({"plan": {**common, "state": state, "model": model}})
    require_same(
        admitted,
        {
            **common,
            "reference": reference,
            "complete_runs": runs,
            "states": [j["plan"] for j in jobs],
        },
    )
    return jobs


def raw_vectors(root, plan, identities, reference_shapes, expected):
    """Verify recorded loading and raw bytes, but never claim fresh tensor validation."""
    root = Path(root)
    verify_file(root / "manifest.json", expected)
    value = read_json(root / "manifest.json")
    if (
        set(value) != {"status", "plan", "observed_loading", "output", "array_metadata"}
        or value["status"] != "complete"
    ):
        raise ValueError("Incomplete archived raw-vector state")
    require_same(value["plan"], plan)
    if (
        set(p.name for p in root.iterdir()) != {"manifest.json", "vectors.npz"}
        or value["output"]["path"] != "vectors.npz"
    ):
        raise ValueError("Archived vector inventory differs")
    observed = value["observed_loading"]
    if (
        set(observed) != {"before", "after", "parameters_unchanged"}
        or observed["parameters_unchanged"] is not True
    ):
        raise ValueError("Missing recorded immutable model loading")
    require_same(observed["before"], observed["after"])
    before = observed["before"]
    tensors = before["state_tensors"]
    require_same(
        before,
        {
            "model_mode": "eval",
            "max_length": 8192,
            "input_execution": INPUT_EXECUTION,
            "saved_dtype_conversion": "bfloat16",
            "state_tensors": tensors,
            "state_tensors_sha256": digest(tensors),
            "saved_weights_matched": True,
        },
    )
    expected_shapes = {MODEL_KEY_PREFIX + key[2:]: shape for key, shape in reference_shapes.items()}
    require_same({name: row["shape"] for name, row in tensors.items()}, expected_shapes)
    for record in tensors.values():
        if set(record) != {"shape", "dtype", "sha256"} or record["dtype"] != "torch.bfloat16":
            raise ValueError("The recorded Dense BF16 tensor schema differs")
        files.identity({"bytes": 0, "sha256": record["sha256"]})
    verify_file(files.ordinary(root / "vectors.npz"), value["output"])
    with np.load(root / "vectors.npz", allow_pickle=False) as source:
        if len(source.files) != len(set(source.files)):
            raise ValueError("Duplicated raw array entries")
        arrays = {name: source[name] for name in source.files}
    require_same(value["array_metadata"], validate_arrays(arrays, identities))
    verify_file(root / "vectors.npz", value["output"])
    verify_file(root / "manifest.json", expected)
    return arrays


@dataclass(frozen=True)
class ReconstructionInput:
    root: Path
    anchor: str
    manifest: dict
    contract: object
    evidence: dict
    admitted: dict
    jobs: list
    runtime: dict

    @classmethod
    def load(cls, root, anchor):
        root = Path(root).absolute()
        manifest = files.inspect(root, anchor)
        running_sources(root, manifest)
        source = root / "source"
        primary = PrimaryV3Contract.load(
            source / "configs/dense_primary_v3_protocol.json", source, root / "training-source"
        )
        contract = FunctionalInferenceContract.load(source / INFERENCE_PROTOCOL, primary)
        require_parent(contract)
        meta = manifest["metadata"]
        expected = {
            "scope": "dense_primary_v3_checkpoint_backed_reconstruction_selection",
            "primary_protocol_sha256": primary.sha256,
            "inference_protocol_sha256": contract.sha256,
            "inference_acceptance_sha256": file_identity(require_parent(contract))["sha256"],
            "dimension_protocol_sha256": contract.dimensions.sha256,
            "original_location_fields_preserved": True,
            "boundary": BOUNDARY,
            "full_raw_vector_reconstruction_repeated_by_transport": False,
            "manuscript_installed": False,
            "scientific_completion": False,
        }
        require_same({key: meta[key] for key in expected}, expected)
        evidence = read_json(root / "inference/evidence.json")
        plan = meta["authoring_plan"]
        require_same(meta["authoring_evidence_sha256"], digest(evidence))
        require_same(
            plan,
            {
                "scope": INFERENCE_SCOPE,
                "inference_protocol_sha256": contract.sha256,
                "primary_protocol_sha256": primary.sha256,
                "evidence_sha256": digest(evidence),
                "scientific_completion": False,
                "manuscript_installation_authorized": False,
            },
        )
        require_same(read_json(root / "inference/manifest.json")["plan"], plan)
        original_runtime = meta["authoring_runtime"]
        runtime = runtime_snapshot(list(original_runtime["packages"]))
        for key in ("python", "torch_cuda", "packages"):
            require_same(runtime[key], original_runtime[key])
        admitted = read_json(root / "vectors/admission.json")
        jobs = vector_admission(contract.dimensions, admitted)
        running_sources(root, manifest)
        return cls(root, anchor, manifest, contract, evidence, admitted, jobs, runtime)

    def recheck(self):
        require_same(files.inspect(self.root, self.anchor), self.manifest)
        running_sources(self.root, self.manifest)
        self.contract.recheck()
