"""Source-bound fixed-probe exports for the complete primary DenseOn matrix.

This entry point never changes training artifacts. It requires all primary runs
to pass their deep training audit before acquiring one cooperative GPU lease.
Each export is tied to checkpoint content, the fixed probe, and observed loading
settings. A valid export cache is reused; a mismatched cache fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .dimension_utilization import _atomic_json, _identity, _validate_export

PROTOCOL = Path("configs/dense_primary_dimension_export_protocol.json")
MATRIX = Path("configs/dense_no_packing_retrain.yaml")
SCIENTIFIC_PROTOCOL = Path("configs/dense_dimension_utilization_protocol.json")
SOURCES = (
    "src/embed_optim/primary_dimension_probe.py",
    "src/embed_optim/dimension_utilization.py",
    "src/embed_optim/probe_export.py",
    "src/embed_optim/probes.py",
    "src/embed_optim/geometry.py",
    "src/embed_optim/aggregate.py",
    "src/embed_optim/config.py",
    "src/embed_optim/corrected_input_execution.py",
    "src/embed_optim/gpu_lease.py",
    "src/embed_optim/runtime.py",
)
ENCODING = {
    "batch_size": 8,
    "model_dtype": "bfloat16",
    "storage_dtype": "float32",
    "device": "cuda:0",
    "flash_attention": True,
    "compressed": False,
    "max_length": 8192,
    "normalized": True,
    "dense_query_prompt": "query: ",
    "dense_document_prompt": "document: ",
    "positive_candidate_index": 0,
}
INPUT_EXECUTION = {
    "mode": "independently_padded",
    "sentence_transformers_can_flatten_inputs": False,
}


@dataclass(frozen=True)
class ExportJob:
    cell: str
    checkpoint: Path
    export: Path
    completion: Path | None = None

    @property
    def receipt(self) -> Path:
        return self.export.with_suffix(".npz.primary.json")


def load_contract(repository: Path) -> dict[str, Any]:
    payload = json.loads((repository / PROTOCOL).read_text(encoding="utf-8"))
    if (
        payload.get("status") != "prospective_primary_dimension_export_lock"
        or payload.get("encoding") != ENCODING
        or payload.get("input_execution") != INPUT_EXECUTION
        or set(payload.get("source_bindings", {})) != set(SOURCES)
    ):
        raise ValueError("Primary dimension export contract differs")
    expected_parents = {
        str(MATRIX),
        str(SCIENTIFIC_PROTOCOL),
        "configs/beir_representation_probe.json",
        "configs/formal_runtime.json",
        "configs/dense_no_packing_execution_protocol.json",
        "configs/dense_no_packing_evaluation_protocol.json",
    }
    if set(payload.get("parent_bindings", {})) != expected_parents:
        raise ValueError("Primary dimension export parents differ")
    for group in ("source_bindings", "parent_bindings"):
        for name, recorded in payload[group].items():
            if _identity(repository / name, repository) != recorded:
                raise ValueError(f"Primary dimension export binding changed: {name}")
    return payload


def planned_cells(configs: list[Any], stages: list[int]) -> list[str]:
    if (
        len(configs) != 12
        or len({item.run_id for item in configs}) != 12
        or any(item.model_family != "dense" or item.dense_can_flatten_inputs for item in configs)
        or stages != [782, 1563, 2345, 3126, 3907]
    ):
        raise ValueError("Primary dimension exports require the complete 12-by-5 matrix")
    return ["pretrained"] + [
        f"{item.run_id}/checkpoint-{step}"
        for item in sorted(configs, key=lambda item: item.run_id)
        for step in stages
    ]


def build_jobs(
    configs: list[Any],
    stages: list[int],
    artifact_root: Path,
    export_root: Path,
    reference: Path,
) -> list[ExportJob]:
    cells = planned_cells(configs, stages)
    by_run = {item.run_id: item for item in configs}
    jobs = [ExportJob("pretrained", reference.resolve(), export_root / "pretrained.npz")]
    for cell in cells[1:]:
        run_id, checkpoint = cell.split("/")
        run_root = (artifact_root / by_run[run_id].output_dir).resolve()
        jobs.append(
            ExportJob(
                cell,
                run_root / checkpoint,
                export_root / f"{cell}.npz",
                run_root / "completed.json",
            )
        )
    return jobs


def _request(job: ExportJob, repository: Path, probe: Path) -> dict[str, Any]:
    from .probe_export import _checkpoint_inputs, _validate_checkpoint_family

    return {
        "cell": job.cell,
        "checkpoint": str(job.checkpoint.resolve()),
        "checkpoint_inputs": _checkpoint_inputs(job.checkpoint),
        "checkpoint_run_config": _validate_checkpoint_family(job.checkpoint, "dense"),
        "completion": None if job.completion is None else _identity(job.completion),
        "protocol": _identity(repository / PROTOCOL, repository),
        "scientific_protocol": _identity(repository / SCIENTIFIC_PROTOCOL, repository),
        "probe_manifest": _identity(probe / "manifest.json"),
        "probe_selection": _identity(probe / "selection.jsonl"),
        "probe_spec": _identity(repository / "configs/beir_representation_probe.json", repository),
        "encoding": ENCODING,
        "input_execution": INPUT_EXECUTION,
    }


@contextmanager
def _verified_loader(receipt_path: Path, receipt: dict[str, Any]) -> Iterator[None]:
    from . import probe_export

    original = probe_export._load_model

    def load(*args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        family = kwargs.get("family", args[0] if args else None)
        if family != "dense" or not hasattr(model, "_first_module"):
            raise TypeError("Primary dimension probe requires a Dense SentenceTransformer")
        first = model._first_module()
        if not hasattr(first, "can_flatten_inputs"):
            raise AttributeError("Dense model does not expose its input execution setting")
        first.can_flatten_inputs = False
        if bool(first.can_flatten_inputs) or model.max_seq_length != 8192:
            raise ValueError("Primary dimension model loading contract could not be verified")
        receipt["status"] = "model_verified"
        receipt["observed_input_execution"] = dict(INPUT_EXECUTION)
        _atomic_json(receipt_path, receipt)
        return model

    probe_export._load_model = load
    try:
        yield
    finally:
        probe_export._load_model = original


def audit_job(
    job: ExportJob,
    repository: Path,
    request: dict[str, Any],
    *,
    require_complete_receipt: bool = True,
) -> dict[str, Any]:
    receipt = json.loads(job.receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("request") != request
        or receipt.get("observed_input_execution") != INPUT_EXECUTION
        or receipt.get("status") != ("complete" if require_complete_receipt else "model_verified")
    ):
        raise ValueError(f"Primary export receipt differs: {job.cell}")
    arrays, manifest = _validate_export(
        job.export, 768, probe_spec=repository / "configs/beir_representation_probe.json"
    )
    if (
        manifest.get("checkpoint") != request["checkpoint"]
        or manifest.get("checkpoint_inputs") != request["checkpoint_inputs"]
        or manifest.get("checkpoint_run_config") != request["checkpoint_run_config"]
        or any(manifest.get("encoding", {}).get(key) != value for key, value in ENCODING.items())
        or any(
            arrays[key].dtype != np.float32 for key in ("query_embeddings", "document_embeddings")
        )
    ):
        raise ValueError(f"Primary export content/encoding contract differs: {job.cell}")
    outputs = {
        "export": _identity(job.export),
        "export_manifest": _identity(job.export.with_suffix(".npz.manifest.json")),
    }
    if require_complete_receipt and receipt.get("outputs") != outputs:
        raise ValueError(f"Primary export outputs changed: {job.cell}")
    return outputs


def run_job(job: ExportJob, repository: Path, probe: Path) -> dict[str, Any]:
    from .probe_export import export_probe

    request = _request(job, repository, probe)
    manifest_path = job.export.with_suffix(".npz.manifest.json")
    if job.receipt.is_file():
        receipt = json.loads(job.receipt.read_text(encoding="utf-8"))
        if receipt.get("request") != request:
            raise ValueError(f"Primary export cache request changed: {job.cell}")
        if receipt.get("status") == "complete":
            return audit_job(job, repository, request)
        if receipt.get("status") not in {"in_progress", "model_verified"}:
            raise ValueError(f"Invalid incomplete primary export receipt: {job.cell}")
        if job.export.is_file() and manifest_path.is_file():
            outputs = audit_job(job, repository, request, require_complete_receipt=False)
            receipt.update(status="complete", outputs=outputs)
            _atomic_json(job.receipt, receipt)
            return outputs
        if job.export.exists() or manifest_path.exists():
            raise FileExistsError(
                f"Partial export requires recovery into a new output root: {job.cell}"
            )
    elif job.export.exists() or manifest_path.exists():
        raise FileExistsError(f"Untagged primary export cache: {job.cell}")
    else:
        receipt = {"schema_version": 1, "status": "in_progress", "request": request}
        _atomic_json(job.receipt, receipt)
    with _verified_loader(job.receipt, receipt):
        export_probe(
            job.checkpoint,
            probe,
            job.export,
            family="dense",
            batch_size=ENCODING["batch_size"],
            model_dtype=ENCODING["model_dtype"],
            storage_dtype=ENCODING["storage_dtype"],
            device=ENCODING["device"],
            flash_attention=ENCODING["flash_attention"],
            probe_spec=repository / "configs/beir_representation_probe.json",
        )
    if _request(job, repository, probe) != request:
        raise ValueError(f"Primary checkpoint or probe changed during export: {job.cell}")
    outputs = audit_job(job, repository, request, require_complete_receipt=False)
    receipt.update(status="complete", outputs=outputs)
    _atomic_json(job.receipt, receipt)
    return outputs


def _training_gate(configs: list[Any], artifact_root: Path) -> dict[str, Any]:
    from .aggregate import audit_dataset_artifacts, audit_training_artifacts
    from .corrected_input_execution import require_corrected_training_receipt

    with _working_directory(artifact_root):
        for config in configs:
            completion = json.loads((config.output_dir / "completed.json").read_text())
            require_corrected_training_receipt(completion)
        dataset = audit_dataset_artifacts(configs)
        if not dataset["complete"]:
            raise ValueError(f"Primary probe data audit failed: {dataset['errors'][:5]}")
        training = audit_training_artifacts(
            configs, deep=True, expected_dataset_fingerprint=dataset["training_view_fingerprint"]
        )
        if not training["complete"]:
            raise ValueError(f"Primary probe training audit failed: {training['errors'][:5]}")
    return {"dataset": dataset, "training": training}


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _audit_matrix(
    jobs: list[ExportJob],
    repository: Path,
    probe: Path,
    export_root: Path,
) -> list[dict[str, Any]]:
    expected = {job.export.resolve() for job in jobs}
    if set(path.resolve() for path in export_root.rglob("*.npz")) != expected:
        raise ValueError("Primary export archive set differs from the exact 61 states")
    rows = []
    sample_reference = None
    for job in jobs:
        outputs = audit_job(job, repository, _request(job, repository, probe))
        with np.load(job.export, allow_pickle=False) as arrays:
            pair = (arrays["sample_ids"], arrays["sample_groups"])
            if sample_reference is None:
                sample_reference = pair
            elif not all(np.array_equal(a, b) for a, b in zip(sample_reference, pair, strict=True)):
                raise ValueError("Primary exports have unpaired samples or task assignments")
        rows.append({"cell": job.cell, **outputs, "receipt": _identity(job.receipt)})
    return rows


def audit_exports(export_root: Path, repository: Path) -> dict[str, Any]:
    """Validate the source-bound export handoff consumed by dimension analysis."""
    from .config import load_matrix

    contract = load_contract(repository)
    path = export_root / "primary_exports.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    scientific = json.loads((repository / SCIENTIFIC_PROTOCOL).read_text())
    configs = load_matrix(repository / MATRIX)
    cells = planned_cells(configs, scientific["inputs"]["checkpoint_stages"])
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("scope") != "primary_dense_fixed_probe_exports"
        or payload.get("scientific_completion") is not False
        or payload.get("protocol") != _identity(repository / PROTOCOL, repository)
        or payload.get("cells") != cells
    ):
        raise ValueError("Primary export handoff coverage or protocol differs")
    artifact_root = Path(payload["artifact_root"])
    spec = json.loads((repository / scientific["inputs"]["probe_spec"]).read_text())
    probe = artifact_root / spec["output"]
    from huggingface_hub import snapshot_download

    from .probe_export import _checkpoint_inputs

    pinned_reference = Path(
        snapshot_download(
            repo_id=contract["model"]["repo_id"],
            revision=contract["model"]["revision"],
            local_files_only=True,
        )
    )
    if _checkpoint_inputs(Path(payload["reference"])) != _checkpoint_inputs(pinned_reference):
        raise ValueError("Primary export pretrained weights differ from the pinned reference")
    jobs = build_jobs(
        configs,
        scientific["inputs"]["checkpoint_stages"],
        artifact_root,
        export_root,
        Path(payload["reference"]),
    )
    if _audit_matrix(jobs, repository, probe, export_root) != payload.get("outputs"):
        raise ValueError("Primary export handoff output identities differ")
    _audit_record(payload["training_audit"], repository)
    training = json.loads((repository / payload["training_audit"]["path"]).read_text())
    expected = {
        "complete": True,
        "verified_runs": 12,
        "expected_runs": 12,
        "verified_checkpoints": 60,
        "expected_checkpoints": 60,
        "deep_validation": True,
        "errors": [],
    }
    if (
        training.get("training") != expected
        or training.get("dataset", {}).get("complete") is not True
        or training.get("dataset", {}).get("verified_rows") != 500_000
    ):
        raise ValueError("Primary export handoff lacks the deep training gate")
    return payload


def _audit_record(record: dict[str, Any], repository: Path) -> None:
    if _identity(repository / record["path"], repository) != record:
        # Output roots may be outside the code checkout, so records there are absolute.
        raise ValueError(f"Primary export identity changed: {record['path']}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    artifact_root = args.artifact_root.resolve()
    contract = load_contract(repository)
    from .config import load_matrix

    configs = load_matrix(repository / MATRIX)
    scientific = json.loads((repository / SCIENTIFIC_PROTOCOL).read_text())
    stages = scientific["inputs"]["checkpoint_stages"]
    cells = planned_cells(configs, stages)
    export_root = args.export_root.resolve()
    missing = [
        item.run_id
        for item in configs
        if not (artifact_root / item.output_dir / "completed.json").is_file()
    ]
    if args.dry_run:
        return {
            "status": "dry_run",
            "scientific_completion": False,
            "cells": cells,
            "training_gate": "missing_completions" if missing else "deep_audit_required",
            "missing_completed_runs": missing,
            "gpu_work_started": False,
            "export_root": str(export_root),
            "protocol": _identity(repository / PROTOCOL),
        }
    if args.audit_only:
        return audit_exports(export_root, repository)
    if missing:
        raise RuntimeError(f"Primary dimension export waits for all 12 runs: {missing}")
    from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens

    gpu = parse_gpu_tokens(args.gpu, expected_count=1)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu[0]
    from .runtime import verify_runtime_spec

    runtime = verify_runtime_spec(repository / "configs/formal_runtime.json")
    training = _training_gate(configs, artifact_root)
    from huggingface_hub import snapshot_download

    model = contract["model"]
    if any(
        item.model_name != model["repo_id"] or item.model_revision != model["revision"]
        for item in configs
    ):
        raise ValueError("Primary probe pretrained reference differs from training")
    reference = Path(snapshot_download(repo_id=model["repo_id"], revision=model["revision"]))
    spec = json.loads((repository / scientific["inputs"]["probe_spec"]).read_text())
    probe = artifact_root / spec["output"]
    jobs = build_jobs(configs, stages, artifact_root, export_root, reference)
    with acquire_gpu_lease(
        gpu,
        lock_dir=artifact_root / "logs/dense-only-runtime/gpu-leases",
        timeout_seconds=60,
        purpose="primary-dimension-probe",
    ):
        for index, job in enumerate(jobs, start=1):
            if load_contract(repository) != contract:
                raise ValueError("Primary export contract changed during execution")
            print(f"Exporting primary state {index}/{len(jobs)}: {job.cell}", flush=True)
            run_job(job, repository, probe)
    outputs = _audit_matrix(jobs, repository, probe, export_root)
    training_path = export_root / "training_audit.json"
    _atomic_json(training_path, training)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "scientific_completion": False,
        "scope": "primary_dense_fixed_probe_exports",
        "protocol": _identity(repository / PROTOCOL, repository),
        "artifact_root": str(artifact_root),
        "reference": str(reference.resolve()),
        "cells": cells,
        "outputs": outputs,
        "runtime": runtime,
        "training_audit": _identity(training_path, repository),
    }
    _atomic_json(export_root / "primary_exports.json", payload)
    return audit_exports(export_root, repository)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--artifact-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("results/dense-primary-dimension-probe/exports/dense"),
    )
    parser.add_argument("--gpu", default="0")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
