"""Read-only primary input/recipe binding for the accepted numerical candidate.

No training, device setup, artifact adoption, remote write or protocol activation.
One actual common data/model snapshot is observed; twelve recipe identities are
derived explicitly from it. This is not twelve independent execution preflights.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFile

from embed_optim import aggregate, train
from embed_optim import dense_numerical_contract as numerical
from embed_optim import dense_run_contract as contract
from embed_optim.config import load_matrix
from scripts.audit_dense_gpu_replay_update import identity
from scripts.audit_prepared_dense_gpu import check_sources

DATA_SHA = "9facc18bcd1cad8378cea94746a95ab09804bdf3610796bf9013cdfcc486aee8"
OLD_PROTOCOL_SHA = "eb106341c72a372d97c0be2064102577c5fbfcd83dd04ea213e05b21fbf991a5"
MODEL = "lightonai/DenseOn-unsupervised"
REVISION = "0edbd55684eb782bce55ee74c95b25c97cbe7f43"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    work, root, experiment = (
        args.workdir.absolute(),
        args.candidate_root.resolve(),
        args.experiment_root.resolve(),
    )
    if any(
        os.environ.get(k) != v
        for k, v in {
            "CUDA_VISIBLE_DEVICES": "",
            "WORLD_SIZE": "4",
            "EMBED_OPTIM_MAX_STEPS": "-1",
            "EMBED_OPTIM_STOP_AFTER_STEP": "-1",
        }.items()
    ):
        raise ValueError("Require explicit CPU-only, unshortened primary argument identity")
    if work.parent != Path("/tmp") or not work.name.startswith("dense-primary-inputs."):
        raise ValueError("Require a fresh temporary preparation namespace")
    if work.is_symlink() or not work.is_dir() or any(work.iterdir()):
        raise ValueError("Preparation namespace must be empty and ordinary")
    manifest = check_sources(args.source_manifest, args.source_sha256, root)
    original_protocol_path = experiment / "configs/dense_no_packing_execution_protocol.json"
    if identity(original_protocol_path)["sha256"] != OLD_PROTOCOL_SHA:
        raise ValueError("Original scientific execution lock changed")
    original_protocol = json.loads(original_protocol_path.read_text())
    data_root = experiment / "data/denseon-sft-500k-seed42"
    if identity(data_root / "manifest.json")["sha256"] != DATA_SHA:
        raise ValueError("Original materialized data manifest changed")
    configs = load_matrix(root / "configs/dense_correctness_candidate.yaml")
    configs = [
        replace(c, dataset_path=str(data_root), output_root=str(work / "unused-output"))
        for c in configs
    ]
    if len(configs) != 12 or {
        (c.model_name, c.model_revision, c.dataset_path) for c in configs
    } != {(MODEL, REVISION, str(data_root))}:
        raise ValueError("Require the twelve shared-source primary recipes")
    dataset_audit = aggregate.audit_dataset_artifacts(configs)
    if dataset_audit["complete"] is not True or dataset_audit["verified_rows"] != 500000:
        raise ValueError("Original full data/row-manifest linkage failed")
    print(json.dumps({"stage": "data_linkage", "verified_rows": 500000}), flush=True)
    first, dataset = contract.prepare(configs[0], train._training_argument_values(configs[0]), 4)
    if len(dataset) != 500000:
        raise ValueError("Actual selected training view is not 500k rows")
    cached = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    remote_entries = {
        e.path: e
        for e in HfApi().list_repo_tree(
            MODEL, revision=REVISION, recursive=True, expand=True, repo_type="model"
        )
        if isinstance(e, RepoFile)
    }
    remote_checked = []
    for item in first["model"]["files"]:
        remote = remote_entries[item["path"]]
        if int(remote.size) != item["bytes"]:
            raise ValueError("Cached initial-model file size differs from pinned HF source")
        if remote.lfs is not None:
            kind, actual, expected = "sha256", item["sha256"], remote.lfs.sha256
        else:
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {item['bytes']}\0".encode())
            with (cached / item["path"]).open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            kind, actual, expected = "git_blob_sha1", digest.hexdigest(), remote.blob_id
        if actual != expected:
            raise ValueError("Cached initial-model bytes differ from the immutable HF revision")
        remote_checked.append(
            {
                "path": item["path"],
                "bytes": item["bytes"],
                "digest_kind": kind,
                "local_digest": actual,
                "remote_digest": expected,
            }
        )
    print(
        json.dumps({"stage": "base_model_remote_digest", "files": len(remote_checked)}), flush=True
    )
    common = {key: first[key] for key in ("schema_version", "scope", "source", "data", "model")}
    runs = []
    for config in configs:
        arguments = train._training_argument_values(config)
        specific = {
            "recipe": contract.recipe_identity(config),
            "execution": contract.execution_identity(arguments, 4),
            "numerical_contract": numerical.receipt(
                config.optimizer, arguments["gradient_accumulation_steps"]
            ),
        }
        if arguments["max_steps"] != -1 or arguments["gradient_accumulation_steps"] != 4:
            raise ValueError("Primary horizon or accumulation was overridden")
        runs.append(
            {
                "run_id": config.run_id,
                "identity_fields": specific,
                "expected_identity_sha256": contract.digest({**common, **specific}),
            }
        )
    if contract.inventory(data_root / "dataset") != first["data"]["files"]:
        raise ValueError("Data payloads changed during preparation")
    if contract.inventory(cached, cache_links=True) != first["model"]["files"]:
        raise ValueError("Initial-model payloads changed during preparation")
    if any(config.output_dir.exists() for config in configs):
        raise ValueError("Read-only input preparation created a training output")
    if check_sources(args.source_manifest, args.source_sha256, root) != manifest:
        raise ValueError("Numerical candidate changed")
    result = {
        "scope": "prepared_dense_primary_actual_input_bindings",
        "schema_version": 1,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "preparation_passed": True,
        "scientific_completion": False,
        "production_deployed": False,
        "model_loaded": False,
        "model_updates_executed": 0,
        "protocol_activated": False,
        "source_manifest": identity(args.source_manifest),
        "candidate_root": str(root),
        "original_scientific_protocol": identity(original_protocol_path),
        "original_analysis": original_protocol["analysis"],
        "original_training": original_protocol["training"],
        "original_evaluation": original_protocol["evaluation"],
        "data_manifest": identity(data_root / "manifest.json"),
        "row_manifest": identity(data_root / "rows.jsonl"),
        "data_linkage_audit": dataset_audit,
        "common_identity": common,
        "runs": runs,
        "initial_model": {
            "repo": MODEL,
            "revision": REVISION,
            "cached_root": str(cached),
            "independent_hf_digest_checks": remote_checked,
        },
        "boundary": "Actual common 500k dataset and immutable untrained base observed once, twelve declared identity recipes derived from it, original row linkage and pinned HF digests verified. No GPU/natural-data training preflight, source publication, protocol activation, model load or primary execution.",
        "source_bindings": [
            identity(Path(inspect.getsourcefile(obj)).resolve())
            for obj in (aggregate, numerical, contract, train, check_sources)
        ]
        + [identity(Path(__file__).resolve())],
    }
    with (work / "result.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "preparation_passed": True,
                "derived_recipe_identities": len(runs),
                "model_updates_executed": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
