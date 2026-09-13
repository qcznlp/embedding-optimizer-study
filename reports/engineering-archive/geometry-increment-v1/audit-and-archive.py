"""Audit eight completed primary geometry runs; add only missing HF files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.hf_api import RepoFile

from embed_optim.artifact_inventory import compare_inventories, file_inventory

RUNS = [
    "padded-adamw-1e-6",
    "padded-adamw-3e-6",
    "padded-adamw-1e-5",
    "padded-adamw-3e-5",
    "padded-muon-1e-4",
    "padded-muon-3e-4",
    "padded-muon-1e-3",
    "padded-muon-3e-3",
]
STAGES = [782, 1563, 2345, 3126, 3907]
REPO = "qcz/embedding-optimizer-study-analysis-artifacts"
PREFIX = "corrected-dense-no-packing-v1/weight-space"
PROTOCOL_SHA = "91d7fa09ebb7e609afc5eb584f499b97baf9718d8cc23ff67eb295adfdd9f27a"


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite geometry value")
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    if isinstance(value, list):
        for item in value:
            finite(item)


def remote(api, revision, selected_prefixes):
    result = {}
    for entry in api.list_repo_tree(
        REPO, repo_type="dataset", path_in_repo=PREFIX, recursive=True, revision=revision
    ):
        if not isinstance(entry, RepoFile):
            continue
        label = str(Path(entry.path).relative_to(PREFIX))
        if not any(label.startswith(p + "/") for p in selected_prefixes):
            continue
        result[label] = {
            "size": entry.size,
            "digest_kind": "sha256" if entry.lfs is not None else "git_blob_sha1",
            "digest": entry.lfs.sha256 if entry.lfs is not None else entry.blob_id,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    root = args.repository.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new audit receipt; previous receipts are immutable")
    geometry_root = root / "results/dense-no-packing-weight-space"
    protocol_path = root / "configs/dense_no_packing_analysis_protocol.json"
    if file_inventory(protocol_path)["sha256"] != PROTOCOL_SHA:
        raise ValueError("Analysis protocol changed")
    protocol = json.loads(protocol_path.read_bytes())
    sources = {}
    for item in [*protocol["source_bindings"].values(), *protocol["parent_bindings"].values()]:
        record = file_inventory(root / item["path"])
        if record["sha256"] != item["sha256"] or (
            "bytes" in item and record["size"] != item["bytes"]
        ):
            raise ValueError("Frozen numerical or parent source changed")
        sources[item["path"]] = record
    inventories = {}
    model_inputs = {}
    tensor_rows = 0
    prefixes = [f"dense/{run}-rank64" for run in RUNS]
    for run_id, prefix in zip(RUNS, prefixes, strict=True):
        directory = geometry_root / prefix
        path = directory / "manifest.json"
        manifest = json.loads(path.read_bytes())
        if (
            manifest["run"]["run_id"] != run_id
            or sorted(map(int, manifest["records"])) != STAGES
            or manifest["analysis_config"]
            != {
                "partitions": ["hidden"],
                "sketch_rank": 64,
                "oversample": 8,
                "power_iterations": 2,
                "seed": 20260903,
            }
            or manifest["partition_summary"]["hidden"] != {"parameters": 110297088, "tensors": 88}
        ):
            raise ValueError("Unexpected geometry run, stages, settings or parameter partition")
        inputs = manifest["inputs"]
        if (
            [x.get("step") for x in inputs if x["kind"] == "checkpoint"] != STAGES
            or len(inputs) != 6
            or inputs[-1]["kind"] != "reference"
        ):
            raise ValueError("Incomplete geometry model inputs")
        for source in inputs:
            if source["kind"] == "checkpoint":
                expected = (
                    root
                    / "outputs/dense-no-packing-v1/dense"
                    / run_id
                    / f"checkpoint-{source['step']}"
                )
                if Path(source["source"]) != expected:
                    raise ValueError("Geometry references a different checkpoint")
            for item in [*source["files"], *source["metadata_files"]]:
                key = item["path"]
                if key not in model_inputs:
                    model_inputs[key] = file_inventory(Path(key))
                observed = model_inputs[key]
                if observed["size"] != item["bytes"] or observed["sha256"] != item["sha256"]:
                    raise ValueError("Geometry input content changed")
        wanted = {"manifest.json"}
        for step in STAGES:
            item = manifest["records"][str(step)]
            label = f"records/checkpoint-{step}.jsonl"
            if item["path"] != label or item["tensors"] != 88:
                raise ValueError("Unexpected checkpoint record identity")
            record = file_inventory(directory / label)
            if record["size"] != item["bytes"] or record["sha256"] != item["sha256"]:
                raise ValueError("Geometry record content differs from its manifest")
            rows = [json.loads(line) for line in (directory / label).read_text().splitlines()]
            if (
                len(rows) != 88
                or len({r["tensor"] for r in rows}) != 88
                or sum(r["parameters"] for r in rows) != 110297088
                or any(r["step"] != step or r["partition"] != "hidden" for r in rows)
                or any("delta_from_reference" not in r for r in rows)
                or (step != STAGES[0] and any("delta_from_previous" not in r for r in rows))
            ):
                raise ValueError("Incomplete per-matrix geometry coverage")
            finite(rows)
            tensor_rows += len(rows)
            wanted.add(label)
        actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
        if actual != wanted:
            raise ValueError("Unexpected file in selected geometry payload")
        for label in sorted(wanted):
            inventories[f"{prefix}/{label}"] = file_inventory(directory / label)
        print(f"Verified {run_id}: five stages, 88 hidden matrices per stage", flush=True)
    api = HfApi()
    parent = api.repo_info(REPO, repo_type="dataset").sha
    before = remote(api, parent, prefixes)
    comparison = compare_inventories(inventories, before)
    if comparison["extra"] or comparison["size_mismatch"] or comparison["digest_mismatch"]:
        raise ValueError("Existing selected remote evidence differs; overwriting is forbidden")
    # Supply immutable in-memory bytes, not potentially changing file handles.
    payloads = {label: (geometry_root / label).read_bytes() for label in comparison["missing"]}
    if any(
        hashlib.sha256(raw).hexdigest() != inventories[label]["sha256"]
        for label, raw in payloads.items()
    ):
        raise ValueError("Local geometry changed before upload")
    revision = parent
    uploaded = False
    if args.upload and payloads:
        commit = api.create_commit(
            REPO,
            repo_type="dataset",
            parent_commit=parent,
            commit_message="Add audited completed Muon primary weight-space stages",
            operations=[
                CommitOperationAdd(path_in_repo=f"{PREFIX}/{label}", path_or_fileobj=raw)
                for label, raw in sorted(payloads.items())
            ],
        )
        revision = commit.oid
        uploaded = True
        print(f"Additions-only geometry commit: {revision}", flush=True)
    after = remote(api, revision, prefixes)
    final = compare_inventories(inventories, after)
    if any(
        file_inventory(geometry_root / label) != record for label, record in inventories.items()
    ):
        raise ValueError("Local geometry changed during archival")
    if args.upload and not final["complete"]:
        raise ValueError("Immutable remote geometry verification failed")
    output = {
        "schema_version": 1,
        "scope": "partial_primary_geometry_content_audit_and_additions_only_archive",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scientific_completion": False,
        "repository": str(root),
        "runs": RUNS,
        "checkpoint_stages": 40,
        "hidden_tensor_rows": tensor_rows,
        "source_bindings": sources,
        "analysis_protocol_sha256": PROTOCOL_SHA,
        "implementation_sha256": file_inventory(Path(__file__))["sha256"],
        "inventory_helper_sha256": file_inventory(
            Path(__import__("embed_optim.artifact_inventory", fromlist=["x"]).__file__)
        )["sha256"],
        "local_inventory": inventories,
        "remote_inventory": after,
        "model_input_inventory": model_inputs,
        "repo_id": REPO,
        "repo_type": "dataset",
        "remote_prefix": PREFIX,
        "parent_commit": parent,
        "verified_commit": revision,
        "uploaded": uploaded,
        "added_files": len(payloads) if uploaded else 0,
        "existing_remote_files_preserved": len(before),
        "inventory": final,
        "boundary": "Raw per-checkpoint weight/displacement geometry only. No 12-run summary, cross-run subspace comparison, retrieval bridge, representation dimension result or scientific optimizer verdict. No existing remote path, card or history was replaced or deleted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: output[k]
                for k in [
                    "checkpoint_stages",
                    "hidden_tensor_rows",
                    "verified_commit",
                    "uploaded",
                    "added_files",
                    "inventory",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
