"""New primary checkpoint backup and evaluation admission; no old-result adoption."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .primary_contract import (
    digest,
    read_json,
    require_same,
)
from .primary_v3_contract import PrimaryV3Contract


def remote_inventory(api, repo, prefix, revision):
    from huggingface_hub.hf_api import RepoFile

    result = {}
    for item in api.list_repo_tree(
        repo, path_in_repo=prefix, revision=revision, repo_type="model", recursive=True, expand=True
    ):
        if not isinstance(item, RepoFile):
            continue
        name = Path(item.path).relative_to(prefix).as_posix()
        result[name] = {
            "bytes": item.size,
            "kind": "sha256" if item.lfs else "git_blob_sha1",
            "digest": item.lfs.sha256 if item.lfs else item.blob_id,
        }
    return result


def compare_remote(checkpoint, files, remote):
    if set(remote) != {x["path"] for x in files}:
        raise ValueError("Immutable remote checkpoint inventory differs")
    for row in files:
        item = remote[row["path"]]
        if item["bytes"] != row["bytes"]:
            raise ValueError("Remote checkpoint size differs")
        if item["kind"] == "sha256":
            expected = row["sha256"]
        elif item["kind"] == "git_blob_sha1":
            h = hashlib.sha1(usedforsecurity=False)
            h.update(f"blob {row['bytes']}\0".encode())
            with (Path(checkpoint) / row["path"]).open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    h.update(block)
            expected = h.hexdigest()
        else:
            raise ValueError("Unknown remote digest format")
        if item["digest"] != expected:
            raise ValueError("Immutable remote checkpoint digest differs")


def backup(contract, checkpoint, run_id, step, receipt_path, *, api=None):
    contract = contract.require_execution()
    checked = contract.checkpoint(checkpoint, run_id, step)
    receipt_path = Path(receipt_path)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("Existing backup receipts are immutable; use audit-backup")
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import RemoteEntryNotFoundError

    api = api or HfApi()
    repo = contract.payload["checkpoint_repository"]
    prefix = f"{contract.payload['checkpoint_prefix']}/{contract.sha256}/{run_id}/checkpoint-{step}"
    head = api.repo_info(repo, repo_type="model").sha
    if not re.fullmatch(r"[0-9a-f]{40}", head or ""):
        raise ValueError("Remote parent must be an immutable commit")
    try:
        existing = remote_inventory(api, repo, prefix, head)
    except RemoteEntryNotFoundError:
        existing = {}
    if existing:
        raise ValueError(
            "Remote prefix already exists; recover/audit its original commit, do not overwrite"
        )
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{prefix}/{row['path']}",
            path_or_fileobj=str(Path(checkpoint) / row["path"]),
        )
        for row in checked["files"]
    ]
    commit = api.create_commit(
        repo,
        repo_type="model",
        operations=operations,
        parent_commit=head,
        commit_message=f"Seal revised Dense {run_id} checkpoint {step} [{contract.sha256[:12]}]",
    )
    if not re.fullmatch(r"[0-9a-f]{40}", getattr(commit, "oid", "") or ""):
        raise ValueError("Upload did not return an immutable commit identity")
    remote = remote_inventory(api, repo, prefix, commit.oid)
    compare_remote(checkpoint, checked["files"], remote)
    require_same(contract.checkpoint(checkpoint, run_id, step), checked)
    receipt = {
        "schema_version": 1,
        "scope": "dense_primary_v3_checkpoint_durability",
        "scientific_completion": False,
        "uploaded_at_utc": datetime.now(UTC).isoformat(),
        "repo_id": repo,
        "repo_type": "model",
        "prefix": prefix,
        "commit_oid": commit.oid,
        "checkpoint": checked,
        "remote_inventory": remote,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def audit_backup(contract, checkpoint, run_id, step, receipt_path, *, api=None):
    # This is read-only and may inspect prepared local evidence, but never uploads,
    # rewrites provenance, or searches by title/HEAD for a replacement upload identity.
    checked = contract.checkpoint(checkpoint, run_id, step)
    receipt = read_json(receipt_path)
    if not re.fullmatch(r"[0-9a-f]{40}", receipt.get("commit_oid", "")):
        raise ValueError("Audit must retain the exact immutable upload commit, not HEAD")
    require_same(receipt["checkpoint"], checked)
    if (
        receipt.get("scope") != "dense_primary_v3_checkpoint_durability"
        or receipt.get("scientific_completion") is not False
    ):
        raise ValueError("Not an intermediate primary durability receipt")
    from huggingface_hub import HfApi

    api = api or HfApi()
    expected_prefix = (
        f"{contract.payload['checkpoint_prefix']}/{contract.sha256}/{run_id}/checkpoint-{step}"
    )
    if (
        receipt["prefix"] != expected_prefix
        or receipt["repo_id"] != contract.payload["checkpoint_repository"]
    ):
        raise ValueError("Backup receipt targets a different primary archive")
    remote = remote_inventory(api, receipt["repo_id"], receipt["prefix"], receipt["commit_oid"])
    compare_remote(checkpoint, checked["files"], remote)
    require_same(remote, receipt["remote_inventory"])
    return {
        "durability_verified": True,
        "scientific_completion": False,
        "commit_oid": receipt["commit_oid"],
    }


def evaluation_plan(contract, checkpoint, run_id, step, results_root):
    checked = contract.checkpoint(checkpoint, run_id, step)
    # Content- and protocol-addressed cache; a historical task score cannot satisfy it.
    key = digest({"protocol": contract.sha256, "checkpoint": checked})
    target = Path(results_root) / key
    return {
        "scope": "dense_primary_v3_beir_admission",
        "protocol_sha256": contract.sha256,
        "checkpoint": checked,
        "cache_key": key,
        "results_root": str(target),
        "tasks": contract.payload["evaluation"]["tasks"],
        "scientific_completion": False,
    }


def evaluate(contract, checkpoint, run_id, step, results_root, gpus, *, runner=subprocess.run):
    contract = contract.require_execution()
    from .evaluate_matrix import _validate_formal_runtime
    from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens

    tokens = parse_gpu_tokens(gpus)
    if not set(tokens).issubset(
        {token for pool in contract.payload["gpu_pools"] for token in pool}
    ):
        raise ValueError("Evaluation requested an undeclared GPU")
    _validate_formal_runtime(
        contract.payload["worker_python"],
        contract.training_root / "configs/dense_correctness_candidate.yaml",
    )
    plan = evaluation_plan(contract, checkpoint, run_id, step, results_root)
    output = Path(plan["results_root"])
    manifest_path = output / "primary_admission.json"
    if manifest_path.exists():
        require_same(read_json(manifest_path), plan)
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation cache lacks its exact primary admission record")
    else:
        output.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x") as handle:
            handle.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    command = [
        contract.payload["worker_python"],
        str(contract.repository / "scripts/eval/dense_no_packing_parallel.py"),
        "--gpus",
        gpus,
        "--results_folder",
        str(output / "dense"),
        "--models",
        str(Path(checkpoint).resolve()),
        "--tasks",
        *plan["tasks"],
        "--log_dir",
        str(output / "logs"),
        "--bf16",
        "--fa2",
        "--local",
        "--decontaminated",
    ]
    with acquire_gpu_lease(
        tokens,
        lock_dir=contract.payload["gpu_lease_root"],
        timeout_seconds=60,
        purpose=f"primary-v3-beir:{run_id}:{step}",
    ):
        result = runner(command, check=False, cwd=contract.repository)
    if result.returncode:
        raise RuntimeError("Primary BEIR worker failed; partial results remain preserved")
    require_same(contract.checkpoint(checkpoint, run_id, step), plan["checkpoint"])
    # Worker success alone is not complete 14-task scoring or publication acceptance.
    return {"worker_exited_zero": True, "scientific_completion": False, "plan": plan}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "backup", "audit-backup", "evaluate"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--gpus")
    args = parser.parse_args(argv)
    contract = PrimaryV3Contract.load(
        args.protocol,
        args.repository,
        args.training_root,
        require_released=args.mode in {"backup", "evaluate"},
    )
    if args.mode == "inspect":
        result = contract.checkpoint(args.checkpoint, args.run_id, args.step)
    elif args.mode in {"backup", "audit-backup"}:
        if args.receipt is None:
            parser.error("backup modes require --receipt")
        result = (backup if args.mode == "backup" else audit_backup)(
            contract, args.checkpoint, args.run_id, args.step, args.receipt
        )
    else:
        if args.results_root is None or args.gpus is None:
            parser.error("evaluate requires --results-root and --gpus")
        result = evaluate(
            contract, args.checkpoint, args.run_id, args.step, args.results_root, args.gpus
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
