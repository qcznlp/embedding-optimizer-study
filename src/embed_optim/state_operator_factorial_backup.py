"""Upload and digest-audit one completed state-by-operator training wave."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from .geometry import SCHEMA_VERSION, _sha256
from .incremental_checkpoint_backup import (
    compare_checkpoint_inventories,
    inventory_digest,
    local_checkpoint_inventory,
    remote_checkpoint_inventory,
)
from .matrix import _run_is_complete
from .state_operator_factorial import CALIBRATION_ROOT, MATRIX_ROOT, SCIENTIFIC_PROTOCOL
from .state_operator_factorial_contract import require_factorial_implementation
from .state_operator_factorial_evaluation import load_cell_configs

DEFAULT_REPO = "qcz/embedding-optimizer-study-checkpoints"
DEFAULT_PREFIX = "state-operator-factorial-v1/dense"
RECEIPT_ROOT = Path("reports/state-operator-factorial/checkpoint-backup")
TRAINING_RECEIPT_ROOT = Path("reports/state-operator-factorial/training")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _existing_commit(
    path: Path,
    *,
    state: str,
    seed: int,
    run_id: str,
    prefix: str,
    digest: str,
) -> tuple[str | None, str | None]:
    if not path.is_file():
        return None, None
    previous = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "source_state": state,
        "order_seed": seed,
        "run_id": run_id,
        "remote_prefix": prefix,
        "inventory_sha256": digest,
    }
    if any(previous.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Existing factorial backup receipt differs: {path}")
    oid = previous.get("commit_oid")
    url = previous.get("commit_url")
    if (not isinstance(oid, str) or not oid) or (not isinstance(url, str) or not url):
        raise RuntimeError(f"Existing factorial backup lacks commit provenance: {path}")
    return oid, url


def backup(args: argparse.Namespace) -> dict[str, Any]:
    implementation_path, implementation = require_factorial_implementation()
    protocol_path, _, matrix_path, configs = load_cell_configs(
        args.state,
        args.seed,
        protocol_path=args.protocol,
        matrix_root=args.matrix_root,
        calibration_root=args.calibration_root,
    )
    training_receipt_path = (
        args.training_receipt_root.resolve() / f"{args.state}-seed{args.seed}.json"
    )
    training_receipt = json.loads(training_receipt_path.read_text(encoding="utf-8"))
    if (
        training_receipt.get("status") != "complete"
        or training_receipt.get("verified_runs") != 2
        or training_receipt.get("verified_checkpoints") != 10
        or training_receipt.get("scientific_protocol_sha256") != _sha256(protocol_path)
        or training_receipt.get("implementation_protocol_sha256") != _sha256(implementation_path)
        or training_receipt.get("matrix_sha256") != _sha256(matrix_path)
    ):
        raise ValueError("Factorial training receipt differs from the completed wave")
    api = HfApi()
    receipts = []
    for config in configs:
        if not _run_is_complete(config):
            raise RuntimeError(f"Refusing to back up incomplete factorial run: {config.run_id}")
        local = local_checkpoint_inventory(config.output_dir)
        if not local:
            raise RuntimeError(f"Completed factorial run has no payload: {config.output_dir}")
        prefix = f"{args.remote_prefix.rstrip('/')}/{args.state}/seed{args.seed}/{config.run_id}"
        receipt_path = (
            args.receipt_root.resolve() / args.state / f"seed{args.seed}" / f"{config.run_id}.json"
        )
        digest = inventory_digest(local)
        commit = None
        if not args.audit_only:
            commit = api.upload_folder(
                repo_id=args.repo_id,
                repo_type="model",
                folder_path=config.output_dir,
                path_in_repo=prefix,
                ignore_patterns=[".cache/**", "*.tmp"],
                commit_message=(
                    f"Back up state-operator {args.state} seed {args.seed} {config.run_id}"
                ),
            )
        remote = remote_checkpoint_inventory(api, repo_id=args.repo_id, prefix=prefix)
        audit = compare_checkpoint_inventories(local, remote)
        if not audit["complete"]:
            raise RuntimeError(
                f"Remote factorial inventory differs for {config.run_id}: "
                f"missing={audit['missing'][:3]} extra={audit['extra'][:3]} "
                f"size={audit['size_mismatch'][:3]} digest={audit['digest_mismatch'][:3]}"
            )
        if args.audit_only:
            commit_oid, commit_url = _existing_commit(
                receipt_path,
                state=args.state,
                seed=args.seed,
                run_id=config.run_id,
                prefix=prefix,
                digest=digest,
            )
        else:
            commit_oid = str(getattr(commit, "oid", "")) or None
            commit_url = str(getattr(commit, "commit_url", "")) or None
            if not commit_oid or not commit_url:
                raise RuntimeError("Hugging Face upload returned no commit provenance")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "role": "state_operator_factorial_checkpoint_durability",
            "source_state": args.state,
            "order_seed": args.seed,
            "run_id": config.run_id,
            "local_root": str(config.output_dir),
            "repo_id": args.repo_id,
            "repo_type": "model",
            "remote_prefix": prefix,
            "inventory_sha256": digest,
            "inventory": audit,
            "scientific_protocol_sha256": _sha256(protocol_path),
            "implementation_protocol_sha256": _sha256(implementation_path),
            "implementation_commit": implementation["implementation_commit"],
            "matrix_sha256": _sha256(matrix_path),
            "training_receipt_sha256": _sha256(training_receipt_path),
            "commit_oid": commit_oid,
            "commit_url": commit_url,
        }
        _atomic_json(receipt_path, receipt)
        receipts.append(receipt)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source_state": args.state,
        "order_seed": args.seed,
        "runs": len(receipts),
        "files": sum(item["inventory"]["local_files"] for item in receipts),
        "bytes": sum(item["inventory"]["local_bytes"] for item in receipts),
        "receipts": [
            {
                "run_id": item["run_id"],
                "commit_oid": item["commit_oid"],
                "commit_url": item["commit_url"],
            }
            for item in receipts
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=("adamw_state", "muon_state"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=SCIENTIFIC_PROTOCOL)
    parser.add_argument("--matrix-root", type=Path, default=MATRIX_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=CALIBRATION_ROOT)
    parser.add_argument("--training-receipt-root", type=Path, default=TRAINING_RECEIPT_ROOT)
    parser.add_argument("--receipt-root", type=Path, default=RECEIPT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--remote-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(backup(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
