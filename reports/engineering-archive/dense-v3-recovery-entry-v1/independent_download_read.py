"""Independent offline file/seal read against the original native backup receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

COMPLETION_SHA = "1d5aa49c6ec8c89c4454b4b9b641aa5e1bffc09a9967dbac6320cbe129d505f3"
INDEX_SHA = "76e1253373e24525044ebbb459aadcb56afa76da3db629e5b00a652ed44dbe89"
RUN = "verified-v3-muon-3e-4"
STEP = 2345


def checked_json(path, sha):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError("Original reference changed")
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous independent reads")
    complete = checked_json(
        args.launch
        / "observations/all-twelve-training-sixty-checkpoints-evaluation-running-20260910.json",
        COMPLETION_SHA,
    )
    cells = complete["checkpoint_receipts_and_final_local_payloads"]["accepted"]["checkpoints"]
    cell = next(c for c in cells if c["run_id"] == RUN and c["step"] == STEP)
    name = f"{RUN}-step-{STEP}.json"
    receipt = checked_json(args.launch / "backup-receipts" / name, cell["receipt_sha256"][name])
    index = checked_json(args.index, INDEX_SHA)
    record = next(r for r in index["checkpoints"] if r["run_id"] == RUN and r["step"] == STEP)
    if receipt["commit_oid"] != record["revision"] or receipt["prefix"] != record["prefix"]:
        raise ValueError("Portable index does not address the original native source")
    root = args.download / receipt["prefix"]
    expected = {f["path"]: f for f in receipt["checkpoint"]["files"]}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(expected) or len(actual) != 20:
        raise ValueError("Downloaded checkpoint has a different actual inventory")
    for name, entry in expected.items():
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Downloaded source must not follow symlinks")
        sha = hashlib.sha256()
        blob = hashlib.sha1(usedforsecurity=False)
        blob.update(f"blob {entry['bytes']}\0".encode())
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                size += len(block)
                sha.update(block)
                blob.update(block)
        if size != entry["bytes"] or sha.hexdigest() != entry["sha256"]:
            raise ValueError("Independent whole-file SHA-256 comparison failed")
        remote = receipt["remote_inventory"][name]
        if remote["digest"] != (
            sha.hexdigest() if remote["kind"] == "sha256" else blob.hexdigest()
        ):
            raise ValueError("Independent original remote-digest comparison failed")
    seal = json.loads((root / "dense_checkpoint_seal.json").read_bytes())
    contract = json.loads((root / "dense_run_contract.json").read_bytes())
    canonical = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if (
        seal["run_identity_sha256"] != hashlib.sha256(canonical).hexdigest()
        or seal["run_identity_sha256"] != cell["run_identity_sha256"]
        or seal["step"] != STEP
    ):
        raise ValueError("Original native checkpoint/run identity differs")
    if sorted(seal["files"], key=lambda f: f["path"]) != sorted(
        (f for n, f in expected.items() if n != "dense_checkpoint_seal.json"),
        key=lambda f: f["path"],
    ):
        raise ValueError("Native seal does not cover the exact downloaded payloads")
    result = {
        "scope": "independent-original-receipt-checkpoint-download-read",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN,
        "step": STEP,
        "revision": receipt["commit_oid"],
        "files": 20,
        "bytes": sum(f["bytes"] for f in expected.values()),
        "download_root": str(args.download),
        "index_sha256": INDEX_SHA,
        "original_native_receipt_sha256": cell["receipt_sha256"][name]
        if name in cell["receipt_sha256"]
        else cell["receipt_sha256"][f"{RUN}-step-{STEP}.json"],
        "native_seal_and_run_identity_verified": True,
        "network_or_gpu_access": False,
        "pickle_or_model_execution": False,
        "scientific_completion": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
