"""List, download or verify the complete corrected DenseOn checkpoint artifacts.

Standalone: verification needs only Python's standard library. Downloading uses
huggingface_hub with anonymous access and immutable revisions. This never imports
model/training code or deserializes pickle files. File integrity is not scientific
admission, whole-run reconstruction or permission to resume a training controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO = "qcz/embedding-optimizer-study-checkpoints"
PROTOCOL = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
PREFIX = f"corrected-dense-correctness-v3/dense/{PROTOCOL}"
STEPS = (782, 1563, 2345, 3126, 3907)
RUNS = tuple(
    f"verified-v3-{operator}-{rate}"
    for operator, rates in (
        ("adamw", ("1e-6", "3e-6", "1e-5", "3e-5")),
        ("muon", ("1e-4", "3e-4", "1e-3", "3e-3")),
        ("normuon", ("1e-4", "3e-4", "1e-3", "3e-3")),
    )
    for rate in rates
)
FILES = {
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "dense_checkpoint_seal.json",
    "dense_numerical_contract.json",
    "dense_run_contract.json",
    "model.safetensors",
    "modules.json",
    "optimizer.pt",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
    "scheduler.pt",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "training_args.bin",
}
SCOPE = "corrected-dense-primary-v3-checkpoint-download-index-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hexadecimal(value, length):
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def file_identity(path):
    require(path.is_file() and not path.is_symlink(), f"Not an ordinary file: {path}")
    size = path.stat().st_size
    sha, blob = hashlib.sha256(), hashlib.sha1(usedforsecurity=False)
    blob.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            blob.update(block)
    require(path.stat().st_size == size, f"File size changed while reading: {path}")
    return {"bytes": size, "sha256": sha.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def validate_index(value):
    require(
        value.get("scope") == SCOPE and value.get("schema_version") == 1,
        "Wrong download-index schema",
    )
    require(
        value.get("repo_id") == REPO and value.get("repo_type") == "model",
        "Wrong checkpoint repository",
    )
    require(value.get("protocol_sha256") == PROTOCOL, "Not the corrected primary v3 protocol")
    require(value.get("scientific_completion") is False, "A recovery index cannot certify science")
    records = value.get("checkpoints")
    require(
        isinstance(records, list) and len(records) == 60, "Require all sixty checkpoint entries"
    )
    cells = []
    total_bytes = 0
    for row in records:
        run, step = row.get("run_id"), row.get("step")
        require(
            run in RUNS and type(step) is int and step in STEPS, "Unknown current checkpoint cell"
        )
        cells.append((run, step))
        require(
            row.get("prefix") == f"{PREFIX}/{run}/checkpoint-{step}",
            "Wrong or unsafe remote prefix",
        )
        require(hexadecimal(row.get("revision"), 40), "Require an immutable HF commit, never main")
        require(hexadecimal(row.get("run_identity_sha256"), 64), "Missing native run identity")
        files = row.get("files")
        require(
            isinstance(files, list) and len(files) == 20, "Require all twenty checkpoint payloads"
        )
        require(
            {f.get("path") for f in files} == FILES,
            "Wrong, duplicate or unsafe checkpoint filenames",
        )
        for entry in files:
            require(
                type(entry.get("bytes")) is int and entry["bytes"] > 0, "Invalid payload byte count"
            )
            require(hexadecimal(entry.get("sha256"), 64), "Missing payload SHA-256")
            remote = entry.get("remote")
            require(
                isinstance(remote, dict) and set(remote) == {"bytes", "kind", "digest"},
                "Missing original remote digest",
            )
            require(remote["bytes"] == entry["bytes"], "Local and remote payload sizes differ")
            require(remote["kind"] in ("sha256", "git_blob_sha1"), "Unknown remote digest kind")
            require(
                hexadecimal(remote["digest"], 64 if remote["kind"] == "sha256" else 40),
                "Invalid remote digest",
            )
            if remote["kind"] == "sha256":
                require(remote["digest"] == entry["sha256"], "Remote LFS digest differs")
        require(
            row.get("total_bytes") == sum(f["bytes"] for f in files), "Checkpoint total differs"
        )
        total_bytes += row["total_bytes"]
    require(
        set(cells) == {(r, s) for r in RUNS for s in STEPS},
        "The sixty-cell grid is incomplete or duplicated",
    )
    require(
        value.get("total_files") == 1200 and value.get("total_bytes") == total_bytes,
        "Whole index totals differ",
    )
    return value


def load_index(path, expected_sha256):
    require(hexadecimal(expected_sha256, 64), "Supply the independently trusted index SHA-256")
    require(path.is_file() and not path.is_symlink(), "Index must be an ordinary file")
    content = path.read_bytes()
    require(
        hashlib.sha256(content).hexdigest() == expected_sha256,
        "Download index authentication failed",
    )
    return validate_index(json.loads(content))


def select(index, *, run_id=None, step=None, all_checkpoints=False):
    require(type(all_checkpoints) is bool, "Invalid all-checkpoints flag")
    require(
        not (all_checkpoints and (run_id is not None or step is not None)),
        "Use --all or a run/stage selector, not both",
    )
    require(all_checkpoints or run_id in RUNS, "Explicitly select --all or --run-id")
    require(
        step is None or type(step) is int and step in STEPS, "Unknown scheduled checkpoint step"
    )
    chosen = [
        r
        for r in index["checkpoints"]
        if all_checkpoints or (r["run_id"] == run_id and (step is None or r["step"] == step))
    ]
    require(
        len(chosen) == (60 if all_checkpoints else 5 if step is None else 1),
        "Selection cardinality differs",
    )
    return chosen


def safe_target(destination, relative):
    require(destination.is_absolute(), "Use an explicit absolute destination")
    fragment = PurePosixPath(relative)
    require(
        not fragment.is_absolute() and all(p not in (".", "..") for p in fragment.parts),
        "Unsafe relative path",
    )
    target = destination.joinpath(*fragment.parts)
    require(
        not any(p.is_symlink() for p in (target, *target.parents)),
        "Refuse symlinked destination paths",
    )
    return target


def verify_file(path, entry):
    actual = file_identity(path)
    require(
        actual["bytes"] == entry["bytes"] and actual["sha256"] == entry["sha256"],
        f"Payload content differs; preserved without overwrite: {path}",
    )
    remote = entry["remote"]
    require(actual[remote["kind"]] == remote["digest"], f"Original remote digest differs: {path}")


def existing_payloads(destination, rows, *, require_complete):
    verified = []
    for row in rows:
        root = safe_target(destination, row["prefix"])
        expected = {f["path"]: f for f in row["files"]}
        if root.exists():
            require(root.is_dir(), f"Not a checkpoint directory: {root}")
            actual = []
            for path in root.rglob("*"):
                require(not path.is_symlink(), f"Unexpected symlink: {path}")
                if path.is_file():
                    actual.append(path.relative_to(root).as_posix())
            require(set(actual) <= set(expected), f"Unexpected checkpoint files; preserved: {root}")
        for name, entry in expected.items():
            path = safe_target(destination, f"{row['prefix']}/{name}")
            if path.exists():
                verify_file(path, entry)
                verified.append(path)
            elif require_complete:
                raise ValueError(f"Missing checkpoint payload: {path}")
    return verified


def download(destination, rows, *, resume=False, workers=2):
    """Download only explicit current cells; preserve partial/corrupt/extra content."""
    require(type(workers) is int and 1 <= workers <= 4, "Use one to four download workers")
    safe_target(destination, ".cache")
    require(not destination.exists() or destination.is_dir(), "Destination is not a directory")
    if destination.exists() and any(destination.iterdir()):
        require(resume, "Destination is not empty; inspect it before explicitly using --resume")
    # Check *all* existing selected files before any network call or file write.
    existing = set(existing_payloads(destination, rows, require_complete=False))
    pending = [
        (row, entry)
        for row in rows
        for entry in row["files"]
        if safe_target(destination, f"{row['prefix']}/{entry['path']}") not in existing
    ]
    if pending:
        from huggingface_hub import hf_hub_download

        destination.mkdir(parents=True, exist_ok=True)

        def transfer(item):
            row, entry = item
            relative = f"{row['prefix']}/{entry['path']}"
            target = safe_target(destination, relative)
            require(not target.exists(), "Target appeared during download; refuse overwrite")
            observed = Path(
                hf_hub_download(
                    repo_id=REPO,
                    repo_type="model",
                    revision=row["revision"],
                    filename=relative,
                    local_dir=destination,
                    token=False,
                    force_download=False,
                    endpoint="https://huggingface.co",
                )
            )
            require(observed == target, "HF client returned a different local path")
            safe_target(destination, relative)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(transfer, pending))
    verified = existing_payloads(destination, rows, require_complete=True)
    return {
        "downloaded_files": len(pending),
        "reused_verified_files": len(existing),
        "verified_files": len(verified),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "verify", "download"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--run-id", choices=RUNS)
    parser.add_argument("--step", type=int, choices=STEPS)
    parser.add_argument("--all", action="store_true", dest="all_checkpoints")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    index = load_index(args.index, args.index_sha256)
    rows = select(index, run_id=args.run_id, step=args.step, all_checkpoints=args.all_checkpoints)
    result = {
        "scope": "corrected-primary-v3-checkpoint-recovery",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": args.action,
        "index_sha256": args.index_sha256,
        "checkpoints": [
            {k: row[k] for k in ("run_id", "step", "revision", "prefix", "total_bytes")}
            for row in rows
        ],
        "scientific_completion": False,
        "gpu_or_model_code_executed": False,
    }
    if args.action != "list":
        require(args.destination is not None, "Provide --destination")
        if args.action == "download":
            result.update(
                download(args.destination, rows, resume=args.resume, workers=args.workers)
            )
        else:
            result["verified_files"] = len(
                existing_payloads(args.destination, rows, require_complete=True)
            )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
