"""Restore the immutable complete DenseOn continuation-probe data snapshot.

No model/GPU import or pickle deserialization. Standard-library verification;
anonymous downloads require huggingface_hub. Checkpoint links are integrity
indexes, not a claim that GPU resume or native training admission was repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO = "qcz/embedding-optimizer-study-analysis-artifacts"
REVISION = "cff3f190e169548931fbd33eadcf1279439798e1"
MANIFEST_SHA = "bcb092fe16e11150abc678a6b1e977497afbc6bf977b9dd9085cebff37004081"
PREFIX = "corrected-dense-correctness-v3/continuation-probes-v1/" + MANIFEST_SHA
MANIFEST_BYTES = 169018
TOTAL_FILES = 638
TOTAL_BYTES = 586119383
MODEL_REPO = "qcz/embedding-optimizer-study-checkpoints"
TRAIN_AUTH = "00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f"
STEPS = (79, 157, 235, 313, 391)
RUNS = tuple(
    f"factorial-v3-{state}-{operator}-seed{seed}"
    for state in ("adamw_state", "muon_state")
    for operator in ("adamw", "muon")
    for seed in (314159, 271828, 161803)
)
MODEL_FILES = {
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "factorial_trainer_component.json",
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


def need(value, message):
    if not value:
        raise ValueError(message)


def safe_name(name):
    need(isinstance(name, str), "Snapshot path must be a string")
    path = PurePosixPath(name)
    need(
        path.parts
        and not path.is_absolute()
        and path.as_posix() == name
        and all(p not in (".", "..") for p in path.parts)
        and "\\" not in name,
        "Unsafe snapshot path",
    )
    return name


def verify_file(path, expected):
    need(
        path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
        "Missing or symlinked payload",
    )
    first = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    last = path.stat()
    need(
        all(
            getattr(first, k) == getattr(last, k)
            for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "Payload changed while reading",
    )
    need(
        last.st_size == expected["bytes"] and digest == expected["sha256"],
        "Payload checksum differs: " + str(path),
    )


def files(root):
    manifest = root / "artifact_manifest.json"
    verify_file(manifest, {"bytes": MANIFEST_BYTES, "sha256": MANIFEST_SHA})
    value = json.loads(manifest.read_text())
    need(
        value["scope"] == "actual-dense-v3-complete-continuation-probe-durability-v1"
        and value["source_code_included"] is False
        and value["raw_example_text_included"] is False
        and value["continuation_runs"] == 12
        and value["checkpoint_probes"] == 60
        and value["reference_states"] == 1
        and value["stage_steps"] == list(STEPS),
        "Wrong snapshot scope",
    )
    result = value["files"]
    for name in result:
        safe_name(name)
    result = {**result, "artifact_manifest.json": {"bytes": MANIFEST_BYTES, "sha256": MANIFEST_SHA}}
    need(
        len(result) == TOTAL_FILES and sum(r["bytes"] for r in result.values()) == TOTAL_BYTES,
        "Incomplete immutable snapshot inventory",
    )
    return result


def verify(root):
    expected = files(root)
    need(not any(p.is_symlink() for p in root.rglob("*")), "Symlinked snapshot")
    need(
        {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(expected),
        "Missing or extra snapshot file",
    )
    for name, binding in sorted(expected.items()):
        verify_file(root / name, binding)
    return {
        "scope": "complete-continuation-probe-artifact-integrity",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO,
        "revision": REVISION,
        "prefix": PREFIX,
        "files": TOTAL_FILES,
        "bytes": TOTAL_BYTES,
        "all_checksums_match": True,
        "source_release": False,
        "gpu_execution": False,
        "scientific_completion": False,
    }


def download(destination):
    need(
        not destination.exists()
        and not any(p.is_symlink() for p in (destination, *destination.parents)),
        "Use a new destination; preserve partial attempts",
    )
    destination.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import hf_hub_download

    def fetch(name):
        path = Path(
            hf_hub_download(
                REPO,
                PREFIX + "/" + name,
                repo_type="dataset",
                revision=REVISION,
                token=False,
                local_dir=destination,
            )
        )
        need(path == destination / PREFIX / name, "Unexpected downloaded location")
        return path

    root = fetch("artifact_manifest.json").parent
    expected = files(root)

    def transfer(name):
        verify_file(fetch(name), expected[name])

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(transfer, sorted(set(expected) - {"artifact_manifest.json"})))
    return root, verify(root)


def checkpoint_index(root):
    """Validate all sixty immutable links against the bound original model inventories."""
    expected = files(root)
    rows = []
    for run in RUNS:
        directory = root / "checkpoints" / run
        for filename in ("artifact_manifest.json", "verified.json"):
            name = "checkpoints/" + run + "/" + filename
            verify_file(directory / filename, expected[name])
        manifest = json.loads((directory / "artifact_manifest.json").read_text())
        remote = json.loads((directory / "verified.json").read_text())
        need(
            manifest["run_id"] == run
            and manifest["training_authorization_sha256"] == TRAIN_AUTH
            and manifest["scientific_completion"] is False
            and remote["all_five_checkpoints"] is True
            and remote["anonymous_remote_metadata_verified"] is True
            and re.fullmatch("[0-9a-f]{40}", remote["commit_oid"]) is not None,
            "Incorrect original checkpoint scope or immutable revision",
        )
        need(
            remote["prefix"] == "dense-v3-state-operator-v1/" + TRAIN_AUTH + "/" + run,
            "Wrong checkpoint remote prefix",
        )
        verify_file(directory / "artifact_manifest.json", remote["artifact_manifest"])
        for step in STEPS:
            prefix = f"run/checkpoint-{step}/"
            selected = {
                name.removeprefix(prefix): value
                for name, value in manifest["files"].items()
                if name.startswith(prefix)
            }
            need(set(selected) == MODEL_FILES, "Incomplete checkpoint payload inventory")
            for name, binding in selected.items():
                safe_name(name)
                need(
                    type(binding["bytes"]) is int
                    and binding["bytes"] > 0
                    and re.fullmatch("[0-9a-f]{64}", binding["sha256"]) is not None,
                    "Invalid original checkpoint size or hash",
                )
            rows.append(
                {
                    "run_id": run,
                    "step": step,
                    "repo_id": MODEL_REPO,
                    "revision": remote["commit_oid"],
                    "prefix": remote["prefix"] + "/" + prefix.rstrip("/"),
                    "files": selected,
                    "bytes": sum(item["bytes"] for item in selected.values()),
                }
            )
    need(len(rows) == 60, "Missing a checkpoint link")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "verify", "list-checkpoints"))
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="download: a new directory; verify/list-checkpoints: snapshot containing artifact_manifest.json",
    )
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt:
        need(not args.receipt.exists(), "Preserve any earlier receipt")
    if args.action == "download":
        root, result = download(args.path.absolute())
        result["downloaded_anonymously"] = True
    elif args.action == "verify":
        root, result = args.path.absolute(), verify(args.path.absolute())
    else:
        root = args.path.absolute()
        rows = checkpoint_index(root)
        result = {
            "scope": "existing-immutable-continuation-checkpoint-links",
            "checkpoints": [{k: v for k, v in row.items() if k != "files"} for row in rows],
            "complete_checkpoint_links": 60,
            "model_payloads_downloaded_here": False,
            "scientific_completion": False,
        }
    result["snapshot_root"] = str(root)
    if args.receipt:
        with args.receipt.open("x") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
