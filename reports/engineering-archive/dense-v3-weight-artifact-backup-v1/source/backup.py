"""Additions-only backup of the exact completed v3 numerical artifacts, never code."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from scripts.restore_primary_v3 import file_identity

REPOSITORY = Path("/root/embedding-optimizer-story-refactor")
EXPERIMENT = Path("/root/embedding-optimizer-v3-experiment")
REPO = "qcz/embedding-optimizer-study-analysis-artifacts"
PARENT = "489c606076b0f8a0ef82adc7b85f1fb29da794bc"
PROTOCOL = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
PREFIX_ROOT = f"corrected-dense-correctness-v3/weight-space/{PROTOCOL}"
INDEX_SHA = "76e1253373e24525044ebbb459aadcb56afa76da3db629e5b00a652ed44dbe89"
HELPER_SHA = "bebdfd3d773e550665660ec12fe7451f70e7e5fd419b1ef21814f774762ef33e"
RECEIPTS = {
    "approximate": (
        "launch/weight-geometry/receipts/approximate.completed.json",
        "6b2eab14aa4add03254ad278ded3764745e4a622230da641fd35b9ba2cb9b3e6",
    ),
    "exact": (
        "launch/weight-geometry/receipts/exact.completed.json",
        "30868950a910502b2f5997e32388415c2ae864441f640da85862684de9321f9e",
    ),
    "update-map": (
        "launch/descriptive-update-map/run/completed.json",
        "9d92e745834a585075b5f6a663d714343091515449acef8b312e17e388a07436",
    ),
}
PLAN = (
    "launch/descriptive-update-map/plan.json",
    "2fd1d215b05a3fc4d0674a4fcd4b9eb22d3706caf85794dce4ed846ba999c1ca",
)
READOUT_SHA = "f2490b643ea77d4aa027133aae1df016b25bd3fbb7a1397184606925ce654c51"
ROOTS = {
    "approximate": EXPERIMENT / "analyses/dense-primary-v3-weight-geometry/approximate",
    "exact": EXPERIMENT / "analyses/dense-primary-v3-weight-geometry/exact",
    "update-map": EXPERIMENT / "analyses/dense-primary-v3-descriptive-update-map",
    "update-map-readout": EXPERIMENT / "analyses/dense-primary-v3-descriptive-update-map-readout",
}
SECRETS = re.compile(
    rb"(wandb_v1_[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def safe_name(name):
    require(isinstance(name, str), "Snapshot paths must be strings")
    path = PurePosixPath(name)
    require(
        bool(path.parts)
        and path.as_posix() == name
        and not path.is_absolute()
        and all(p not in (".", "..") for p in path.parts)
        and "\\" not in name,
        "Unsafe snapshot path",
    )
    return name


def read(path, sha=None):
    require(
        path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
        "Original input is not an ordinary file",
    )
    raw = path.read_bytes()
    require(sha is None or hashlib.sha256(raw).hexdigest() == sha, "Original receipt changed")
    return json.loads(raw)


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def select_sources():
    require(
        file_identity(REPOSITORY / "scripts/restore_primary_v3.py")["sha256"] == HELPER_SHA,
        "Transport hashing implementation changed",
    )
    records = {label: read(EXPERIMENT / path, sha) for label, (path, sha) in RECEIPTS.items()}
    selected = {}
    for label, root in ROOTS.items():
        if label in ("approximate", "exact"):
            complete = records[label]
            require(
                complete["checkpoints"] == 60
                and complete["complete_runs"] == 12
                and complete["native_numeric_readback_passed"] is True
                and complete["scientific_completion"] is False,
                "Incomplete genuine weight measurement",
            )
            wanted = dict(complete["outputs"])
            summary = read(root / "summary.json", wanted["summary.json"]["sha256"])
            require(
                summary["primary_protocol_sha256"] == PROTOCOL
                and len(summary["raw_bindings"]) == 132,
                "Wrong native geometry population",
            )
            wanted.update(
                {
                    row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
                    for row in summary["raw_bindings"]
                }
            )
            require(len(wanted) == 137, "Incomplete native geometry file set")
        elif label == "update-map":
            require(
                records[label]["actual_tensor_measurement_completed"] is True,
                "Actual update map was not completed",
            )
            wanted = records[label]["outputs"]
            require(len(wanted) == 10, "Incomplete native map")
        else:
            report = read(root / "readout.json", READOUT_SHA)
            wanted = {**report["outputs"], "readout.json": file_identity(root / "readout.json")}
            require(
                len(wanted) == 7
                and report["all_original_pair_rows_checked"] == 1770
                and report["scientific_completion"] is False,
                "Incomplete original readout",
            )
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        require(actual == set(wanted), "Native output inventory differs; preserve and investigate")
        require(not any(p.is_symlink() for p in root.rglob("*")), "Native symlink refused")
        for name, expected in wanted.items():
            selected[f"{label}/{safe_name(name)}"] = (root / name, expected)
    require(len(selected) == 291, "Do not narrow the complete numerical snapshot")
    for label, (name, sha) in RECEIPTS.items():
        selected[f"provenance/{label}.completed.json"] = (EXPERIMENT / name, {"sha256": sha})
    read(EXPERIMENT / PLAN[0], PLAN[1])
    selected["provenance/map-plan.json"] = (EXPERIMENT / PLAN[0], {"sha256": PLAN[1]})
    index = REPOSITORY / "docs/primary-v3-checkpoints.json"
    require(len(read(index, INDEX_SHA)["checkpoints"]) == 60, "Wrong primary source index")
    selected["provenance/primary-v3-checkpoints.json"] = (index, {"sha256": INDEX_SHA})
    document = Path(__file__).with_name("ARTIFACT_README.md")
    selected["README.md"] = (document, file_identity(document))
    require(len(selected) == 297, "Unexpected snapshot scope")
    return selected


def compare_file(path, expected):
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked payload refused")
    got = file_identity(path)
    require(
        all(got[key] == value for key, value in expected.items()), f"File content differs: {path}"
    )
    return got


def scan_text(path):
    if path.suffix in (".json", ".csv", ".md", ".svg"):
        raw = path.read_bytes()
        require(SECRETS.search(raw) is None, f"Credential-shaped content refused: {path}")
        if path.suffix == ".json":

            def visit(value):
                if isinstance(value, str):
                    require(
                        re.search(r"(^|\n)(from \S+ import |import \w|def \w+\(|class \w+)", value)
                        is None,
                        "Embedded executable source is outside scope",
                    )
                elif isinstance(value, dict):
                    for item in value.values():
                        visit(item)
                elif isinstance(value, list):
                    for item in value:
                        visit(item)

            visit(json.loads(raw))


def prepare(work):
    destination = work / "staging"
    require(not destination.exists(), "Preserve the earlier staging attempt")
    sources = select_sources()
    require(
        shutil.disk_usage(work).free > 12_000_000_000, "Need space for staging and real recovery"
    )
    destination.mkdir()
    files = {}
    for name, (source, expected) in sorted(sources.items()):
        require(
            source.suffix
            in {".json", ".csv", ".safetensors", ".npz", ".png", ".pdf", ".svg", ".md"},
            "Source-code or unexpected payload suffix refused",
        )
        compare_file(source, expected)
        scan_text(source)
        target = destination / safe_name(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = compare_file(target, expected)
    manifest = {
        "schema_version": 1,
        "scope": "complete-primary-v3-native-weight-artifact-backup",
        "created_at_utc": stamp(),
        "primary_protocol_sha256": PROTOCOL,
        "scientific_completion": False,
        "source_code_included": False,
        "runs": 12,
        "stages_per_run": 5,
        "hidden_matrices": 88,
        "native_files": 291,
        "payload_files": len(files),
        "payload_bytes": sum(f["bytes"] for f in files.values()),
        "files": files,
        "boundary": "Unchanged complete numerical artifacts only; no code release, new model execution, full retrieval or functional/causal inference.",
    }
    write_new(destination / "artifact_manifest.json", manifest)
    manifest_id = file_identity(destination / "artifact_manifest.json")
    preflight = {
        "scope": "source-bound-additions-only-weight-backup-preflight",
        "prepared_at_utc": stamp(),
        "repo_id": REPO,
        "repo_type": "dataset",
        "parent_revision": PARENT,
        "prefix": f"{PREFIX_ROOT}/{manifest_id['sha256']}",
        "manifest": manifest_id,
        "native_files": 291,
        "upload_files": len(files) + 1,
        "upload_bytes": manifest["payload_bytes"] + manifest_id["bytes"],
        "source": file_identity(Path(__file__)),
        "helper_sha256": HELPER_SHA,
        "scientific_completion": False,
        "remote_mutations": False,
    }
    write_new(work / "preflight.json", preflight)
    return preflight


def prepared(work):
    preflight = read(work / "preflight.json")
    require(
        preflight["source"] == file_identity(Path(__file__))
        and preflight["helper_sha256"] == HELPER_SHA,
        "Prepared transport source changed",
    )
    require(
        preflight["repo_id"] == REPO and preflight["parent_revision"] == PARENT,
        "Prepared remote target changed",
    )
    manifest = read(work / "staging/artifact_manifest.json", preflight["manifest"]["sha256"])
    sources = select_sources()
    require(
        set(manifest["files"]) == set(sources), "Prepared population differs from native sources"
    )
    for name, (_, original) in sources.items():
        require(
            all(manifest["files"][name][key] == value for key, value in original.items()),
            "Prepared payload identity differs from original native evidence",
        )
    require(
        preflight["prefix"] == f"{PREFIX_ROOT}/{preflight['manifest']['sha256']}",
        "Prepared snapshot prefix changed",
    )
    expected = {**manifest["files"], "artifact_manifest.json": preflight["manifest"]}
    require(
        len(expected) == 298
        and manifest["scientific_completion"] is False
        and manifest["source_code_included"] is False,
        "Prepared snapshot scope changed",
    )
    actual = {
        p.relative_to(work / "staging").as_posix()
        for p in (work / "staging").rglob("*")
        if p.is_file()
    }
    require(actual == set(expected), "Staged inventory changed")
    for name, identity in expected.items():
        compare_file(work / "staging" / safe_name(name), identity)
    return preflight, expected


def root_inventory(api, revision):
    return {
        e.path: {
            "kind": type(e).__name__,
            "tree_id": getattr(e, "tree_id", None),
            "blob_id": getattr(e, "blob_id", None),
            "bytes": getattr(e, "size", None),
        }
        for e in api.list_repo_tree(
            REPO, repo_type="dataset", revision=revision, recursive=False, token=False
        )
    }


def compare_remote(expected, actual):
    require(set(actual) == set(expected), "Remote inventory differs")
    for name, record in actual.items():
        kind = record["kind"]
        require(
            kind in ("sha256", "git_blob_sha1")
            and record["bytes"] == expected[name]["bytes"]
            and record["digest"] == expected[name][kind],
            "Remote file content differs",
        )


def upload(work):
    require(
        not (work / "upload-started.json").exists(),
        "An upload was attempted; inspect its exact outcome instead of blindly retrying",
    )
    preflight, expected = prepared(work)
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(endpoint="https://huggingface.co", token=True)
    require(api.whoami(token=True).get("name") == "qcz", "Configured HF owner differs")
    require(
        api.repo_info(REPO, repo_type="dataset", token=True).sha == PARENT,
        "Remote parent changed; no write attempted",
    )
    before = root_inventory(api, PARENT)
    require("corrected-dense-correctness-v3" not in before, "Existing namespace; inspect first")
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{preflight['prefix']}/{name}",
            path_or_fileobj=str(work / "staging" / name),
        )
        for name in sorted(expected)
    ]
    write_new(
        work / "upload-started.json",
        {
            "started_at_utc": stamp(),
            "preflight": preflight,
            "files": len(operations),
            "original_root": before,
        },
    )
    commit = api.create_commit(
        REPO,
        repo_type="dataset",
        parent_commit=PARENT,
        operations=operations,
        token=True,
        num_threads=2,
        commit_message="Back up complete corrected v3 weight-analysis artifacts (no source release)",
    )
    require(re.fullmatch("[0-9a-f]{40}", commit.oid) is not None, "Missing immutable commit")
    uploaded = {
        "uploaded_at_utc": stamp(),
        "repo_id": REPO,
        "repo_type": "dataset",
        "revision": commit.oid,
        "prefix": preflight["prefix"],
        "manifest": preflight["manifest"],
        "files": len(expected),
        "scientific_completion": False,
        "durability_verified": False,
    }
    write_new(work / "upload.json", uploaded)
    actual = {}
    for item in api.list_repo_tree(
        REPO,
        repo_type="dataset",
        revision=commit.oid,
        path_in_repo=preflight["prefix"],
        recursive=True,
        token=False,
    ):
        if type(item).__name__ == "RepoFile":
            name = item.path.removeprefix(preflight["prefix"] + "/")
            actual[name] = {
                "bytes": item.size,
                "kind": "sha256" if item.lfs else "git_blob_sha1",
                "digest": item.lfs.sha256 if item.lfs else item.blob_id,
            }
    compare_remote(expected, actual)
    after = root_inventory(api, commit.oid)
    require(
        {k: after[k] for k in before} == before
        and set(after) - set(before) == {"corrected-dense-correctness-v3"},
        "Original remote trees or root files changed",
    )
    prepared(work)
    result = {
        **uploaded,
        "verified_at_utc": stamp(),
        "durability_verified": True,
        "remote_inventory": actual,
        "old_root_entries_unchanged": len(before),
        "source_code_uploaded": False,
        "remote_paths_deleted_or_overwritten": False,
    }
    write_new(work / "remote-audit.json", result)
    return {k: v for k, v in result.items() if k != "remote_inventory"}


def download_verify(work):
    destination = work / "download"
    require(not destination.exists(), "Preserve an earlier download attempt")
    preflight, expected = prepared(work)
    audit = read(work / "remote-audit.json")
    require(
        audit["durability_verified"] is True
        and audit["prefix"] == preflight["prefix"]
        and re.fullmatch("[0-9a-f]{40}", audit["revision"]) is not None,
        "Require the original completed immutable remote audit",
    )
    destination.mkdir()
    from huggingface_hub import hf_hub_download

    def transfer(name):
        filename = f"{audit['prefix']}/{safe_name(name)}"
        path = Path(
            hf_hub_download(
                REPO,
                repo_type="dataset",
                revision=audit["revision"],
                filename=filename,
                local_dir=destination,
                token=False,
                endpoint="https://huggingface.co",
                force_download=False,
            )
        )
        require(path == destination / filename, "Different downloaded path")
        compare_file(path, expected[name])

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(transfer, sorted(expected)))
    root = destination / audit["prefix"]
    require(
        {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(expected),
        "Actual recovered inventory differs",
    )
    result = {
        "scope": "actual-complete-anonymous-weight-artifact-download-verification",
        "completed_at_utc": stamp(),
        "revision": audit["revision"],
        "prefix": audit["prefix"],
        "download_root": str(destination),
        "files": len(expected),
        "bytes": sum(f["bytes"] for f in expected.values()),
        "manifest": preflight["manifest"],
        "all_payload_hashes_match": True,
        "gpu_or_model_execution": False,
        "same_physical_host": True,
        "scientific_completion": False,
    }
    write_new(work / "download-verified.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "upload", "download-verify"))
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root = args.workdir
    require(
        root.parent == Path("/tmp")
        and root.name.startswith("dense-v3-weight-artifact-backup.")
        and root.is_dir()
        and not root.is_symlink(),
        "Require a new exact task temporary root",
    )
    result = {"prepare": prepare, "upload": upload, "download-verify": download_verify}[
        args.action
    ](root)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
