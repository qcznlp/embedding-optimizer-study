"""Read/recover the observed commit; preserve the original root-preservation failure.

No remote write is implemented. This separates byte-exact payload durability
from the original unchanged-root guard, which remains failed and unmodified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BACKUP_SHA = "4dbfac65e1aaee5f004bd2af1d4cc4200d9d193329c1d7aaf0858d45bfbdad41"
HELPER = Path("/root/embedding-optimizer-story-refactor/scripts/restore_primary_v3.py")
HELPER_SHA = "bebdfd3d773e550665660ec12fe7451f70e7e5fd419b1ef21814f774762ef33e"
COMMIT = "3c95da08a4c817d5bcd58b5c84df777716a02ca9"
MANIFEST_SHA = "9a69191e47b25eca459a08eb1a3ddb2958d0f652e94b3c089e1c5685a3cf7d6e"
SOURCE = Path(__file__).with_name("backup.py")
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != BACKUP_SHA:
    raise ValueError("Original upload source changed")
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError("Original hashing helper changed")
spec = importlib.util.spec_from_file_location("original_training_backup", SOURCE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def attributes(before, after, prefix):
    targets = [
        prefix + "/native/inputs/admission.json",
        prefix + "/native/training-trajectories.pdf",
    ]
    suffix = "".join(name + " filter=lfs diff=lfs merge=lfs -text\n" for name in targets).encode()
    b.require(
        after == before + suffix,
        "Require exactly two new literal-path LFS rules, old bytes unchanged",
    )
    return targets


def original_content(before, after, old_subtree, new_subtree):
    excluded = {b.NAMESPACE, ".gitattributes"}
    b.require(set(before) == set(after), "Remote root population changed")
    b.require(
        {k: v for k, v in before.items() if k not in excluded}
        == {k: v for k, v in after.items() if k not in excluded},
        "Old remote payload/card entry changed",
    )
    b.require(
        set(new_subtree) == set(old_subtree) | {b.ADDITION} and b.ADDITION not in old_subtree,
        "Unexpected corrected subtree population",
    )
    b.require(
        {k: new_subtree[k] for k in old_subtree} == old_subtree,
        "Old corrected weight subtree changed",
    )
    return len(before) - 2, len(old_subtree)


def reconcile(work):
    preflight, expected = b.prepared(work)
    b.require(preflight["manifest"]["sha256"] == MANIFEST_SHA, "Wrong original upload population")
    upload = b.read(work / "upload.json")
    b.require(
        upload["revision"] == COMMIT
        and upload["prefix"] == preflight["prefix"]
        and upload["manifest"] == preflight["manifest"]
        and upload["files"] == 149,
        "Wrong observed immutable commit",
    )
    b.require(
        not (work / "remote-audit.json").exists(), "Do not manufacture the failed original audit"
    )
    from huggingface_hub import HfApi

    api = HfApi(endpoint="https://huggingface.co", token=False)
    start = b.read(work / "upload-started.json")
    before = b.root_inventory(api, b.PARENT)
    after = b.root_inventory(api, COMMIT)
    old_subtree = b.root_inventory(api, b.PARENT, b.NAMESPACE)
    new_subtree = b.root_inventory(api, COMMIT, b.NAMESPACE)
    b.require(
        before == start["original_root"] and old_subtree == start["original_corrected_subtree"],
        "Original observed remote parent differs",
    )
    roots, subtrees = original_content(before, after, old_subtree, new_subtree)
    attribute_files = []
    for revision, inventory in ((b.PARENT, before), (COMMIT, after)):
        path = work / ("attribute-read-" + revision) / ".gitattributes"
        identity = b.file_identity(path)
        b.require(
            identity["bytes"] == inventory[".gitattributes"]["bytes"]
            and identity["git_blob_sha1"] == inventory[".gitattributes"]["blob_id"],
            "Downloaded attributes differ from immutable remote Git contents",
        )
        attribute_files.append({"revision": revision, "path": str(path), **identity})
    targets = attributes(
        Path(attribute_files[0]["path"]).read_bytes(),
        Path(attribute_files[1]["path"]).read_bytes(),
        preflight["prefix"],
    )
    actual = {}
    for item in api.list_repo_tree(
        b.REPO,
        repo_type="dataset",
        revision=COMMIT,
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
    b.compare_remote(expected, actual)
    for target in targets:
        b.require(
            actual[target.removeprefix(preflight["prefix"] + "/")]["kind"] == "sha256",
            "New attribute rule is not for an actual uploaded LFS file",
        )
    try:
        b.compare_preserved(before, after, old_subtree, new_subtree)
    except ValueError as error:
        b.require(
            str(error) == "Original remote root entries changed", "Different original refusal"
        )
    else:
        raise ValueError("The original unchanged-root failure must remain a failure")
    result = {
        "scope": "read-only-observed-commit-payload-and-attribute-reconciliation",
        "observed_at_utc": b.stamp(),
        "revision": COMMIT,
        "parent_revision": b.PARENT,
        "prefix": preflight["prefix"],
        "manifest": preflight["manifest"],
        "files": len(expected),
        "remote_inventory": actual,
        "payload_digests_verified": True,
        "original_root_byte_preservation_passed": False,
        "original_upload_process_exit_code": 1,
        "original_remote_audit_created": False,
        "old_root_payload_and_card_entries_unchanged": roots,
        "old_corrected_subtrees_unchanged": subtrees,
        "attribute_files": attribute_files,
        "original_attribute_bytes_are_unchanged_prefix": True,
        "new_literal_lfs_targets": targets,
        "remote_mutations_by_this_read": False,
        "new_write_or_guard_relaxation": False,
        "source_code_published": False,
        "scientific_completion": False,
        "source": b.file_identity(Path(__file__).resolve()),
    }
    b.write_new(work / "observed-commit-verification.json", result)
    return result, expected


def recover(work):
    b.require(
        not (work / "download").exists()
        and not (work / "observed-commit-verification.json").exists(),
        "Preserve existing recovery attempt",
    )
    record, expected = reconcile(work)
    destination = work / "download"
    destination.mkdir()
    from huggingface_hub import hf_hub_download

    def transfer(name):
        filename = record["prefix"] + "/" + b.safe_name(name)
        path = Path(
            hf_hub_download(
                b.REPO,
                repo_type="dataset",
                revision=COMMIT,
                filename=filename,
                token=False,
                endpoint="https://huggingface.co",
                local_dir=destination,
            )
        )
        b.require(path == destination / filename, "Different downloaded payload path")
        b.compare_file(path, expected[name])

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(transfer, sorted(expected)))
    root = destination / record["prefix"]
    b.require(
        {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == set(expected),
        "Incomplete recovered file population",
    )
    result = {
        "scope": "actual-complete-anonymous-training-artifact-recovery",
        "completed_at_utc": b.stamp(),
        "revision": COMMIT,
        "prefix": record["prefix"],
        "download_root": str(destination),
        "files": len(expected),
        "bytes": sum(v["bytes"] for v in expected.values()),
        "manifest": record["manifest"],
        "all_payload_hashes_match": True,
        "physical_second_host": False,
        "original_root_byte_preservation_passed": False,
        "original_failed_audit_unchanged": True,
        "remote_mutations_by_this_read": False,
        "model_or_gpu_execution": False,
        "source_publication": False,
        "scientific_completion": False,
    }
    b.write_new(work / "download-verified.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    work = parser.parse_args().workdir
    b.require(
        work.is_absolute()
        and work.parent == Path("/tmp")
        and work.name.startswith("dense-v3-training-artifact-backup.")
        and work.is_dir()
        and not any(p.is_symlink() for p in (work, *work.parents)),
        "Require exact owned work root",
    )
    print(json.dumps(recover(work), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
