"""Authenticate frozen evaluator sources without trusting the current checkout.

Historical result directories bind scores to byte-level source identities in
``evaluation_runtime.json``.  Those identities remain valid when later analysis
code changes, provided every recorded source is recoverable from either the
packaged content-addressed archive or a reachable Git blob at the recorded path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import evaluation_source_archive

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX_SHA1 = re.compile(r"[0-9a-f]{40}")

# The discovery archive predates the GPU lease module.  Its exact eight-label
# topology is intentional and must not be widened by current analysis code.
DISCOVERY_ARCHIVE_LABELS = frozenset(
    {
        "scripts/eval/dense_parallel.py",
        "scripts/eval/dense_sequential.py",
        "scripts/eval/late_interaction.py",
        "src/embed_optim/aggregate.py",
        "src/embed_optim/decontamination.py",
        "src/embed_optim/evaluate_matrix.py",
        "src/embed_optim/evaluation_utils.py",
        "src/embed_optim/pylate_compat.py",
    }
)

# A non-archived manifest may only use the complete source topology emitted by
# the current evaluator.  Individual Git-authenticated files cannot be mixed
# into an arbitrary subset or extended with an unrecognized worker input.
CURRENT_SOURCE_LABELS = frozenset(
    {
        *DISCOVERY_ARCHIVE_LABELS,
        "src/embed_optim/gpu_lease.py",
    }
)


class EvaluationSourceProvenanceError(ValueError):
    """Raised when evaluator source provenance cannot be authenticated."""


def _git_blob_sha1(content: bytes) -> str:
    header = b"blob " + str(len(content)).encode("ascii") + b"\0"
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324 - Git object identity


def _recorded_manifest(source_files: object) -> dict[str, dict[str, str | int]]:
    if not isinstance(source_files, dict) or not source_files:
        raise EvaluationSourceProvenanceError("source manifest is empty or not an object")
    normalized: dict[str, dict[str, str | int]] = {}
    for label, identity in source_files.items():
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(identity, dict)
            or set(identity) != {"bytes", "sha256"}
        ):
            raise EvaluationSourceProvenanceError(
                "source manifest labels and identities do not have the exact schema"
            )
        size = identity.get("bytes")
        digest = identity.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or _HEX_SHA256.fullmatch(digest) is None
        ):
            raise EvaluationSourceProvenanceError(
                f"source identity for {label!r} has invalid bytes/SHA-256"
            )
        normalized[label] = {"bytes": size, "sha256": digest}
    return normalized


def _archive_bundle_id(records: Mapping[str, Mapping[str, str | int]]) -> str:
    canonical = json.dumps(
        {
            "schema_version": evaluation_source_archive.ARCHIVE_SCHEMA_VERSION,
            "sources": records,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validated_archive() -> tuple[dict[str, dict[str, str | int]], dict[str, bytes]]:
    """Decode and independently validate every archived byte and content address."""

    if evaluation_source_archive.ARCHIVE_SCHEMA_VERSION != 1:
        raise EvaluationSourceProvenanceError("unsupported evaluator source archive schema")
    records = evaluation_source_archive.SOURCE_RECORDS
    if not isinstance(records, dict) or set(records) != DISCOVERY_ARCHIVE_LABELS:
        raise EvaluationSourceProvenanceError(
            "packaged evaluator archive has an unexpected exact label mapping"
        )
    if _archive_bundle_id(records) != evaluation_source_archive.BUNDLE_ID:
        raise EvaluationSourceProvenanceError(
            "packaged evaluator archive metadata does not match its bundle address"
        )
    try:
        sources = evaluation_source_archive.archived_sources()
    except (ValueError, TypeError, OSError) as error:
        raise EvaluationSourceProvenanceError(
            f"packaged evaluator archive payload is unreadable: {error}"
        ) from error
    if set(sources) != set(records):
        raise EvaluationSourceProvenanceError(
            "packaged evaluator archive payload has an unexpected exact label mapping"
        )

    normalized: dict[str, dict[str, str | int]] = {}
    for label, record in records.items():
        if not isinstance(record, dict) or set(record) != {
            "bytes",
            "git_blob_sha1",
            "origin_commit",
            "sha256",
        }:
            raise EvaluationSourceProvenanceError(
                f"packaged archive record for {label!r} has an invalid schema"
            )
        size = record.get("bytes")
        digest = record.get("sha256")
        blob_id = record.get("git_blob_sha1")
        origin_commit = record.get("origin_commit")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or _HEX_SHA256.fullmatch(digest) is None
            or not isinstance(blob_id, str)
            or _HEX_SHA1.fullmatch(blob_id) is None
            or not isinstance(origin_commit, str)
            or _HEX_SHA1.fullmatch(origin_commit) is None
        ):
            raise EvaluationSourceProvenanceError(
                f"packaged archive record for {label!r} has invalid identities"
            )
        content = sources[label]
        if not isinstance(content, bytes):
            raise EvaluationSourceProvenanceError(f"packaged archive member {label!r} is not bytes")
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise EvaluationSourceProvenanceError(
                f"packaged archive member {label!r} does not match bytes/SHA-256"
            )
        if _git_blob_sha1(content) != blob_id:
            raise EvaluationSourceProvenanceError(
                f"packaged archive member {label!r} does not match its Git blob address"
            )
        normalized[label] = {"bytes": size, "sha256": digest}
    return normalized, sources


def archived_evaluation_source_manifest() -> dict[str, dict[str, str | int]]:
    """Return the fully validated historical discovery source manifest."""

    manifest, _ = _validated_archive()
    return manifest


def _git(
    repo_root: Path,
    *arguments: str,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    environment = os.environ.copy()
    # Never turn a provenance audit into a network fetch in a partial clone.
    environment["GIT_NO_LAZY_FETCH"] = "1"
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=text,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EvaluationSourceProvenanceError(
            f"Git provenance lookup failed for {repo_root}: {error}"
        ) from error


def _reachable_git_blob(
    repo_root: Path, label: str, identity: Mapping[str, str | int]
) -> str | None:
    objects = _git(repo_root, "rev-list", "--objects", "--all", "--", label, text=True)
    candidates: set[str] = set()
    for line in objects.stdout.splitlines():
        object_id, separator, object_path = line.partition(" ")
        if separator and object_path == label and _HEX_SHA1.fullmatch(object_id):
            candidates.add(object_id)
    for object_id in sorted(candidates):
        try:
            kind = _git(repo_root, "cat-file", "-t", object_id, text=True).stdout.strip()
            if kind != "blob":
                continue
            content = _git(repo_root, "cat-file", "blob", object_id).stdout
        except EvaluationSourceProvenanceError:
            # A shallow/partial clone can enumerate an unavailable object.  It
            # is not authenticated unless its bytes are locally recoverable.
            continue
        if (
            len(content) == identity["bytes"]
            and hashlib.sha256(content).hexdigest() == identity["sha256"]
            and _git_blob_sha1(content) == object_id
        ):
            return object_id
    return None


def _authenticate_reachable_git_sources(
    source_files: Mapping[str, Mapping[str, str | int]], repo_root: Path
) -> dict[str, str]:
    try:
        top_level = Path(
            _git(repo_root, "rev-parse", "--show-toplevel", text=True).stdout.strip()
        ).resolve()
    except EvaluationSourceProvenanceError as error:
        raise EvaluationSourceProvenanceError(
            "source manifest is not archived and no usable Git history is available"
        ) from error
    blobs: dict[str, str] = {}
    for label in sorted(source_files):
        blob_id = _reachable_git_blob(top_level, label, source_files[label])
        if blob_id is None:
            raise EvaluationSourceProvenanceError(
                f"source {label!r} is untracked, unavailable, or does not match a reachable Git blob"
            )
        blobs[label] = blob_id
    return blobs


def verify_packaged_archive_git_origins(repo_root: str | Path) -> dict[str, Any]:
    """Verify that archive members are exact blobs at their declared origin commits."""

    _, sources = _validated_archive()
    root = Path(repo_root).resolve()
    blobs: dict[str, str] = {}
    for label, record in evaluation_source_archive.SOURCE_RECORDS.items():
        revision = f"{record['origin_commit']}:{label}"
        observed = _git(root, "rev-parse", revision, text=True).stdout.strip()
        if observed != record["git_blob_sha1"]:
            raise EvaluationSourceProvenanceError(
                f"archive origin {revision!r} resolves to an unexpected Git object"
            )
        content = _git(root, "cat-file", "blob", observed).stdout
        if content != sources[label]:
            raise EvaluationSourceProvenanceError(
                f"archive member {label!r} differs from its declared historical Git blob"
            )
        blobs[label] = observed
    return {
        "schema_version": 1,
        "complete": True,
        "bundle_id": evaluation_source_archive.BUNDLE_ID,
        "source_files": len(blobs),
        "git_blobs": blobs,
    }


def verify_evaluation_source_manifest(
    source_files: object,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate an evaluator source manifest against immutable source bytes.

    The packaged discovery archive is authoritative for its exact historical
    label mapping.  Other accepted manifests must have the complete current
    evaluator topology and every identity must match a locally reachable Git
    blob recorded at that exact repository path.
    """

    recorded = _recorded_manifest(source_files)
    labels = frozenset(recorded)
    if labels == DISCOVERY_ARCHIVE_LABELS:
        archived, _ = _validated_archive()
        if recorded != archived:
            raise EvaluationSourceProvenanceError(
                "recorded source identities differ from the authenticated archive "
                "for this exact label mapping"
            )
        return {
            "schema_version": 1,
            "complete": True,
            "method": "packaged-content-addressed-archive",
            "bundle_id": evaluation_source_archive.BUNDLE_ID,
            "source_files": len(recorded),
            "git_blobs": {
                label: record["git_blob_sha1"]
                for label, record in evaluation_source_archive.SOURCE_RECORDS.items()
            },
        }

    if labels != CURRENT_SOURCE_LABELS:
        missing = sorted(CURRENT_SOURCE_LABELS - labels)
        unexpected = sorted(labels - CURRENT_SOURCE_LABELS)
        raise EvaluationSourceProvenanceError(
            "source manifest has no authenticated exact label mapping: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if repo_root is None:
        raise EvaluationSourceProvenanceError(
            "non-archived source manifest requires locally available Git history"
        )
    blobs = _authenticate_reachable_git_sources(recorded, Path(repo_root).resolve())
    return {
        "schema_version": 1,
        "complete": True,
        "method": "reachable-git-blobs",
        "bundle_id": None,
        "source_files": len(recorded),
        "git_blobs": blobs,
    }
