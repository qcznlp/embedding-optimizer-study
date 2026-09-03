import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from embed_optim import evaluation_source_archive
from embed_optim.aggregate import EVALUATION_PACKAGES, _evaluation_runtime
from embed_optim.evaluate_matrix import EVALUATION_SOURCE_MODULES
from embed_optim.evaluation_source_provenance import (
    CURRENT_SOURCE_LABELS,
    DISCOVERY_ARCHIVE_LABELS,
    EvaluationSourceProvenanceError,
    archived_evaluation_source_manifest,
    verify_evaluation_source_manifest,
    verify_packaged_archive_git_origins,
)


def _identity(content: bytes) -> dict[str, str | int]:
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _committed_source_fixture(repo: Path) -> dict[str, dict[str, str | int]]:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    manifest = {}
    for label in sorted(CURRENT_SOURCE_LABELS):
        content = f"tracked evaluator source: {label}\n".encode()
        path = repo / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest[label] = _identity(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "record evaluator sources")
    return manifest


def test_packaged_archive_authenticates_without_git_history(tmp_path):
    manifest = archived_evaluation_source_manifest()

    assert set(manifest) == DISCOVERY_ARCHIVE_LABELS
    audit = verify_evaluation_source_manifest(manifest, repo_root=tmp_path)
    assert audit == {
        "schema_version": 1,
        "complete": True,
        "method": "packaged-content-addressed-archive",
        "bundle_id": "8b0db1303ac68789f2d566b708a7b9c24c670d6528e1f19979c6a07501a0f942",
        "source_files": 8,
        "git_blobs": {
            label: record["git_blob_sha1"]
            for label, record in evaluation_source_archive.SOURCE_RECORDS.items()
        },
    }


def test_runtime_archive_verification_never_rewrites_frozen_manifest(tmp_path):
    payload = {
        "schema_version": 2,
        "python": "/historical/python",
        "versions": {package: "1" for package in EVALUATION_PACKAGES},
        "source_files": archived_evaluation_source_manifest(),
    }
    path = tmp_path / "evaluation_runtime.json"
    path.write_text(json.dumps(payload, sort_keys=False, separators=(", ", ": ")) + "\n")
    frozen_bytes = path.read_bytes()

    assert _evaluation_runtime(tmp_path) == payload["versions"]
    assert path.read_bytes() == frozen_bytes


def test_packaged_archive_rejects_tampered_recorded_identity():
    manifest = json.loads(json.dumps(archived_evaluation_source_manifest()))
    manifest["src/embed_optim/aggregate.py"]["sha256"] = "0" * 64

    with pytest.raises(EvaluationSourceProvenanceError, match="differ from the authenticated"):
        verify_evaluation_source_manifest(manifest)


def test_packaged_archive_validates_payload_bytes(monkeypatch):
    packed = evaluation_source_archive._PACKED_B85
    replacement = "0" if packed[-1] != "0" else "1"
    monkeypatch.setattr(evaluation_source_archive, "_PACKED_B85", packed[:-1] + replacement)

    with pytest.raises(EvaluationSourceProvenanceError, match="archive payload is unreadable"):
        archived_evaluation_source_manifest()


def test_source_manifest_rejects_unknown_or_partial_label_mappings():
    manifest = archived_evaluation_source_manifest()
    manifest.pop("scripts/eval/late_interaction.py")

    with pytest.raises(EvaluationSourceProvenanceError, match="exact label mapping"):
        verify_evaluation_source_manifest(manifest)


def test_git_fallback_accepts_only_reachable_blobs_at_exact_paths(tmp_path):
    repo = tmp_path / "repo"
    manifest = _committed_source_fixture(repo)

    audit = verify_evaluation_source_manifest(manifest, repo_root=repo)
    assert audit["method"] == "reachable-git-blobs"
    assert audit["source_files"] == len(CURRENT_SOURCE_LABELS)
    assert set(audit["git_blobs"]) == CURRENT_SOURCE_LABELS

    changed = dict(manifest)
    untracked = repo / "src/embed_optim/evaluate_matrix.py"
    untracked.write_bytes(untracked.read_bytes() + b"uncommitted\n")
    changed["src/embed_optim/evaluate_matrix.py"] = _identity(untracked.read_bytes())
    with pytest.raises(EvaluationSourceProvenanceError, match="untracked, unavailable"):
        verify_evaluation_source_manifest(changed, repo_root=repo)


def test_current_evaluator_emits_the_authenticated_git_fallback_topology():
    emitted_labels = {
        *EVALUATION_SOURCE_MODULES,
        "scripts/eval/dense_parallel.py",
        "scripts/eval/dense_sequential.py",
        "scripts/eval/late_interaction.py",
    }
    assert emitted_labels == CURRENT_SOURCE_LABELS


def test_packaged_archive_matches_declared_historical_git_blobs():
    repo = Path(__file__).resolve().parents[1]
    if not (repo / ".git").exists():
        pytest.skip("historical Git objects are unavailable in a source archive")
    audit = verify_packaged_archive_git_origins(repo)
    assert audit["complete"] is True
    assert audit["source_files"] == 8
