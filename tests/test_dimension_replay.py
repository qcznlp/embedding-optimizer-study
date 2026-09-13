from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from embed_optim import dimension_archive as archive
from embed_optim import dimension_replay as replay
from embed_optim import dimension_utilization as dimensions
from embed_optim import primary_dimension_probe as primary
from embed_optim.artifact_inventory import compare_inventories, file_inventory
from embed_optim.config import load_matrix

ROOT = Path(__file__).resolve().parents[1]


def _refresh_contract(path):
    value = json.loads(path.read_text())
    for group in ("source_bindings", "parent_bindings"):
        for name, record in value[group].items():
            record.update(dimensions._identity(path.parents[1] / name, path.parents[1]))
    dimensions._atomic_json(path, value)


def _seal(root):
    payload = root / "payload"
    files = {
        path.relative_to(payload).as_posix(): file_inventory(path)
        for path in sorted(payload.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "scope": "primary_dense_dimension_reconstruction",
        "scientific_completion": False,
        "synthetic_test_fixture": True,
        "files": files,
    }
    data = archive._json_bytes(manifest)
    (root / "archive_manifest.json").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def small_archive(tmp_path_factory):
    """61 states / 224 paired queries / 14 tasks, with 8D synthetic vectors for speed.

    Actual primary exports remain 768D and require the full authoring gate. This
    fixture tests complete panel replay with real numerical kernels, not encoding.
    """
    root = tmp_path_factory.mktemp("dimension-replay") / "downloaded"
    payload = root / "payload"
    exporter = json.loads((ROOT / primary.PROTOCOL).read_text())
    names = {
        *archive.SOURCES,
        *primary.SOURCES,
        *exporter["parent_bindings"],
        str(archive.PROTOCOL),
        str(primary.PROTOCOL),
        "src/embed_optim/__init__.py",
    }
    for name in names:
        destination = payload / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    protocol_path = payload / primary.SCIENTIFIC_PROTOCOL
    protocol = json.loads(protocol_path.read_text())
    protocol["inputs"]["embedding_dimension"] = 8
    dimensions._atomic_json(protocol_path, protocol)
    ids = np.arange(224, dtype=np.int64)
    groups = np.asarray([f"task-{i // 16:02}" for i in range(224)])
    expected = {
        "manifest_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
        "selected_sample_ids_sha256": hashlib.sha256(
            "".join(f"{i}\n" for i in ids).encode()
        ).hexdigest(),
        "task_counts": {f"task-{i:02}": 16 for i in range(14)},
    }
    probe_path = payload / "configs/beir_representation_probe.json"
    dimensions._atomic_json(probe_path, {"expected": expected})
    _refresh_contract(payload / primary.PROTOCOL)
    _refresh_contract(payload / archive.PROTOCOL)
    cells = primary.planned_cells(
        load_matrix(payload / primary.MATRIX), protocol["inputs"]["checkpoint_stages"]
    )
    exports = payload / archive.EXPORT_ROOT
    records = []
    for index, cell in enumerate(cells):
        rng = np.random.default_rng(1701 + index)
        queries = rng.normal(size=(224, 8)).astype(np.float32)
        documents = rng.normal(size=(224, 8, 8)).astype(np.float32)
        documents[:, 0] += queries * 0.35
        queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
        documents /= np.linalg.norm(documents, axis=-1, keepdims=True)
        path = exports / f"{cell}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            sample_ids=ids,
            sample_groups=groups,
            query_embeddings=queries,
            document_embeddings=documents,
        )
        manifest_path = path.with_suffix(".npz.manifest.json")
        dimensions._atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "family": "dense",
                "encoding": {"positive_candidate_index": 0},
                "output": dimensions._identity(path, payload),
                "probe": {**expected, "frozen_spec": {"sha256": dimensions._sha256(probe_path)}},
            },
        )
        receipt = path.with_suffix(".npz.primary.json")
        dimensions._atomic_json(
            receipt, {"synthetic_encoding": True, "scientific_completion": False}
        )
        records.append(
            {
                "cell": cell,
                "export": dimensions._identity(path, payload),
                "export_manifest": dimensions._identity(manifest_path, payload),
                "receipt": dimensions._identity(receipt, payload),
            }
        )
    handoff = exports / "primary_exports.json"
    dimensions._atomic_json(
        handoff,
        {
            "status": "complete",
            "scope": "primary_dense_fixed_probe_exports",
            "cells": cells,
            "outputs": records,
            "scientific_completion": False,
            "synthetic_encoding": True,
        },
    )
    args = Namespace(
        repository=payload,
        input_root=exports,
        protocol=protocol_path,
        output_dir=payload / archive.release.DIMENSION_DIR,
        analysis_scope="corrected",
        skip_rotation=False,
    )
    paths = [exports / "pretrained.npz", *sorted(exports.glob("*/checkpoint-*.npz"))]
    dimensions._compute_features(args, protocol, paths, 60, dimensions._identity(handoff, payload))
    return root, _seal(root)


def test_replay_recomputes_the_complete_panel_offline_from_only_downloaded_files(
    small_archive, tmp_path
):
    original, sha = small_archive
    root = tmp_path / "different-machine" / "renamed-snapshot"
    shutil.copytree(original, root)
    output = tmp_path / "fresh-reconstruction"
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "payload/src"),
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    program = (
        "import socket, sys; "
        "deny=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('unexpected network')); "
        "socket.socket.connect=deny; socket.getaddrinfo=deny; "
        "from embed_optim.dimension_replay import main; "
        f"main({['--archive-root', str(root), '--expected-sha256', sha, '--output-dir', str(output)]!r}); "
        "assert 'torch' not in sys.modules; assert 'datasets' not in sys.modules"
    )
    checked = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["complete"] is True and result["features_recomputed"] is True
    assert result["scientific_completion"] is False and result["checkpoint_revalidated"] is False
    assert {
        k: v["rows"] for k, v in result["comparisons"].items() if k != "coordinate_attribution"
    } == {
        "checkpoint_summary": 61,
        "task_summary": 854,
        "random_removal": 68320,
        "rotation_summary": 546,
    }
    assert all(
        v["exact_cells"] for k, v in result["comparisons"].items() if k != "coordinate_attribution"
    )
    manifest = json.loads((output / "summary_manifest.json").read_text())
    assert manifest["scope"] == "reconstruction_dense_dimension_utilization"
    assert manifest["primary_export_handoff"] is None
    from embed_optim.dimension_publication import _load_dimension_tables

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        _load_dimension_tables(output, root / "payload", {})
    assert archive.audit_download(root, sha)["complete"] is True


def test_replay_rejects_changed_running_code_even_with_a_resealed_outer_manifest(
    small_archive, tmp_path
):
    original, _ = small_archive
    root = tmp_path / "changed"
    shutil.copytree(original, root)
    path = root / "payload/src/embed_optim/dimension_utilization.py"
    path.write_text(path.read_text() + "# different implementation\n")
    sha = _seal(root)
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="Running replay implementation differs"):
        replay.replay(root, sha, output)
    assert not output.exists()


def test_replay_never_overwrites_archive_or_existing_outputs(small_archive, tmp_path):
    root, sha = small_archive
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("preserve")
    for destination in (existing, root / "new-report"):
        with pytest.raises(FileExistsError, match="new output directory"):
            replay.replay(root, sha, destination)
    assert (existing / "keep.txt").read_text() == "preserve"


def test_csv_comparison_checks_values_not_only_digests_and_sizes(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("run_id,task,value\nmuon,task-0,0.12\n")
    right.write_text("run_id,task,value\nmuon,task-0,0.13\n")
    assert replay.compare_csv(left, right)["complete"] is False
    right.write_text("run_id,task,value\nadamw,task-0,0.12\n")
    assert replay.compare_csv(left, right)["complete"] is False


def test_replay_records_numerical_mismatch_and_preserves_original_files(small_archive, tmp_path):
    original, _ = small_archive
    root = tmp_path / "numerical-mismatch"
    shutil.copytree(original, root)
    report = root / "payload" / archive.release.DIMENSION_DIR
    table = report / "checkpoint_summary.csv"
    with table.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["baseline_margin"] = str(float(rows[0]["baseline_margin"]) + 0.01)
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = report / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["checkpoint_summary"] = dimensions._identity(table, root / "payload")
    dimensions._atomic_json(manifest_path, manifest)
    sha = _seal(root)
    result = replay.replay(root, sha, tmp_path / "mismatch-output")
    assert result["status"] == "mismatch" and result["complete"] is False
    assert result["comparisons"]["checkpoint_summary"]["mismatched_cells"] == 1
    assert archive.audit_download(root, sha)["complete"] is True


def test_lightweight_inventory_matches_checkpoint_digest_format(tmp_path):
    from embed_optim.incremental_checkpoint_backup import (
        _file_digests,
        compare_checkpoint_inventories,
    )

    path = tmp_path / "payload.bin"
    path.write_bytes(b"same digest convention\n")
    record = file_inventory(path)
    assert (record["sha256"], record["git_blob_sha1"]) == _file_digests(path)
    local = {"payload.bin": record}
    remote = {
        "payload.bin": {"size": record["size"], "digest_kind": "sha256", "digest": record["sha256"]}
    }
    assert compare_inventories(local, remote) == compare_checkpoint_inventories(local, remote)
