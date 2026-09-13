from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from embed_optim import dimension_publication as publication
from embed_optim import dimension_release as release
from embed_optim.config import load_matrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_release(tmp_path, monkeypatch):
    """Synthetic full panels; only the expensive model/embedding audit is stubbed."""
    root = tmp_path / "producer" / "embedding-optimizer-study"
    root.mkdir(parents=True)
    shutil.copytree(ROOT / "src", root / "src")
    shutil.copytree(ROOT / "configs", root / "configs")
    monkeypatch.setattr(
        publication, "__file__", str(root / "src/embed_optim/dimension_publication.py")
    )
    monkeypatch.setattr(release, "__file__", str(root / "src/embed_optim/dimension_release.py"))
    monkeypatch.setattr(
        publication, "audit_outputs", lambda *_args: {"fixture": "synthetic encoding"}
    )
    protocol = json.loads((root / release.PROTOCOL).read_text())
    spec = json.loads((root / "configs/beir_representation_probe.json").read_text())
    tasks = sorted(spec["expected"]["task_counts"])
    steps = protocol["inputs"]["checkpoint_stages"]
    configs = load_matrix(root / "configs/dense_no_packing_retrain.yaml")
    checkpoints = [
        {"run_id": "pretrained", "optimizer": "pretrained", "learning_rate": "", "step": 0}
    ]
    checkpoints += [
        {
            "run_id": c.run_id,
            "optimizer": c.optimizer.name,
            "learning_rate": c.optimizer.lr,
            "step": step,
        }
        for c in configs
        for step in steps
    ]
    task_rows, rotations, random_rows, outcomes = [], [], [], []
    for index, row in enumerate(checkpoints):
        optimizer = -1 if index == 0 else publication.OPTIMIZERS.index(row["optimizer"])
        row.update(
            {
                feature: 0.4 + 0.03 * math.sin(index + k) + 0.02 * optimizer
                for k, feature in enumerate(publication.PRIMARY_FEATURES)
            }
        )
        for task_index, task in enumerate(tasks):
            task_row = {
                **row,
                "task": task,
                **{
                    feature: row[feature] + 0.005 * optimizer * (task_index + 1) * (k + 1)
                    for k, feature in enumerate(publication.PRIMARY_FEATURES)
                },
            }
            task_rows.append(task_row)
            if row["step"] in (0, steps[-1]):
                rotations += [
                    {**task_row, "rotation_seed": seed}
                    for seed in protocol["rotation_control"]["seeds"]
                ]
            for fraction in protocol["random_removal"]["removed_fractions"]:
                random_rows += [
                    {
                        **row,
                        "task": task,
                        "removed_fraction": fraction,
                        "draw": draw,
                        "relative_shortlist_ndcg": 0.98 + 0.01 * math.sin(index),
                    }
                    for draw in range(20)
                ]
        if index:
            outcomes.append(
                {
                    "run_id": row["run_id"],
                    "optimizer": row["optimizer"],
                    "learning_rate": row["learning_rate"],
                    "tasks": 14,
                    "stage": steps.index(row["step"]) + 1,
                    "mean_ndcg_at_10": 0.4 + 0.01 * optimizer + 0.03 * math.sin(index),
                }
            )
    dimension_dir = root / release.DIMENSION_DIR
    tables = {
        "checkpoint_summary": checkpoints,
        "task_summary": task_rows,
        "rotation_summary": rotations,
        "random_removal": random_rows,
    }
    outputs = {
        name: publication._atomic_csv(dimension_dir / f"{name}.csv", rows)
        for name, rows in tables.items()
    }
    outputs["coordinate_attribution"] = publication._atomic_text(
        dimension_dir / "synthetic-attribution.txt", "Synthetic coordinates; no model inference.\n"
    )
    export_root = root / "results/dense-primary-dimension-probe/exports/dense"
    receipts = []
    for index in range(61):
        manifest = export_root / f"state-{index}.npz.manifest.json"
        receipt = export_root / f"state-{index}.npz.primary.json"
        publication._atomic_json(manifest, {"synthetic_state": index})
        publication._atomic_json(
            receipt, {"synthetic_state": index, "scientific_completion": False}
        )
        receipts.append(
            {
                "export_manifest": publication._identity(manifest, root),
                "receipt": publication._identity(receipt, root),
            }
        )
    training = export_root / "training_audit.json"
    publication._atomic_json(training, {"fixture": "no actual training"})
    handoff = export_root / "primary_exports.json"
    publication._atomic_json(
        handoff,
        {
            "status": "complete",
            "outputs": receipts,
            "training_audit": publication._identity(training, root),
        },
    )
    publication._atomic_json(
        dimension_dir / "summary_manifest.json",
        {
            "status": "complete",
            "scope": "corrected_dense_dimension_utilization",
            "coverage": {"checkpoint_exports": 60, "tasks": 14, "dimensions": 768},
            "claim_boundary": protocol["claim_boundary"],
            "outputs": outputs,
            "protocol": publication._identity(root / release.PROTOCOL, root),
            "implementation": publication._identity(
                root / "src/embed_optim/dimension_utilization.py", root
            ),
            "primary_export_handoff": publication._identity(handoff, root),
        },
    )
    outcome_dir = root / release.OUTCOME_DIR
    scores = publication._atomic_csv(outcome_dir / "run_stage_scores.csv", outcomes)
    publication._atomic_json(
        outcome_dir / "summary_manifest.json",
        {
            "status": "complete",
            "scope": "corrected_dense_no_packing",
            "coverage": {"runs": 12, "checkpoints": 60, "tasks": 14, "task_units": 840},
            "outputs": {"run_stage_scores": scores},
        },
    )
    publication.build_report(release.publication_args(root))
    release.build_closure(root)
    return root


def test_portable_release_recomputes_in_a_renamed_clone_without_model_payloads(
    synthetic_release, tmp_path
):
    root = synthetic_release
    assert not (root / "outputs").exists()
    assert not (root / "data").exists()
    result = release.audit_closure(root)
    assert result["scientific_completion"] is False
    assert result["checkpoint_revalidated"] is False
    clone = tmp_path / "renamed-clean-clone"
    shutil.copytree(root, clone)
    env = dict(
        os.environ,
        PYTHONPATH=str(clone / "src"),
        CUDA_VISIBLE_DEVICES="",
        OPENBLAS_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    command = [sys.executable, "-m", "embed_optim.dimension_publication", "--portable-audit"]
    checked = subprocess.run(
        command, cwd=clone, env=env, capture_output=True, text=True, timeout=60
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["checkpoints"] == 60
    # A still-present valid producer must not mask a corrupt file in the clone.
    damaged = clone / release.PUBLICATION_DIR / "primary_contrasts.csv"
    damaged.write_text(damaged.read_text() + "changed\n")
    rejected = subprocess.run(
        command, cwd=clone, env=env, capture_output=True, text=True, timeout=60
    )
    assert rejected.returncode != 0


def test_portable_mode_requires_authoring_closure(synthetic_release):
    root = synthetic_release
    manifest = root / release.MANIFEST
    manifest.rename(manifest.with_suffix(".retained"))
    with pytest.raises(FileNotFoundError):
        publication.audit_report(release.publication_args(root), portable=True)


def test_paper_gate_rejects_missing_or_unclosed_dimension_findings(tmp_path):
    from embed_optim.paper_audit import _dimension_publication_status

    assert _dimension_publication_status(tmp_path)["complete"] is False
    summary = tmp_path / release.PUBLICATION_DIR / "summary_manifest.json"
    publication._atomic_json(summary, {"status": "complete"})
    result = _dimension_publication_status(tmp_path)
    assert result["complete"] is False
    assert result["status"] == "invalid_or_stale"


def test_closure_rejects_changed_provenance_even_if_the_listing_is_edited(synthetic_release):
    root = synthetic_release
    manifest_path = root / release.MANIFEST
    payload = json.loads(manifest_path.read_text())
    path = root / "results/dense-primary-dimension-probe/exports/dense/state-0.npz.primary.json"
    path.write_text("changed execution provenance\n")
    for index, record in enumerate(payload["files"]):
        if record["path"] == str(path.relative_to(root)):
            payload["files"][index] = publication._identity(path, root)
    publication._atomic_json(manifest_path, payload)
    with pytest.raises(ValueError, match="identity mismatch"):
        release.audit_closure(root)


def test_archive_closes_real_publication_tables_and_all_61_synthetic_vectors(synthetic_release):
    from embed_optim import dimension_archive as archive
    from embed_optim.primary_dimension_probe import planned_cells

    root = synthetic_release
    shutil.copy(ROOT / "pyproject.toml", root / "pyproject.toml")
    handoff_path = root / archive.EXPORT_ROOT / "primary_exports.json"
    handoff = json.loads(handoff_path.read_text())
    protocol = json.loads((root / release.PROTOCOL).read_text())
    cells = planned_cells(
        load_matrix(root / "configs/dense_no_packing_retrain.yaml"),
        protocol["inputs"]["checkpoint_stages"],
    )
    handoff.update(scope="primary_dense_fixed_probe_exports", cells=cells)
    for index, (cell, row) in enumerate(zip(cells, handoff["outputs"], strict=True)):
        path = root / archive.EXPORT_ROOT / f"{cell}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic encoding transport fixture {index}".encode())
        row.update(cell=cell, export=publication._identity(path, root))
    publication._atomic_json(handoff_path, handoff)
    manifest_path = root / release.DIMENSION_DIR / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["primary_export_handoff"] = publication._identity(handoff_path, root)
    publication._atomic_json(manifest_path, manifest)
    publication.build_report(release.publication_args(root))
    release.build_closure(root)
    # Only model/embedding inference is synthetic. The actual closure, statistical
    # recomputation and source-derived archive selector are exercised together.
    payload, selected = archive.build_bundle(root)
    assert payload["cells"] == cells
    assert (
        len(
            [
                path
                for path in selected.values()
                if path.is_relative_to(root / archive.EXPORT_ROOT) and path.suffix == ".npz"
            ]
        )
        == 61
    )
    assert str(release.PUBLICATION_DIR / "primary_contrasts.csv") in payload["files"]
    assert str(release.MANIFEST) in payload["files"]
    changed = root / archive.EXPORT_ROOT / "pretrained.npz"
    changed.write_bytes(b"changed after closure")
    with pytest.raises(ValueError, match="identity mismatch"):
        archive.build_bundle(root)
