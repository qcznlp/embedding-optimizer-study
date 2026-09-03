from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from embed_optim.geometry import _sha256
from embed_optim.temporal_short_branch_predictors import (
    FIELDS,
    _matrix_metrics,
    audit_predictors,
    build_predictors,
)


def _audit(
    manifest: Path,
    *,
    protocol: Path,
    matrix: Path,
    output: Path,
    cache_dir: Path,
) -> dict:
    return audit_predictors(
        manifest,
        protocol=protocol,
        analysis_protocol=protocol,
        families=("dense",),
        scope_amendment=protocol,
        experiment_matrix=matrix,
        matrix_dir=None,
        output_csv=output,
        cache_dir=cache_dir,
    )


def test_exact_spectrum_metrics_have_normalized_energy_partition():
    metrics = _matrix_metrics(torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5])))
    assert 0 < metrics["stable"] <= 1
    assert 0 < metrics["entropy"] <= 1
    assert metrics["head"] + metrics["middle"] + metrics["tail"] == pytest.approx(1.0)
    assert metrics["energy"] == pytest.approx(21.25)


def test_missing_checkpoint_writes_pending_and_audit_rejects(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("runs: []")
    missing = tmp_path / "missing"
    jobs = [{"seed": 1, "operator": "adamw", "stage": 1, "reference": missing, "current": missing}]
    monkeypatch.setattr(
        "embed_optim.temporal_short_branch_predictors._jobs", lambda *a: (jobs, tmp_path / "unused")
    )
    monkeypatch.setattr("embed_optim.temporal_short_branch_predictors._load_spec", lambda path: {})
    monkeypatch.setattr(
        "embed_optim.temporal_short_branch_predictors.resolve_scope",
        lambda families, scope: (("dense",), {"path": str(scope)}),
    )
    manifest = tmp_path / "predictors.manifest.json"
    result = build_predictors(
        protocol=protocol,
        analysis_protocol=protocol,
        families=("dense",),
        scope_amendment=protocol,
        experiment_matrix=matrix,
        matrix_dir=None,
        output_csv=tmp_path / "p.csv",
        manifest_path=manifest,
        cache_dir=tmp_path / "cache",
        device="cpu",
    )
    assert result["status"] == "pending-not-claimable"
    with pytest.raises(RuntimeError, match="pending"):
        _audit(
            manifest,
            protocol=protocol,
            matrix=matrix,
            output=tmp_path / "p.csv",
            cache_dir=tmp_path / "cache",
        )


def test_complete_predictor_manifest_is_hash_bound(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("runs: []")
    matrix_manifest = tmp_path / "matrix-manifest.json"
    matrix_manifest.write_text("{}")
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "model.safetensors").write_text("reference")
    current = tmp_path / "current"
    current.mkdir()
    (current / "model.safetensors").write_text("current")
    jobs = [
        {
            "seed": seed,
            "operator": operator,
            "stage": stage,
            "reference": reference,
            "current": current,
        }
        for seed in (314159, 271828, 161803)
        for operator in ("adamw", "muon", "normuon")
        for stage in range(1, 6)
    ]
    monkeypatch.setattr(
        "embed_optim.temporal_short_branch_predictors._jobs", lambda *a: (jobs, matrix_manifest)
    )
    monkeypatch.setattr("embed_optim.temporal_short_branch_predictors._load_spec", lambda path: {})
    monkeypatch.setattr(
        "embed_optim.temporal_short_branch_predictors.resolve_scope",
        lambda families, scope: (("dense",), {"path": str(scope)}),
    )
    calls = []

    def fake_predictors(*args):
        calls.append(args)
        return (
            {
                "stable": 0.5,
                "entropy": 0.6,
                "head": 0.7,
                "middle": 0.2,
                "tail": 0.1,
                "row_cv": 0.3,
                "update_frobenius_norm": 2.0,
                "weight_frobenius_norm": 3.0,
            },
            [
                {
                    "name": f"layers.{index}",
                    "shape": [1, 1 if index < 87 else 110_297_001],
                    "parameters": 1 if index < 87 else 110_297_001,
                }
                for index in range(88)
            ],
        )

    monkeypatch.setattr(
        "embed_optim.temporal_short_branch_predictors.checkpoint_predictors", fake_predictors
    )
    monkeypatch.setattr(
        "embed_optim.temporal_short_branch_predictors._checkpoint_identity",
        lambda path: {
            "path": str(path),
            "files": [
                {
                    "path": str((path / "model.safetensors").resolve()),
                    "bytes": (path / "model.safetensors").stat().st_size,
                    "sha256": _sha256(path / "model.safetensors"),
                }
            ],
        },
    )
    output = tmp_path / "predictors.csv"
    manifest = tmp_path / "predictors.manifest.json"
    cache_dir = tmp_path / "cache"
    result = build_predictors(
        protocol=protocol,
        analysis_protocol=protocol,
        families=("dense",),
        scope_amendment=protocol,
        experiment_matrix=matrix,
        matrix_dir=None,
        output_csv=output,
        manifest_path=manifest,
        cache_dir=cache_dir,
        device="cpu",
    )
    assert result["complete"] is True
    assert len(calls) == 45
    assert result["output"]["fields"] == FIELDS
    assert (
        _audit(
            manifest,
            protocol=protocol,
            matrix=matrix,
            output=output,
            cache_dir=cache_dir,
        )["output"]["rows"]
        == 45
    )
    build_predictors(
        protocol=protocol,
        analysis_protocol=protocol,
        families=("dense",),
        scope_amendment=protocol,
        experiment_matrix=matrix,
        matrix_dir=None,
        output_csv=output,
        manifest_path=manifest,
        cache_dir=cache_dir,
        device="cpu",
    )
    assert len(calls) == 45
    original_manifest = json.loads(manifest.read_text())
    alternate_protocol = tmp_path / "alternate-protocol.json"
    alternate_protocol.write_bytes(protocol.read_bytes())
    with pytest.raises(RuntimeError, match="CLI protocol/scope bindings"):
        audit_predictors(
            manifest,
            protocol=protocol,
            analysis_protocol=alternate_protocol,
            families=("dense",),
            scope_amendment=protocol,
            experiment_matrix=matrix,
            matrix_dir=None,
            output_csv=output,
            cache_dir=cache_dir,
        )
    altered = json.loads(json.dumps(original_manifest))
    altered["sources"][0]["reference"]["path"] = str(tmp_path / "self-consistent-swap")
    manifest.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="canonical job paths"):
        _audit(
            manifest,
            protocol=protocol,
            matrix=matrix,
            output=output,
            cache_dir=cache_dir,
        )
    manifest.write_text(json.dumps(original_manifest))
    altered = dict(original_manifest)
    altered["sources"] = []
    manifest.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="source identities"):
        _audit(
            manifest,
            protocol=protocol,
            matrix=matrix,
            output=output,
            cache_dir=cache_dir,
        )
    manifest.write_text(json.dumps(original_manifest))
    rows = output.read_text().splitlines()
    rows[-1] = rows[-2]
    output.write_text("\n".join(rows) + "\n")
    altered = dict(original_manifest)
    altered["output"] = {
        **altered["output"],
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
    }
    manifest.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="grid"):
        _audit(
            manifest,
            protocol=protocol,
            matrix=matrix,
            output=output,
            cache_dir=cache_dir,
        )
