from __future__ import annotations

import json
from pathlib import Path

from embed_optim.config import load_matrix
from embed_optim.confirmatory_matrix import (
    audit_confirmatory_matrices,
    generate_confirmatory_matrices,
)
from embed_optim.geometry import _sha256


def _fixture(tmp_path: Path, monkeypatch):
    root = Path(__file__).parents[1]
    protocol = json.loads((root / "configs" / "confirmatory_protocol.json").read_text())
    validation_spec = root / "configs" / "validation_probe.json"
    discovery_matrix = root / "configs" / "experiment.yaml"
    discovery_runs = load_matrix(discovery_matrix)
    selected = []
    for family in ("dense", "late"):
        for optimizer in ("adamw", "muon", "normuon"):
            candidates = sorted(
                (
                    run
                    for run in discovery_runs
                    if run.model_family == family and run.optimizer.name == optimizer
                ),
                key=lambda run: run.optimizer.lr,
            )
            winner = candidates[0 if family == "dense" else -1]
            selected.append(
                {
                    "family": family,
                    "optimizer": optimizer,
                    "run_id": winner.run_id,
                    "learning_rate": winner.optimizer.lr,
                    "validation_contrastive_loss": 1.0,
                    "validation_positive_margin": 0.0,
                    "optimizer_config": winner.as_dict()["optimizer"],
                    "model_name": winner.model_name,
                    "model_revision": winner.model_revision,
                }
            )
    selection_path = tmp_path / "recipe_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "selection_rule": {},
                "validation_spec": {
                    "path": str(validation_spec),
                    "sha256": _sha256(validation_spec),
                },
                "selected": selected,
            }
        ),
        encoding="utf-8",
    )
    protocol["recipe_selection"]["source"] = str(selection_path)
    protocol["training"]["matrix_output_dir"] = str(tmp_path / "matrices")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {},
        "negative_pool": {},
        "query_positive_identity_sha256": "q" * 64,
        "views": [
            {
                "seed": seed,
                "manifest_sha256": f"{index + 1:064x}",
                "rows": 500000,
            }
            for index, seed in enumerate(protocol["confirmatory_data"]["seeds"])
        ],
    }
    monkeypatch.setattr(
        "embed_optim.confirmatory_matrix.audit_confirmatory_data", lambda _path: receipt
    )
    return protocol_path, discovery_matrix, validation_spec, tmp_path / "matrices"


def test_generate_and_audit_three_family_specific_seed_matrices(tmp_path: Path, monkeypatch):
    protocol, discovery, validation, output = _fixture(tmp_path, monkeypatch)

    manifest = generate_confirmatory_matrices(
        protocol,
        experiment_matrix=discovery,
        validation_spec=validation,
        output_dir=output,
    )
    audit = audit_confirmatory_matrices(
        protocol,
        experiment_matrix=discovery,
        validation_spec=validation,
        output_dir=output,
    )

    assert len(manifest["matrices"]) == 3
    assert audit == {
        "status": "complete",
        "manifest_sha256": _sha256(output / "manifest.json"),
        "matrices": 3,
        "runs": 18,
    }
    for seed in (314159, 271828, 161803):
        runs = load_matrix(output / f"seed{seed}.yaml")
        assert len(runs) == 6
        assert {run.seed for run in runs} == {seed}
        assert {run.model_family for run in runs} == {"dense", "late"}
        assert {run.optimizer.name for run in runs} == {"adamw", "muon", "normuon"}
        for optimizer in ("adamw", "muon", "normuon"):
            dense = next(
                run
                for run in runs
                if run.model_family == "dense" and run.optimizer.name == optimizer
            )
            late = next(
                run
                for run in runs
                if run.model_family == "late" and run.optimizer.name == optimizer
            )
            assert dense.optimizer.lr != late.optimizer.lr
