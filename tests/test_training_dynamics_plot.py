from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from embed_optim.training_dynamics_plot import plot_training_dynamics


def _copy_summary(tmp_path: Path) -> Path:
    source = Path("reports/training-dynamics")
    destination = tmp_path / "reports" / "training-dynamics"
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination)
    return destination


def test_training_dynamics_plot_binds_complete_sources(tmp_path: Path):
    summary = _copy_summary(tmp_path)

    result = plot_training_dynamics(summary)

    assert result["complete"] is True
    assert result["sources"]["stages"]["rows"] == 120
    assert result["sources"]["optimizer_systems"]["rows"] == 6
    for name in ("training_loss_dynamics.svg", "system_tradeoffs.svg"):
        path = summary / name
        assert path.read_text(encoding="utf-8").startswith("<?xml")
        assert result["outputs"][name]["bytes"] == path.stat().st_size


def test_training_dynamics_plot_rejects_tampered_table(tmp_path: Path):
    summary = _copy_summary(tmp_path)
    table = summary / "stage_dynamics.csv"
    table.write_text(table.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its manifest"):
        plot_training_dynamics(summary)


def test_training_dynamics_plot_rejects_incomplete_manifest(tmp_path: Path):
    summary = _copy_summary(tmp_path)
    manifest_path = summary / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"]["checkpoints"] = 119
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="strict complete discovery summary"):
        plot_training_dynamics(summary)
