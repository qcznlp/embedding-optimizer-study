from __future__ import annotations

import json
from pathlib import Path

import pytest
import tomllib

from embed_optim import three_regime_diagnostic
from embed_optim.three_regime_diagnostic import build_diagnostic, load_protocol


def test_protocol_discloses_outcome_visibility_and_claim_boundary() -> None:
    path, protocol = load_protocol("configs/three_regime_diagnostic.json")

    assert path.name == "three_regime_diagnostic.json"
    assert protocol["timing"]["confirmatory_final_beir_visible"] is True
    assert protocol["timing"]["confirmatory_dynamics_results_visible"] is False
    assert protocol["timing"]["candidate_breadth_data_or_scores_visible"] is False
    assert "post-hoc" in protocol["claim_boundary"]
    assert "cannot alter" in protocol["claim_boundary"]


def test_diagnostic_reconstructs_all_dense_runs_and_both_reversals(tmp_path: Path) -> None:
    manifest = build_diagnostic(output_dir=tmp_path / "report")

    assert manifest["runs"] == 12
    assert manifest["comparisons"] == 2
    assert manifest["decision"] == "observed_for_both_muon_family_optimizers"

    rows = (tmp_path / "report" / "high_dose_contrasts.csv").read_text().splitlines()
    assert len(rows) == 3
    assert all(line.endswith(",True") for line in rows[1:])
    assert not any("late" in line.lower() for line in rows)


def test_audit_recomputes_outputs_and_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "report"
    expected = build_diagnostic(output_dir=output)
    audited = build_diagnostic(output_dir=output, audit_only=True)
    assert audited == expected

    summary = output / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output differs"):
        build_diagnostic(output_dir=output, audit_only=True)


def test_protocol_rejects_changed_manifest_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path.cwd()
    monkeypatch.setattr(three_regime_diagnostic, "_repository_root", lambda _path: repository)
    payload = json.loads(Path("configs/three_regime_diagnostic.json").read_text())
    payload["sources"]["mechanism_bridge"]["manifest_sha256"] = "0" * 64
    protocol = tmp_path / "configs" / "three_regime_diagnostic.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest differs"):
        load_protocol(protocol)


def test_distribution_declares_cli_protocol_and_audited_reports() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["embed-optim-three-regime-diagnostic"] == (
        "embed_optim.three_regime_diagnostic:main"
    )
    data_files = project["tool"]["setuptools"]["data-files"]
    assert (
        "configs/three_regime_diagnostic.json"
        in data_files["share/embedding-optimizer-study/configs"]
    )
    assert data_files["share/embedding-optimizer-study/reports/three-regime-diagnostic"] == [
        "reports/three-regime-diagnostic/*.csv",
        "reports/three-regime-diagnostic/*.json",
        "reports/three-regime-diagnostic/*.md",
    ]


def test_checked_in_json_provenance_is_checkout_portable() -> None:
    report = Path("reports/three-regime-diagnostic")

    for path in report.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "/root/" not in text
        assert "embedding-optimizer-study-story" not in text
