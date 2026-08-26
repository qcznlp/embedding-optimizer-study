from __future__ import annotations

from pathlib import Path

import pytest

from embed_optim.paper_audit import _macros, audit_paper


def test_macro_parser_rejects_duplicate_definition(tmp_path: Path):
    path = tmp_path / "results.tex"
    path.write_text(
        "\\newcommand{\\Metric}{1}\n\\newcommand{\\Metric}{2}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate paper result macro"):
        _macros(path)


def test_current_paper_constants_match_strict_sources():
    result = audit_paper()

    assert result["complete"] is False
    assert result["pending_headlines"] == [
        "CommonStateHeadline",
        "ConfirmationHeadline",
        "DiscoveryHeadline",
        "InterventionHeadline",
        "RepresentationHeadline",
    ]
    assert result["constant_macros"]["NumDiscoveryRuns"] == "24"
    assert result["constant_macros"]["NumDiscoveryUnits"] == "1680"
    assert result["constant_macros"]["NumWeightPairs"] == "40"
    assert result["constant_macros"]["MuonFamilyThroughputRatioRange"] == "0.9348--0.9946"
    assert result["constant_macros"]["MuonFamilyStateRatioRange"] == "0.6299--0.6420"
    discovery_evidence = result["evidence"]["DiscoveryHeadline"]
    assert len(discovery_evidence) == 3
    assert discovery_evidence[1]["complete"] is True
    assert discovery_evidence[2]["complete"] is True


def test_strict_paper_audit_rejects_pending_headlines():
    with pytest.raises(ValueError, match="Paper is not final"):
        audit_paper(strict=True)
