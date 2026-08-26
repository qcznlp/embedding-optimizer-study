from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from embed_optim.paper_audit import (
    _complete_manifest,
    _macros,
    audit_paper,
    expected_constant_macros,
    load_paper_claim_protocol,
)


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
    assert result["constant_sources"]["dataset_manifest"]["sha256"] == (
        "9facc18bcd1cad8378cea94746a95ab09804bdf3610796bf9013cdfcc486aee8"
    )
    assert result["constant_sources"]["dataset_contract"]["path"].endswith(
        "configs/training_data_contract.json"
    )
    discovery_evidence = result["evidence"]["DiscoveryHeadline"]
    assert len(discovery_evidence) == 4
    assert discovery_evidence[1]["complete"] is True
    assert discovery_evidence[2]["complete"] is True
    assert discovery_evidence[3]["complete"] is False
    assert result["claim_protocol"]["status"] == "prospective_completion_lock"
    assert result["claim_protocol"]["amendments"][0]["headline_contract_changed"] is False
    assert len(result["claim_protocol"]["source_bindings"]) == 10
    assert result["paper_results"]["complete"] is False


def test_paper_claim_protocol_freezes_result_contingent_language_before_completion():
    path, protocol, sources = load_paper_claim_protocol()

    assert path.name == "paper_claim_protocol.json"
    assert protocol["freeze_context"]["strict_beir_valid_units"] == 168
    assert protocol["freeze_context"]["complete_retrieval_matrix_visible"] is False
    assert protocol["freeze_context"]["formal_common_state_output_visible"] is False
    assert protocol["amendments"] == [
        {
            "amended_at": "2026-08-26T19:51:44Z",
            "scope": "documentation_only_weight_spectrum_tier_correction",
            "reason": (
                "Correct the bound paper plan to state that the completed 120-checkpoint "
                "trajectory tier used sketch-rank 0 and that exact spectra are reserved for the "
                "frozen common-state subset; no result-selection or interpretation rule changed."
            ),
            "previous_source_sha256": (
                "2d61c1c1a150269986dbc41786f5b10c7304b45d23148278959ef3d75b72c888"
            ),
            "updated_source_sha256": (
                "adf12c547e4c337a5acb94657b7f6c4207da550c9f2f46ea3ea5098f3e418ce4"
            ),
            "strict_beir_valid_units": 196,
            "strict_beir_expected_units": 1680,
            "complete_retrieval_matrix_visible": False,
            "formal_common_state_output_visible": False,
            "formal_representation_output_visible": False,
            "formal_functional_intervention_output_visible": False,
            "hybrid_adamw_output_visible": False,
            "short_branch_output_visible": False,
            "confirmatory_output_visible": False,
            "headline_contract_changed": False,
            "result_contingent_story_map_changed": False,
        }
    ]
    assert set(protocol["headline_contract"]) == {
        "DiscoveryHeadline",
        "CommonStateHeadline",
        "RepresentationHeadline",
        "InterventionHeadline",
        "ConfirmationHeadline",
    }
    assert (
        "otherwise inconclusive"
        in protocol["headline_contract"]["ConfirmationHeadline"]["selection_rule"]
    )
    assert len(sources) == 10


def test_strict_paper_audit_rejects_pending_headlines():
    with pytest.raises(ValueError, match="Paper is not final"):
        audit_paper(strict=True)


def test_paper_constants_use_distributable_receipt_without_local_500k_data(tmp_path: Path):
    files = [
        "configs/experiment.yaml",
        "configs/training_data_contract.json",
        "configs/confirmatory_protocol.json",
        "configs/hybrid_adamw_control.json",
        "configs/representation_probe.json",
        "configs/short_branch_protocol.json",
        "configs/validation_probe.json",
        "reports/training-dynamics/summary_manifest.json",
        "reports/training-dynamics/optimizer_system_summary.csv",
        "reports/weight-space/summary_manifest.json",
        "reports/weight-space/optimizer_pair_contrast_trajectory.csv",
    ]
    for relative in files:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, destination)

    constants, sources = expected_constant_macros(
        tmp_path / "configs/experiment.yaml",
        tmp_path / "reports/weight-space",
        tmp_path / "reports/training-dynamics",
        repo_root=tmp_path,
    )

    assert constants["NumTrainingQueries"] == "500{,}000"
    assert constants["NumHardNegatives"] == "7"
    assert sources["dataset_manifest"]["local_byte_verification"] is False


def test_strict_evidence_requires_boolean_complete(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version":1,"status":"complete"}\n', encoding="utf-8")
    assert _complete_manifest(path) is False

    path.write_text('{"schema_version":1,"complete":true,"status":"complete"}\n', encoding="utf-8")
    assert _complete_manifest(path) is True


def test_coverage_uses_its_exact_legacy_contract(tmp_path: Path):
    path = tmp_path / "coverage.json"
    path.write_text(
        """{
          "complete": true,
          "observed_results": 1680,
          "expected_results": 1680,
          "observed_checkpoint_summaries": 120,
          "expected_checkpoint_summaries": 120,
          "missing": [],
          "unexpected": []
        }\n""",
        encoding="utf-8",
    )
    assert _complete_manifest(path) is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["missing"] = ["dense/example/1/task"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _complete_manifest(path) is False


def test_retrieval_dynamics_manifest_requires_full_hashed_contract(tmp_path: Path):
    path = tmp_path / "reports" / "retrieval-dynamics" / "summary_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":1,"complete":true}\n', encoding="utf-8")

    assert _complete_manifest(path) is False


def test_outcome_manifest_requires_final_blog_and_source_hash_contract(tmp_path: Path):
    path = tmp_path / "reports" / "outcome-summary.manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":1,"complete":true}\n', encoding="utf-8")

    assert _complete_manifest(path) is False


def test_paper_results_manifest_requires_generated_tex_and_all_evidence(tmp_path: Path):
    path = tmp_path / "reports" / "paper-results.manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":1,"complete":true}\n', encoding="utf-8")

    assert _complete_manifest(path) is False
