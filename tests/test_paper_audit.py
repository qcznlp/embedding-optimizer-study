from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from embed_optim.paper_audit import (
    PAPER_RESULT_TABLE_PATHS,
    PAPER_SOURCE_TABLE_PATHS,
    _complete_manifest,
    _final_document_language_problems,
    _macros,
    _paper_result_tables_complete,
    _paper_source_tables_complete,
    audit_paper,
    expected_constant_macros,
    load_paper_claim_protocol,
    main,
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
    assert isinstance(discovery_evidence[3]["complete"], bool)
    assert result["claim_protocol"]["status"] == "prospective_completion_lock"
    assert result["claim_protocol"]["amendments"][0]["headline_contract_changed"] is False
    assert len(result["claim_protocol"]["source_bindings"]) == 11
    assert result["paper_results"]["complete"] is False
    assert result["document_language_problems"]


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
        },
        {
            "amended_at": "2026-08-27T05:54:04Z",
            "scope": "prospective_confirmatory_inference_correction",
            "reason": (
                "Correct the bound paper plan to match the retained aggregate MTEB evidence, "
                "which supports seed-by-task but not query-level inference, and prospectively "
                "fix Bonferroni familywise intervals over all six prespecified contrasts before "
                "any confirmatory output exists."
            ),
            "previous_source_sha256": (
                "adf12c547e4c337a5acb94657b7f6c4207da550c9f2f46ea3ea5098f3e418ce4"
            ),
            "updated_source_sha256": (
                "f77c22170144adcab3364f9e19167984727ba6edf8899dcf421158ef0588cdf0"
            ),
            "strict_beir_valid_units": 322,
            "strict_beir_expected_units": 1680,
            "complete_retrieval_matrix_visible": False,
            "formal_common_state_output_visible": False,
            "formal_representation_output_visible": False,
            "formal_functional_intervention_output_visible": False,
            "hybrid_adamw_output_visible": False,
            "short_branch_output_visible": False,
            "confirmatory_output_visible": False,
            "headline_contract_changed": True,
            "result_contingent_story_map_changed": False,
        },
        {
            "amended_at": "2026-08-27T06:37:30Z",
            "scope": "prospective_rope_basis_symmetry_correction",
            "reason": (
                "Correct the bound paper plan so the attention basis diagnostic uses only "
                "split-half SO(2) rotations that commute with ModernBERT RoPE, bind the exact "
                "prospective diagnostic protocol, and reject the invalid claim that arbitrary "
                "orthogonal head rotations preserve post-RoPE logits."
            ),
            "previous_source_sha256": (
                "f77c22170144adcab3364f9e19167984727ba6edf8899dcf421158ef0588cdf0"
            ),
            "updated_source_sha256": (
                "3296f4882f1a68e96a0ee4a1608bc47155b776d5078dc40d6cfb654e096cc0c3"
            ),
            "strict_beir_valid_units": 340,
            "strict_beir_expected_units": 1680,
            "complete_retrieval_matrix_visible": False,
            "formal_common_state_output_visible": False,
            "formal_basis_output_visible": False,
            "formal_representation_output_visible": False,
            "formal_functional_intervention_output_visible": False,
            "hybrid_adamw_output_visible": False,
            "short_branch_output_visible": False,
            "confirmatory_output_visible": False,
            "headline_contract_changed": False,
            "result_contingent_story_map_changed": False,
        },
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
    assert (
        "familywise interval for headline sign language"
        in protocol["headline_contract"]["ConfirmationHeadline"]["selection_rule"]
    )
    assert len(sources) == 11


def test_strict_paper_audit_rejects_pending_headlines():
    with pytest.raises(ValueError, match="Paper is not final"):
        audit_paper(strict=True)


def test_strict_paper_audit_cli_reports_pending_evidence(capsys):
    with pytest.raises(SystemExit, match="1"):
        main(["--strict"])

    result = json.loads(capsys.readouterr().out)
    assert result["complete"] is False
    assert result["strict"] is True
    assert result["pending_headlines"] == [
        "CommonStateHeadline",
        "ConfirmationHeadline",
        "DiscoveryHeadline",
        "InterventionHeadline",
        "RepresentationHeadline",
    ]


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


def test_retrieval_dynamics_manifest_accepts_expanded_hashed_contract(tmp_path: Path):
    root = tmp_path
    report = root / "reports" / "retrieval-dynamics"
    report.mkdir(parents=True)

    def record(relative: str, content: str, *, rows: int | None = None):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        item = {
            "path": relative,
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        if rows is not None:
            item["rows"] = rows
        return item

    matrix = record("configs/experiment.yaml", "matrix\n")
    training = record("reports/training-dynamics/summary_manifest.json", "training\n")
    training_table = record("reports/training-dynamics/run_summary.csv", "runs\n")
    coverage = record(
        "reports/coverage.json",
        json.dumps(
            {
                "complete": True,
                "observed_results": 1680,
                "expected_results": 1680,
                "observed_checkpoint_summaries": 120,
                "expected_checkpoint_summaries": 120,
                "missing": [],
                "unexpected": [],
            }
        ),
    )
    protocol_payload = {
        "status": "prospective_completion_lock",
        "freeze_context": {
            "strict_beir_valid_units": 160,
            "strict_beir_expected_units": 1680,
            "complete_retrieval_matrix_visible": False,
        },
        "reference_target": {
            "uses_muon_or_normuon_outcomes": False,
            "uses_confirmation_outcomes": False,
        },
        "matrix": {"sha256": matrix["sha256"]},
        "training_summary": {"sha256": training["sha256"]},
    }
    protocol = record(
        "configs/retrieval_dynamics_protocol.json",
        json.dumps(protocol_payload),
    )
    outputs = {
        "checkpoint_dynamics": record(
            "reports/retrieval-dynamics/checkpoint_dynamics.csv", "checkpoints\n", rows=120
        ),
        "run_first_passage": record(
            "reports/retrieval-dynamics/run_first_passage.csv", "runs\n", rows=24
        ),
        "optimizer_first_passage": record(
            "reports/retrieval-dynamics/optimizer_first_passage.csv", "groups\n", rows=6
        ),
        "best_config_task_comparison": record(
            "reports/retrieval-dynamics/best_config_task_comparison.csv",
            "tasks\n",
            rows=28,
        ),
        "best_config_task_delta_dynamics": record(
            "reports/retrieval-dynamics/best_config_task_delta_dynamics.csv",
            "deltas\n",
            rows=280,
        ),
        "task_delta_stability": record(
            "reports/retrieval-dynamics/task_delta_stability.csv",
            "stability\n",
            rows=16,
        ),
        "quality_vs_useful_wall_time": record(
            "reports/retrieval-dynamics/quality_vs_useful_wall_time.svg", "<svg/>\n"
        ),
    }
    evaluation_results = [
        record(f"results/evaluation/result-{index}.json", f"{index}\n") for index in range(1680)
    ]
    manifest = {
        "schema_version": 1,
        "complete": True,
        "coverage": {
            "runs": 24,
            "checkpoints": 120,
            "tasks": 14,
            "evaluation_units": 1680,
            "optimizer_family_groups": 6,
            "best_config_task_delta_rows": 280,
            "adjacent_stage_task_stability_rows": 16,
        },
        "outputs": outputs,
        "sources": {
            "frozen_protocol": protocol,
            "matrix": matrix,
            "strict_coverage": coverage,
            "training_summary": training,
            "training_run_table": training_table,
            "evaluation_results": evaluation_results,
        },
    }
    path = report / "summary_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert _complete_manifest(path) is True

    (report / "task_delta_stability.csv").write_text("changed\n", encoding="utf-8")
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


def test_result_table_audit_requires_exact_hashes_and_no_pending_markers(tmp_path: Path):
    records = []
    for relative in PAPER_RESULT_TABLE_PATHS:
        output = tmp_path / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("final result table\n", encoding="utf-8")
        records.append(
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )

    assert _paper_result_tables_complete(tmp_path, records) is True

    pending = tmp_path / PAPER_RESULT_TABLE_PATHS[0]
    pending.write_text("\\ResultPending{not final}\n", encoding="utf-8")
    records[0]["bytes"] = pending.stat().st_size
    records[0]["sha256"] = hashlib.sha256(pending.read_bytes()).hexdigest()
    assert _paper_result_tables_complete(tmp_path, records) is False


def test_final_document_language_audit_rejects_only_declared_stale_phrases(tmp_path: Path):
    blog = tmp_path / "docs/blog.md"
    paper = tmp_path / "paper/main.tex"
    blog.parent.mkdir(parents=True)
    paper.parent.mkdir(parents=True)
    blog.write_text("Final prose.\n", encoding="utf-8")
    paper.write_text("Final manuscript.\n", encoding="utf-8")

    assert _final_document_language_problems(tmp_path) == []

    blog.write_text("Results will be inserted here after evaluation.\n", encoding="utf-8")
    paper.write_text("The final analysis will report everything.\n", encoding="utf-8")
    problems = _final_document_language_problems(tmp_path)

    assert problems == [
        "docs/blog.md: Results will be inserted here",
        "paper/main.tex: The final analysis will report",
    ]


def test_paper_source_table_audit_requires_all_thirteen_tables_in_declared_order(
    tmp_path: Path,
):
    records = []
    for relative in PAPER_SOURCE_TABLE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source={relative}\n", encoding="utf-8")
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    assert len(records) == 13
    assert _paper_source_tables_complete(tmp_path, records) is True
    assert _paper_source_tables_complete(tmp_path, records[:-1]) is False
    assert _paper_source_tables_complete(tmp_path, list(reversed(records))) is False

    first = tmp_path / PAPER_SOURCE_TABLE_PATHS[0]
    first.write_text("changed\n", encoding="utf-8")
    assert _paper_source_tables_complete(tmp_path, records) is False
