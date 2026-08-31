from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from embed_optim import paper_audit as paper_audit_module
from embed_optim.causal_chain_rendering import (
    causal_chain_paper_contract,
    render_causal_chain_headline_fragment,
)
from embed_optim.paper_audit import (
    BLOG_MARKERS,
    PAPER_APPENDIX_GENERATED_INPUTS,
    PAPER_DEFINITION_GENERATED_INPUTS,
    PAPER_DISCOVERY_FIGURE_CAPTION,
    PAPER_DISCOVERY_FIGURE_INCLUDES,
    PAPER_DISCOVERY_FIGURE_LABEL,
    PAPER_MAIN_GENERATED_INPUTS,
    PAPER_MAIN_REQUIRED_ONCE,
    PAPER_RESULT_TABLE_PATHS,
    PAPER_SOURCE_TABLE_PATHS,
    _causal_chain_evidence,
    _causal_chain_source_complete,
    _complete_manifest,
    _final_document_language_problems,
    _macros,
    _paper_figures_complete,
    _paper_main_topology_complete,
    _paper_result_tables_complete,
    _paper_results_complete,
    _paper_source_tables_complete,
    _renderer_marker_blocks_complete,
    _spectral_transplant_complete,
    _tail_stability_complete,
    audit_paper,
    expected_constant_macros,
    load_paper_claim_protocol,
    main,
)
from embed_optim.scope import resolve_scope

REPOSITORY = Path(__file__).resolve().parents[1]


def test_active_manuscript_sources_do_not_overclaim_hybrid_identification():
    active_sources = (
        REPOSITORY / "paper" / "main.tex",
        REPOSITORY / "README.md",
        REPOSITORY / "docs" / "blog.md",
        REPOSITORY / "docs" / "naacl-dense-paper-plan.md",
    )
    forbidden = (
        "isolates the matrix rule",
        "separate the matrix rule from parameter routing",
        "orthogonalized hidden-matrix rule under matched routing",
        "matrix transform, not routing, causes",
        "matrix rule matters",
    )

    for path in active_sources:
        source = path.read_text(encoding="utf-8").lower()
        assert not any(phrase in source for phrase in forbidden), path


def test_macro_parser_rejects_duplicate_definition(tmp_path: Path):
    path = tmp_path / "results.tex"
    path.write_text(
        "\\newcommand{\\Metric}{1}\n\\newcommand{\\Metric}{2}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate paper result macro"):
        _macros(path)


def test_audit_resolves_scope_amendment_against_repo_root_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "checkout"
    expected = repository / "configs/scope.json"
    captured: dict[str, Path] = {}

    def capture(_families, scope_path):
        captured["path"] = Path(scope_path)
        raise RuntimeError("captured")

    monkeypatch.setattr("embed_optim.paper_audit.resolve_scope", capture)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="captured"):
        audit_paper(
            repo_root=repository,
            families=("dense",),
            scope_amendment=Path("configs/scope.json"),
        )

    assert captured["path"] == expected.resolve()


def test_audit_uses_one_causal_evidence_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    original = paper_audit_module.load_causal_chain_evidence
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(paper_audit_module, "load_causal_chain_evidence", counted)

    audit_paper(
        families=("dense",),
        scope_amendment="configs/dense_scope_amendment.json",
    )

    assert calls == 1


@pytest.fixture
def checked_in_dense_audit():
    return audit_paper(
        families=("dense",),
        scope_amendment="configs/dense_scope_amendment.json",
    )


def test_current_dense_paper_constants_match_strict_sources(checked_in_dense_audit):
    result = checked_in_dense_audit

    assert isinstance(result["complete"], bool)
    assert result["constant_macros"]["NumDiscoveryRuns"] == "12"
    assert result["constant_macros"]["NumDiscoveryUnits"] == "840"
    assert result["constant_macros"]["NumWeightPairs"] == "20"
    assert result["constant_macros"]["MuonFamilyThroughputRatioRange"] == "0.9348--0.9489"
    assert result["constant_macros"]["MuonFamilyStateRatioRange"] == "0.6299--0.6304"
    assert result["constant_sources"]["dataset_manifest"]["sha256"] == (
        "9facc18bcd1cad8378cea94746a95ab09804bdf3610796bf9013cdfcc486aee8"
    )
    assert result["constant_sources"]["dataset_contract"]["path"].endswith(
        "configs/training_data_contract.json"
    )
    discovery_evidence = result["evidence"]["DiscoveryHeadline"]
    assert len(discovery_evidence) == 5
    assert discovery_evidence[1]["complete"] is True
    assert discovery_evidence[2]["complete"] is True
    assert isinstance(discovery_evidence[3]["complete"], bool)
    assert discovery_evidence[4]["complete"] is False
    assert result["claim_protocol"]["status"] == "prospective_completion_lock"
    assert result["claim_protocol"]["amendments"][0]["headline_contract_changed"] is False
    assert len(result["claim_protocol"]["source_bindings"]) == 11
    assert isinstance(result["paper_results"]["complete"], bool)
    assert set(result["blog_marker_blocks"]) == set(BLOG_MARKERS)
    assert result["document_language_problems"] == []


@pytest.fixture
def synthetic_pending_audit(monkeypatch):
    parsed = _macros(Path("paper/results.tex"))
    for name in (
        "DiscoveryHeadline",
        "CommonStateHeadline",
        "RepresentationHeadline",
        "InterventionHeadline",
        "ConfirmationHeadline",
    ):
        parsed[name] = "\\ResultPending{synthetic pending fixture}"
    monkeypatch.setattr("embed_optim.paper_audit._macros", lambda _path: parsed)
    return audit_paper(
        families=("dense",),
        scope_amendment="configs/dense_scope_amendment.json",
    )


def test_synthetic_pending_checked_in_state_stays_incomplete(synthetic_pending_audit):
    assert synthetic_pending_audit["complete"] is False
    assert synthetic_pending_audit["pending_headlines"] == [
        "CommonStateHeadline",
        "ConfirmationHeadline",
        "DiscoveryHeadline",
        "InterventionHeadline",
        "RepresentationHeadline",
    ]


@pytest.fixture
def synthetic_future_final_audit(monkeypatch):
    parsed = _macros(Path("paper/results.tex"))
    for name in (
        "DiscoveryHeadline",
        "CommonStateHeadline",
        "RepresentationHeadline",
        "InterventionHeadline",
        "ConfirmationHeadline",
    ):
        parsed[name] = "audited final result"
    monkeypatch.setattr("embed_optim.paper_audit._macros", lambda _path: parsed)
    monkeypatch.setattr(
        "embed_optim.paper_audit._complete_manifest", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._blog_marker_audit",
        lambda *_args, **_kwargs: {name: {"complete": True} for name in BLOG_MARKERS},
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._final_document_language_problems",
        lambda _root: [],
    )

    def causal_receipt(root, causal_cache=None):
        if causal_cache is not None:
            causal_cache[Path(root).resolve()] = (
                {"repository_root": str(Path(root).resolve())},
                None,
            )
        return {
            "temporal_short_branch": {"complete": True},
            "dose_band": {"complete": True},
        }

    monkeypatch.setattr("embed_optim.paper_audit._causal_chain_evidence", causal_receipt)
    monkeypatch.setattr(
        "embed_optim.paper_audit._causal_snapshot_still_current",
        lambda *_args, **_kwargs: True,
    )
    return audit_paper(
        families=("dense",),
        scope_amendment="configs/dense_scope_amendment.json",
    )


def test_synthetic_future_final_state_can_pass_every_completion_gate(
    synthetic_future_final_audit,
):
    assert synthetic_future_final_audit["complete"] is True
    assert synthetic_future_final_audit["pending_headlines"] == []
    assert synthetic_future_final_audit["incomplete_evidence"] == []
    assert synthetic_future_final_audit["incomplete_blog_marker_blocks"] == []


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


def test_strict_paper_audit_rejects_pending_headlines(synthetic_pending_audit):
    assert synthetic_pending_audit["complete"] is False
    with pytest.raises(ValueError, match="Paper is not final"):
        audit_paper(
            strict=True,
            families=("dense",),
            scope_amendment="configs/dense_scope_amendment.json",
        )


def test_strict_paper_audit_cli_reports_pending_evidence(capsys, synthetic_pending_audit):
    assert synthetic_pending_audit["complete"] is False
    with pytest.raises(SystemExit, match="1"):
        main(
            [
                "--strict",
                "--families",
                "dense",
                "--scope-amendment",
                "configs/dense_scope_amendment.json",
            ]
        )

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


def test_dense_scope_constants_filter_only_after_full_source_audit():
    constants, sources = expected_constant_macros(
        families=("dense",),
        scope_amendment="configs/dense_scope_amendment.json",
    )

    assert constants["NumDiscoveryRuns"] == "12"
    assert constants["NumDiscoveryCheckpoints"] == "60"
    assert constants["NumDiscoveryUnits"] == "840"
    assert constants["NumWeightPairs"] == "20"
    assert constants["MuonFamilyThroughputRatioRange"] == "0.9348--0.9489"
    assert constants["MuonFamilyStateRatioRange"] == "0.6299--0.6304"
    assert sources["scope_amendment"]["status"] == ("user_directed_post_hoc_scope_amendment")


def test_dense_hybrid_manifest_requires_scope_and_exact_four_run_coverage(tmp_path: Path):
    families, scope = resolve_scope(("dense",), "configs/dense_scope_amendment.json")
    report = tmp_path / "reports" / "hybrid-adamw"
    report.mkdir(parents=True)
    table = report / "final_summary.csv"
    table.write_text("header\nrow1\nrow2\nrow3\nrow4\n", encoding="utf-8")
    record = {
        "path": str(table),
        "rows": 4,
        "bytes": table.stat().st_size,
        "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
    }
    path = report / "summary_manifest.json"
    payload = {
        "schema_version": 1,
        "complete": True,
        "families": ["dense"],
        "scope_amendment": scope,
        "evaluations": {
            "native_five_stage_units": 280,
            "native_final_units": 56,
            "hybrid_final_units": 56,
            "tasks": 14,
        },
        "outputs": {"final_summary": record},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _complete_manifest(
        path,
        families=families,
        scope_amendment="configs/dense_scope_amendment.json",
    )
    payload["evaluations"]["hybrid_final_units"] = 55
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not _complete_manifest(
        path,
        families=families,
        scope_amendment="configs/dense_scope_amendment.json",
    )


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


def test_rendered_causal_verdict_must_match_strict_branch_status_and_boundary(
    tmp_path: Path, monkeypatch
):
    root = tmp_path
    cases = {
        "temporal_short_branch": {
            "path": root / "reports/temporal-short-branch/summary_manifest.json",
            "payload": {
                "schema_version": 1,
                "status": "complete",
                "complete": True,
                "claimable": True,
                "decision": {"spectral_temporal_bridge_supported": False},
                "claim_boundary": "not formal mediation",
            },
        },
        "dose_band": {
            "path": root / "reports/dose-band/summary_manifest.json",
            "payload": {
                "schema_version": 1,
                "status": "complete",
                "complete": True,
                "claimability": "claimable",
                "supported": True,
                "claim_boundary": "fixed-state only",
            },
        },
    }
    for label, case in cases.items():
        path = case["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(case["payload"]), encoding="utf-8")
        expected_status = "negative" if label == "temporal_short_branch" else "supported"
        supported = expected_status == "supported"
        identity = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        branch = {
            "complete": True,
            "claimable": True,
            "supported": supported,
            "status": expected_status,
            "claim_boundary": case["payload"]["claim_boundary"],
            "manifest": identity,
        }
        monkeypatch.setattr(
            "embed_optim.paper_audit.load_causal_chain_evidence",
            lambda *_args, _label=label, _branch=branch, **_kwargs: {
                "complete": True,
                _label: _branch,
            },
        )
        record = {
            "path": str(path.relative_to(root)),
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
            "status": expected_status,
            "claimable": True,
            "supported": supported,
            "claim_boundary": case["payload"]["claim_boundary"],
        }
        assert _causal_chain_source_complete(root, record, path, label)
        assert not _causal_chain_source_complete(
            root,
            {**record, "status": "supported" if expected_status == "negative" else "negative"},
            path,
            label,
        )
        assert not _causal_chain_source_complete(
            root, {**record, "claim_boundary": "changed"}, path, label
        )


def test_causal_audit_preserves_completed_branch_in_partial_state(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "embed_optim.paper_audit.load_causal_chain_evidence",
        lambda *_args, **_kwargs: {
            "complete": False,
            "temporal_short_branch": {
                "complete": True,
                "claimable": True,
                "status": "supported",
                "supported": True,
                "claim_boundary": "temporal boundary",
            },
            "dose_band": {
                "complete": False,
                "claimable": False,
                "status": "pending",
                "supported": None,
                "claim_boundary": "dose boundary",
            },
        },
    )

    evidence = _causal_chain_evidence(tmp_path)

    assert evidence["temporal_short_branch"]["complete"] is True
    assert evidence["temporal_short_branch"]["claimable"] is True
    assert evidence["dose_band"]["complete"] is False
    assert evidence["dose_band"]["claimable"] is False


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


def test_paper_figure_audit_binds_both_panels_to_dense_coverage(tmp_path: Path):
    paths = {
        "dense_training_dynamics_by_run": tmp_path
        / "reports/dense-discovery/figures/dense-training-dynamics-by-run.png",
        "dense_lr_sensitivity": tmp_path
        / "reports/dense-discovery/figures/dense-lr-sensitivity.png",
    }
    outputs = {}
    panels = []
    for name, path in paths.items():
        record = _hashed_record(path, f"{name}\n")
        record["path"] = str(path.relative_to(tmp_path))
        outputs[name] = dict(record)
        panels.append({"name": name, **record})
    coverage = tmp_path / "reports/dense-discovery/coverage.json"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_text(json.dumps({"outputs": outputs}), encoding="utf-8")
    discovery = tmp_path / "paper/generated/discovery.tex"
    discovery.parent.mkdir(parents=True, exist_ok=True)
    discovery.write_text(
        "\n".join(
            (
                *PAPER_DISCOVERY_FIGURE_INCLUDES,
                PAPER_DISCOVERY_FIGURE_CAPTION,
                PAPER_DISCOVERY_FIGURE_LABEL,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    contract = {"source_manifest": _hashed_record(coverage, coverage.read_text()), "panels": panels}

    assert _paper_figures_complete(tmp_path, contract) is True
    canonical = discovery.read_text(encoding="utf-8")
    for token in (*PAPER_DISCOVERY_FIGURE_INCLUDES, PAPER_DISCOVERY_FIGURE_CAPTION):
        discovery.write_text(canonical.replace(token, "", 1), encoding="utf-8")
        assert _paper_figures_complete(tmp_path, contract) is False
    discovery.write_text(canonical, encoding="utf-8")
    paths["dense_lr_sensitivity"].write_text("drift\n", encoding="utf-8")
    assert _paper_figures_complete(tmp_path, contract) is False


def test_paper_main_topology_requires_every_generated_input_and_conclusion(tmp_path: Path):
    main = tmp_path / "paper/main.tex"
    main.parent.mkdir(parents=True)
    canonical = (
        "\n".join(
            (
                r"\input{results}",
                *PAPER_DEFINITION_GENERATED_INPUTS,
                *PAPER_MAIN_GENERATED_INPUTS,
                r"\CausalChainSummaryTable",
                r"\section{Conclusion}",
                r"\ResultConclusion",
                r"\label{paper-main-end}",
                r"\section{Limitations}",
                r"\section{Ethical Considerations}",
                r"\bibliography{references}",
                r"\appendix",
                r"\section{Artifact and Reproducibility}",
                *PAPER_APPENDIX_GENERATED_INPUTS,
                r"\CausalChainDiagnostics",
                r"\input{generated/retrieval-dynamics-extension}",
            )
        )
        + "\n"
    )
    main.write_text(canonical, encoding="utf-8")

    assert _paper_main_topology_complete(tmp_path) is True
    for token in PAPER_MAIN_REQUIRED_ONCE:
        main.write_text(canonical.replace(token, "", 1), encoding="utf-8")
        assert _paper_main_topology_complete(tmp_path) is False
    main.write_text(canonical + PAPER_MAIN_GENERATED_INPUTS[0] + "\n", encoding="utf-8")
    assert _paper_main_topology_complete(tmp_path) is False

    main.write_text(
        canonical.replace(
            r"\section{Ethical Considerations}",
            r"\section{Ethics and Reproducibility}",
            1,
        ),
        encoding="utf-8",
    )
    assert _paper_main_topology_complete(tmp_path) is False

    main.write_text(
        canonical.replace(
            r"\section{Ethical Considerations}",
            r"\section{Reproducibility}\section{Ethical Considerations}",
            1,
        ),
        encoding="utf-8",
    )
    assert _paper_main_topology_complete(tmp_path) is False


@pytest.mark.parametrize(
    "bypass",
    (
        r"\section[Reproducibility]{Additional Reproducibility}",
        r"\section*{Additional Reproducibility}",
        r"\input{additional-reproducibility}",
        r"\include{additional-reproducibility}",
    ),
)
def test_paper_main_topology_rejects_exempt_region_bypasses(tmp_path: Path, bypass: str) -> None:
    main = tmp_path / "paper/main.tex"
    main.parent.mkdir(parents=True)
    canonical = "\n".join(
        (
            r"\input{results}",
            *PAPER_DEFINITION_GENERATED_INPUTS,
            *PAPER_MAIN_GENERATED_INPUTS,
            r"\CausalChainSummaryTable",
            r"\section{Conclusion}",
            r"\ResultConclusion",
            r"\label{paper-main-end}",
            r"\section{Limitations}",
            r"\section{Ethical Considerations}",
            r"\bibliography{references}",
            r"\appendix",
            r"\section{Artifact and Reproducibility}",
            *PAPER_APPENDIX_GENERATED_INPUTS,
            r"\CausalChainDiagnostics",
            r"\input{generated/retrieval-dynamics-extension}",
        )
    )
    main.write_text(
        canonical.replace(r"\section{Limitations}", f"{bypass}\n\\section{{Limitations}}") + "\n",
        encoding="utf-8",
    )

    assert _paper_main_topology_complete(tmp_path) is False


def test_paper_main_topology_uses_only_active_latex_source(tmp_path: Path) -> None:
    main = tmp_path / "paper/main.tex"
    main.parent.mkdir(parents=True)
    canonical = "\n".join(
        (
            r"\input{results}",
            *PAPER_DEFINITION_GENERATED_INPUTS,
            *PAPER_MAIN_GENERATED_INPUTS,
            r"\CausalChainSummaryTable",
            r"\section{Conclusion}",
            r"\ResultConclusion",
            r"\label{paper-main-end}",
            "% \\section[Hidden]{Hidden section}",
            "% \\section*{Hidden starred section}",
            "% \\input{hidden-input}",
            "% \\include{hidden-include}",
            r"\section{Limitations}",
            r"\section{Ethical Considerations}",
            r"\bibliography{references}",
            r"\appendix",
            r"\section{Artifact and Reproducibility}",
            *PAPER_APPENDIX_GENERATED_INPUTS,
            r"\CausalChainDiagnostics",
            r"\input{generated/retrieval-dynamics-extension}",
            "% \\section{Conclusion}",
        )
    )
    main.write_text(canonical + "\n", encoding="utf-8")

    assert _paper_main_topology_complete(tmp_path) is True


def test_final_document_language_audit_rejects_only_declared_stale_phrases(tmp_path: Path):
    readme = tmp_path / "README.md"
    blog = tmp_path / "docs/blog.md"
    paper = tmp_path / "paper/main.tex"
    blog.parent.mkdir(parents=True)
    paper.parent.mkdir(parents=True)
    readme.write_text("Final repository status.\n", encoding="utf-8")
    blog.write_text("Final prose.\n", encoding="utf-8")
    paper.write_text("Final manuscript.\n", encoding="utf-8")

    assert _final_document_language_problems(tmp_path) == []

    readme.write_text(
        "Status: remaining DenseOn confirmation is running. Once complete, the supplemental view changes.\n",
        encoding="utf-8",
    )
    blog.write_text(
        "Results will be inserted here after evaluation. When complete, those 728 rows change.\n",
        encoding="utf-8",
    )
    paper.write_text(
        "The final analysis will report everything; it is intentionally left unresolved.\n",
        encoding="utf-8",
    )
    problems = _final_document_language_problems(tmp_path)

    assert problems == [
        "README.md: remaining DenseOn confirmation is running",
        "README.md: Once complete, the supplemental",
        "docs/blog.md: Results will be inserted here",
        "docs/blog.md: When complete, those 728 rows",
        "paper/main.tex: The final analysis will report",
        "paper/main.tex: intentionally left unresolved",
    ]


def test_paper_source_table_audit_requires_all_twenty_four_tables_in_declared_order(
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

    assert len(records) == 24
    assert _paper_source_tables_complete(tmp_path, records) is True
    assert _paper_source_tables_complete(tmp_path, records[:-1]) is False
    assert _paper_source_tables_complete(tmp_path, list(reversed(records))) is False

    first = tmp_path / PAPER_SOURCE_TABLE_PATHS[0]
    first.write_text("changed\n", encoding="utf-8")
    assert _paper_source_tables_complete(tmp_path, records) is False


def _hashed_record(path: Path, content: str = "source\n") -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_paper_dynamics_extension_contract_is_exact_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_rows = [["Confirmatory AdamW", "3", "0.1000", "0.2000", "0.3000", "0.4000", "0.5000"]]
    expected_latex = (
        "\\includegraphics{../reports/dense-retrieval-dynamics/"
        "five_stage_retrieval_dynamics.pdf}\n"
        "This descriptive artifact is not an inference input.\n"
    )
    expected_markdown = (
        "[Source-bound CSV](../reports/dense-retrieval-dynamics/"
        "five_stage_retrieval_dynamics.csv)\n\n"
        "![Five-stage dynamics](../reports/dense-retrieval-dynamics/"
        "five_stage_retrieval_dynamics.svg)"
    )
    monkeypatch.setattr(paper_audit_module, "load_publication_rows", lambda _root: ([], {}))
    monkeypatch.setattr(
        paper_audit_module, "summarize_publication_rows", lambda _rows: summary_rows
    )
    monkeypatch.setattr(
        paper_audit_module, "render_publication_latex", lambda _rows: expected_latex
    )
    monkeypatch.setattr(
        paper_audit_module, "render_publication_markdown", lambda _rows: expected_markdown
    )

    report = tmp_path / "reports/dense-retrieval-dynamics"
    manifest_record = _hashed_record(report / "summary_manifest.json", "{}\n")
    csv_record = _hashed_record(report / "five_stage_retrieval_dynamics.csv", "csv\n")
    svg_record = _hashed_record(report / "five_stage_retrieval_dynamics.svg", "svg\n")
    pdf_path = report / "five_stage_retrieval_dynamics.pdf"
    pdf_record = _hashed_record(pdf_path, "pdf\n")
    generated_record = _hashed_record(
        tmp_path / "paper/generated/retrieval-dynamics-extension.tex", expected_latex
    )
    main = tmp_path / "paper/main.tex"
    main.write_text("\\input{generated/retrieval-dynamics-extension}\n", encoding="utf-8")
    begin, end = paper_audit_module.DYNAMICS_EXTENSION_MARKERS
    blog = tmp_path / "docs/blog.md"
    blog.parent.mkdir(parents=True)
    blog.write_text(f"{begin}\n\n{expected_markdown}\n\n{end}\n", encoding="utf-8")
    block = blog.read_text(encoding="utf-8").split(end, 1)[0] + end
    block_bytes = block.encode("utf-8")
    contract = {
        "manifest": manifest_record,
        "trajectory_csv": csv_record,
        "figure_svg": svg_record,
        "figure_pdf": pdf_record,
        "generated_tex": generated_record,
        "blog": {
            "path": "docs/blog.md",
            "markers": [begin, end],
            "block_bytes": len(block_bytes),
            "block_sha256": hashlib.sha256(block_bytes).hexdigest(),
        },
        "summary_rows": summary_rows,
        "role": "descriptive-only",
        "formal_inference_reads_joined_outputs": False,
    }

    assert paper_audit_module._paper_dynamics_extension_complete(tmp_path, contract) is True

    pdf_path.write_text("tampered pdf\n", encoding="utf-8")
    assert paper_audit_module._paper_dynamics_extension_complete(tmp_path, contract) is False
    pdf_path.write_text("pdf\n", encoding="utf-8")
    assert paper_audit_module._paper_dynamics_extension_complete(tmp_path, contract) is True

    blog.write_text(f"{begin}\n\n{expected_markdown}\n\nextra\n\n{end}\n", encoding="utf-8")
    assert paper_audit_module._paper_dynamics_extension_complete(tmp_path, contract) is False


@pytest.mark.parametrize(
    ("fresh_verdict", "fresh_label", "forged_label"),
    (
        ("supported", "supported", "claimable negative"),
        ("not_supported_claimable_negative", "claimable negative", "supported"),
    ),
)
def test_paper_audit_binds_causal_headline_and_rejects_main_text_overclaim(
    tmp_path: Path,
    monkeypatch,
    fresh_verdict: str,
    fresh_label: str,
    forged_label: str,
):
    evidence = {
        "complete": True,
        "overall_verdict": fresh_verdict,
        "temporal_short_branch": {
            "criteria_rows": [{"passed": True} for _ in range(5)],
            "estimate_rows": [
                {
                    "outcome": outcome,
                    "predictor": "update_tail_energy_fraction",
                    "relative_rmse_improvement": improvement,
                }
                for outcome, improvement in (
                    ("validation_loss_p95", 0.125),
                    ("unseen_margin_p05", 0.25),
                )
            ],
        },
        "dose_band": {
            "decision_counts": {
                "loss_dose_monotone_anchors": 8,
                "margin_dose_monotone_anchors": 7,
                "tail_band_anchors": 6,
                "basis_control_anchors": 9,
            },
            "rmse_rows": [
                {"predictor": predictor, "rmse_improvement": improvement}
                for predictor, improvement in (
                    ("baseline", 0.0),
                    ("spectrum_loss", 0.3),
                    ("spectrum_margin", 0.2),
                    ("basis_loss", 0.1),
                    ("basis_margin", 0.05),
                )
            ],
            "bridge_rows": [
                {
                    "spectrum_predictor": spectrum,
                    "spectrum_rmse_improvement": spectrum_gain,
                    "matched_basis_rmse_improvement": basis_gain,
                }
                for spectrum, spectrum_gain, basis_gain in (
                    ("spectrum_loss", 0.3, 0.1),
                    ("spectrum_margin", 0.2, 0.05),
                )
            ],
        },
    }
    causal_latex = "fresh causal table\n"
    causal_display = {"complete": True, "evidence_sha256": "fresh"}
    monkeypatch.setattr(
        "embed_optim.paper_audit._causal_evidence_snapshot",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit.causal_chain_display_contract",
        lambda _evidence: causal_display,
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit.render_causal_chain_latex",
        lambda _evidence: causal_latex,
    )
    monkeypatch.setattr("embed_optim.paper_audit._strict_evidence_paths", lambda _families: {})
    monkeypatch.setattr(
        "embed_optim.paper_audit._paper_source_tables_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._paper_result_tables_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._paper_figures_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._paper_dynamics_extension_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._paper_systems_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._confirmation_rows",
        lambda *_args, **_kwargs: ([], Path("confirmation.csv"), {}),
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._hybrid_rows",
        lambda *_args, **_kwargs: ([], Path("hybrid.csv"), {}),
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._tail_stability_rows",
        lambda *_args, **_kwargs: ([], [], (), {}),
    )
    conclusion = {
        "status": "complete",
        "plain": "final evidence-bound conclusion",
        "markdown": "final evidence-bound conclusion",
    }
    monkeypatch.setattr(
        "embed_optim.paper_audit.build_final_conclusion_contract",
        lambda *_args, **_kwargs: conclusion,
    )
    monkeypatch.setattr(
        "embed_optim.paper_audit._causal_chain_source_complete",
        lambda *_args, **_kwargs: True,
    )

    protocol = tmp_path / "configs/paper_claim_protocol.json"
    protocol.parent.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "configs/paper_claim_protocol.json",
        protocol,
    )
    protocol_record = {
        "path": str(protocol),
        "bytes": protocol.stat().st_size,
        "sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "status": "prospective_completion_lock",
        "frozen_at": "2026-08-25T00:00:00Z",
    }
    main_tex = tmp_path / "paper/main.tex"
    main_tex.parent.mkdir(parents=True)
    paper_contract = causal_chain_paper_contract()
    ordered_topology = (
        r"\input{results}",
        *PAPER_DEFINITION_GENERATED_INPUTS,
        *PAPER_MAIN_GENERATED_INPUTS,
        r"\CausalChainSummaryTable",
        r"\section{Conclusion}",
        r"\ResultConclusion",
        r"\label{paper-main-end}",
        r"\section{Limitations}",
        r"\section{Ethical Considerations}",
        r"\bibliography{references}",
        r"\appendix",
        r"\section{Artifact and Reproducibility}",
        *PAPER_APPENDIX_GENERATED_INPUTS,
        r"\CausalChainDiagnostics",
        r"\input{generated/retrieval-dynamics-extension}",
    )
    canonical_main_text = "\n".join(
        (
            *ordered_topology,
            *(token for token in paper_contract["required_once"] if token not in ordered_topology),
            *paper_contract["required_boundary_substrings"],
        )
    )
    main_tex.write_text(canonical_main_text + "\n", encoding="utf-8")
    causal_table = tmp_path / PAPER_RESULT_TABLE_PATHS[-1]
    causal_table.parent.mkdir(parents=True, exist_ok=True)
    causal_table.write_text(causal_latex, encoding="utf-8")
    result_path = tmp_path / "paper/results.tex"
    fragment = render_causal_chain_headline_fragment(evidence)
    headlines = {
        name: f"final {name}"
        for name in (
            "DiscoveryHeadline",
            "CommonStateHeadline",
            "RepresentationHeadline",
            "InterventionHeadline",
            "ConfirmationHeadline",
        )
    }
    headlines["InterventionHeadline"] = "intervention base." + fragment

    def write_results() -> dict[str, object]:
        content = "".join(
            f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in headlines.items()
        )
        content += "\\newcommand{\\ResultConclusion}{final evidence-bound conclusion}\n"
        return _hashed_record(result_path, content)

    results_record = write_results()
    manifest_path = tmp_path / "reports/paper-results.manifest.json"
    manifest_path.parent.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "complete": True,
        "claim_protocol": protocol_record,
        "evidence_manifests": [],
        "source_tables": [],
        "result_tables": [],
        "headlines": dict(headlines),
        "results_tex": results_record,
        "causal_chain": {"temporal_short_branch": {}, "dose_band": {}},
        "causal_chain_display": causal_display,
        "figures": {},
        "conclusion": conclusion,
        "conclusion_macro": {
            "name": "ResultConclusion",
            "value": "final evidence-bound conclusion",
        },
        "claim_boundary": (
            "do not convert descriptive checkpoint associations into causal evidence"
        ),
    }
    assert _paper_results_complete(manifest_path, payload, ("dense", "late"), None) is True

    for token in PAPER_MAIN_REQUIRED_ONCE:
        main_tex.write_text(canonical_main_text.replace(token, "", 1) + "\n", encoding="utf-8")
        assert _paper_results_complete(manifest_path, payload, ("dense", "late"), None) is False
    main_tex.write_text(canonical_main_text + "\n", encoding="utf-8")

    forged = headlines["InterventionHeadline"].replace(
        f"joint result {fresh_label}.", f"joint result {forged_label}."
    )
    assert forged != headlines["InterventionHeadline"]
    headlines["InterventionHeadline"] = forged
    payload["headlines"] = dict(headlines)
    payload["results_tex"] = write_results()

    assert _paper_results_complete(manifest_path, payload, ("dense", "late"), None) is False

    headlines["InterventionHeadline"] = "intervention base." + fragment
    payload["headlines"] = dict(headlines)
    payload["results_tex"] = write_results()
    main_tex.write_text(
        canonical_main_text + "\nThe causal chain proves formal mediation.\n",
        encoding="utf-8",
    )
    assert all(main_tex.read_text().count(token) == 1 for token in paper_contract["required_once"])
    assert _paper_results_complete(manifest_path, payload, ("dense", "late"), None) is False


def _csv_record(
    path: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _generic_rows(count: int) -> list[dict[str, object]]:
    return [{"row_id": index} for index in range(count)]


def _marker_record(text: str, markers: tuple[str, str]) -> dict[str, object]:
    start = text.index(markers[0])
    stop = text.index(markers[1], start) + len(markers[1])
    block = text[start:stop].encode()
    return {
        "begin_marker": markers[0],
        "end_marker": markers[1],
        "encoding": "utf-8",
        "bytes": len(block),
        "sha256": hashlib.sha256(block).hexdigest(),
    }


def test_renderer_marker_hashes_are_block_local_not_whole_blog(tmp_path: Path):
    blog = tmp_path / "docs/blog.md"
    blog.parent.mkdir(parents=True)
    text = (
        "editable introduction\n"
        f"{BLOG_MARKERS['results'][0]}\nresult body\n{BLOG_MARKERS['results'][1]}\n"
        f"{BLOG_MARKERS['systems'][0]}\nsystems body\n{BLOG_MARKERS['systems'][1]}\n"
        "editable conclusion\n"
    )
    blog.write_text(text, encoding="utf-8")
    record = {
        "schema_version": 1,
        "path": "docs/blog.md",
        "blocks": {
            name: _marker_record(text, BLOG_MARKERS[name]) for name in ("results", "systems")
        },
    }

    assert _renderer_marker_blocks_complete(
        tmp_path,
        record,
        {name: BLOG_MARKERS[name] for name in ("results", "systems")},
    )
    blog.write_text("new preface\n" + text, encoding="utf-8")
    assert _renderer_marker_blocks_complete(
        tmp_path,
        record,
        {name: BLOG_MARKERS[name] for name in ("results", "systems")},
    )
    blog.write_text(("new preface\n" + text).replace("result body", "tampered"), encoding="utf-8")
    assert not _renderer_marker_blocks_complete(
        tmp_path,
        record,
        {name: BLOG_MARKERS[name] for name in ("results", "systems")},
    )


def test_tail_stability_future_manifest_requires_every_hashed_source_and_row(
    tmp_path: Path,
):
    report = tmp_path / "reports/tail-stability"
    report.mkdir(parents=True)
    scope = {"status": "test-dense-scope"}
    expected_outputs = {
        "discovery_anchor_tail": 30,
        "discovery_family_contrasts": 2,
        "discovery_cross_tail": 20,
        "discovery_cross_tail_summary": 2,
        "short_branch_checkpoint_tail": 45,
        "short_branch_checkpoint_contrasts": 30,
        "short_branch_final_summary": 2,
    }
    outputs = {}
    for name, count in expected_outputs.items():
        rows = _generic_rows(count)
        if name in {"discovery_cross_tail_summary", "short_branch_final_summary"}:
            rows = [
                {"family": "dense", "challenger": challenger, "reference": "adamw"}
                for challenger in ("muon", "normuon")
            ]
        outputs[name] = _csv_record(report / f"{name}.csv", rows)
    outputs["readme"] = _hashed_record(report / "README.md", "audited tail report\n")

    source_root = tmp_path / "tail-sources"
    discovery_sources = []
    for index in range(10):
        discovery_sources.append(
            {
                "label": f"anchor-{index}",
                "manifest": _hashed_record(source_root / f"discovery-{index}-manifest.json"),
                "sample_metrics": _hashed_record(source_root / f"discovery-{index}-samples.jsonl"),
            }
        )
    short_sources = [{"short_branch_summary": _hashed_record(source_root / "short-summary.json")}]
    for index in range(45):
        short_sources.append(
            {
                "label": f"validation-{index}",
                "validation_manifest": _hashed_record(
                    source_root / f"validation-{index}-manifest.json"
                ),
                "validation_samples": _hashed_record(
                    source_root / f"validation-{index}-samples.jsonl"
                ),
            }
        )
    for index in range(45):
        short_sources.append(
            {
                "label": f"unseen-{index}",
                "unseen_metrics": _hashed_record(source_root / f"unseen-{index}.json"),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "complete": True,
        "discovery_complete": True,
        "short_branch_confirmation_complete": True,
        "analysis_status": "post_hoc_discovery_with_prospective_short_branch_confirmation",
        "families": ["dense"],
        "scope_amendment": scope,
        "protocol": _hashed_record(tmp_path / "configs/tail.json"),
        "discovery_anchors": 10,
        "discovery_anchor_operator_rows": 30,
        "discovery_contrasts": 2,
        "discovery_cross_tail_rows": 20,
        "discovery_cross_tail_summaries": 2,
        "short_branch_checkpoint_rows": 45,
        "short_branch_contrast_rows": 30,
        "short_branch_final_rows": 2,
        "pending_reason": None,
        "discovery_sources": discovery_sources,
        "short_branch_sources": short_sources,
        "outputs": outputs,
        "claim_boundary": "post-hoc decomposition, not BEIR mediation",
    }
    path = report / "summary_manifest.json"

    assert _tail_stability_complete(path, manifest, ("dense",), scope)
    Path(discovery_sources[0]["sample_metrics"]["path"]).write_text("tampered\n", encoding="utf-8")
    assert not _tail_stability_complete(path, manifest, ("dense",), scope)


def test_spectral_transplant_future_manifest_requires_full_grid_and_source_hashes(
    tmp_path: Path,
):
    report = tmp_path / "reports/spectral-transplant"
    report.mkdir(parents=True)
    scope = {"status": "test-dense-scope"}
    output_counts = {
        "anchor_condition_effects": 100,
        "family_condition_summary": 60,
        "anchor_factorial_effects": 60,
        "family_factorial_summary": 6,
        "anchor_spectral_path": 300,
        "family_spectral_path": 30,
        "anchor_band_effects": 180,
        "family_band_summary": 18,
        "anchor_query_tail_effects": 90,
        "family_query_tail_summary": 9,
    }
    metrics = (
        "contrastive_loss",
        "positive_score",
        "hardest_negative_score",
        "positive_margin",
        "reciprocal_rank",
        "top1_accuracy",
    )
    conditions = (
        "muon-native",
        "adam-basis__spectrum-lambda-0.25",
        "adam-basis__spectrum-lambda-0.50",
        "adam-basis__spectrum-lambda-0.75",
        "adam-basis__muon-spectrum",
        "muon-basis__adam-spectrum",
        "adam-basis__muon-head-spectrum",
        "adam-basis__muon-middle-spectrum",
        "adam-basis__muon-tail-spectrum",
    )
    outputs = {}
    for name, count in output_counts.items():
        rows = _generic_rows(count)
        if name == "family_factorial_summary":
            rows = [{"family": "dense", "metric": metric} for metric in metrics]
        elif name == "family_query_tail_summary":
            rows = [{"family": "dense", "condition": condition} for condition in conditions]
        outputs[name] = _csv_record(report / f"{name}.csv", rows)

    source_root = tmp_path / "spectral-sources"
    sources = [
        {
            "label": f"anchor-{index}",
            "manifest": _hashed_record(source_root / f"anchor-{index}-manifest.json"),
            "sample_metrics": _hashed_record(source_root / f"anchor-{index}-samples.jsonl"),
        }
        for index in range(10)
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "complete": True,
        "analysis_status": "post_hoc_explanatory_intervention",
        "families": ["dense"],
        "scope_amendment": scope,
        "spectral_transplant_spec": _hashed_record(tmp_path / "configs/spectral.json"),
        "common_state_spec": _hashed_record(tmp_path / "configs/common-state.json"),
        "anchors": 10,
        "anchor_effect_records": 100,
        "anchor_tail_effect_records": 90,
        "tail_protocol": {
            "status": "frozen-before-spectral-transplant-output",
            "tail_fraction": 0.05,
            "tail_count": 12,
        },
        "sources": sources,
        "outputs": outputs,
        "claim_boundary": "fixed-state intervention, not retrieval mediation",
    }
    path = report / "summary_manifest.json"

    assert _spectral_transplant_complete(path, manifest, ("dense",), scope)
    outputs["family_query_tail_summary"]["rows"] = 8
    assert not _spectral_transplant_complete(path, manifest, ("dense",), scope)
