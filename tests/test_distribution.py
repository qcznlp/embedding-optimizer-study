from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path, PurePosixPath

from embed_optim.distribution_audit import (
    _checkout_path_findings,
    _data_files,
    _dense_dynamics_distribution_problems,
)

ROOT = Path(__file__).parents[1]


def test_distribution_scanner_rejects_producer_checkout_paths() -> None:
    assert _checkout_path_findings(
        "wheel",
        "report.json",
        b'{"path":"' + b"/root" + b'/embedding-optimizer-study/report"}',
    )
    assert not _checkout_path_findings("wheel", "report.json", b'{"path":"reports/report.json"}')


def _installed_data_paths() -> dict[str, PurePosixPath]:
    pyproject = (ROOT / "pyproject.toml").read_text()
    section = pyproject.split("[tool.setuptools.data-files]", 1)[1].split("\n[", 1)[0]
    groups = re.findall(r'^"([^"]+)" = \[(.*?)^\]$', section, flags=re.MULTILINE | re.DOTALL)
    installed = {}
    for destination, sources in groups:
        for pattern in re.findall(r'^\s+"([^"]+)",?$', sources, flags=re.MULTILINE):
            matches = (
                sorted(path for path in ROOT.glob(pattern) if path.is_file())
                if any(character in pattern for character in "*?[")
                else [ROOT / pattern]
            )
            for path in matches:
                source = path.relative_to(ROOT).as_posix()
                installed[source] = PurePosixPath(destination) / path.name
    return installed


def _resolve_relative(source: PurePosixPath, target: str) -> PurePosixPath:
    return PurePosixPath(posixpath.normpath(str(source.parent / target)))


def test_distribution_preserves_weight_space_document_links() -> None:
    installed = _installed_data_paths()
    expected_links = {
        "docs/blog.md": (
            "../configs/dense_scope_amendment.json",
            "../reports/dense-discovery/figures/dense-training-dynamics.png",
            "../reports/dense-discovery/figures/dense-training-dynamics-by-run.png",
            "../reports/dense-discovery/figures/dense-lr-sensitivity.png",
            "../reports/weight-space/optimizer_pair_contrast_trajectory.svg",
            "../reports/weight-space/optimizer_geometry_phase.svg",
            "../reports/weight-space/README.md",
            "naacl-dense-paper-plan.md",
            "naacl-paper-plan.md",
        ),
        "docs/naacl-dense-paper-plan.md": (),
        "docs/naacl-paper-plan.md": (
            "../reports/weight-space/optimizer_pair_contrasts.csv",
            "../reports/weight-space/optimizer_pair_contrast_trajectory.csv",
            "../reports/weight-space/optimizer_pair_contrast_trajectory.svg",
            "../reports/weight-space/optimizer_geometry_phase.svg",
        ),
        "docs/completion-gates.md": (
            "../configs/dense_scope_amendment.json",
            "../reports/weight-space/summary_manifest.json",
            "blog.md",
        ),
    }

    installed_targets = set(installed.values())
    for source, links in expected_links.items():
        installed_source = installed[source]
        for link in links:
            assert _resolve_relative(installed_source, link) in installed_targets


def test_distribution_bundles_frozen_representation_probe_spec() -> None:
    installed = _installed_data_paths()
    source = "configs/representation_probe.json"
    assert installed[source] == PurePosixPath(
        "share/embedding-optimizer-study/configs/representation_probe.json"
    )
    spec = json.loads((ROOT / source).read_text())
    assert spec["count"] == 1024
    assert spec["seed"] == 1729
    assert spec["expected"]["manifest_sha256"] == (
        "40953eb60bb5dbfa02d9abde5e634bb3221ee934d7ca654c5e5a7e961b39eed2"
    )


def test_distribution_bundles_prospective_beir_probe_protocol() -> None:
    installed = _installed_data_paths()
    source = "configs/beir_representation_probe.json"
    assert installed[source] == PurePosixPath(
        "share/embedding-optimizer-study/configs/beir_representation_probe.json"
    )
    spec = json.loads((ROOT / source).read_text())
    assert len(spec["tasks"]) == 14
    assert sum(task["query_count"] for task in spec["tasks"]) == 224
    assert {task["candidate_pool_count"] for task in spec["tasks"]} == {24}
    assert {task["split"] for task in spec["tasks"] if task["name"] == "MSMARCO"} == {"dev"}
    assert spec["expected"]["manifest_sha256"] == (
        "89fb514b73d3e3c06a6e68d7042b40e99d9ca02ac6220216f43363622b6d9b0d"
    )


def test_distribution_bundles_frozen_common_state_spectrum_protocol() -> None:
    installed = _installed_data_paths()
    source = "configs/common_state_spectrum_probe.json"
    assert installed[source] == PurePosixPath(
        "share/embedding-optimizer-study/configs/common_state_spectrum_probe.json"
    )
    spec = json.loads((ROOT / source).read_text())
    assert spec["selection"]["expected_anchors"] == 20
    assert spec["selection"]["expected_tensors_per_anchor"] == 6
    assert spec["selection"]["expected_spectra"] == 360
    assert spec["freeze_context"]["formal_common_state_outputs_already_observed"] is False
    assert spec["freeze_context"]["partial_beir_results_already_observed"] is True


def test_distribution_bundles_prospective_basis_sensitivity_protocol() -> None:
    installed = _installed_data_paths()
    source = "configs/basis_sensitivity.json"
    assert installed[source] == PurePosixPath(
        "share/embedding-optimizer-study/configs/basis_sensitivity.json"
    )
    spec = json.loads((ROOT / source).read_text())
    assert spec["selection"]["expected_records"] == 540
    assert spec["selection"]["expected_head_records"] == 3240
    assert spec["architecture"]["qkv_tensor_template"] == ("0.layers.{layer}.attn.Wqkv.weight")
    assert spec["freeze_context"]["formal_basis_output_visible"] is False


def test_distribution_bundles_training_receipt_and_retrieval_dynamics_protocol() -> None:
    installed = _installed_data_paths()
    training_source = "configs/training_data_contract.json"
    retrieval_source = "configs/retrieval_dynamics_protocol.json"
    assert installed[training_source] == PurePosixPath(
        "share/embedding-optimizer-study/configs/training_data_contract.json"
    )
    assert installed[retrieval_source] == PurePosixPath(
        "share/embedding-optimizer-study/configs/retrieval_dynamics_protocol.json"
    )
    training = json.loads((ROOT / training_source).read_text())
    retrieval = json.loads((ROOT / retrieval_source).read_text())
    assert training["total_queries"] == 500_000
    assert training["sampled_negatives"] == 7
    assert len(training["independent_protocol_bindings"]) == 5
    assert retrieval["freeze_context"]["strict_beir_valid_units"] == 160
    assert retrieval["freeze_context"]["complete_retrieval_matrix_visible"] is False


def test_distribution_bundles_frozen_paper_claim_protocol() -> None:
    installed = _installed_data_paths()
    source = "configs/paper_claim_protocol.json"
    assert installed[source] == PurePosixPath(
        "share/embedding-optimizer-study/configs/paper_claim_protocol.json"
    )
    protocol = json.loads((ROOT / source).read_text())
    assert protocol["freeze_context"]["strict_beir_valid_units"] == 168
    assert protocol["freeze_context"]["complete_retrieval_matrix_visible"] is False
    assert len(protocol["source_bindings"]) == 11
    assert set(protocol["headline_contract"]) == {
        "DiscoveryHeadline",
        "CommonStateHeadline",
        "RepresentationHeadline",
        "InterventionHeadline",
        "ConfirmationHeadline",
    }


def test_distribution_bundles_causal_chain_runtime_contracts_and_entry_points() -> None:
    installed = _installed_data_paths()
    expected_configs = {
        "configs/causal_chain_analysis.json",
        "configs/mechanism_analysis_integration.json",
        "configs/loss_retrieval_diagnostic.json",
    }
    assert expected_configs <= set(installed)
    assert {installed[source] for source in expected_configs} == {
        PurePosixPath("share/embedding-optimizer-study/configs") / Path(source).name
        for source in expected_configs
    }

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = pyproject.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    for command in (
        "embed-optim-dose-band-analysis",
        "embed-optim-temporal-short-branch-predictors",
        "embed-optim-temporal-short-branch",
    ):
        assert f"{command} = " in scripts


def test_distribution_bundles_dense_retrieval_dynamics_extension() -> None:
    installed = _installed_data_paths()
    source = "configs/dense_retrieval_dynamics_extension.json"
    assert installed[source] == PurePosixPath(
        "share/embedding-optimizer-study/configs/dense_retrieval_dynamics_extension.json"
    )
    contract = json.loads((ROOT / source).read_text(encoding="utf-8"))
    assert contract["evaluation"]["dynamics_stages"] == [1, 2, 3, 4]
    assert contract["evaluation"]["formal_inference_stage"] == 5
    assert contract["evaluation"]["expected_additional_units"] == 728
    assert contract["evaluation"]["formal_inference_uses_dynamics_rows"] is False

    scripts = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    scripts = scripts.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    assert "embed-optim-evaluate-dense-retrieval-dynamics = " in scripts
    assert "embed-optim-summarize-dense-retrieval-dynamics = " in scripts
    placeholder = "reports/dense-retrieval-dynamics/README.md"
    assert installed[placeholder] == PurePosixPath(
        "share/embedding-optimizer-study/reports/dense-retrieval-dynamics/README.md"
    )


def test_completed_dense_dynamics_distribution_requires_manifest_csv_svg_pdf(tmp_path: Path):
    report = tmp_path / "reports/dense-retrieval-dynamics"
    report.mkdir(parents=True)
    (report / "README.md").write_text("placeholder\n", encoding="utf-8")
    manifest = report / "summary_manifest.json"
    manifest.write_text('{"complete":true}\n', encoding="utf-8")
    text = (
        "[tool.setuptools.data-files]\n"
        '"share/project/reports/dense-retrieval-dynamics" = [\n'
        '  "reports/dense-retrieval-dynamics/*",\n'
        "]\n"
    )
    pending = _data_files(text, tmp_path)
    problems = _dense_dynamics_distribution_problems(tmp_path, pending)
    assert len(problems) == 3

    for suffix in ("csv", "svg", "pdf"):
        (report / f"five_stage_retrieval_dynamics.{suffix}").write_text(
            f"{suffix}\n", encoding="utf-8"
        )
    complete = _data_files(text, tmp_path)
    assert _dense_dynamics_distribution_problems(tmp_path, complete) == []


def test_distribution_bundles_result_safe_paper_sources() -> None:
    installed = _installed_data_paths()
    paper_files = (
        "paper/Makefile",
        "paper/README.md",
        "paper/main.tex",
        "paper/references.bib",
        "paper/results.tex",
    )
    result_tables = (
        "paper/generated/causal-chain.tex",
        "paper/generated/common-state.tex",
        "paper/generated/confirmation.tex",
        "paper/generated/diagnostics.tex",
        "paper/generated/discovery.tex",
        "paper/generated/intervention.tex",
        "paper/generated/per-task.tex",
        "paper/generated/representation.tex",
        "paper/generated/retrieval-dynamics-extension.tex",
    )
    for source in paper_files:
        assert installed[source] == PurePosixPath(
            "share/embedding-optimizer-study"
        ) / PurePosixPath(source)
    for source in result_tables:
        assert installed[source] == PurePosixPath(
            "share/embedding-optimizer-study"
        ) / PurePosixPath(source)

    main = (ROOT / "paper/main.tex").read_text()
    makefile = (ROOT / "paper/Makefile").read_text()
    assert (
        "EXTENDED_RESULT_FIGURE := $(wildcard "
        "../reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.pdf)" in makefile
    )
    assert "$(EXTENDED_RESULT_FIGURE)" in makefile
    for source in result_tables:
        table = PurePosixPath(source).stem
        assert f"\\input{{generated/{table}}}" in main
        assert f"generated/{table}.tex" in makefile
        # Checked-in artifacts may represent either the audited pending state or
        # the audited final state. Distribution safety requires complete,
        # non-empty tables; strict pending/final consistency is owned by the
        # paper-results manifest and paper audit.
        assert (ROOT / source).read_text().strip()


def test_distribution_bundles_project_governance_documents() -> None:
    installed = _installed_data_paths()
    documents = (
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
    )
    for source in documents:
        assert installed[source] == PurePosixPath(
            "share/embedding-optimizer-study"
        ) / PurePosixPath(source)
