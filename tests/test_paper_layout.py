import re
from pathlib import Path

import pytest

from embed_optim.paper_layout import (
    APPENDIX_FLOAT_LABELS,
    FLOAT_LABELS,
    MAIN_END_LABEL,
    MAIN_TEXT_FLOAT_LABELS,
    audit_paper_layout,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FLOAT_ROWS = {
    "tab:discovery-results": 3,
    "tab:common-state-results": 2,
    "tab:intervention-results": 1,
    "tab:causal-chain-summary": 3,
    "tab:confirmation-results": 3,
    "tab:corrected-primary": 3,
    "tab:claim-firewall": 8,
    "tab:training-systems-results": 3,
    "tab:basis-sensitivity-results": 3,
    "tab:representation-results": 3,
    "tab:tail-identity-results": 2,
    "tab:tail-persistence-results": 2,
    "tab:spectral-factorial-results": 2,
    "tab:spectral-tail-results": 3,
    "tab:causal-temporal-diagnostics": 6,
    "tab:causal-temporal-estimates": 16,
    "tab:causal-temporal-pairs": 6,
    "tab:causal-dose-diagnostics": 6,
    "tab:causal-dose-anchors": 10,
    "tab:causal-forward-rmse": 5,
    "tab:extended-retrieval-dynamics": 4,
    "tab:denseon-per-task-results": 14,
    "tab:task-delta-stability": 8,
    "tab:corrected-bridge": 9,
    "tab:corrected-sensitivity": 2,
}


def _aux(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "paper/build/main.aux"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _complete_aux(
    tmp_path: Path,
    *,
    main_end_page: int = 8,
    main_float_pages: dict[str, int] | None = None,
    appendix_float_pages: dict[str, int] | None = None,
    extra: str = "",
) -> Path:
    main_pages = {label: 8 for label in MAIN_TEXT_FLOAT_LABELS} | (main_float_pages or {})
    appendix_pages = {label: 9 for label in APPENDIX_FLOAT_LABELS} | (appendix_float_pages or {})
    lines = [rf"\newlabel{{{MAIN_END_LABEL}}}{{{{10}}{{{main_end_page}}}}}"]
    lines.extend(rf"\newlabel{{{label}}}{{{{1}}{{{page}}}}}" for label, page in main_pages.items())
    lines.extend(
        rf"\newlabel{{{label}}}{{{{1}}{{{page}}}}}" for label, page in appendix_pages.items()
    )
    if extra:
        lines.append(extra)
    return _aux(tmp_path, "\n".join(lines) + "\n")


def test_layout_gate_accepts_main_end_on_page_eight(tmp_path: Path):
    aux = _complete_aux(tmp_path)

    result = audit_paper_layout(tmp_path / "paper")

    assert result == {
        "complete": True,
        "aux_path": str(aux.resolve()),
        "label": MAIN_END_LABEL,
        "main_end_page": 8,
        "max_main_page": 8,
        "main_float_pages": {label: 8 for label in MAIN_TEXT_FLOAT_LABELS},
        "appendix_float_pages": {label: 9 for label in APPENDIX_FLOAT_LABELS},
    }


def test_layout_gate_rejects_main_end_on_page_nine(tmp_path: Path):
    _complete_aux(tmp_path, main_end_page=9)

    with pytest.raises(ValueError, match="page 9, beyond the 8-page limit"):
        audit_paper_layout(tmp_path / "paper")


def test_layout_gate_rejects_missing_or_duplicate_main_end_label(tmp_path: Path):
    aux = _aux(tmp_path, r"\relax" + "\n")
    with pytest.raises(ValueError, match="found 0"):
        audit_paper_layout(tmp_path / "paper")

    label = rf"\newlabel{{{MAIN_END_LABEL}}}{{{{10}}{{8}}}}"
    aux.write_text(label + "\n" + label + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="found 2"):
        audit_paper_layout(tmp_path / "paper")


def test_layout_gate_rejects_main_float_deferred_past_page_limit(tmp_path: Path):
    escaped = MAIN_TEXT_FLOAT_LABELS[-1]
    _complete_aux(tmp_path, main_float_pages={escaped: 9})

    with pytest.raises(ValueError, match="Main-text floats land after"):
        audit_paper_layout(tmp_path / "paper")


def test_layout_gate_rejects_main_float_after_earlier_main_endpoint(tmp_path: Path):
    escaped = MAIN_TEXT_FLOAT_LABELS[-1]
    _complete_aux(tmp_path, main_end_page=7, main_float_pages={escaped: 8})

    with pytest.raises(ValueError, match="endpoint on page 7"):
        audit_paper_layout(tmp_path / "paper")


def test_layout_gate_rejects_nonpositive_float_page(tmp_path: Path):
    invalid = MAIN_TEXT_FLOAT_LABELS[0]
    _complete_aux(tmp_path, main_float_pages={invalid: 0})

    with pytest.raises(ValueError, match="must be positive"):
        audit_paper_layout(tmp_path / "paper")


def test_layout_gate_rejects_appendix_float_before_main_endpoint(tmp_path: Path):
    premature = APPENDIX_FLOAT_LABELS[0]
    _complete_aux(tmp_path, appendix_float_pages={premature: 8})

    with pytest.raises(ValueError, match="Appendix floats do not land after"):
        audit_paper_layout(tmp_path / "paper")


def test_corpus_size_diagnostic_is_frozen_as_appendix_only():
    label = "fig:corpus-size-diagnostic"

    assert label in APPENDIX_FLOAT_LABELS
    assert label not in MAIN_TEXT_FLOAT_LABELS


def test_corrected_result_topology_preserves_one_main_answer_and_appendix_detail():
    assert "tab:corrected-primary" in MAIN_TEXT_FLOAT_LABELS
    assert "tab:corrected-bridge" in APPENDIX_FLOAT_LABELS
    assert "tab:corrected-sensitivity" in APPENDIX_FLOAT_LABELS


def test_layout_gate_rejects_missing_or_unclassified_float(tmp_path: Path):
    aux = _complete_aux(tmp_path)
    text = aux.read_text(encoding="utf-8")
    missing = FLOAT_LABELS[0]
    aux.write_text(
        "\n".join(line for line in text.splitlines() if f"{{{missing}}}" not in line) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="float-label topology differs"):
        audit_paper_layout(tmp_path / "paper")

    _complete_aux(
        tmp_path,
        extra=r"\newlabel{tab:unclassified}{{1}{9}}",
    )
    with pytest.raises(ValueError, match="tab:unclassified"):
        audit_paper_layout(tmp_path / "paper")


def test_checked_in_sources_reserve_final_float_topology_and_row_counts():
    sources = [ROOT / "paper/main.tex", *sorted((ROOT / "paper/generated").glob("*.tex"))]
    texts = {path: path.read_text(encoding="utf-8") for path in sources}
    observed_labels = []
    observed_rows = {}
    table_pattern = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.DOTALL)
    for text in texts.values():
        observed_labels.extend(re.findall(r"\\label\{((?:tab|fig):[^{}]+)\}", text))
        for table in table_pattern.findall(text):
            labels = re.findall(r"\\label\{(tab:[^{}]+)\}", table)
            assert len(labels) == 1
            body = table.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
            observed_rows[labels[0]] = sum(
                line.rstrip().endswith(r"\\") for line in body.splitlines()
            )

    assert len(observed_labels) == len(set(observed_labels))
    assert set(observed_labels) == set(FLOAT_LABELS)
    assert observed_rows == EXPECTED_FLOAT_ROWS


def test_final_results_fill_every_headline_and_conclusion_without_pending_placeholders():
    results = (ROOT / "paper/results.tex").read_text(encoding="utf-8")
    expected_fragments = {
        "DiscoveryHeadline": ("median throughput ratios", "training was not faster"),
        "CommonStateHeadline": ("normalized exact stable ranks", "0.5772/0.4153"),
        "RepresentationHeadline": ("trailing-training-loss-to-BEIR", "-0.684"),
        "InterventionHeadline": (
            "0.0024 (3/0/0)/-0.0007 (1/0/2)",
            "joint result claimable negative",
        ),
        "ConfirmationHeadline": ("[-0.0464, -0.0138]", "inconclusive"),
        "ResultConclusion": ("routing-matched hybrid AdamW", "universal optimizer ranking"),
    }
    for macro, fragments in expected_fragments.items():
        match = re.search(rf"^\\newcommand\{{\\{macro}\}}\{{(.+)\}}$", results, re.MULTILINE)
        assert match is not None
        value = match.group(1)
        assert not value.startswith(r"\ResultPending{")
        assert all(fragment in value for fragment in fragments)
    assert r"\ResultPending{" not in results

    confirmation = (ROOT / "paper/generated/confirmation.tex").read_text(encoding="utf-8")
    assert r"\phantom{" not in confirmation
    assert r"\ResultPending{" not in confirmation
    causal = (ROOT / "paper/generated/causal-chain.tex").read_text(encoding="utf-8")
    assert "gain L/M -0.237277/-0.688154" in causal
    assert "support L/M/T/B 0/0/0/2 of 10" in causal
    assert "gaps -1.508471e-04/-7.124088e-05" in causal
    assert "Temporal spectral bridge rejected" in causal
    assert "Frozen component account rejected" in causal
    assert "No forward bridge; fixed-state conclusion only" in causal
    candidate = (ROOT / "paper/generated/candidate-breadth.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\CandidateBreadthConclusion}" in candidate
    assert r"\newcommand{\CandidateBreadthDiscussion}" in candidate
    assert r"\newcommand{\CandidateBreadthFigure}" in candidate
    assert r"\label{fig:candidate-breadth}" in candidate
    assert "nested-candidate decision was not supported" in candidate
    assert "prerequisite width-7 bridge failed" in candidate
    assert "maximum error 8.286419" in candidate
    assert "Candidate-breadth uncertainty and paired prevalence" in candidate
    assert r"Width 7 $\rightarrow$ 2,048 loss/margin deltas" in candidate


def test_release_runs_layout_gate_only_after_pdf_build():
    makefile = (ROOT / "paper/Makefile").read_text(encoding="utf-8")
    all_target = makefile.split("\nall:", 1)[1].split("\n\nrelease:", 1)[0]
    assert "$(BUILD)/main.pdf" in all_target
    assert "embed_optim.paper_layout" in all_target
    release = makefile.split("\nrelease:\n", 1)[1].split("\n\nvendor:", 1)[0]

    assert "cd .. && $(PYTHON) -m embed_optim.paper_results --repo-root ." in release
    assert release.index("$(MAKE) $(BUILD)/main.pdf") < release.index("-m embed_optim.paper_layout")
