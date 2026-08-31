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
    "tab:claim-firewall": 7,
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

    with pytest.raises(ValueError, match="Main-text floats land beyond"):
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


def test_release_runs_layout_gate_only_after_pdf_build():
    makefile = (ROOT / "paper/Makefile").read_text(encoding="utf-8")
    release = makefile.split("\nrelease:\n", 1)[1].split("\n\nvendor:", 1)[0]

    assert release.index("$(MAKE) $(BUILD)/main.pdf") < release.index("-m embed_optim.paper_layout")
    assert "paper_layout" not in makefile.split("\nrelease:\n", 1)[0]
