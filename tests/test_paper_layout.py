from pathlib import Path

import pytest

from embed_optim.paper_layout import MAIN_END_LABEL, audit_paper_layout

ROOT = Path(__file__).resolve().parents[1]


def _aux(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "paper/build/main.aux"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_layout_gate_accepts_main_end_on_page_eight(tmp_path: Path):
    aux = _aux(tmp_path, rf"\newlabel{{{MAIN_END_LABEL}}}{{{{10}}{{8}}}}" + "\n")

    result = audit_paper_layout(tmp_path / "paper")

    assert result == {
        "complete": True,
        "aux_path": str(aux.resolve()),
        "label": MAIN_END_LABEL,
        "main_end_page": 8,
        "max_main_page": 8,
    }


def test_layout_gate_rejects_main_end_on_page_nine(tmp_path: Path):
    _aux(tmp_path, rf"\newlabel{{{MAIN_END_LABEL}}}{{{{10}}{{9}}}}" + "\n")

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


def test_release_runs_layout_gate_only_after_pdf_build():
    makefile = (ROOT / "paper/Makefile").read_text(encoding="utf-8")
    release = makefile.split("\nrelease:\n", 1)[1].split("\n\nvendor:", 1)[0]

    assert release.index("$(MAKE) $(BUILD)/main.pdf") < release.index("-m embed_optim.paper_layout")
    assert "paper_layout" not in makefile.split("\nrelease:\n", 1)[0]
