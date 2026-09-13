"""Synthetic compiled-label controls; not a real PDF or scientific publication."""

import pytest

from embed_optim import paper_layout as original
from embed_optim import primary_v3_paper_layout as layout


def auxiliary(tmp_path):
    paper = tmp_path / "paper"
    (paper / "build").mkdir(parents=True)
    rows = {original.MAIN_END_LABEL: 7}
    rows.update({name: 4 for name in layout.MAIN_TEXT_FLOAT_LABELS})
    rows.update({name: 9 for name in layout.APPENDIX_FLOAT_LABELS})
    return paper, rows


def write_aux(paper, rows):
    (paper / "build/main.aux").write_text(
        "".join("\\newlabel{" + name + "}{{1}{" + str(page) + "}}\n" for name, page in rows.items())
    )


def test_complete_layout_requires_original_floats_and_separate_exact_table(tmp_path):
    paper, rows = auxiliary(tmp_path)
    write_aux(paper, rows)
    result = layout.audit_paper_layout(paper)
    assert result["complete"] is True and result["scientific_completion"] is False
    assert result["main_end_page"] == 7 and result["max_main_page"] == 8
    assert set(result["appendix_float_pages"]) == set(layout.APPENDIX_FLOAT_LABELS)
    with pytest.raises(ValueError, match="topology"):
        original.audit_paper_layout(paper)
    del rows["tab:exact-geometry-sensitivity"]
    write_aux(paper, rows)
    assert original.audit_paper_layout(paper)["complete"] is True
    with pytest.raises(ValueError, match="topology"):
        layout.audit_paper_layout(paper)


@pytest.mark.parametrize("name", layout.FLOAT_LABELS)
@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_region"])
def test_each_required_float_must_be_unique_present_and_in_its_region(tmp_path, name, change):
    paper, rows = auxiliary(tmp_path)
    if change == "missing":
        del rows[name]
    elif change == "wrong_region":
        rows[name] = 9 if name in layout.MAIN_TEXT_FLOAT_LABELS else 7
    write_aux(paper, rows)
    if change == "duplicate":
        with (paper / "build/main.aux").open("a") as stream:
            stream.write("\\newlabel{" + name + "}{{1}{9}}\n")
    with pytest.raises(ValueError):
        layout.audit_paper_layout(paper)


@pytest.mark.parametrize("page", ["i", 0, 9])
def test_main_endpoint_cannot_be_invalid_or_exceed_eight_pages(tmp_path, page):
    paper, rows = auxiliary(tmp_path)
    rows[original.MAIN_END_LABEL] = page
    write_aux(paper, rows)
    with pytest.raises(ValueError):
        layout.audit_paper_layout(paper)


def test_extra_float_is_not_silently_ignored(tmp_path):
    paper, rows = auxiliary(tmp_path)
    rows["tab:unregistered"] = 9
    write_aux(paper, rows)
    with pytest.raises(ValueError, match="unexpected"):
        layout.audit_paper_layout(paper)


def test_auxiliary_symlink_is_not_a_compilation_receipt(tmp_path):
    paper, rows = auxiliary(tmp_path)
    write_aux(paper, rows)
    path = paper / "build/main.aux"
    path.rename(tmp_path / "retained.aux")
    path.symlink_to(tmp_path / "retained.aux")
    with pytest.raises(ValueError, match="ordinary"):
        layout.audit_paper_layout(paper)
