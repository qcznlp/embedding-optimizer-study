"""Fresh-document parsing/refusal checks; no primary/factorial admission is simulated."""

from pathlib import Path

import pytest

from embed_optim import primary_v3_manuscript_build as build


def captions():
    values = [("figure", 1, 3), ("figure", 2, 5)]
    values += [("table", index, 4 if index == 1 else 9) for index in range(1, 8)]
    return values


def aux(values):
    return "".join(
        "\\@writefile{" + {"figure": "lof", "table": "lot"}[kind] + "}{"
        "\\contentsline {"
        + kind
        + "}{\\numberline {"
        + str(number)
        + "}{caption}}{"
        + str(page)
        + "}{anchor}\\protected@file@percent }\n"
        for kind, number, page in values
    )


def test_caption_inventory_includes_unlabelled_functional_predictor_table():
    result = build.float_inventory(aux(captions()), 7, 11)
    assert len(result["figure"]) == 2 and len(result["table"]) == 7
    assert result["table"][4]["number"] == 5


@pytest.mark.parametrize("position", range(9))
@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_region"])
def test_every_caption_is_counted_and_located(position, kind):
    values = captions()
    if kind == "missing":
        values.pop(position)
    elif kind == "duplicate":
        values.insert(position, values[position])
    else:
        category, number, page = values[position]
        values[position] = (category, number, 9 if page <= 7 else 7)
    with pytest.raises(ValueError):
        build.float_inventory(aux(values), 7, 11)


@pytest.mark.parametrize("page", [0, -1, 12, "iv", "x", "١"])
def test_caption_pages_must_belong_to_actual_pdf(page):
    values = captions()
    values[0] = ("figure", 1, page)
    with pytest.raises(ValueError):
        build.float_inventory(aux(values), 7, 11)


@pytest.mark.parametrize("change", ["kind", "prefix", "number", "unclosed"])
def test_malformed_caption_records_refuse(change):
    value = aux(captions())
    if change == "kind":
        value = value.replace(r"\contentsline {figure}", r"\contentsline {table}", 1)
    elif change == "prefix":
        value = value.replace(r"\contentsline", r"\unknown", 1)
    elif change == "number":
        value = value.replace(r"\numberline {1}", r"\numberline {x}", 1)
    else:
        value = value[:-2]
    with pytest.raises(ValueError):
        build.float_inventory(value, 7, 11)


def font_text(kind="Type 1", embedded="yes"):
    return "name type encoding emb sub uni object ID\n-------------------\n" + (
        f"ABC+Font {kind} Custom {embedded} yes yes 12 0\n"
    )


def test_actual_fonts_must_be_embedded_non_type3_and_nonempty():
    assert len(build.font_inventory(font_text())) == 1


@pytest.mark.parametrize(
    "text",
    ["", "name\n---\n", font_text("Type 3"), font_text(embedded="no"), "name\n---\ninvalid\n"],
)
def test_font_failures_cannot_be_proved_by_substring_absence(text):
    with pytest.raises(ValueError):
        build.font_inventory(text)


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    paper = tmp_path / "paper"
    for name in (*build.TEX_INPUTS, "build/main.bbl"):
        path = paper / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic compiler-input parser fixture")
    system = tmp_path / "system"
    system.mkdir()
    (system / "class.cls").write_text("Synthetic system source")
    monkeypatch.setattr(build, "SYSTEM_TEX_ROOTS", (system,))
    lines = ["PWD " + str(paper)]
    lines += ["INPUT " + name for name in sorted(build.TEX_INPUTS)]
    lines += ["INPUT build/main.bbl", "INPUT " + str(system / "class.cls")]
    return paper, lines, system


def test_fresh_compiler_requires_all_declared_actual_inputs(recorder):
    paper, lines, _ = recorder
    result = build.recorded_inputs(paper, "\n".join(lines))
    assert set(result["document"]) == build.TEX_INPUTS
    assert len(result["system"]) == 1 and result["generated"] == ["build/main.bbl"]


@pytest.mark.parametrize("name", sorted(build.TEX_INPUTS))
def test_omitted_compiler_input_is_not_a_fresh_build_proof(recorder, name):
    paper, lines, _ = recorder
    lines.remove("INPUT " + name)
    with pytest.raises(ValueError):
        build.recorded_inputs(paper, "\n".join(lines))


@pytest.mark.parametrize(
    "change",
    [
        "pwd",
        "duplicate_pwd",
        "foreign_pwd",
        "old_tree",
        "extra_local",
        "missing_bbl",
        "missing_system",
    ],
)
def test_recorder_does_not_consume_old_locations_or_incomplete_sources(recorder, change):
    paper, lines, system = recorder
    if change == "pwd":
        lines[0] = "PWD /some/old/paper"
    elif change == "duplicate_pwd":
        lines.append(lines[0])
    elif change == "foreign_pwd":
        lines.append("PWD /some/old/paper")
    elif change == "old_tree":
        path = paper.parent / "old-manuscript.tex"
        path.write_text("Not a declared compiler input")
        lines.append("INPUT " + str(path))
    elif change == "extra_local":
        (paper / "undeclared.tex").write_text("Unrelated manuscript content")
        lines.append("INPUT undeclared.tex")
    elif change == "missing_bbl":
        lines.remove("INPUT build/main.bbl")
    else:
        lines.remove("INPUT " + str(system / "class.cls"))
    with pytest.raises(ValueError):
        build.recorded_inputs(paper, "\n".join(lines))


def test_existing_pending_repository_fails_before_output_or_compiler(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "not-created"

    def should_not_compile(*args, **kwargs):
        pytest.fail("The incomplete paper reached the compiler")

    monkeypatch.setattr(build.subprocess, "run", should_not_compile)
    includes = {name: (root / "paper" / name).read_bytes() for name in build.document.RESULT_FILES}
    with pytest.raises(ValueError):
        build.compile_document(root, includes, b"not the active constants", output)
    assert not output.exists()


def test_compilation_is_explicitly_cpu_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU only"):
        build.compile_document(tmp_path, {}, b"", tmp_path / "absent")
    assert list(tmp_path.iterdir()) == []
