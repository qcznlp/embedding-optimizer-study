"""Current document entry and unchanged complete-topology controls."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import tomllib

from embed_optim import complete_paper_document as document
from embed_optim import current_paper as entry

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "paper/current"


@pytest.fixture
def paper(tmp_path):
    target = tmp_path / "source/paper"
    shutil.copytree(CURRENT, target)
    return target


def inspect(paper):
    includes = {name: (paper / name).read_bytes() for name in document.RESULT_FILES}
    return document.inspect_sources(paper.parent, includes, (paper / "results.tex").read_bytes())


def test_original_complete_component_is_byte_identical():
    assert (
        hashlib.sha256(Path(document.__file__).read_bytes()).hexdigest() == entry.COMPONENT_SHA256
    )


def test_complete_current_snapshot_and_semantic_read(paper):
    _, raw, snapshot = entry.read_snapshot(paper)
    assert hashlib.sha256(raw).hexdigest() == entry.SNAPSHOT_SHA256
    assert len(snapshot["inputs"]) == 12
    result = inspect(paper)
    assert result["actual_abstract"]["words_conservative"] == 158
    assert result["final_release_verified"] is False
    assert result["upstream_scientific_admission_performed"] is False


@pytest.mark.parametrize("name", document.BUILD_INPUTS)
def test_every_changed_input_is_refused_before_output_creation(paper, tmp_path, monkeypatch, name):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    path = paper / name
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="input differs"):
        entry.build_current_paper(paper, output)
    assert not output.exists()


@pytest.mark.parametrize("name", (*document.BUILD_INPUTS, entry.SNAPSHOT_NAME))
def test_every_missing_input_is_refused(paper, name):
    (paper / name).unlink()
    with pytest.raises(ValueError, match="ordinary manuscript"):
        entry.read_snapshot(paper)


def test_rehashing_an_edited_snapshot_cannot_admit_changed_results(paper):
    snapshot_path = paper / entry.SNAPSHOT_NAME
    value = json.loads(snapshot_path.read_text())
    name = document.RESULT_FILES[0]
    source = paper / name
    source.write_bytes(source.read_bytes() + b"\n% altered\n")
    value["inputs"][name] = entry._identity(source.read_bytes())
    snapshot_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Not the reviewed"):
        entry.read_snapshot(paper)


@pytest.mark.parametrize("name", ("main.tex", "generated", "vendor", entry.SNAPSHOT_NAME))
def test_symlinked_inputs_and_directories_refused(paper, tmp_path, name):
    source = paper / name
    relocated = tmp_path / "relocated"
    source.rename(relocated)
    source.symlink_to(relocated, target_is_directory=relocated.is_dir())
    with pytest.raises(ValueError, match="ordinary manuscript"):
        entry.read_snapshot(paper)


@pytest.mark.parametrize("visible", (None, "0", "0,1,2,3"))
def test_gpu_visible_document_invocation_refused(paper, tmp_path, monkeypatch, visible):
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        entry.build_current_paper(paper, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ("relative", "existing", "source-child", "symlink", "traversal"))
def test_unsafe_or_reused_output_refused(paper, tmp_path, monkeypatch, kind):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "output"
    if kind == "relative":
        output = Path("new-relative")
    elif kind == "existing":
        output.mkdir()
        (output / "keep.txt").write_text("preserve")
    elif kind == "source-child":
        output = paper / "output"
    elif kind == "symlink":
        output.symlink_to(tmp_path / "not-present", target_is_directory=True)
    elif kind == "traversal":
        output = tmp_path / "source/../output"
    with pytest.raises(ValueError):
        entry.build_current_paper(paper, output)
    if kind == "existing":
        assert (output / "keep.txt").read_text() == "preserve"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (r"\input{generated/recipe-sensitivity}", ""),
        (r"\WeightRecipeSensitivityTable", ""),
        (r"\FunctionalRecipeSensitivityTable", ""),
        ("figures/weight-to-retrieval-map.pdf", "figures/unbound.pdf"),
        (r"\begin{abstract}", r"\begin{abstract}" + " word" * 201),
        (r"\begin{document}", r"\newcommand{\CorrectedConclusionFinding}{fake}\begin{document}"),
        (r"\begin{document}", r"\input{unbound}\begin{document}"),
        (r"\section{Introduction}", r"\section{Introduction} PENDING"),
        (r"\section{Introduction}", r"\section{Introduction} padding bug"),
    ],
)
def test_unchanged_semantic_checker_rejects_invalid_document(paper, old, new):
    path = paper / "main.tex"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))
    with pytest.raises(ValueError):
        inspect(paper)


def test_failed_compiler_preserves_output_and_does_not_claim_success(paper, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def fail(*args):
        raise RuntimeError("intentional compiler failure")

    monkeypatch.setattr(document, "compile_fresh", fail)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="intentional"):
        entry.build_current_paper(paper, output)
    failure = json.loads((output / "failed.json").read_text())
    assert failure["automatic_retry"] is False and failure["outputs_preserved"] is True
    assert (output / "paper/main.tex").read_bytes() == (paper / "main.tex").read_bytes()
    assert not (output / "current-paper.json").exists()


def test_all_current_document_inputs_are_declared_in_distribution():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declarations = config["tool"]["setuptools"]["data-files"]
    declared = {p for rows in declarations.values() for p in rows}
    required = {
        "paper/current/" + n for n in (*document.BUILD_INPUTS, entry.SNAPSHOT_NAME, "README.md")
    }
    assert required <= declared
    assert (
        config["project"]["scripts"]["embed-optim-build-current-paper"]
        == "embed_optim.current_paper:main"
    )


def test_new_make_target_keeps_legacy_release_unchanged():
    makefile = (ROOT / "paper/Makefile").read_text()
    assert ".DEFAULT_GOAL := all" in makefile
    target = makefile.split("\ncurrent:\n", 1)[1].split("\nall:", 1)[0]
    assert "embed_optim.current_paper" in target
    assert "--paper-dir current" in target
    assert "--output" in target
    release = makefile.split("\nrelease:\n", 1)[1].split("\n\nvendor:", 1)[0]
    assert "embed_optim.paper_layout" in release
    assert "embed_optim.paper_audit --strict" in release
