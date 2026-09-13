"""Compile a complete paper copy with explicit synthetic results, never the manuscript."""

import argparse
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from embed_optim.paper_audit import _abstract_word_budget
from embed_optim.paper_layout import audit_paper_layout
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.state_operator_factorial_publication import _render_latex
from scripts.audit_dense_natural_data import handoff


def run(args):
    root, work = args.repository.resolve(), args.workdir.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not work.is_dir() or list(work.iterdir()):
        raise ValueError("Require a new empty CPU-only synthetic layout directory")
    before = handoff()
    assert file_identity(args.generation)["sha256"] == args.generation_sha256
    generation = read_json(args.generation)
    assert (
        generation["passed"] is True and generation["upstream_primary_admission_simulated"] is True
    )
    assert generation["actual_checkpoint_backed_positive_authoring_verified"] is False
    for row in generation["sources"]:
        verify_file(row["path"], row)
    records = {row["path"]: row for row in generation["generated_artifacts"]}
    originals = []
    for path in sorted((root / "paper").rglob("*")):
        if "build" in path.relative_to(root / "paper").parts:
            continue
        assert not path.is_symlink()
        if path.is_file():
            originals.append({"path": str(path), **file_identity(path)})
    paper = work / "paper"
    shutil.copytree(root / "paper", paper, ignore=shutil.ignore_patterns("build"))
    for name in ("optimizer-primary.tex", "dimension-utilization.tex"):
        source = args.generation.parent / "synthetic-evidence" / name
        verify_file(source, records[str(source)])
        (paper / "generated" / name).write_bytes(source.read_bytes())
    helper = root / "tests/test_state_operator_factorial_completion.py"
    estimands = runpy.run_path(str(helper))["_estimands"](
        "supported_positive", "supported_positive", "supported_positive"
    )
    (paper / "generated/state-operator-factorial.tex").write_text(_render_latex(estimands))
    manuscript = (paper / "main.tex").read_text()
    assert manuscript.count(r"\maketitle") == 1
    manuscript = manuscript.replace(
        r"\maketitle",
        r"\maketitle\begin{center}\textbf{SYNTHETIC LAYOUT TEST -- NOT EXPERIMENT RESULTS}\end{center}",
    )
    (paper / "main.tex").write_text(manuscript)
    for name in ("acl.sty", "acl_natbib.bst"):
        assert (paper / "vendor" / name).is_file()
    result = subprocess.run(
        ["make", "all", "PYTHON=" + sys.executable],
        cwd=paper,
        capture_output=True,
        text=True,
        timeout=55,
        env={
            **os.environ,
            "TEXINPUTS": str(paper / "vendor") + "//:",
            "BSTINPUTS": str(paper / "vendor") + "//:",
        },
    )
    (work / "make-stdout.txt").write_text(result.stdout)
    (work / "make-stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout[-5000:]
    layout = audit_paper_layout(paper)
    abstract = _abstract_word_budget(work)
    assert layout["complete"] is True and abstract["complete"] is True
    log = (paper / "build/main.log").read_text()
    assert "Overfull" not in log
    pdf = paper / "build/main.pdf"
    fonts = subprocess.run(
        ["pdffonts", str(pdf)], capture_output=True, text=True, check=True
    ).stdout
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    assert "Type 3" not in fonts and "Type 1" in fonts
    (work / "fonts.txt").write_text(fonts)
    (work / "pdfinfo.txt").write_text(info)
    for row in originals:
        verify_file(row["path"], row)
    require_same(handoff(), before)
    receipt = {
        "scope": "engineering_complete_synthetic_paper_layout",
        "passed": True,
        "complete_synthetic_paper_layout_verified": True,
        "layout": layout,
        "abstract_conservative_budget": abstract,
        "overfull_boxes": False,
        "type3_fonts": False,
        "primary_and_functional_include_source": {
            "path": str(args.generation),
            "sha256": args.generation_sha256,
        },
        "factorial_inputs": {"scope": "synthetic_layout_only", "estimands": estimands},
        "actual_primary_results": False,
        "manuscript_installed": False,
        "strict_manuscript_publication_audit_passed": False,
        "scientific_completion": False,
        "original_paper_bindings": originals,
        "sources": [
            {"path": str(path), **file_identity(path)}
            for path in (
                Path(__file__).resolve(),
                helper,
                root / "src/embed_optim/paper_layout.py",
                root / "src/embed_optim/paper_audit.py",
                root / "src/embed_optim/state_operator_factorial_publication.py",
            )
        ],
        "artifacts": [
            {"path": str(path), **file_identity(path)}
            for path in sorted(work.rglob("*"))
            if path.is_file()
        ],
        "post_execution_dispatchers": before,
    }
    write_new(work / "layout.json", receipt)
    print(
        {
            "passed": True,
            "main_end_page": layout["main_end_page"],
            "scientific_completion": False,
            **file_identity(work / "layout.json"),
        },
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "workdir", "generation"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generation-sha256", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
