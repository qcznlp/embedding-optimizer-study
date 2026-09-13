"""Fresh compilation of verified active-v3 document inputs, never scientific release.

The caller must first perform actual complete primary/factorial admission. This
document component prevents a stale auxiliary file or unrelated PDF from proving
layout compliance. It changes only a new output directory outside the source tree.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from . import paper_audit, paper_layout
from . import primary_v3_manuscript as document
from . import primary_v3_paper_layout as layout
from .primary_contract import file_identity, require_same, verify_file
from .primary_v3_validation_io import write_new

BUILD_INPUTS = (
    "main.tex",
    "results.tex",
    *document.RESULT_FILES,
    "references.bib",
    "vendor/acl.sty",
    "vendor/acl_natbib.bst",
    "figures/optimizer-weight-dimension-map.pdf",
)
TEX_INPUTS = set(BUILD_INPUTS) - {"references.bib", "vendor/acl_natbib.bst"}
SYSTEM_TEX_ROOTS = tuple(
    Path(value)
    for value in (
        "/etc/texmf",
        "/usr/share/texmf",
        "/usr/share/texlive",
        "/usr/share/fonts",
        "/var/lib/texmf",
    )
)
FONT_ROW = re.compile(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$")


def float_inventory(aux, main_end_page, pdf_pages):
    """Cover all nine captions, including the functional table without its own label."""
    observed = {"figure": [], "table": []}
    for match in re.finditer(r"\\@writefile\{(lof|lot)\}", aux):
        body, _ = document.brace_group(aux, match.end())
        prefix = re.match(r"\\contentsline\s*", body)
        if prefix is None:
            raise ValueError("Malformed compiled caption entry")
        kind, position = document.brace_group(body, prefix.end())
        caption, position = document.brace_group(body, position)
        page, _ = document.brace_group(body, position)
        numberline = re.match(r"\\numberline\s*", caption)
        if kind != {"lof": "figure", "lot": "table"}[match[1]] or numberline is None:
            raise ValueError("Compiled caption kind differs")
        number, _ = document.brace_group(caption, numberline.end())
        if (
            not number.isascii()
            or not number.isdecimal()
            or not page.isascii()
            or not page.isdecimal()
        ):
            raise ValueError("Compiled float number/page is not a positive integer")
        number, page = int(number), int(page)
        if number < 1 or not 1 <= page <= pdf_pages:
            raise ValueError("Compiled float is outside the current PDF")
        observed[kind].append({"number": number, "page": page, "caption_tex": caption})
    if [row["number"] for row in observed["figure"]] != [1, 2] or [
        row["number"] for row in observed["table"]
    ] != list(range(1, 8)):
        raise ValueError("Complete active-v3 caption inventory differs")
    if any(row["page"] > main_end_page for row in observed["figure"]):
        raise ValueError("An active-v3 main figure is deferred beyond the main text")
    if observed["table"][0]["page"] > main_end_page or any(
        row["page"] <= main_end_page for row in observed["table"][1:]
    ):
        raise ValueError("An active-v3 table is in the wrong manuscript region")
    return observed


def font_inventory(text):
    lines = text.splitlines()
    if len(lines) < 3 or not lines[0].startswith("name") or not lines[1].startswith("---"):
        raise ValueError("Missing or malformed current PDF font inventory")
    fonts = []
    for line in lines[2:]:
        match = FONT_ROW.search(line)
        if not line.strip() or match is None or match[1] != "yes" or "Type 3" in line:
            raise ValueError("An actual PDF font is missing, unembedded, or Type 3")
        fonts.append(line)
    if not fonts:
        raise ValueError("Empty current PDF font inventory")
    return fonts


def recorded_inputs(paper, text):
    """Resolve compiler records only inside the fresh copy or standard system TeX roots."""
    paper = document.ordinary(paper, directory=True)
    if [line for line in text.splitlines() if line.startswith("PWD ")] != ["PWD " + str(paper)]:
        raise ValueError("Compiler record does not identify the fresh document directory")
    selected, systems, generated = {}, {}, set()
    for line in text.splitlines():
        if not line.startswith("INPUT "):
            continue
        path = Path(line[6:])
        if not path.is_absolute():
            path = paper / path
        resolved = path.resolve(strict=True)
        if resolved.is_relative_to(paper):
            name = resolved.relative_to(paper).as_posix()
            document.ordinary(path)
            if name in TEX_INPUTS:
                selected[name] = file_identity(path)
            elif name in {"build/main.aux", "build/main.out", "build/main.bbl"}:
                generated.add(name)
            else:
                raise ValueError("Undeclared project file consumed by the fresh compiler")
        elif any(resolved.is_relative_to(root) for root in SYSTEM_TEX_ROOTS):
            if not resolved.is_file():
                raise ValueError("Invalid system TeX dependency")
            systems[str(resolved)] = file_identity(resolved)
        else:
            raise ValueError("Compiler consumed a file outside the declared document/system roots")
    require_same(sorted(selected), sorted(TEX_INPUTS))
    if not systems or "build/main.bbl" not in generated:
        raise ValueError("Incomplete compiler source/bibliography record")
    return {"document": selected, "system": systems, "generated": sorted(generated)}


def compile_document(repository, expected_includes, constants, output):
    """Compile exactly the verified inputs; a prior PDF/auxiliary receipt is not accepted."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Manuscript verification is CPU only")
    root = document.ordinary(repository, directory=True)
    initial = document.inspect_sources(root, expected_includes, constants)
    output = Path(output).absolute()
    if (
        output.exists()
        or any(p.is_symlink() for p in (output, *output.parents))
        or output == root
        or output in root.parents
        or root in output.parents
    ):
        raise ValueError("Use a new document verification directory outside the source tree")
    sources = {
        name: file_identity(document.ordinary(root / "paper" / name)) for name in BUILD_INPUTS
    }
    require_same({name: sources[name] for name in initial["inputs"]}, initial["inputs"])
    implementations = {
        str(Path(module.__file__).resolve()): file_identity(module.__file__)
        for module in (document, layout, paper_audit, paper_layout)
    }
    implementations[str(Path(__file__).resolve())] = file_identity(__file__)
    paper = output / "paper"
    paper.mkdir(parents=True, exist_ok=False)
    for name, identity in sources.items():
        source, destination = root / "paper" / name, paper / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())
        verify_file(source, identity)
        verify_file(destination, identity)
    environment = {
        **os.environ,
        "TEXINPUTS": str(paper / "vendor") + "//:",
        "BSTINPUTS": str(paper / "vendor") + "//:",
    }
    command = [
        "latexmk",
        "-pdf",
        "-bibtex",
        "-no-shell-escape",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-recorder",
        "-outdir=build",
        "main.tex",
    ]
    result = subprocess.run(
        command, cwd=paper, env=environment, capture_output=True, text=True, timeout=180
    )
    for name, text in (
        ("compile-stdout.txt", result.stdout),
        ("compile-stderr.txt", result.stderr),
    ):
        with (output / name).open("x") as stream:
            stream.write(text)
    if result.returncode:
        raise ValueError("Fresh manuscript compilation failed; exact logs are retained")
    build = paper / "build"
    compiled = layout.audit_paper_layout(paper)
    log = document.ordinary(build / "main.log").read_text()
    if "Overfull" in log or re.search(
        r"(?:Citation|Reference) .+ undefined|There were undefined references", log
    ):
        raise ValueError(
            "Fresh manuscript has overflowing boxes or unresolved citations/references"
        )
    blg = document.ordinary(build / "main.blg").read_text()
    if re.findall(r"^Database file #\d+: (.+)$", blg, re.MULTILINE) != [
        "references.bib"
    ] or re.findall(r"^The style file: (.+)$", blg, re.MULTILINE) != ["acl_natbib.bst"]:
        raise ValueError("Fresh bibliography input/style differs")
    pdf = document.ordinary(build / "main.pdf")
    font_text = subprocess.run(
        ["pdffonts", str(pdf)], capture_output=True, text=True, check=True, timeout=30
    ).stdout
    info = subprocess.run(
        ["pdfinfo", str(pdf)], capture_output=True, text=True, check=True, timeout=30
    ).stdout
    page_counts = re.findall(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    if len(page_counts) != 1 or int(page_counts[0]) < compiled["main_end_page"]:
        raise ValueError("Invalid current PDF page count")
    fonts = font_inventory(font_text)
    captions = float_inventory(
        document.ordinary(build / "main.aux").read_text(),
        compiled["main_end_page"],
        int(page_counts[0]),
    )
    inputs = recorded_inputs(paper, document.ordinary(build / "main.fls").read_text())
    require_same(inputs["document"], {name: sources[name] for name in TEX_INPUTS})
    for name, identity in sources.items():
        verify_file(root / "paper" / name, identity)
        verify_file(paper / name, identity)
    for name, identity in implementations.items():
        verify_file(name, identity)
    for name, identity in inputs["system"].items():
        verify_file(name, identity)
    require_same(document.inspect_sources(root, expected_includes, constants), initial)
    for name, text in (("fonts.txt", font_text), ("pdfinfo.txt", info)):
        with (output / name).open("x") as stream:
            stream.write(text)
    receipt = {
        "scope": "dense_primary_v3_fresh_document_component",
        "compiled_pdf_verified": True,
        "document": initial,
        "source_inputs": sources,
        "executing_sources": implementations,
        "compiler_command": command,
        "compiler_inputs": inputs,
        "layout": compiled,
        "all_caption_inventory": captions,
        "font_count": len(fonts),
        "pdf_pages": int(page_counts[0]),
        "upstream_scientific_admission_performed": False,
        "final_release_verified": False,
        "artifacts": {
            p.relative_to(output).as_posix(): file_identity(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write_new(output / "document.json", receipt)
    return receipt
