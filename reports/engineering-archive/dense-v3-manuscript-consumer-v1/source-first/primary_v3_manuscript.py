"""Actual active-manuscript bytes and expanded abstract, without scientific admission.

The complete v3 consumer must supply freshly reconstructed primary, dimension,
and routed-factorial includes. This module is its document component, not a
replacement for checkpoint admission, portable reconstruction, or release review.
It deliberately exposes no final-release CLI or user-supplied success callback.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import paper_audit as legacy
from .primary_contract import digest, file_identity, require_same, verify_file

RESULT_FILES = (
    "generated/optimizer-primary.tex",
    "generated/dimension-utilization.tex",
    "generated/state-operator-factorial.tex",
)
CONSTANTS = {
    "NumPrimaryRuns": 12,
    "NumPrimaryCheckpoints": 60,
    "NumBEIRTasks": 14,
    "NumPrimaryUnits": 840,
    "NumTrainingQueries": 500000,
    "NumHardNegatives": 7,
    "ContextLength": 8192,
}
ACTIVE_INPUTS = ("results", *(name.removesuffix(".tex") for name in RESULT_FILES))
ABSTRACT_MACROS = ("CorrectedAbstractFinding", "StateOperatorAbstractFinding")
DEFINITION = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\*?\s*")
CONTROL = re.compile(r"\\([A-Za-z@]+|.)", re.DOTALL)
INPUT = re.compile(r"\\(input|include)\s*\{([^{}]+)\}")
INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[[^\[\]]*\])?\s*\{([^{}]+)\}")
FORBIDDEN_CONTROL = re.compile(
    r"\\(?:def|gdef|edef|xdef|catcode|csname|endcsname|expandafter|input|include|"
    r"openin|openout|read|write|immediate|special|directlua|pdfobj|pdfliteral)"
    r"(?![A-Za-z@])"
)
FORMATTERS = {"emph", "textbf", "textit", "textrm", "textsf", "texttt", "mbox"}
LITERAL_COMMANDS = {"%": "%", "&": "&", "_": "_", "#": "#", "{": "{", "}": "}"}


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not (
        path.is_dir() if directory else path.is_file()
    ):
        raise ValueError("Require an ordinary manuscript input")
    return path


def brace_group(text, position):
    """Read one balanced group; escaped braces do not change the nesting depth."""
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "{":
        raise ValueError("Expected a braced manuscript argument")
    start, depth, position = position + 1, 1, position + 1
    while position < len(text):
        if text[position] == "\\":
            match = CONTROL.match(text, position)
            if match is None:
                raise ValueError("Truncated manuscript control")
            position = match.end()
            continue
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position], position + 1
        position += 1
    raise ValueError("Unclosed manuscript argument")


def definitions(text):
    """Parse top-level generated definitions, not commands nested in their bodies."""
    text = legacy._strip_latex_comments(text)
    result, position = {}, 0
    while position < len(text):
        match = DEFINITION.search(text, position)
        if match is None:
            break
        macro, end = brace_group(text, match.end())
        if re.fullmatch(r"\\[A-Za-z@]+", macro) is None:
            raise ValueError("Invalid generated macro name")
        name = macro[1:]
        while end < len(text) and text[end].isspace():
            end += 1
        arity = 0
        if end < len(text) and text[end] == "[":
            argument = re.match(r"\[([0-9])\]", text[end:])
            if argument is None:
                raise ValueError("Unsupported generated macro argument declaration")
            arity, end = int(argument[1]), end + argument.end()
        body, position = brace_group(text, end)
        result.setdefault(name, []).append(
            {"kind": match[0].split("{")[0].strip(), "arity": arity, "body": body}
        )
    return result


def expand_plain(text, macros, *, stack=()):
    """Expand the restricted abstract vocabulary; unknown TeX is an error, not zero words."""
    if len(stack) > 32:
        raise ValueError("Recursive abstract expansion")
    output, position = [], 0
    while position < len(text):
        character = text[position]
        if character == "\\":
            match = CONTROL.match(text, position)
            if match is None:
                raise ValueError("Truncated abstract command")
            name, position = match[1], match.end()
            if name in LITERAL_COMMANDS:
                output.append(LITERAL_COMMANDS[name])
            elif name in FORMATTERS:
                body, position = brace_group(text, position)
                output.append(expand_plain(body, macros, stack=stack))
            elif name in macros:
                if name in stack:
                    raise ValueError("Recursive abstract macro")
                output.append(expand_plain(macros[name], macros, stack=(*stack, name)))
            else:
                raise ValueError("Unsupported abstract command: " + name)
        elif character == "{":
            body, position = brace_group(text, position)
            output.append(expand_plain(body, macros, stack=stack))
        elif character in "}$#":
            raise ValueError("Unexpected abstract grouping, math or parameter token")
        else:
            output.append(character)
            position += 1
    return "".join(output)


def abstract_readout(main, includes):
    """Count the actual result-expanded abstract with the existing conservative tokenizer."""
    source = legacy._strip_latex_comments(main)
    matches = re.findall(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.DOTALL)
    if len(matches) != 1:
        raise ValueError("Require exactly one abstract")
    body = matches[0]
    declared = {}
    for content in includes:
        for name, values in definitions(content).items():
            declared.setdefault(name, []).extend(values)
    macros = {}
    for name in ABSTRACT_MACROS:
        values = declared.get(name, [])
        if (
            len(values) != 1
            or values[0]["arity"] != 0
            or not values[0]["kind"].startswith(r"\newcommand")
            or len(re.findall(r"\\" + name + r"(?![A-Za-z@])", body)) != 1
        ):
            raise ValueError("An abstract finding is absent, repeated, redefined or parameterized")
        macros[name] = values[0]["body"]
    # No other custom command can silently change the actual result text or its count.
    expanded = expand_plain(body, macros)
    words = legacy.ABSTRACT_WORD_PATTERN.findall(expanded)
    if len(words) > legacy.ABSTRACT_WORD_LIMIT:
        raise ValueError("Actual expanded abstract exceeds 200 words")
    return {
        "expanded_text": expanded,
        "words_conservative": len(words),
        "limit": legacy.ABSTRACT_WORD_LIMIT,
        "dynamic_words": {
            name: len(legacy.ABSTRACT_WORD_PATTERN.findall(expand_plain(value, macros)))
            for name, value in macros.items()
        },
        "counting_rule": "Actual expansion; ASCII alphanumeric tokens, apostrophe compounds; decimals and hyphens split",
    }


def render_constants(primary):
    """Generate only active v3 constants. Reading a draft is not a training admission."""
    rows = primary.inputs["runs"]
    if len(rows) != CONSTANTS["NumPrimaryRuns"] or len({r["run_id"] for r in rows}) != 12:
        raise ValueError("Require all twelve distinct primary identities")
    recipes = []
    for row in rows:
        run_id = row["run_id"]
        if not run_id.startswith("verified-v3-"):
            raise ValueError("Historical run cannot supply active manuscript constants")
        expected = primary.expected_identity(run_id)
        if expected["data"]["rows"] != CONSTANTS["NumTrainingQueries"]:
            raise ValueError("Primary training row count differs")
        recipes.append(expected["recipe"])
    require_same(primary.payload["checkpoint_steps"], [782, 1563, 2345, 3126, 3907])
    if len(primary.payload["beir_task_revisions"]) != CONSTANTS["NumBEIRTasks"]:
        raise ValueError("Primary task population differs")
    rates = {name: [] for name in ("adamw", "muon", "normuon")}
    for recipe in recipes:
        if recipe["max_length"] != 8192 or recipe["hard_negatives"] != 7:
            raise ValueError("Primary context/negative constants differ")
        if recipe["optimizer"]["name"] not in rates:
            raise ValueError("Non-primary optimizer in the manuscript population")
        rates[recipe["optimizer"]["name"]].append(recipe["optimizer"]["lr"])
    require_same(
        {name: sorted(values) for name, values in rates.items()},
        {
            "adamw": [1e-6, 3e-6, 1e-5, 3e-5],
            "muon": [1e-4, 3e-4, 1e-3, 3e-3],
            "normuon": [1e-4, 3e-4, 1e-3, 3e-3],
        },
    )
    lines = ["% Active v3 constants only; no historical findings or scientific admission."]
    for name, value in CONSTANTS.items():
        formatted = f"{value:,}".replace(",", "{,}")
        lines.append("\\newcommand{\\" + name + "}{" + formatted + "}")
    return ("\n".join(lines) + "\n").encode()


def inspect_sources(repository, expected_includes, constants):
    """Compare all active bytes before semantic checks; supply no upstream success bit."""
    root = ordinary(repository, directory=True)
    paper = ordinary(root / "paper", directory=True)
    require_same(sorted(expected_includes), sorted(RESULT_FILES))
    expected = {"results.tex": constants, **expected_includes}
    if any(type(content) is not bytes for content in expected.values()):
        raise ValueError("Expected manuscript artifacts must be freshly generated bytes")
    paths = {name: ordinary(paper / name) for name in ("main.tex", *expected)}
    before = {name: file_identity(path) for name, path in paths.items()}
    texts = {}
    for name, path in paths.items():
        content = path.read_bytes()
        if name in expected and content != expected[name]:
            raise ValueError("Installed manuscript differs from regenerated evidence: " + name)
        texts[name] = legacy._strip_latex_comments(content.decode("utf-8"))
    main = texts["main.tex"]
    if tuple(match[2] for match in INPUT.finditer(main)) != ACTIVE_INPUTS:
        raise ValueError("Active manuscript input topology differs")
    unmatched = INPUT.sub("", main)
    if re.search(r"\\(?:input|include)(?![A-Za-z@])", unmatched):
        raise ValueError("Unbraced or indirect manuscript input")
    if INCLUDEGRAPHICS.findall(main) != ["figures/optimizer-weight-dimension-map.pdf"]:
        raise ValueError("Active manuscript figure topology differs")
    if not legacy._paper_main_topology_complete(root):
        raise ValueError("Active manuscript result/section ordering differs")
    for name, text in texts.items():
        if r"\ResultPending" in text or re.search(r"\b(?:PENDING|TODO|TBD)\b", text):
            raise ValueError("Unresolved active manuscript finding: " + name)
        for label, pattern in legacy.MANUSCRIPT_ENGINEERING_PATTERNS.items():
            if pattern.search(text):
                raise ValueError("Implementation narrative in the manuscript: " + label)
        inspected = INPUT.sub("", text) if name == "main.tex" else text
        if FORBIDDEN_CONTROL.search(inspected):
            raise ValueError("Indirect or unsafe manuscript control: " + name)
    actual_constants = definitions(texts["results.tex"])
    if set(actual_constants) != set(CONSTANTS):
        raise ValueError("Historical or missing active manuscript constants")
    for name, value in CONSTANTS.items():
        expected_body = f"{value:,}".replace(",", "{,}")
        if actual_constants[name] != [
            {"kind": r"\newcommand", "arity": 0, "body": expected_body}
        ]:
            raise ValueError("Active constant value differs: " + name)
    for name in CONSTANTS:
        for path, text in texts.items():
            if path != "results.tex" and name in definitions(text):
                raise ValueError("Active constant redefined outside results.tex")
    abstract = abstract_readout(main, [texts[name] for name in RESULT_FILES])
    conservative = legacy._abstract_word_budget(root)
    if not conservative["complete"]:
        raise ValueError("The unchanged conservative abstract budget also fails")
    for name, path in paths.items():
        verify_file(path, before[name])
    return {
        "scope": "dense_primary_v3_active_manuscript_document_component",
        "active_source_bytes_verified": True,
        "inputs": before,
        "expected_generated_sha256": {name: digest(content) for name, content in expected.items()},
        "actual_abstract": abstract,
        "conservative_abstract_budget": conservative,
        "upstream_scientific_admission_performed": False,
        "compiled_pdf_verified": False,
        "final_release_verified": False,
    }
