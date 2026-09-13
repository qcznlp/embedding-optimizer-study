"""Document-component controls only; the synthetic text admits no experiment."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_v3_manuscript as document
from embed_optim.primary_v3_contract import PrimaryV3Contract

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(scope="module")
def primary():
    return PrimaryV3Contract.load(ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE)


def definition(name, text="Synthetic document-component text.", *, command="newcommand"):
    return "\\" + command + "{\\" + name + "}{" + text + "}\n"


def result_includes():
    results = {}
    for filename, names in document.RESULT_MACROS.items():
        source = "".join(definition(name) for name in names)
        if filename == document.RESULT_FILES[0]:
            for name in ("CorrectedGeometryBridgeFinding", "CorrectedGeometryBridgeTable"):
                source += definition(name, "Synthetic wrapped geometry.", command="renewcommand")
        results[filename] = source.encode()
    return results


@pytest.fixture
def manuscript(primary, tmp_path):
    paper = tmp_path / "paper"
    (paper / "generated").mkdir(parents=True)
    (paper / "main.tex").write_bytes((ROOT / "paper/main.tex").read_bytes())
    constants = document.render_constants(primary)
    includes = result_includes()
    for name, content in {"results.tex": constants, **includes}.items():
        (paper / name).write_bytes(content)
    return tmp_path, includes, constants


def edit_main(root, change):
    path = root / "paper/main.tex"
    path.write_text(change(path.read_text()))


def test_actual_declared_grid_generates_only_active_constants(primary):
    text = document.render_constants(primary).decode()
    macros = document.definitions(text)
    assert set(macros) == set(document.CONSTANTS)
    assert macros["NumTrainingQueries"][0]["body"] == "500{,}000"
    assert "Headline" not in text and "Discovery" not in text and "ResultPending" not in text


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "historical",
        "data",
        "negatives",
        "stages",
        "tasks",
        "context",
        "rate",
        "optimizer",
    ],
)
def test_constants_require_complete_declared_v3_population(primary, change):
    payload, inputs = copy.deepcopy(primary.payload), copy.deepcopy(primary.inputs)
    identities = {row["run_id"]: primary.expected_identity(row["run_id"]) for row in inputs["runs"]}
    first = inputs["runs"][0]["run_id"]
    if change == "missing":
        inputs["runs"].pop()
    elif change == "duplicate":
        inputs["runs"][-1] = inputs["runs"][0]
    elif change == "historical":
        inputs["runs"][0]["run_id"] = "padded-adamw-1e-6"
    elif change == "data":
        identities[first]["data"]["rows"] = 499999
    elif change == "negatives":
        identities[first]["data"]["selected_columns"].remove("negative_6")
    elif change == "stages":
        payload["checkpoint_steps"][-1] -= 1
    elif change == "tasks":
        payload["beir_task_revisions"].pop(next(iter(payload["beir_task_revisions"])))
    elif change == "context":
        identities[first]["recipe"]["max_length"] = 512
    elif change == "rate":
        identities[first]["recipe"]["optimizer"]["lr"] = 0.7
    else:
        identities[first]["recipe"]["optimizer"]["name"] = "hybrid_adamw"
    controlled = SimpleNamespace(
        payload=payload, inputs=inputs, expected_identity=identities.__getitem__
    )
    with pytest.raises(ValueError):
        document.render_constants(controlled)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("{a{b}c}", "a{b}c"),
        (r" {a\{b\}c}", r"a\{b\}c"),
        (r"{a\textbf{b}c} extra", r"a\textbf{b}c"),
    ],
)
def test_balanced_group_preserves_nested_and_escaped_braces(text, expected):
    assert document.brace_group(text, 0)[0] == expected


@pytest.mark.parametrize("text", ["", "word", "{x", "{x\\", "{{x}"])
def test_malformed_groups_are_refused(text):
    with pytest.raises(ValueError):
        document.brace_group(text, 0)


def test_definitions_skip_comments_and_nested_definitions():
    text = "% " + definition("Ignored") + definition("Outer", definition("Nested"))
    text += definition("Visible", "95\\% and {nested}")
    assert set(document.definitions(text)) == {"Outer", "Visible"}


@pytest.mark.parametrize(
    "text", [r"\newcommand{bad}{x}", r"\newcommand{\A}[many]{x}", r"\newcommand{\A}{x"]
)
def test_malformed_generated_definitions_refuse(text):
    with pytest.raises(ValueError):
        document.definitions(text)


def test_plain_expansion_counts_visible_formatter_arguments():
    assert (
        document.expand_plain(r"\A{} \textbf{two words} 95\%", {"A": "one"}) == "one two words 95%"
    )


@pytest.mark.parametrize(
    "text, macros",
    [
        (r"\A", {"A": r"\B", "B": r"\A"}),
        (r"\input{secret}", {}),
        (r"\unknown{many words}", {}),
        (r"\textbf", {}),
        ("$math$", {}),
        ("bad}", {}),
        ("bad#1", {}),
        ("bad\\", {}),
    ],
)
def test_unknown_or_recursive_abstract_is_not_counted_as_empty(text, macros):
    with pytest.raises(ValueError):
        document.expand_plain(text, macros)


def test_document_positive_is_explicitly_not_primary_admission(manuscript):
    root, includes, constants = manuscript
    result = document.inspect_sources(root, includes, constants)
    assert result["active_source_bytes_verified"] is True
    assert result["actual_abstract"]["words_conservative"] < 200
    assert result["upstream_scientific_admission_performed"] is False
    assert result["compiled_pdf_verified"] is False
    assert result["final_release_verified"] is False


def test_actual_abstract_counterexample_passes_old_reserve_but_fails_expansion(manuscript):
    root, includes, constants = manuscript
    original = definition("CorrectedAbstractFinding").encode()
    changed = definition("CorrectedAbstractFinding", "oversized " * 180).encode()
    includes[document.RESULT_FILES[0]] = includes[document.RESULT_FILES[0]].replace(
        original, changed
    )
    (root / "paper" / document.RESULT_FILES[0]).write_bytes(includes[document.RESULT_FILES[0]])
    assert document.legacy._abstract_word_budget(root)["complete"] is True
    with pytest.raises(ValueError, match="Actual expanded abstract exceeds"):
        document.inspect_sources(root, includes, constants)


@pytest.mark.parametrize("name", ["results.tex", *document.RESULT_FILES])
def test_every_installed_artifact_must_equal_regenerated_bytes(manuscript, name):
    root, includes, constants = manuscript
    path = root / "paper" / name
    path.write_bytes(path.read_bytes() + b"% a rehashed but ungenerated edit\n")
    with pytest.raises(ValueError, match="differs from regenerated"):
        document.inspect_sources(root, includes, constants)


@pytest.mark.parametrize("name", ["main.tex", "results.tex", *document.RESULT_FILES])
def test_every_document_input_rejects_symlinks(manuscript, name):
    root, includes, constants = manuscript
    path = root / "paper" / name
    saved = root / (path.name + ".retained")
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(ValueError, match="ordinary"):
        document.inspect_sources(root, includes, constants)


@pytest.mark.parametrize("name", ["main.tex", "results.tex", *document.RESULT_FILES])
def test_every_document_input_must_remain_present(manuscript, name):
    root, includes, constants = manuscript
    path = root / "paper" / name
    path.rename(root / (path.name + ".retained"))
    with pytest.raises(ValueError, match="ordinary"):
        document.inspect_sources(root, includes, constants)


@pytest.mark.parametrize(
    "extra",
    [
        r"\input{generated/discovery}",
        r"\input other.tex",
        r"\include{secret}",
        r"\includegraphics{old-results.pdf}",
        r"\openout1=unsafe.tex",
        r"\write18{anything}",
        r"\csname input\endcsname{secret}",
        r"\renewcommand{\CorrectedAbstractFinding}{fabricated}",
        r"\let\CorrectedAbstractFinding\DiscoveryHeadline",
        r"\ResultPending{missing}",
        "TODO",
        "padding",
        "packing",
        "candidate-breadth",
        "kernel bug",
    ],
)
def test_foreign_pending_indirect_and_engineering_content_is_refused(manuscript, extra):
    root, includes, constants = manuscript
    edit_main(
        root,
        lambda text: text.replace(
            r"\section{Introduction}", extra + "\n" + r"\section{Introduction}"
        ),
    )
    with pytest.raises(ValueError):
        document.inspect_sources(root, includes, constants)


def test_comments_do_not_become_rendered_incidents(manuscript):
    root, includes, constants = manuscript
    edit_main(root, lambda text: "% padding and TODO are provenance comments\n" + text)
    assert (
        document.inspect_sources(root, includes, constants)["active_source_bytes_verified"] is True
    )


@pytest.mark.parametrize("name", ["NumTrainingQueries", "NumBEIRTasks", "ContextLength"])
def test_constants_cannot_be_redefined_in_an_include(manuscript, name):
    root, includes, constants = manuscript
    filename = document.RESULT_FILES[1]
    includes[filename] += definition(name, "wrong", command="renewcommand").encode()
    (root / "paper" / filename).write_bytes(includes[filename])
    with pytest.raises(ValueError, match="constant redefined"):
        document.inspect_sources(root, includes, constants)


@pytest.mark.parametrize(
    "filename, name", [(f, n) for f, names in document.RESULT_MACROS.items() for n in names]
)
def test_each_active_result_macro_is_required(manuscript, filename, name):
    root, includes, constants = manuscript
    includes[filename] = includes[filename].replace(definition(name).encode(), b"")
    (root / "paper" / filename).write_bytes(includes[filename])
    with pytest.raises(ValueError, match="active result macro"):
        document.inspect_sources(root, includes, constants)


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "reordered", "duplicate", "parameterized", "redefined", "unknown_command"],
)
def test_abstract_structure_and_expansion_rejects_ambiguous_findings(manuscript, change):
    root, includes, _ = manuscript
    main = (root / "paper/main.tex").read_text()
    if change == "missing":
        main = main.replace(r"\CorrectedAbstractFinding{}", "")
    elif change == "extra":
        main += r"\begin{abstract}Extra\end{abstract}"
    elif change == "reordered":
        # A macro invoked in the abstract must not be defined only inside another definition.
        key = document.RESULT_FILES[0]
        includes[key] = includes[key].replace(
            definition("CorrectedAbstractFinding").encode(),
            definition("Outer", definition("CorrectedAbstractFinding")).encode(),
        )
    elif change == "duplicate":
        main = main.replace(
            r"\CorrectedAbstractFinding{}",
            r"\CorrectedAbstractFinding{}\CorrectedAbstractFinding{}",
        )
    elif change == "parameterized":
        key = document.RESULT_FILES[0]
        includes[key] = includes[key].replace(
            b"{\\CorrectedAbstractFinding}{", b"{\\CorrectedAbstractFinding}[1]{"
        )
    elif change == "redefined":
        includes[document.RESULT_FILES[0]] += definition(
            "CorrectedAbstractFinding", command="renewcommand"
        ).encode()
    else:
        main = main.replace(
            r"\CorrectedAbstractFinding{}", r"\unknown{not zero words}\CorrectedAbstractFinding{}"
        )
    with pytest.raises(ValueError):
        document.abstract_readout(main, [content.decode() for content in includes.values()])


def test_file_changed_during_semantic_inspection_is_not_readmitted(manuscript, monkeypatch):
    root, includes, constants = manuscript
    original = document.abstract_readout

    def change_after_read(*args):
        result = original(*args)
        (root / "paper/results.tex").write_bytes(constants + b"% changed after read\n")
        return result

    monkeypatch.setattr(document, "abstract_readout", change_after_read)
    with pytest.raises(ValueError):
        document.inspect_sources(root, includes, constants)


def test_actual_pending_paper_is_not_accepted_as_complete_document(primary):
    includes = {name: (ROOT / "paper" / name).read_bytes() for name in document.RESULT_FILES}
    # Existing historical constants fail before any caller can reinterpret these placeholders.
    with pytest.raises(ValueError, match="differs from regenerated"):
        document.inspect_sources(ROOT, includes, document.render_constants(primary))
