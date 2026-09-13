"""Regressions for the integrated, numerically unchanged continuation-claim wording."""

import ast
import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "reports/paper-review/factorial-claims-v1"


def load_renderer(which):
    path = ROOT / which / "state_operator_factorial_publication.py"
    spec = importlib.util.spec_from_file_location(f"embed_optim._claim_review_{which}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def candidate():
    from embed_optim import state_operator_factorial_publication

    return state_operator_factorial_publication


@pytest.fixture(scope="module")
def cases():
    return {c["name"]: c for c in json.loads((ROOT / "counterexamples.json").read_text())["cases"]}


def test_only_the_three_publication_text_functions_change():
    before = ast.parse((ROOT / "before/state_operator_factorial_publication.py").read_text())
    after = ast.parse(
        (REPOSITORY / "src/embed_optim/state_operator_factorial_publication.py").read_text()
    )
    allowed = {"_interpretation", "_render_latex", "_expected_manifest"}
    for tree in [before, after]:
        tree.body = [
            node
            for node in tree.body
            if not isinstance(node, ast.FunctionDef) or node.name not in allowed
        ]
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)


def test_reproduces_existing_unsupported_additivity(cases):
    case = cases["positive_main_effects_with_supported_negative_interaction"]
    rows = case["estimands"]
    assert rows["state_operator_interaction"]["decision"] == "supported_negative"
    assert rows["state_operator_interaction"]["point_estimate"] == -0.375
    assert "additive" in load_renderer("before")._interpretation(rows)


def test_does_not_turn_negative_interaction_into_additivity(candidate, cases):
    rows = cases["positive_main_effects_with_supported_negative_interaction"]["estimands"]
    text = candidate._interpretation(rows)
    assert "do not imply an additive response" in text
    assert "interaction must be interpreted separately" in text
    assert "supports additive" not in text


def test_positive_interaction_can_be_less_harmful_not_beneficial(candidate, cases):
    case = cases["positive_interaction_but_muon_harms_both_states"]
    assert all(a < 0 and b < 0 for a, b in case["simple_operator_effects_by_seed"])
    assert case["estimands"]["state_operator_interaction"]["point_estimate"] == 0.125
    text = candidate._interpretation(case["estimands"])
    assert "need not mean Muon helps from either state" in text
    assert "does not establish full-trajectory co-adaptation" in text
    assert "supporting the predeclared closed-loop" not in text


def test_inconclusive_operator_does_not_prove_state_dominance(candidate, cases):
    rows = cases["state_supported_operator_larger_but_uncertain"]["estimands"]
    assert rows["weight_state_effect"]["decision"] == "supported_positive"
    assert rows["operator_effect"]["decision"] == "inconclusive"
    assert rows["operator_effect"]["point_estimate"] > rows["weight_state_effect"]["point_estimate"]
    text = candidate._interpretation(rows)
    assert "does not establish its dominance" in text
    assert "not equivalence to zero" in text
    assert "primarily inherited" not in text


def test_manuscript_candidate_defines_endpoint_contrasts_and_respects_abstract_budget(tmp_path):
    from embed_optim.paper_audit import _abstract_word_budget

    text = (REPOSITORY / "paper/current/main.tex").read_text()
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(text)
    assert _abstract_word_budget(tmp_path)["complete"]
    assert "Y_{MM}-Y_{AA}=\\Delta_S+\\Delta_O" in text
    assert "marginal, not simultaneous intervals" in text
    assert "does not match the first realized shuffled-batch update" in text
    assert "pair is fixed, not selected from the new BEIR outcomes" in text
    assert "positive interaction can mean" in text
    assert "Neither result isolates a measured geometric mediator or decomposes the" in text
    assert "full training gain" in text
    assert "next optimizer carry the effect" not in text
    assert not re.search(r"packing|padding|execution-path incidents", text, flags=re.IGNORECASE)


@pytest.mark.parametrize(
    "decisions",
    itertools.product(["supported_positive", "supported_negative", "inconclusive"], repeat=3),
)
def test_every_decision_combination_retains_estimates_and_bounds(candidate, decisions):
    from embed_optim.paper_audit import ABSTRACT_RESULT_WORD_RESERVE, ABSTRACT_WORD_PATTERN

    rows = {}
    for key, decision in zip(candidate.ESTIMANDS, decisions, strict=True):
        point, lower, upper = {
            "supported_positive": (0.02, 0.01, 0.03),
            "supported_negative": (-0.02, -0.03, -0.01),
            "inconclusive": (0.0, -0.01, 0.01),
        }[decision]
        rows[key] = {
            "estimand": key,
            "point_estimate": point,
            "bootstrap_ci_95_lower": lower,
            "bootstrap_ci_95_upper": upper,
            "decision": decision,
        }
    original = json.dumps(rows, sort_keys=True)
    text = candidate._render_latex(rows)
    assert json.dumps(rows, sort_keys=True) == original
    for key in candidate.ESTIMANDS:
        assert candidate._effect(rows[key]) in text
        assert candidate._decision_label(rows[key]["decision"]) in text
    for name in ["AbstractFinding", "MechanismFinding", "ConclusionFinding", "AppendixTable"]:
        assert f"\\newcommand{{\\StateOperator{name}}}" in text
    assert "separates the gain" not in text
    assert "first-update scale matching" not in text
    assert "calibration-probe scale matching" in text
    assert "marginal 95\\% intervals" in text
    assert "not simultaneous family-wise intervals" in text
    assert "They do not decompose the primary-training contrast" in text
    assert text.count(" \\\\") == 4
    abstract = text.split("\\newcommand{\\StateOperatorAbstractFinding}{%\n", 1)[1].split(
        "}\n\\newcommand", 1
    )[0]
    assert (
        len(ABSTRACT_WORD_PATTERN.findall(abstract))
        <= ABSTRACT_RESULT_WORD_RESERVE["state_operator_finding"]
    )


def test_complete_synthetic_candidate_paper_compiles_within_layout_gate(
    candidate, cases, tmp_path, monkeypatch
):
    from test_corrected_publication import _evidence
    from test_dimension_publication import synthetic_publication

    from embed_optim.corrected_publication import render_latex
    from embed_optim.paper_layout import audit_paper_layout

    assert shutil.which("latexmk") and shutil.which("make"), "Require the actual paper toolchain"
    args, _, _, _ = synthetic_publication.__wrapped__(tmp_path / "dimension-fixture", monkeypatch)
    paper = tmp_path / "synthetic-claim-review-only" / "paper"
    shutil.copytree(ROOT.parents[2] / "paper", paper, ignore=shutil.ignore_patterns("build"))
    text = (
        (REPOSITORY / "paper/main.tex")
        .read_text()
        .replace(
            r"\maketitle",
            r"\maketitle\begin{center}\textbf{SYNTHETIC CLAIM/LAYOUT TEST -- NOT EXPERIMENT RESULTS}\end{center}",
        )
    )
    (paper / "main.tex").write_text(text)
    (paper / "generated/dimension-utilization.tex").write_text(args.paper_output.read_text())
    (paper / "generated/optimizer-primary.tex").write_text(render_latex(_evidence()))
    rows = cases["positive_main_effects_with_supported_negative_interaction"]["estimands"]
    (paper / "generated/state-operator-factorial.tex").write_text(candidate._render_latex(rows))
    result = subprocess.run(
        # Deliberate synthetic placeholders belong to the preserved historical
        # template; the default now builds only the authenticated real paper.
        ["make", "legacy-all", f"PYTHON={sys.executable}"],
        cwd=paper,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "TEXINPUTS": f"{paper / 'vendor'}//:",
            "BSTINPUTS": f"{paper / 'vendor'}//:",
        },
    )
    assert result.returncode == 0, result.stdout[-7000:] + result.stderr[-2000:]
    layout = audit_paper_layout(paper)
    assert layout["complete"]
    assert "Overfull \\hbox" not in (paper / "build/main.log").read_text()
    (paper / "candidate-layout-audit.json").write_text(
        json.dumps({**layout, "scientific_completion": False, "synthetic_inputs": True}, indent=2)
        + "\n"
    )


def test_pending_generator_contains_no_scores_and_is_reproducible(tmp_path):
    from scripts.render_state_operator_pending import render_pending, write_pending

    output = write_pending(tmp_path)
    assert output.read_text() == render_pending()
    assert output.read_text().count(r"\ResultPending{") == 3
    assert "0.0000" not in output.read_text()
    assert "first-update scale matching" not in output.read_text()
    assert write_pending(tmp_path).read_text() == render_pending()


@pytest.mark.parametrize("name", ["summary_manifest.json", "publication_manifest.json"])
def test_pending_generator_refuses_any_existing_scientific_output(tmp_path, name):
    from scripts.render_state_operator_pending import write_pending

    output = write_pending(tmp_path)
    before = output.read_bytes()
    blocker = tmp_path / "reports/state-operator-factorial" / name
    blocker.parent.mkdir(parents=True)
    blocker.write_text("{}")
    with pytest.raises(ValueError, match="after summary/publication exists"):
        write_pending(tmp_path)
    assert output.read_bytes() == before


def test_pending_generator_never_overwrites_a_real_include(tmp_path):
    from scripts.render_state_operator_pending import OUTPUT, write_pending

    output = tmp_path / OUTPUT
    output.parent.mkdir(parents=True)
    output.write_text("Existing source-bound result.\n")
    with pytest.raises(ValueError, match="non-development"):
        write_pending(tmp_path)
    assert output.read_text() == "Existing source-bound result.\n"


def test_pending_generator_rejects_a_symlink_without_modifying_its_target(tmp_path):
    from scripts.render_state_operator_pending import OUTPUT, write_pending

    output = tmp_path / "repository" / OUTPUT
    output.parent.mkdir(parents=True)
    other = tmp_path / "other.tex"
    other.write_text("Untouched.\n")
    output.symlink_to(other)
    with pytest.raises(ValueError, match="inside the repository"):
        write_pending(tmp_path / "repository")
    assert other.read_text() == "Untouched.\n"


def test_conceptual_figure_labels_do_not_claim_gain_components():
    text = (REPOSITORY / "scripts/plot_weight_space_dimension_map.py").read_text()
    assert "Endpoint contrasts" in text
    assert "source state (averaged)" in text
    assert "probe-calibrated scale" in text
    assert "What carries retrieval value?" not in text
    assert "Mechanism intervention:" not in text


def test_actual_conceptual_figure_has_no_box_text_overflow(tmp_path, monkeypatch):
    from scripts import plot_weight_space_dimension_map as plot

    boxes = []
    original_box = plot._box
    original_close = plot.plt.close

    def capture_box(axis, *args, **kwargs):
        original_box(axis, *args, **kwargs)
        boxes.append((axis.texts[-1], axis.patches[-1]))

    monkeypatch.setattr(plot, "_box", capture_box)
    monkeypatch.setattr(plot.plt, "close", lambda figure: None)
    pdf = tmp_path / "concept.pdf"
    try:
        plot.render(pdf, tmp_path / "concept.png")
        figure = boxes[0][0].figure
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for text, patch in boxes:
            if not text.get_text():
                continue
            label = text.get_window_extent(renderer)
            border = patch.get_window_extent(renderer)
            assert label.x0 >= border.x0, text.get_text()
            assert label.x1 <= border.x1, text.get_text()
            assert label.y0 >= border.y0, text.get_text()
            assert label.y1 <= border.y1, text.get_text()
        # Default PDF creation timestamps are deliberately absent so independent
        # rerendering does not create source drift solely from wall-clock time.
        assert b"/CreationDate" not in pdf.read_bytes()
        assert b"/ModDate" not in pdf.read_bytes()
    finally:
        for text, _ in boxes:
            original_close(text.figure)
