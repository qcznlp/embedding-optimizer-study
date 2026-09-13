"""Offline rehearsal of the exact proposed live syntax-only repair, never deployment."""

import ast
import difflib
import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

from embed_optim import completion_contract_migration as migration

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports/engineering-archive/publication-header"
RECEIPTS = ROOT / "reports/experiment-integrity"
RENDERER = "src/embed_optim/corrected_publication.py"
PROTOCOL = "configs/dense_no_packing_publication_protocol.json"


@pytest.fixture
def projected_hotfix(tmp_path):
    original_source = (ARCHIVE / "renderer-source.py").read_text()
    assert hashlib.sha256(original_source.encode()).hexdigest() == (
        "a2f9aaed850373a0bc54466769f0513944eb7c8e54c76c2a6d0c1aaddcc364e6"
    )
    assert original_source.count(r"\\\\n\\midrule") == 3
    source = original_source.replace(r"\\\\n\\midrule", r"\\\\\n\\midrule")
    ast.parse(source)
    assert hashlib.sha256(source.encode()).hexdigest() == (
        "ec299f29bb76b73228eb5d026cc014a78e35bcacaed688906ad0b02668dd1af6"
    )
    original_protocol = (ARCHIVE / "publication-protocol.json").read_text()
    protocol = json.loads(original_protocol)
    protocol["source_bindings"]["corrected_publication"].update(
        bytes=len(source.encode()), sha256=hashlib.sha256(source.encode()).hexdigest()
    )
    protocol["latex_header_syntax_amendment"] = {
        "reason": "Replace literal backslash-n after all three table-header row terminators with an actual newline. Publication syntax only.",
        "affected_headers": 3,
        "scientific_contract_changed": False,
        "primary_publication_outputs_visible": False,
    }
    protocol_text = json.dumps(protocol, indent=2) + "\n"
    expected_patch = ""
    for path, before, after in (
        (RENDERER, original_source, source),
        (PROTOCOL, original_protocol, protocol_text),
    ):
        expected_patch += "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
    assert (RECEIPTS / "live-publication-header-hotfix.patch").read_text() == expected_patch
    old = json.loads((ARCHIVE / "main-ledger.json").read_text())["contract"]
    projected = deepcopy(old)
    for record in projected["sources"]:
        if record["path"] == PROTOCOL:
            record.update(
                bytes=len(protocol_text.encode()),
                sha256=hashlib.sha256(protocol_text.encode()).hexdigest(),
            )
    projected.pop("sha256")
    projected["sha256"] = hashlib.sha256(
        json.dumps(projected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    proposal = json.loads((RECEIPTS / "live-publication-header-migration-draft.json").read_text())
    assert projected["sha256"] == proposal["to_contract_sha256"]
    assert old["sha256"] == proposal["from_contract_sha256"]
    (tmp_path / PROTOCOL).parent.mkdir(parents=True)
    (tmp_path / PROTOCOL).write_text(protocol_text)
    return original_source, source, old, projected, proposal


def test_exact_live_patch_changes_only_three_newline_escapes_and_one_protocol(
    projected_hotfix, tmp_path
):
    original, source, old, projected, proposal = projected_hotfix
    assert len(source) == len(original) + 3
    assert old["steps"] == projected["steps"]
    assert old["arguments"] == projected["arguments"]
    changed = [
        a["path"] for a, b in zip(old["sources"], projected["sources"], strict=True) if a != b
    ]
    assert changed == [PROTOCOL]
    migration.validate_transition(old, projected, proposal, tmp_path)


def test_rehearsal_preserves_exact_original_ledger_and_all_existing_backups(
    projected_hotfix, tmp_path, monkeypatch
):
    _, _, _, projected, proposal = projected_hotfix
    original = (ARCHIVE / "main-ledger.json").read_bytes()
    ledger = json.loads(original)
    path = tmp_path / "logs/pipeline-ledger.json"
    path.parent.mkdir()
    path.write_bytes(original)
    protocol_path = tmp_path / "migration.json"
    protocol_path.write_text(json.dumps(proposal))
    # The counterfactual contract above uses the original host paths; no live files
    # or controllers are changed to manufacture that counterfactual.
    monkeypatch.setattr(migration, "current_contract", lambda *_: projected)
    result = migration.migrate_ledger(path, protocol_path, tmp_path)
    assert result["status"] == "migrated"
    assert (path.parent / proposal["archive_basename"]).read_bytes() == original
    after = json.loads(path.read_text())
    for key in ("backups", "complete_runs", "steps", "complete", "started_at_utc"):
        assert after[key] == ledger[key]
    assert after["contract_migrations"][:-1] == ledger["contract_migrations"]
    assert migration.migrate_ledger(path, protocol_path, tmp_path)["status"] == "already_migrated"


@pytest.mark.parametrize("mutation", ["arguments", "steps", "source", "from_hash", "scientific"])
def test_exact_rehearsal_rejects_undeclared_changes(projected_hotfix, tmp_path, mutation):
    _, _, old, projected, proposal = projected_hotfix
    if mutation == "arguments":
        projected["arguments"]["gpus"] = "0"
    elif mutation == "steps":
        projected["steps"].reverse()
    elif mutation == "source":
        projected["sources"][0]["sha256"] = "0" * 64
    elif mutation == "from_hash":
        old["sha256"] = "0" * 64
    else:
        proposal["scientific_contract_changed"] = True
    with pytest.raises(ValueError):
        migration.validate_transition(old, projected, proposal, tmp_path)


def _load_renderer(source):
    module = ModuleType("embed_optim._offline_header_fixture")
    module.__package__ = "embed_optim"
    exec(compile(source, "archived-publication-renderer.py", "exec"), module.__dict__)
    return module


def test_successor_gate_must_track_the_exact_migrated_main_without_weakening_completion(
    projected_hotfix, tmp_path
):
    from embed_optim.state_operator_factorial_completion import _main_complete

    _, _, _, projected, _ = projected_hotfix
    gate = json.loads((ARCHIVE / "factorial-main-gate.json").read_text())
    proposed_gate = {**gate, "contract_sha256": projected["sha256"]}
    assert {k: v for k, v in proposed_gate.items() if k != "contract_sha256"} == {
        k: v for k, v in gate.items() if k != "contract_sha256"
    }
    ledger = {
        "scope": "corrected_dense_no_packing_completion",
        "status": "complete",
        "complete": True,
        "training_runs_complete": 12,
        "training_runs_expected": 12,
        "contract": {"sha256": projected["sha256"]},
        "steps": [{"name": name, "complete": True} for name in gate["required_steps"]],
        "backups": {run_id: {"complete": True} for run_id in gate["required_run_ids"]},
    }
    path = tmp_path / "synthetic-complete-main.json"
    path.write_text(json.dumps(ledger))
    assert _main_complete(path, {"main_completion_gate": gate}) is False
    assert _main_complete(path, {"main_completion_gate": proposed_gate}) is True
    for mutation in ("old_contract", "incomplete_training", "unfinished_step", "missing_backup"):
        invalid = deepcopy(ledger)
        if mutation == "old_contract":
            invalid["contract"]["sha256"] = gate["contract_sha256"]
        elif mutation == "incomplete_training":
            invalid["training_runs_complete"] = 11
        elif mutation == "unfinished_step":
            invalid["steps"][-1]["complete"] = False
        else:
            invalid["backups"].pop(gate["required_run_ids"][0])
        path.write_text(json.dumps(invalid))
        assert _main_complete(path, {"main_completion_gate": proposed_gate}) is False


def test_all_three_real_renderer_headers_fail_before_and_compile_after(projected_hotfix, tmp_path):
    from test_corrected_publication import _evidence

    compiler = shutil.which("pdflatex")
    if compiler is None:
        pytest.skip("pdflatex is not installed")
    original, source, _, _, _ = projected_hotfix
    old_latex = _load_renderer(original).render_latex(_evidence())
    new_latex = _load_renderer(source).render_latex(_evidence())
    assert old_latex.count(r"\\n\midrule") == 3
    assert new_latex == old_latex.replace(r"\\n\midrule", "\\\\\n\\midrule")
    outcomes = []
    for name, latex in (("before", old_latex), ("after", new_latex)):
        directory = tmp_path / name
        directory.mkdir()
        document = directory / "fixture.tex"
        document.write_text(
            "\\documentclass{article}\n\\usepackage{booktabs}\n"
            + latex
            + "\n\\begin{document}\nSYNTHETIC SYNTAX TEST -- NOT EXPERIMENT RESULTS\n"
            + "\\CorrectedMainSection\n\\CorrectedGeometryBridgeTable\n"
            + "\\CorrectedExecutionSensitivityTable\n\\end{document}\n"
        )
        result = subprocess.run(
            [compiler, "-interaction=nonstopmode", "-halt-on-error", "fixture.tex"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=60,
        )
        outcomes.append(result)
    assert outcomes[0].returncode != 0 and "Misplaced \\noalign" in outcomes[0].stdout
    assert outcomes[1].returncode == 0, outcomes[1].stdout[-3000:]
