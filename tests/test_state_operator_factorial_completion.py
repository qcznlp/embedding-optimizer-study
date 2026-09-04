from __future__ import annotations

import csv
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from embed_optim.paper_audit import _state_operator_publication_status
from embed_optim.state_operator_factorial_completion import (
    _main_complete,
    load_completion_protocol,
    pipeline_steps,
)
from embed_optim.state_operator_factorial_publication import (
    _interpretation,
    _load_protocol,
    _render_latex,
    render,
)

ROOT = Path(__file__).resolve().parents[1]
COMPLETION_PROTOCOL = (
    ROOT / "configs/dense_no_packing_state_operator_factorial_completion_protocol.json"
)
PUBLICATION_PROTOCOL = (
    ROOT / "configs/dense_no_packing_state_operator_factorial_publication_protocol.json"
)


def _args() -> Namespace:
    return Namespace(
        python="/usr/bin/python3",
        beir_gpu_pools="0,1,2,3;4,5,6,7",
        checkpoint_repo="qcz/embedding-optimizer-study-checkpoints",
        checkpoint_prefix="state-operator-factorial-v1/dense",
    )


def _row(decision: str, point: float) -> dict[str, object]:
    lower, upper = {
        "supported_positive": (point - 0.01, point + 0.01),
        "supported_negative": (point - 0.01, point + 0.01),
        "inconclusive": (-0.01, 0.01),
    }[decision]
    return {
        "point_estimate": point,
        "bootstrap_ci_95_lower": lower,
        "bootstrap_ci_95_upper": upper,
        "decision": decision,
    }


def _estimands(
    weight: str = "inconclusive",
    operator: str = "inconclusive",
    interaction: str = "inconclusive",
) -> dict[str, dict[str, object]]:
    points = {
        "supported_positive": 0.02,
        "supported_negative": -0.02,
        "inconclusive": 0.0,
    }
    return {
        "weight_state_effect": _row(weight, points[weight]),
        "operator_effect": _row(operator, points[operator]),
        "state_operator_interaction": _row(interaction, points[interaction]),
    }


def test_checked_in_completion_contract_is_source_bound_and_exact() -> None:
    protocol = load_completion_protocol(COMPLETION_PROTOCOL, ROOT)
    steps = pipeline_steps(_args(), ROOT, protocol)

    assert protocol["visibility_at_freeze"]["factorial_outputs_visible"] is False
    assert len(steps) == 36
    assert [step.name for step in steps[:8]] == [
        "protocol-audit",
        "source-checkpoint-durability",
        "calibration-01",
        "calibration-02",
        "calibration-03",
        "calibration-04",
        "matrix-generate",
        "matrix-audit",
    ]
    training = [step for step in steps if step.name.startswith("training-")]
    backups = [step for step in steps if step.name.startswith("checkpoint-backup-")]
    beir = [step for step in steps if step.name.startswith("full-beir-")]
    assert len(training) == len(backups) == len(beir) == 6
    assert [step.parallel_group for step in beir] == [
        "full-beir-pair-1",
        "full-beir-pair-1",
        "full-beir-pair-2",
        "full-beir-pair-2",
        "full-beir-pair-3",
        "full-beir-pair-3",
    ]
    assert all("--remote-prefix" in step.command for step in backups)
    assert steps[-9].name == "summary"
    assert steps[-8].name == "publication-render"
    assert steps[-7].name == "paper-release"


def test_main_completion_gate_requires_the_entire_exact_parent(tmp_path: Path) -> None:
    protocol = load_completion_protocol(COMPLETION_PROTOCOL, ROOT)
    gate = protocol["main_completion_gate"]
    ledger = {
        "scope": gate["scope"],
        "status": "complete",
        "complete": True,
        "training_runs_complete": 12,
        "training_runs_expected": 12,
        "contract": {"sha256": gate["contract_sha256"]},
        "steps": [{"name": name, "complete": True} for name in gate["required_steps"]],
        "backups": {run_id: {"complete": True} for run_id in gate["required_run_ids"]},
    }
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    assert _main_complete(path, protocol) is True

    mutations = (
        ("status", "finalizing"),
        ("complete", False),
        ("training_runs_complete", 11),
    )
    for field, value in mutations:
        changed = dict(ledger)
        changed[field] = value
        path.write_text(json.dumps(changed), encoding="utf-8")
        assert _main_complete(path, protocol) is False
    changed = json.loads(json.dumps(ledger))
    changed["steps"][-1]["complete"] = False
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert _main_complete(path, protocol) is False
    changed = json.loads(json.dumps(ledger))
    changed["backups"].pop(next(iter(changed["backups"])))
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert _main_complete(path, protocol) is False


def test_checked_in_publication_contract_is_result_blind_and_bound() -> None:
    protocol = _load_protocol(PUBLICATION_PROTOCOL, ROOT)
    visibility = protocol["visibility_at_freeze"]
    assert visibility["corrected_factorial_summary_visible"] is False
    assert visibility["corrected_factorial_publication_visible"] is False
    assert set(protocol["expected_outputs"]) == {
        "paper_latex",
        "publication_manifest",
    }
    assert "state-operator-factorial" in (ROOT / "paper/main.tex").read_text(encoding="utf-8")


def test_paper_audit_allows_only_pending_before_summary(tmp_path: Path) -> None:
    fixture = tmp_path / "paper/generated/state-operator-factorial.tex"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("\\ResultPending{prospective}\n", encoding="utf-8")
    assert _state_operator_publication_status(tmp_path) == {
        "complete": True,
        "status": "prospective_pending",
        "summary_present": False,
    }

    summary = tmp_path / "reports/state-operator-factorial/summary_manifest.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}\n", encoding="utf-8")
    status = _state_operator_publication_status(tmp_path)
    assert status["complete"] is False
    assert status["status"] == "invalid_or_stale"


@pytest.mark.parametrize(
    ("weight", "operator", "interaction", "expected"),
    (
        ("supported_positive", "inconclusive", "inconclusive", "inherited"),
        ("inconclusive", "supported_positive", "inconclusive", "continuation transform"),
        ("supported_positive", "supported_positive", "inconclusive", "additive"),
        ("inconclusive", "inconclusive", "supported_positive", "closed-loop"),
        ("inconclusive", "supported_negative", "inconclusive", "opposite direction"),
        ("inconclusive", "inconclusive", "inconclusive", "does not support"),
    ),
)
def test_publication_decision_map_is_fixed_before_results(
    weight: str,
    operator: str,
    interaction: str,
    expected: str,
) -> None:
    interpretation = _interpretation(_estimands(weight, operator, interaction))
    assert expected in interpretation


def test_latex_renderer_propagates_all_estimands_to_the_story() -> None:
    rows = _estimands("supported_positive", "inconclusive", "supported_positive")
    latex = _render_latex(rows)

    assert "\\ResultPending" not in latex
    assert latex.count("+0.0200") >= 2
    assert latex.count("inconclusive") >= 1
    assert "closed-loop state--operator feedback" in latex
    assert "\\newcommand{\\StateOperatorAbstractFinding}" in latex
    assert "\\newcommand{\\StateOperatorMechanismFinding}" in latex
    assert "\\newcommand{\\StateOperatorConclusionFinding}" in latex
    assert "\\newcommand{\\StateOperatorAppendixTable}" in latex
    assert latex.count(" \\\\") == 4


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_publication_renderer_requires_complete_hashed_summary(tmp_path: Path) -> None:
    scientific = tmp_path / "configs/scientific.json"
    implementation = tmp_path / "configs/implementation.json"
    bound_source = tmp_path / "src/renderer.py"
    for path in (scientific, implementation, bound_source):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    protocol_path = tmp_path / "configs/publication.json"
    protocol = {
        "status": "prospective_state_operator_publication_lock",
        "parent_bindings": {
            "scientific_protocol": _file_record(scientific, tmp_path),
            "implementation_protocol": _file_record(implementation, tmp_path),
        },
        "source_bindings": {"renderer": _file_record(bound_source, tmp_path)},
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    report = tmp_path / "reports/state-operator-factorial"
    specifications = {
        "beir_seed_task_scores": (["row"], [{"row": index} for index in range(168)]),
        "factorial_cell_summary": (
            ["state", "operator"],
            [
                {"state": state, "operator": operator}
                for state in ("adamw_state", "muon_state")
                for operator in ("adamw", "muon")
            ],
        ),
        "estimand_seed_task_contrasts": (
            ["row"],
            [{"row": index} for index in range(126)],
        ),
        "estimand_summary": (
            [
                "estimand",
                "point_estimate",
                "bootstrap_ci_95_lower",
                "bootstrap_ci_95_upper",
                "decision",
                "bootstrap_samples",
                "bootstrap_seed",
                "seed_clusters",
                "task_clusters",
            ],
            [
                {
                    "estimand": name,
                    "point_estimate": 0.02,
                    "bootstrap_ci_95_lower": 0.01,
                    "bootstrap_ci_95_upper": 0.03,
                    "decision": "supported_positive",
                    "bootstrap_samples": 100000,
                    "bootstrap_seed": 20260904,
                    "seed_clusters": 3,
                    "task_clusters": 14,
                }
                for name in (
                    "weight_state_effect",
                    "operator_effect",
                    "state_operator_interaction",
                )
            ],
        ),
        "probe_checkpoint_metrics": (["row"], [{"row": index} for index in range(60)]),
        "probe_task_metrics": (["row"], [{"row": index} for index in range(840)]),
    }
    outputs = {}
    for name, (fields, rows) in specifications.items():
        path = report / f"{name}.csv"
        _csv(path, fields, rows)
        outputs[name] = _file_record(path, tmp_path)
    summary_path = report / "summary_manifest.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "scientific_protocol": _file_record(scientific, tmp_path),
                "implementation_protocol": _file_record(implementation, tmp_path),
                "coverage": {
                    "training_runs": 12,
                    "beir_seed_task_scores": 168,
                    "estimand_seed_task_contrasts": 126,
                    "estimands": 3,
                    "probe_checkpoints": 60,
                    "probe_task_rows": 840,
                },
                "inference": {"samples": 100000, "seed": 20260904},
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    paper = tmp_path / "paper/generated/state-operator-factorial.tex"
    publication = report / "publication_manifest.json"

    result = render(
        repo_root=tmp_path,
        protocol_path=protocol_path,
        paper_path=paper,
        manifest_path=publication,
    )

    assert result["status"] == "complete"
    assert result["interpretation"].startswith("Muon-created weights")
    assert "\\ResultPending" not in paper.read_text(encoding="utf-8")
    assert (
        render(
            repo_root=tmp_path,
            protocol_path=protocol_path,
            paper_path=paper,
            manifest_path=publication,
            audit_only=True,
        )
        == result
    )

    with (report / "estimand_summary.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(ValueError, match="summary output changed"):
        render(
            repo_root=tmp_path,
            protocol_path=protocol_path,
            paper_path=paper,
            manifest_path=publication,
            audit_only=True,
        )
