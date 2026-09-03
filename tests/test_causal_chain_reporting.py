from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from embed_optim import causal_chain_reporting as reporting
from embed_optim.causal_chain_rendering import (
    causal_chain_display_contract,
    render_causal_chain_latex,
    render_causal_chain_markdown,
)
from embed_optim.causal_chain_reporting import (
    CausalChainReportingError,
    load_causal_chain_evidence,
)
from embed_optim.paper_audit import _causal_snapshot_still_current
from embed_optim.temporal_short_branch import analyze_rows, support_decision

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _record(path: Path, *, root: Path | None = None) -> dict:
    resolved = path.resolve()
    declared = str(resolved) if root is None else resolved.relative_to(root.resolve()).as_posix()
    return {
        "path": declared,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _declared_identity_paths(value) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value) and isinstance(value["path"], str):
            paths.append(value["path"])
        for child in value.values():
            paths.extend(_declared_identity_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_declared_identity_paths(child))
    return paths


def _protocol(root: Path) -> Path:
    path = root / reporting.PROTOCOL_PATH
    for relative in (str(reporting.PROTOCOL_PATH), *reporting.CAUSAL_SOURCE_PATHS):
        source = REPOSITORY_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return path


def _temporal_outputs(root: Path, protocol: Path) -> None:
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    temporal_protocol = protocol_payload["temporal_short_branch"]
    spec = {
        "seeds": list(reporting.SEEDS),
        "operators": list(reporting.OPERATORS),
        "predictors": list(reporting.PREDICTORS),
        "negative_controls": list(reporting.CONTROLS),
        "outcomes": list(reporting.OUTCOMES),
        "beneficial_direction": {
            "validation_loss_p95": "negative",
            "unseen_margin_p05": "positive",
        },
        "analysis": {"primary_predictor": reporting.PREDICTORS[0]},
    }
    predictor_rows = []
    outcome_rows = []
    operator_scale = {"adamw": 0.0, "muon": 0.7, "normuon": 1.2}
    for seed_index, seed in enumerate(reporting.SEEDS):
        for operator in reporting.OPERATORS:
            displacement = seed_index * 0.2 + operator_scale[operator] * (1 + seed_index * 0.1)
            for stage in range(1, 6):
                row = {"family": "dense", "seed": seed, "operator": operator, "stage": stage}
                for predictor_index, predictor in enumerate(reporting.PREDICTORS, start=1):
                    row[predictor] = displacement * predictor_index
                row["update_frobenius_norm"] = 10 + seed_index + stage * 0.01
                row["weight_frobenius_norm"] = 20 + seed_index + stage * 0.02
                predictor_rows.append(row)
                outcome_rows.append(
                    {
                        "family": "dense",
                        "seed": seed,
                        "operator": operator,
                        "stage": stage,
                        "validation_loss_p95": 8 - 2 * displacement,
                        "unseen_margin_p05": 1 + 1.5 * displacement,
                    }
                )
    paired, loso, estimates = analyze_rows(spec, predictor_rows, outcome_rows)
    decision = support_decision(spec, paired, estimates)
    directory = root / reporting.TEMPORAL_DIR
    _csv(directory / "paired_contrasts.csv", paired)
    _csv(directory / "loso_predictions.csv", loso)
    _csv(directory / "estimates.csv", estimates)
    (directory / "README.md").write_text("# complete temporal report\n", encoding="utf-8")
    sources = []
    for index, relative in enumerate(reporting.TEMPORAL_SOURCE_PATHS):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"source {index}\n", encoding="utf-8")
        sources.append(_record(source, root=root))
    manifest = {
        "schema_version": 1,
        "complete": True,
        "status": "complete",
        "claimable": True,
        "family": "dense",
        "protocol": _record(protocol, root=root),
        "sources": sources,
        "coverage": {
            "seeds": 3,
            "operators": 3,
            "checkpoints": 45,
            "paired_contrasts": 6,
            "loso_predictions": 96,
        },
        "outputs": {
            name: _record(directory / name, root=root)
            for name in (
                "paired_contrasts.csv",
                "loso_predictions.csv",
                "estimates.csv",
                "README.md",
            )
        },
        "decision": decision,
        "claim_rule": temporal_protocol["analysis"]["primary_support_rule"]["decision"],
        "claim_boundary": temporal_protocol["claim_boundary"],
    }
    _json(directory / "summary_manifest.json", manifest)


def _dose_outputs(root: Path, protocol: Path) -> None:
    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    dose_protocol = protocol_payload["dose_band"]
    directory = root / reporting.DOSE_DIR
    anchor_rows = []
    for anchor in reporting._ordered_anchors():
        values = {
            "family": "dense",
            "anchor": anchor,
            "loss_dose_monotone": True,
            "margin_dose_monotone": True,
            "tail_band_best_both_metrics": True,
            "basis_swap_negative_control": True,
            "loss_lambda_0.00": 0.0,
            "loss_lambda_0.25": -0.25,
            "loss_lambda_0.50": -0.5,
            "loss_lambda_0.75": -0.75,
            "loss_lambda_1.00": -1.0,
            "margin_lambda_0.00": 0.0,
            "margin_lambda_0.25": 0.25,
            "margin_lambda_0.50": 0.5,
            "margin_lambda_0.75": 0.75,
            "margin_lambda_1.00": 1.0,
            "loss_band_head": -0.1,
            "loss_band_middle": -0.2,
            "loss_band_tail": -0.3,
            "margin_band_head": 0.1,
            "margin_band_middle": 0.2,
            "margin_band_tail": 0.3,
            "anchor_passed": True,
        }
        anchor_rows.append({field: values[field] for field in reporting.ANCHOR_FIELDS})
    heldout = []
    rates = {"adamw-lr1e-5": 1e-5, "muon-lr1e-3": 1e-3, "normuon-lr1e-3": 1e-3}
    for run_index, run in enumerate(reporting.SOURCE_RUNS):
        for task_index, task in enumerate(reporting.TASKS):
            for transition_index, (transition, step) in enumerate(
                reporting.TRANSITION_STEPS.items()
            ):
                observed = 0.1 + run_index * 0.02 + task_index * 0.001 + transition_index * 0.01
                values = {
                    "family": "dense",
                    "held_out_run": run,
                    "held_out_learning_rate": rates[run],
                    "task": task,
                    "transition": transition,
                    "anchor": f"dense/{run}/checkpoint-{step}",
                    "observed_increment": observed,
                    "baseline_prediction": 0.0,
                    "fold": "leave-one-run-and-learning-rate-out",
                    "spectrum_loss_prediction": observed,
                    "spectrum_margin_prediction": observed * 0.5,
                    "basis_loss_prediction": 0.0,
                    "basis_margin_prediction": 0.0,
                }
                heldout.append({field: values[field] for field in reporting.HELDOUT_FIELDS})
    _csv(directory / "anchor_tests.csv", anchor_rows, reporting.ANCHOR_FIELDS)
    _csv(directory / "heldout_predictions.csv", heldout, reporting.HELDOUT_FIELDS)
    (directory / "report.md").write_text("# complete dose report\n", encoding="utf-8")
    parsed = reporting._parse_heldout(
        reporting._read_csv(
            directory / "heldout_predictions.csv", reporting.HELDOUT_FIELDS, 84, "test"
        )
    )
    rmse, improvements = reporting._dose_rmse(parsed)
    canonical_sources = [root / relative for relative in reporting.DOSE_CANONICAL_SOURCE_PATHS]
    spectral_manifest, spectral_table, evaluation_input, evaluation_manifest = canonical_sources
    spectral_manifest.parent.mkdir(parents=True, exist_ok=True)
    evaluation_manifest.parent.mkdir(parents=True, exist_ok=True)
    _json(evaluation_manifest, {"complete": True})
    evaluation_input.write_text("evaluation\n", encoding="utf-8")
    raw_sources = []
    for index in range(10):
        source = root / "results/spectral-transplant" / f"anchor-{index}.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f'{{"anchor": {index}}}\n', encoding="utf-8")
        raw_sources.append(source)
    condition_values = {
        "muon-native": (-0.9, 0.9),
        "adam-basis__spectrum-lambda-0.25": (-0.25, 0.25),
        "adam-basis__spectrum-lambda-0.50": (-0.5, 0.5),
        "adam-basis__spectrum-lambda-0.75": (-0.75, 0.75),
        "adam-basis__muon-spectrum": (-1.0, 1.0),
        "muon-basis__adam-spectrum": (-0.2, 0.2),
        "adam-basis__muon-head-spectrum": (-0.1, 0.1),
        "adam-basis__muon-middle-spectrum": (-0.2, 0.2),
        "adam-basis__muon-tail-spectrum": (-0.3, 0.3),
    }
    spectral_rows = [
        {
            "family": "dense",
            "anchor": anchor,
            "condition": condition,
            "p95_pairwise_loss_contrast": loss,
            "p05_pairwise_margin_contrast": margin,
        }
        for anchor in reporting._ordered_anchors()
        for condition, (loss, margin) in condition_values.items()
    ]
    _csv(spectral_table, spectral_rows)
    spectral_table_record = {**_record(spectral_table, root=root), "rows": 90}
    _json(
        spectral_manifest,
        {
            "complete": True,
            "outputs": {"anchor_query_tail_effects": spectral_table_record},
            "sources": [
                {"label": anchor, "sample_metrics": _record(source, root=root)}
                for anchor, source in zip(reporting._ordered_anchors(), raw_sources, strict=True)
            ],
        },
    )
    bridge_supported = reporting._bridge_supported(improvements)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "complete": True,
        "claimability": "claimable",
        "supported": bridge_supported,
        "falsification": "overall_chain_supported"
        if bridge_supported
        else "overall_chain_not_supported",
        "local_supported": True,
        "forward_bridge_supported": bridge_supported,
        "criterion": "at_least_8_of_10_anchors",
        "decisions": [
            {
                "family": "dense",
                "anchors": 10,
                "supporting_anchors": 10,
                "loss_dose_monotone_anchors": 10,
                "margin_dose_monotone_anchors": 10,
                "basis_control_anchors": 10,
                "prespecified_band": "tail",
                "tail_band_anchors": 10,
                "threshold": 8,
                "local_supported": True,
            }
        ],
        "prediction_protocol": "separate OLS predictors over task+transition baseline with leave-one-run/LR-out folds",
        "forward_bridge": {
            "rmse": rmse,
            "rmse_improvement": improvements,
            "supported": bridge_supported,
        },
        "claim_boundary": dose_protocol["claim_boundary"],
        "prediction_rows": 84,
        "expected_prediction_rows": 84,
        "protocol": {
            "protocol": _record(protocol, root=root),
            "source_bindings": [
                _record(root / relative, root=root) for relative in reporting.CAUSAL_SOURCE_PATHS
            ],
        },
        "evaluation_manifest": _record(evaluation_manifest, root=root),
        "evaluation_input": _record(evaluation_input, root=root),
        "sources": [_record(path, root=root) for path in (*canonical_sources, *raw_sources)],
        "outputs": {
            name: _record(directory / name, root=root)
            for name in ("anchor_tests.csv", "heldout_predictions.csv", "report.md")
        },
    }
    _json(directory / "summary_manifest.json", manifest)


@pytest.fixture
def complete_root(tmp_path: Path) -> Path:
    protocol = _protocol(tmp_path)
    _temporal_outputs(tmp_path, protocol)
    _dose_outputs(tmp_path, protocol)
    return tmp_path


def _rewrite_csv(root: Path, branch: Path, name: str, mutate) -> None:
    path = root / branch / name
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    mutate(rows)
    _csv(path, rows, fields)
    manifest_path = root / branch / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][name] = _record(path, root=root)
    _json(manifest_path, manifest)


def test_complete_evidence_exposes_stable_full_reporting_rows(complete_root: Path) -> None:
    evidence = load_causal_chain_evidence(complete_root, allow_pending=False)

    assert evidence["complete"] is evidence["claimable"] is True
    assert evidence["status"] == "complete"
    assert evidence["verdict"] == evidence["overall_verdict"] == "supported"
    protocol = json.loads((complete_root / reporting.PROTOCOL_PATH).read_text(encoding="utf-8"))
    assert evidence["claim_boundary"] == {
        "temporal_short_branch": protocol["temporal_short_branch"]["claim_boundary"],
        "dose_band": protocol["dose_band"]["claim_boundary"],
    }
    temporal = evidence["temporal_short_branch"]
    dose = evidence["dose_band"]
    assert len(temporal["paired_rows"]) == 6
    assert len(temporal["loso_rows"]) == 96
    assert len(temporal["estimate_rows"]) == 16
    assert len(temporal["criteria_rows"]) == 5
    assert len(temporal["rmse_rows"]) == 16
    assert len(dose["anchor_rows"]) == len(dose["anchor_criteria_rows"]) == 10
    assert len(dose["criteria_rows"]) == 4
    assert len(dose["heldout_rows"]) == 84
    assert len(dose["rmse_rows"]) == 5
    assert len(dose["bridge_rows"]) == 2
    assert [Path(row["path"]).name for row in evidence["source_table_records"]] == [
        "paired_contrasts.csv",
        "loso_predictions.csv",
        "estimates.csv",
        "anchor_tests.csv",
        "heldout_predictions.csv",
    ]


def test_manifest_output_object_order_is_not_semantic(complete_root: Path) -> None:
    for relative in (reporting.TEMPORAL_DIR, reporting.DOSE_DIR):
        manifest_path = complete_root / relative / "summary_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = manifest["outputs"]
        manifest["outputs"] = {name: outputs[name] for name in reversed(tuple(outputs))}
        _json(manifest_path, manifest)

    evidence = load_causal_chain_evidence(complete_root, allow_pending=False)

    assert evidence["complete"] is evidence["claimable"] is True


def test_canonical_consumers_display_the_full_numeric_contract(complete_root: Path) -> None:
    evidence = load_causal_chain_evidence(complete_root, allow_pending=False)

    detailed = render_causal_chain_markdown(evidence, detailed=True, heading_level=3)
    compact = render_causal_chain_markdown(evidence, detailed=False, heading_level=3)
    latex = render_causal_chain_latex(evidence)
    contract = causal_chain_display_contract(evidence)

    primary = next(
        row
        for row in evidence["temporal_short_branch"]["estimate_rows"]
        if row["outcome"] == "validation_loss_p95"
        and row["predictor"] == "update_tail_energy_fraction"
    )
    visible_number = f"{primary['relative_rmse_improvement']:+.6g}"
    assert visible_number in detailed
    assert visible_number in compact
    assert visible_number in latex
    assert detailed.count("| validation loss p95 |") == 8
    assert "All 10 fixed-state anchors" in detailed
    assert "All 84 held-run predictions" in detailed
    assert "84 leave-source-run-out" in latex
    assert latex.count(r"\newcommand{\CausalChainSummaryTable}") == 1
    assert latex.count(r"\newcommand{\CausalChainDiagnostics}") == 1
    summary = latex.split(r"\newcommand{\CausalChainSummaryTable}", 1)[1].split(
        r"\newcommand{\CausalChainDiagnostics}", 1
    )[0]
    assert r"\scriptsize" in summary
    causal_rows = {}
    for table in re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", latex, flags=re.DOTALL):
        labels = re.findall(r"\\label\{(tab:[^{}]+)\}", table)
        assert len(labels) == 1
        body = table.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
        causal_rows[labels[0]] = sum(line.rstrip().endswith(r"\\") for line in body.splitlines())
    assert causal_rows == {
        "tab:causal-chain-summary": 3,
        "tab:causal-temporal-diagnostics": 6,
        "tab:causal-temporal-estimates": 16,
        "tab:causal-temporal-pairs": 6,
        "tab:causal-dose-diagnostics": 6,
        "tab:causal-dose-anchors": 10,
        "tab:causal-forward-rmse": 5,
    }
    expected_counts = {
        "complete": True,
        "temporal_criteria": 5,
        "temporal_paired_rows": 6,
        "temporal_estimates": 16,
        "temporal_loso_rows": 96,
        "dose_criteria": 4,
        "dose_anchors": 10,
        "dose_rmse_rows": 5,
        "dose_bridge_rows": 2,
        "dose_heldout_rows": 84,
        "source_tables": 5,
    }
    assert {name: contract[name] for name in expected_counts} == expected_counts


def test_complete_causal_latex_is_portable_pdflatex(complete_root: Path) -> None:
    evidence = load_causal_chain_evidence(complete_root, allow_pending=False)
    generated = render_causal_chain_latex(evidence)
    assert generated.isascii()
    executable = shutil.which("pdflatex")
    if executable is None:
        pytest.skip("pdflatex is not installed")
    document = complete_root / "causal-smoke.tex"
    document.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n" + generated + "\\begin{document}\n"
        "\\CausalChainSummaryTable\n"
        "\\clearpage\n"
        "\\CausalChainDiagnostics\n"
        "\\end{document}\n",
        encoding="ascii",
    )
    completed = subprocess.run(
        [executable, "-interaction=nonstopmode", "-halt-on-error", document.name],
        cwd=complete_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout[-4000:]
    log_text = document.with_suffix(".log").read_text(encoding="utf-8")
    assert "Overfull \\hbox" not in log_text


def test_coefficient_criterion_displays_the_values_used_by_the_decision(
    complete_root: Path,
) -> None:
    evidence = copy.deepcopy(load_causal_chain_evidence(complete_root, allow_pending=False))
    primary = next(
        row
        for row in evidence["temporal_short_branch"]["estimate_rows"]
        if row["outcome"] == "validation_loss_p95"
        and row["predictor"] == "update_tail_energy_fraction"
    )
    primary["muon_coefficient_label_only"] = 0.0
    primary["muon_coefficient_with_predictor"] = 0.25
    primary["muon_absolute_coefficient_shrinkage"] = 0.0

    rendered = render_causal_chain_markdown(evidence, detailed=False)

    criterion_line = next(
        line for line in rendered.splitlines() if line.startswith("| coefficient_behavior |")
    )
    assert "muon abs(beta)=0 to 0.25" in criterion_line
    assert "gap -2.500000e-01" in criterion_line


def test_near_tie_decisions_display_unrounded_margins(complete_root: Path) -> None:
    evidence = copy.deepcopy(load_causal_chain_evidence(complete_root, allow_pending=False))
    estimates = evidence["temporal_short_branch"]["estimate_rows"]
    primary = next(
        row
        for row in estimates
        if row["outcome"] == "validation_loss_p95"
        and row["predictor"] == "update_tail_energy_fraction"
    )
    control = next(
        row
        for row in estimates
        if row["outcome"] == "validation_loss_p95" and row["predictor"] == "update_frobenius_norm"
    )
    primary["relative_rmse_improvement"] = 0.12345641
    control["relative_rmse_improvement"] = 0.12345639
    bridge = evidence["dose_band"]["bridge_rows"][0]
    bridge["spectrum_rmse_improvement"] = 0.12345641
    bridge["matched_basis_rmse_improvement"] = 0.12345639

    rendered = render_causal_chain_markdown(evidence, detailed=False)

    assert "decision gaps=+2.000000e-08" in rendered
    assert "| +2.000000e-08 | pass |" in rendered


def test_signed_zero_is_not_presented_as_positive_evidence(complete_root: Path) -> None:
    evidence = copy.deepcopy(load_causal_chain_evidence(complete_root, allow_pending=False))
    primary = next(
        row
        for row in evidence["temporal_short_branch"]["estimate_rows"]
        if row["outcome"] == "validation_loss_p95"
        and row["predictor"] == "update_tail_energy_fraction"
    )
    primary["relative_rmse_improvement"] = 0.0

    rendered = render_causal_chain_markdown(evidence, detailed=False)

    assert "validation loss p95=0 (decision gap +0.000000e+00)" in rendered
    assert "validation loss p95=+0" not in rendered


def test_pending_joint_chain_preserves_a_complete_branch_result(complete_root: Path) -> None:
    evidence = copy.deepcopy(load_causal_chain_evidence(complete_root, allow_pending=False))
    evidence["complete"] = evidence["claimable"] = False
    evidence["dose_band"] = {
        "complete": False,
        "pending_reason": "dose evidence is still running",
    }

    rendered = render_causal_chain_markdown(evidence, detailed=False)

    assert "**Temporal shared-start:** supported" in rendered
    assert "**Dose/band bridge:** pending, not claimable" in rendered
    assert "No joint causal-chain claim is permitted" in rendered


def test_fixed_state_summary_is_not_downgraded_by_forward_failure(complete_root: Path) -> None:
    evidence = copy.deepcopy(load_causal_chain_evidence(complete_root, allow_pending=False))
    evidence["overall_verdict"] = "not_supported_claimable_negative"
    evidence["dose_band"]["supported"] = False
    evidence["dose_band"]["forward_bridge_supported"] = False

    latex = render_causal_chain_latex(evidence)
    fixed_state = next(
        line for line in latex.splitlines() if line.startswith("Fixed-state component")
    )

    assert "; supported & Fixed-weight component attribution supported" in fixed_state
    assert "Forward retrieval" in latex
    assert "fail & No forward bridge; fixed-state conclusion only" in latex


def test_negative_summary_rows_reject_positive_inference_language(complete_root: Path) -> None:
    evidence = copy.deepcopy(load_causal_chain_evidence(complete_root, allow_pending=False))
    evidence["overall_verdict"] = "not_supported_claimable_negative"
    evidence["temporal_short_branch"]["supported"] = False
    evidence["dose_band"]["supported"] = False
    evidence["dose_band"]["local_supported"] = False
    evidence["dose_band"]["forward_bridge_supported"] = False

    latex = render_causal_chain_latex(evidence)

    assert "Temporal spectral bridge rejected" in latex
    assert "Frozen component account rejected" in latex
    assert "No forward bridge; fixed-state conclusion only" in latex
    assert "Fixed-weight component attribution supported" not in latex


def test_evidence_hash_binds_raw_predictions_and_source_identities(complete_root: Path) -> None:
    evidence = load_causal_chain_evidence(complete_root, allow_pending=False)
    baseline = causal_chain_display_contract(evidence)["evidence_sha256"]

    loso_changed = copy.deepcopy(evidence)
    loso_changed["temporal_short_branch"]["loso_rows"][0]["mediator_prediction"] += 1e-6
    assert causal_chain_display_contract(loso_changed)["evidence_sha256"] != baseline

    heldout_changed = copy.deepcopy(evidence)
    heldout_changed["dose_band"]["heldout_rows"][0]["spectrum_loss_prediction"] += 1e-6
    assert causal_chain_display_contract(heldout_changed)["evidence_sha256"] != baseline

    source_changed = copy.deepcopy(evidence)
    source_changed["source_table_records"][0]["sha256"] = "0" * 64
    assert causal_chain_display_contract(source_changed)["evidence_sha256"] != baseline


def test_evidence_hash_is_independent_of_a_parent_named_reports(complete_root: Path) -> None:
    evidence = load_causal_chain_evidence(complete_root, allow_pending=False)
    baseline = causal_chain_display_contract(evidence)["evidence_sha256"]
    relocated = copy.deepcopy(evidence)
    relocated_root = Path("/tmp/reports/relocated-project")
    relocated["repository_root"] = str(relocated_root)
    for record in relocated["source_table_records"]:
        relative = Path(record["path"]).relative_to(complete_root)
        record["path"] = str(relocated_root / relative)

    assert causal_chain_display_contract(relocated)["evidence_sha256"] == baseline


def test_persisted_complete_tree_strict_loads_after_repository_relocation(
    complete_root: Path,
) -> None:
    evidence = load_causal_chain_evidence(complete_root, allow_pending=False)
    baseline = causal_chain_display_contract(evidence)["evidence_sha256"]
    persisted_manifests = (
        complete_root / reporting.TEMPORAL_DIR / "summary_manifest.json",
        complete_root / reporting.DOSE_DIR / "summary_manifest.json",
        complete_root / reporting.DOSE_CANONICAL_SOURCE_PATHS[0],
    )
    declared_paths = [
        path
        for manifest_path in persisted_manifests
        for path in _declared_identity_paths(json.loads(manifest_path.read_text()))
    ]
    assert declared_paths
    assert all(not Path(path).is_absolute() for path in declared_paths)
    assert all(Path(path).as_posix() == path for path in declared_paths)

    relocated_root = complete_root.parent / f"{complete_root.name}-relocated"
    shutil.copytree(complete_root, relocated_root)
    relocated = load_causal_chain_evidence(relocated_root, allow_pending=False)

    assert causal_chain_display_contract(relocated)["evidence_sha256"] == baseline
    assert all(
        Path(record["path"]).is_relative_to(relocated_root)
        for record in relocated["source_table_records"]
    )


def test_legacy_absolute_path_prefers_active_checkout(tmp_path: Path) -> None:
    active = tmp_path / "renamed-checkout"
    active.mkdir()
    expected = active / "results" / "suite" / "sample_metrics.jsonl"
    legacy = Path("/root") / "embedding-optimizer-study" / "results/suite/sample_metrics.jsonl"

    candidates = reporting._candidate_paths(
        str(legacy),
        active,
        active / "reports",
    )

    assert candidates == (expected.resolve(),)


def test_exit_snapshot_rehashes_non_table_branch_outputs(complete_root: Path) -> None:
    evidence = load_causal_chain_evidence(complete_root, allow_pending=False)
    assert _causal_snapshot_still_current(complete_root, evidence)
    report = Path(evidence["dose_band"]["outputs"]["report.md"]["path"])
    report.write_text(report.read_text(encoding="utf-8") + "drift\n", encoding="utf-8")

    assert not _causal_snapshot_still_current(complete_root, evidence)


def test_missing_manifests_are_explicitly_pending_but_can_be_disallowed(tmp_path: Path) -> None:
    empty = load_causal_chain_evidence(tmp_path)
    assert empty["status"] == "pending"
    assert empty["protocol"] is None

    _protocol(tmp_path)

    evidence = load_causal_chain_evidence(tmp_path)

    assert evidence["status"] == "pending"
    assert evidence["complete"] is evidence["claimable"] is False
    assert evidence["verdict"] == "pending_not_claimable"
    assert evidence["source_table_records"] == []
    with pytest.raises(CausalChainReportingError, match="evidence is pending"):
        load_causal_chain_evidence(tmp_path, allow_pending=False)


def test_declared_output_hash_and_path_are_strict(complete_root: Path) -> None:
    estimates = complete_root / reporting.TEMPORAL_DIR / "estimates.csv"
    estimates.write_text(estimates.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CausalChainReportingError, match="content identity differs"):
        load_causal_chain_evidence(complete_root)

    # Restoring bytes but declaring an alternate path must still fail path provenance.
    _temporal_outputs(complete_root, complete_root / reporting.PROTOCOL_PATH)
    manifest_path = complete_root / reporting.TEMPORAL_DIR / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["estimates.csv"]["path"] = manifest["sources"][0]["path"]
    _json(manifest_path, manifest)
    with pytest.raises(CausalChainReportingError, match="does not resolve"):
        load_causal_chain_evidence(complete_root)


def test_frozen_protocol_hash_and_eight_source_bindings_are_authoritative(
    complete_root: Path,
) -> None:
    protocol = complete_root / reporting.PROTOCOL_PATH
    protocol.write_text(protocol.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CausalChainReportingError, match="protocol is not schema-1 Dense"):
        load_causal_chain_evidence(complete_root)

    _protocol(complete_root)
    bound_source = complete_root / reporting.CAUSAL_SOURCE_PATHS[0]
    bound_source.write_text(bound_source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CausalChainReportingError, match="source binding differs"):
        load_causal_chain_evidence(complete_root)


def test_duplicate_and_nonfinite_rows_are_rejected_even_when_hash_is_updated(
    complete_root: Path,
) -> None:
    def duplicate_estimate(rows: list[dict]) -> None:
        rows[1]["outcome"] = rows[0]["outcome"]
        rows[1]["predictor"] = rows[0]["predictor"]
        rows[1]["predictor_kind"] = rows[0]["predictor_kind"]

    _rewrite_csv(complete_root, reporting.TEMPORAL_DIR, "estimates.csv", duplicate_estimate)
    with pytest.raises(CausalChainReportingError, match="duplicates or incomplete"):
        load_causal_chain_evidence(complete_root)

    _temporal_outputs(complete_root, complete_root / reporting.PROTOCOL_PATH)

    def nonfinite_prediction(rows: list[dict]) -> None:
        rows[0]["spectrum_loss_prediction"] = "nan"

    _rewrite_csv(complete_root, reporting.DOSE_DIR, "heldout_predictions.csv", nonfinite_prediction)
    with pytest.raises(CausalChainReportingError, match="non-finite"):
        load_causal_chain_evidence(complete_root)


def test_temporal_and_dose_status_contradictions_are_rejected(complete_root: Path) -> None:
    temporal_path = complete_root / reporting.TEMPORAL_DIR / "summary_manifest.json"
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    temporal["decision"]["criteria"]["negative_control"] = not temporal["decision"]["criteria"][
        "negative_control"
    ]
    _json(temporal_path, temporal)
    with pytest.raises(CausalChainReportingError, match="criterion contradicts"):
        load_causal_chain_evidence(complete_root)

    _temporal_outputs(complete_root, complete_root / reporting.PROTOCOL_PATH)
    dose_path = complete_root / reporting.DOSE_DIR / "summary_manifest.json"
    dose = json.loads(dose_path.read_text(encoding="utf-8"))
    dose["forward_bridge"]["supported"] = not dose["forward_bridge"]["supported"]
    _json(dose_path, dose)
    with pytest.raises(CausalChainReportingError, match="forward bridge contradicts"):
        load_causal_chain_evidence(complete_root)


def test_basis_control_is_recomputed_from_canonical_spectral_contrasts(
    complete_root: Path,
) -> None:
    table = complete_root / reporting.DOSE_CANONICAL_SOURCE_PATHS[1]
    with table.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    target = next(
        row
        for row in rows
        if row["anchor"] == "dense/pretrained" and row["condition"] == "muon-basis__adam-spectrum"
    )
    target["p95_pairwise_loss_contrast"] = "-2"
    _csv(table, rows, fields)

    spectral_manifest_path = complete_root / reporting.DOSE_CANONICAL_SOURCE_PATHS[0]
    spectral = json.loads(spectral_manifest_path.read_text(encoding="utf-8"))
    spectral["outputs"]["anchor_query_tail_effects"] = {
        **_record(table, root=complete_root),
        "rows": 90,
    }
    _json(spectral_manifest_path, spectral)
    dose_manifest_path = complete_root / reporting.DOSE_DIR / "summary_manifest.json"
    dose = json.loads(dose_manifest_path.read_text(encoding="utf-8"))
    dose["sources"][0] = _record(spectral_manifest_path, root=complete_root)
    dose["sources"][1] = _record(table, root=complete_root)
    _json(dose_manifest_path, dose)

    with pytest.raises(CausalChainReportingError, match="basis negative-control decision"):
        load_causal_chain_evidence(complete_root)


def test_matched_bridge_rule_does_not_compare_against_unmatched_control(
    complete_root: Path,
) -> None:
    evidence = load_causal_chain_evidence(complete_root)
    rows = evidence["dose_band"]["bridge_rows"]

    assert rows[0]["spectrum_predictor"] == "spectrum_loss"
    assert rows[0]["matched_basis_control"] == "basis_loss"
    assert rows[0]["passed"] is True
    assert evidence["dose_band"]["forward_bridge_supported"] is True
