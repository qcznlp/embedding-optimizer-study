from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from embed_optim import dimension_publication as publication
from embed_optim.config import load_matrix
from embed_optim.dimension_publication import _validate_panel_rows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "configs/dense_dimension_utilization_protocol.json").read_text())


def _checkpoint_rows():
    rows = [{"run_id": "pretrained", "optimizer": "pretrained", "learning_rate": "", "step": "0"}]
    for config in load_matrix(ROOT / "configs/dense_no_packing_retrain.yaml"):
        for step in PROTOCOL["inputs"]["checkpoint_stages"]:
            rows.append(
                {
                    "run_id": config.run_id,
                    "optimizer": config.optimizer.name,
                    "learning_rate": str(config.optimizer.lr),
                    "step": str(step),
                }
            )
    return rows


@pytest.mark.parametrize("mutation", [None, "duplicate", "optimizer", "stage", "rate"])
def test_publication_requires_exact_primary_cells_with_unchanged_row_count(mutation):
    rows = _checkpoint_rows()
    if mutation == "duplicate":
        rows[-1] = dict(rows[-2])
    elif mutation == "optimizer":
        rows[-1]["optimizer"] = "muon"
    elif mutation == "stage":
        rows[-1]["step"] = "1234"
    elif mutation == "rate":
        rows[-1]["learning_rate"] = "0.123"
    if mutation is None:
        _validate_panel_rows(rows, "checkpoint_summary", ROOT, PROTOCOL)
    else:
        with pytest.raises(ValueError):
            _validate_panel_rows(rows, "checkpoint_summary", ROOT, PROTOCOL)


@pytest.fixture
def synthetic_publication(tmp_path, monkeypatch):
    """Use synthetic upstream tables; exercise actual inference and rendering."""
    checkpoints = _checkpoint_rows()
    task_rows, rotation_rows, random_rows, outcomes = [], [], [], []
    spec = json.loads((ROOT / PROTOCOL["inputs"]["probe_spec"]).read_text())
    tasks = sorted(spec["expected"]["task_counts"])
    for index, row in enumerate(checkpoints):
        optimizer = -1 if index == 0 else publication.OPTIMIZERS.index(row["optimizer"])
        stage = (
            0 if index == 0 else PROTOCOL["inputs"]["checkpoint_stages"].index(int(row["step"])) + 1
        )
        values = {
            feature: 0.4 + 0.03 * math.sin(index + k) + optimizer * 0.02
            for k, feature in enumerate(publication.PRIMARY_FEATURES)
        }
        row.update(values)
        if index:
            outcomes.append(
                {
                    "run_id": row["run_id"],
                    "stage": stage,
                    "mean_ndcg_at_10": 0.4
                    + 0.01 * optimizer
                    + 0.005 * stage
                    + 0.03 * math.sin(index),
                }
            )
        for task_index, task in enumerate(tasks):
            task_row = {
                **row,
                "task": task,
                **{
                    feature: value + 0.005 * optimizer * (task_index + 1) * (k + 1)
                    for k, (feature, value) in enumerate(values.items())
                },
            }
            task_rows.append(task_row)
            if stage in (0, 5):
                for seed in PROTOCOL["rotation_control"]["seeds"]:
                    rotation_rows.append({**task_row, "rotation_seed": seed})
            for fraction in PROTOCOL["random_removal"]["removed_fractions"]:
                for draw in range(20):
                    random_rows.append(
                        {
                            **row,
                            "task": task,
                            "draw": draw,
                            "removed_fraction": fraction,
                            "relative_shortlist_ndcg": 0.98 + 0.01 * math.sin(index),
                        }
                    )
    tables = {
        "checkpoint_summary": checkpoints,
        "task_summary": task_rows,
        "rotation_summary": rotation_rows,
        "random_removal": random_rows,
    }
    dimension_source = {"fixture": "synthetic dimension input"}
    outcome_source = {"fixture": "synthetic outcome input"}
    monkeypatch.setattr(
        publication, "_load_dimension_tables", lambda *a: (tables, dimension_source)
    )
    monkeypatch.setattr(publication, "_load_outcomes", lambda *a: (outcomes, outcome_source))
    args = Namespace(
        repository=ROOT,
        protocol=ROOT / "configs/dense_dimension_utilization_protocol.json",
        dimension_dir=tmp_path / "dimension",
        outcomes_dir=tmp_path / "outcomes",
        output_dir=tmp_path / "publication",
        paper_output=tmp_path / "dimension.tex",
    )
    manifest = publication.build_report(args)
    return args, manifest, tables, outcomes


def test_publication_audit_recomputes_inference_and_exact_rendering(synthetic_publication):
    args, manifest, _, _ = synthetic_publication
    assert publication.audit_report(args) == manifest
    assert manifest["coverage"]["primary_contrasts"] == 9
    assert manifest["coverage"]["bridge_features"] == 4
    assert manifest["coverage"]["figure_points"] == 68


def test_figure_uses_all_rates_tasks_masks_and_pretrained_reference(synthetic_publication):
    _, _, tables, _ = synthetic_publication
    points = publication._figure_points(tables, ROOT, PROTOCOL)
    assert len(points) == 68
    assert sum(row["role"] == "identity_reference" for row in points) == 4
    for point in points:
        assert point["run_count"] == (1 if point["optimizer"] == "pretrained" else 4)
        assert point["task_count"] == 14
        if point["role"] == "identity_reference":
            assert point["mean"] == 1 and point["x"] == 0
            continue
        name = "random_removal" if point["feature"] == "relative_shortlist_ndcg" else "task_summary"
        rows = [
            row
            for row in tables[name]
            if row["optimizer"] == point["optimizer"] and int(row["step"]) == point["step"]
        ]
        if name == "random_removal":
            rows = [row for row in rows if 100 * float(row["removed_fraction"]) == point["x"]]
            assert point["mask_draws"] == 20
        values = [float(row[point["feature"]]) for row in rows]
        assert point["mean"] == math.fsum(values) / len(values)


@pytest.mark.parametrize("mutation", ["duplicate", "missing_rate", "nonfinite", "missing_mask"])
def test_figure_cannot_silently_drop_or_reweight_cells(synthetic_publication, mutation):
    _, _, tables, _ = synthetic_publication
    if mutation == "duplicate":
        tables["task_summary"][-1] = dict(tables["task_summary"][-2])
    elif mutation == "missing_rate":
        tables["task_summary"] = [
            row for row in tables["task_summary"] if row["run_id"] != "padded-muon-3e-3"
        ]
    elif mutation == "missing_mask":
        tables["random_removal"].pop()
    else:
        tables["task_summary"][-1][publication.PRIMARY_FEATURES[0]] = float("nan")
    with pytest.raises(ValueError):
        publication._figure_points(tables, ROOT, PROTOCOL)


def test_rehashed_plot_points_cannot_bypass_exact_reconstruction(synthetic_publication):
    args, manifest, _, _ = synthetic_publication
    path = Path(manifest["outputs"]["figure_points"]["path"])
    rows = publication._read_csv(path)
    rows[-1]["mean"] = "123.456"
    manifest["outputs"]["figure_points"] = publication._atomic_csv(path, rows)
    publication._atomic_json(args.output_dir / "summary_manifest.json", manifest)
    with pytest.raises(ValueError, match="figure_points"):
        publication.audit_report(args)


def test_native_vector_coordinates_are_audited_not_just_figure_hashes(synthetic_publication):
    args, manifest, _, _ = synthetic_publication
    text = args.paper_output.read_text()
    assert text.count(r"\nextgroupplot") == 4
    assert text.count(r"\addplot+") == 16
    assert "full-corpus retrieval scores" in text
    assert "not confidence intervals" in text
    args.paper_output.write_text(text.replace("coordinates {", "coordinates {(1,99) ", 1))
    manifest["outputs"]["paper_latex"].update(publication._identity(args.paper_output, ROOT))
    publication._atomic_json(args.output_dir / "summary_manifest.json", manifest)
    with pytest.raises(ValueError, match="manuscript"):
        publication.audit_report(args)


@pytest.mark.parametrize("mutation", ["upstream", "decision", "table", "paper", "inventory"])
def test_rehashing_generated_outputs_cannot_bypass_recomputation(synthetic_publication, mutation):
    args, manifest, tables, _ = synthetic_publication
    if mutation == "upstream":
        tables["task_summary"][-1][publication.PRIMARY_FEATURES[0]] += 0.5
    elif mutation == "decision":
        values = manifest["decisions"]["constructive_native_coordinate_use"]
        values["muon"] = not values["muon"]
    elif mutation == "inventory":
        manifest["outputs"].pop("rotation_contrasts")
    else:
        name = "paper_latex" if mutation == "paper" else "primary_contrasts"
        path = Path(manifest["outputs"][name]["path"])
        path.write_text(path.read_text() + "\nchanged\n")
        manifest["outputs"][name].update(publication._identity(path, args.repository))
    publication._atomic_json(args.output_dir / "summary_manifest.json", manifest)
    with pytest.raises(ValueError):
        publication.audit_report(args)


def test_rendered_dimension_findings_compile(synthetic_publication):
    args, _, _, _ = synthetic_publication
    executable = shutil.which("pdflatex")
    if executable is None:
        pytest.skip("pdflatex is not installed")
    latex = args.paper_output.read_text()
    assert r"With 50\%" in latex
    assert "Decision " + "\\\\\n" + r"\midrule" in latex
    document = args.output_dir / "fixture.tex"
    document.write_text(
        "\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage{pgfplots}\n"
        "\\usepgfplotslibrary{groupplots}\n\\pgfplotsset{compat=1.18}\n"
        + latex
        + "\n\\begin{document}\n\\DimensionUtilizationFinding\n"
        + "\\DimensionRetrievalBridgeFinding\n\\DimensionConclusionFinding\n"
        + "\\DimensionUtilizationFigure\n\\DimensionUtilizationAppendixTable\n\\end{document}\n"
    )
    result = subprocess.run(
        [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(args.output_dir),
            str(document),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout[-4000:]


def test_complete_synthetic_findings_fit_the_naacl_paper_layout(synthetic_publication, tmp_path):
    """Render every primary include; a pending box is not a final-layout test."""
    from test_corrected_publication import _evidence
    from test_state_operator_factorial_completion import _estimands

    from embed_optim.corrected_publication import render_latex
    from embed_optim.paper_layout import audit_paper_layout
    from embed_optim.state_operator_factorial_publication import _render_latex

    if shutil.which("latexmk") is None or shutil.which("make") is None:
        pytest.skip("latexmk/make is not installed")
    args, _, _, _ = synthetic_publication
    paper = tmp_path / "synthetic-layout-only" / "paper"
    shutil.copytree(ROOT / "paper", paper, ignore=shutil.ignore_patterns("build"))
    (paper / "generated/dimension-utilization.tex").write_text(args.paper_output.read_text())
    (paper / "generated/optimizer-primary.tex").write_text(render_latex(_evidence()))
    (paper / "generated/state-operator-factorial.tex").write_text(
        _render_latex(_estimands("supported_positive", "supported_positive", "supported_positive"))
    )
    manuscript = (paper / "main.tex").read_text()
    manuscript = manuscript.replace(
        r"\maketitle",
        r"\maketitle\begin{center}\textbf{SYNTHETIC LAYOUT TEST -- NOT EXPERIMENT RESULTS}\end{center}",
    )
    (paper / "main.tex").write_text(manuscript)
    result = subprocess.run(
        ["make", "all", f"PYTHON={sys.executable}"],
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
    assert result.returncode == 0, result.stdout[-6000:]
    layout = audit_paper_layout(paper)
    assert layout["complete"] and layout["main_float_pages"]["fig:dimension-utility"] <= 8
    assert "Overfull \\hbox" not in (paper / "build/main.log").read_text()
    (paper / "layout-audit.json").write_text(
        json.dumps(
            {
                **layout,
                "scientific_completion": False,
                "synthetic_inputs": True,
            },
            indent=2,
        )
        + "\n"
    )
