from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.geometry import _sha256
from embed_optim.mechanism_report import (
    _read_declared_csv,
    ensure_retrieval_dynamics,
    render_mechanism_report,
)


@pytest.fixture(autouse=True)
def _isolated_repository_root(tmp_path: Path) -> None:
    """Keep pending causal evidence inside each synthetic repository."""

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _declared(path: Path, rows: int) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_declared_csv_rebases_historical_checkout_path(tmp_path: Path) -> None:
    repository = tmp_path / "renamed-checkout"
    (repository / "pyproject.toml").parent.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    table = repository / "reports" / "summary" / "table.csv"
    _write_csv(table, [{"value": "1"}])
    record = _declared(table, 1)
    record["path"] = str(Path("/root") / "embedding-optimizer-study" / "reports/summary/table.csv")
    manifest = {"outputs": {"table": record}}

    rows, observed = _read_declared_csv(
        table.parent,
        manifest,
        "table",
        required_fields={"value"},
    )

    assert rows == [{"value": "1"}]
    assert observed == table.resolve()


def _common_state(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = []
    for family in ("dense", "late"):
        for operator_index, operator in enumerate(("muon", "normuon"), start=1):
            for anchor in range(10):
                rows.append(
                    {
                        "family": family,
                        "label": f"{family}/anchor-{anchor}",
                        "update_operator": operator,
                        "row_norm_cv_parameter_weighted_to_adamw_ratio": 0.5 + operator_index * 0.1,
                        "top_1pct_row_energy_parameter_weighted_to_adamw_ratio": 0.6
                        + operator_index * 0.1,
                        "approx_stable_rank_parameter_weighted_to_adamw_ratio": 1.1
                        + operator_index * 0.1,
                        "spectral_norm_parameter_weighted_to_adamw_ratio": 0.9
                        + operator_index * 0.1,
                        "cosine_with_adamw_parameter_weighted": 0.7 - operator_index * 0.1,
                    }
                )
    table = root / "anchor_contrasts.csv"
    _write_csv(table, rows)
    counts = {
        "gradient_tensor_metrics": 1_760,
        "update_tensor_metrics": 5_280,
        "pairwise_tensor_cosines": 5_280,
        "gradient_anchor_metrics": 20,
        "anchor_metrics": 60,
        "pairwise_anchor_cosines": 60,
        "update_gradient_contrasts": 60,
        "anchor_contrasts": 40,
    }
    outputs = {name: {"rows": count} for name, count in counts.items()}
    outputs["anchor_contrasts"] = _declared(table, len(rows))
    (root / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "allow_partial": False,
                "expected_anchors": 20,
                "valid_anchors": 20,
                "missing_labels": [],
                "outputs": outputs,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return root


def _spectra(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = []
    for family in ("dense", "late"):
        for operator_index, operator in enumerate(("adamw", "muon", "normuon"), start=1):
            for index in range(60):
                rows.append(
                    {
                        "family": family,
                        "label": f"{family}/anchor-{index // 6}",
                        "update_operator": operator,
                        "tensor": f"tensor-{index % 6}",
                        "rank": 10,
                        "stable_rank": 4 + operator_index * 0.5,
                        "entropy_effective_rank": 5 + operator_index * 0.5,
                        "condition_number": 10 + operator_index,
                    }
                )
    table = root / "spectrum_metrics.csv"
    _write_csv(table, rows)
    (root / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "allow_partial": False,
                "expected_anchors": 20,
                "valid_anchors": 20,
                "expected_spectra": 360,
                "valid_spectra": 360,
                "missing_labels": [],
                "outputs": {"spectrum_metrics": _declared(table, len(rows))},
            },
            sort_keys=True,
        )
        + "\n"
    )
    return root


def _basis(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = [
        {
            "family": family,
            "optimizer": optimizer,
            "records": 90,
            "median_mapped_direction_cosine": 0.99 - optimizer_index * 0.01,
            "median_mapped_relative_frobenius_error": 0.01 + optimizer_index * 0.01,
            "maximum_mapped_relative_frobenius_error": 0.02 + optimizer_index * 0.01,
            "median_absolute_norm_ratio_error": 0.001 + optimizer_index * 0.001,
            "median_predicted_descent_relative_error": 0.003 + optimizer_index * 0.001,
            "median_head_spectrum_relative_l2_error": 0.004 + optimizer_index * 0.001,
            "maximum_functional_invariance_error": 1e-14,
        }
        for family in ("dense", "late")
        for optimizer_index, optimizer in enumerate(("adamw", "muon", "normuon"))
    ]
    table = root / "summary.csv"
    _write_csv(table, rows)
    protocol = root / "basis.json"
    protocol.write_text("{}\n", encoding="utf-8")
    (root / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "protocol": {
                    "path": str(protocol.resolve()),
                    "sha256": _sha256(protocol),
                },
                "coverage": {
                    "anchors": 20,
                    "tensor_sequences": 60,
                    "records": 540,
                    "head_records": 3_240,
                    "summary_rows": 6,
                },
                "outputs": {"summary": _declared(table, len(rows))},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _bridge(root: Path) -> Path:
    root.mkdir(parents=True)
    checkpoints = []
    for family in ("dense", "late"):
        for optimizer_index, optimizer in enumerate(("adamw", "muon", "normuon")):
            for learning_rate in range(4):
                for stage in range(1, 6):
                    checkpoints.append(
                        {
                            "model_family": family,
                            "optimizer": optimizer,
                            "stage": stage,
                            "mean_training_loss": 0.5 - stage * 0.01,
                            "training_margin_mean": 0.1 + optimizer_index * 0.01 + stage * 0.001,
                            "unseen_margin_mean": 0.2 + optimizer_index * 0.01 + stage * 0.001,
                            "unseen_query_normalized_effective_rank": 0.3 + optimizer_index * 0.01,
                            "unseen_reference_top1_agreement": 0.9 - stage * 0.01,
                            "unseen_document_token_coverage_mean": (
                                0.5 + optimizer_index * 0.01 if family == "late" else ""
                            ),
                            "mean_beir_ndcg_at_10": 0.25
                            + optimizer_index * 0.01
                            + learning_rate * 0.001,
                        }
                    )
    checkpoint_path = root / "checkpoint_bridge.csv"
    _write_csv(checkpoint_path, checkpoints)
    selected = [
        (family, predictor, outcome)
        for family in ("dense", "late")
        for predictor, outcome in (
            ("reference_delta_row_cv_parameter_weighted", "unseen_margin_mean"),
            ("unseen_margin_mean", "mean_beir_ndcg_at_10"),
            ("unseen_query_normalized_effective_rank", "mean_beir_ndcg_at_10"),
        )
    ]
    selected.append(("late", "unseen_document_token_coverage_mean", "mean_beir_ndcg_at_10"))
    selected.extend(
        (family, "mean_training_loss", "mean_beir_ndcg_at_10") for family in ("dense", "late")
    )
    correlations = [
        {
            "model_family": family,
            "scope": "all_optimizers",
            "optimizer": "all",
            "analysis": "within_run_first_differences",
            "predictor": predictor,
            "outcome": outcome,
            "observations": 48,
            "spearman_rho": 0.25,
        }
        for family, predictor, outcome in selected
    ]
    for family, target in (("dense", 96), ("late", 120)):
        present = sum(row["model_family"] == family for row in correlations)
        for index in range(present, target):
            correlations.append(
                {
                    "model_family": family,
                    "scope": "optimizer",
                    "optimizer": "adamw",
                    "analysis": "checkpoint_levels",
                    "predictor": f"fixture-{family}-{index}",
                    "outcome": "mean_beir_ndcg_at_10",
                    "observations": 20,
                    "spearman_rho": 0.0,
                }
            )
    correlation_path = root / "descriptive_correlations.csv"
    _write_csv(correlation_path, correlations)
    transitions = [
        {
            "model_family": family,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "stage": stage,
            "previous_stage": stage - 1,
        }
        for family in ("dense", "late")
        for optimizer in ("adamw", "muon", "normuon")
        for learning_rate in range(4)
        for stage in range(2, 6)
    ]
    transition_path = root / "within_run_changes.csv"
    _write_csv(transition_path, transitions)
    source = root / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    reference = {"path": str(source.resolve()), "sha256": _sha256(source)}
    (root / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "checkpoints": 120,
                "within_run_transitions": 96,
                "correlations": 216,
                "sources": {
                    "weight_space": {"manifest": reference, "table": reference},
                    "training_dynamics": {"manifest": reference, "table": reference},
                    "representation": {
                        "training": {"summary_manifest": reference},
                        "unseen": {"summary_manifest": reference},
                    },
                    "evaluation": {"coverage": reference, "table": reference},
                    "loss_retrieval_protocol": reference,
                },
                "outputs": {
                    "checkpoint_bridge": _declared(checkpoint_path, len(checkpoints)),
                    "descriptive_correlations": _declared(correlation_path, len(correlations)),
                    "within_run_changes": _declared(transition_path, len(transitions)),
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return root


def _figure(path: Path, *, spectra: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<svg/>\n", encoding="utf-8")
    source = path.with_name(f"{path.stem}-source.json")
    source.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source_summary_manifest": {
            "path": str(source.resolve()),
            "sha256": _sha256(source),
        },
        "output": {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        },
    }
    if spectra:
        manifest.update({"anchors": 20, "spectra": 360})
    else:
        manifest["complete"] = True
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _retrieval(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = []
    for family in ("dense", "late"):
        for optimizer_index, optimizer in enumerate(("adamw", "muon", "normuon")):
            rows.append(
                {
                    "model_family": family,
                    "optimizer": optimizer,
                    "learning_rate_points": 4,
                    "adamw_median_final_target": 0.4,
                    "points_reaching_target": 4 - optimizer_index,
                    "points_right_censored": optimizer_index,
                    "fastest_observed_useful_wall_time_hours": 2.0 + optimizer_index,
                    "median_observed_useful_wall_time_hours": 3.0 + optimizer_index,
                    "target_definition": "within-family-median-of-four-adamw-final-points",
                    "interpolation": "none-five-observed-checkpoints-only",
                }
            )
    table = root / "optimizer_first_passage.csv"
    _write_csv(table, rows)
    figure = root / "quality_vs_useful_wall_time.svg"
    figure.write_text("<svg/>\n", encoding="utf-8")
    source = root / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    reference = {"path": str(source.resolve()), "sha256": _sha256(source)}
    sources = {
        "frozen_protocol": reference,
        "matrix": reference,
        "strict_coverage": reference,
        "training_summary": reference,
        "training_run_table": reference,
        "evaluation_results": [reference for _ in range(1_680)],
    }
    (root / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "coverage": {
                    "runs": 24,
                    "checkpoints": 120,
                    "tasks": 14,
                    "evaluation_units": 1_680,
                    "optimizer_family_groups": 6,
                    "best_config_task_delta_rows": 280,
                    "adjacent_stage_task_stability_rows": 16,
                },
                "sources": sources,
                "outputs": {
                    "optimizer_first_passage": _declared(table, len(rows)),
                    "quality_vs_useful_wall_time": {
                        "path": str(figure.resolve()),
                        "bytes": figure.stat().st_size,
                        "sha256": _sha256(figure),
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _inputs(tmp_path: Path):
    common = _common_state(tmp_path / "common")
    basis = _basis(tmp_path / "basis")
    spectra = _spectra(tmp_path / "spectra")
    bridge = _bridge(tmp_path / "bridge")
    retrieval = _retrieval(tmp_path / "reports" / "retrieval-dynamics")
    figures = (
        _figure(tmp_path / "figures" / "spectra.svg", spectra=True),
        _figure(tmp_path / "figures" / "representation.svg"),
        _figure(tmp_path / "figures" / "late.svg"),
    )
    return common, basis, spectra, bridge, retrieval, figures


def _scope_amendment(root: Path, *, bad_hash: bool = False) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    project = root / "pyproject.toml"
    project.write_text("[project]\nname = 'mechanism-fixture'\n", encoding="utf-8")
    digest = "0" * 64 if bad_hash else _sha256(project)
    amendment = root / "configs" / "dense_scope_amendment.json"
    amendment.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "user_directed_post_hoc_scope_amendment",
                "amended_at_utc": "2026-08-30T20:14:42Z",
                "active_scope": {"families": ["dense"]},
                "claim_boundary": "Dense-only post-hoc claim boundary for the fixture.",
                "source_bindings": [{"path": "pyproject.toml", "sha256": digest} for _ in range(6)],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return amendment


@pytest.fixture(autouse=True)
def _accept_fixture_basis_audit(monkeypatch):
    def audit(_protocol, *, output_dir, **_kwargs):
        return json.loads((Path(output_dir) / "summary_manifest.json").read_text())

    monkeypatch.setattr("embed_optim.mechanism_report.audit_basis_sensitivity", audit)


def test_mechanism_report_strictly_renders_fixed_report(tmp_path: Path):
    common, basis, spectra, bridge, retrieval, figures = _inputs(tmp_path)
    output = tmp_path / "reports" / "mechanism-summary.md"

    manifest = render_mechanism_report(
        common,
        spectra,
        bridge,
        retrieval,
        output,
        basis_dir=basis,
        spectrum_figure=figures[0],
        representation_figure=figures[1],
        late_token_figure=figures[2],
    )
    first = output.read_bytes()
    repeated = render_mechanism_report(
        common,
        spectra,
        bridge,
        retrieval,
        output,
        basis_dir=basis,
        spectrum_figure=figures[0],
        representation_figure=figures[1],
        late_token_figure=figures[2],
    )

    assert manifest["complete"] is True
    assert "families" not in manifest
    assert "scope_amendment" not in manifest
    assert repeated == manifest
    assert output.read_bytes() == first
    assert "Same-state optimizer fingerprints" in output.read_text()
    assert "Function-preserving basis sensitivity" in output.read_text()
    assert "Retrieval time to an AdamW reference" in output.read_text()
    assert "first seven geometry associations were fixed" in output.read_text()
    assert "trailing training loss (post-hoc)" in output.read_text()
    assert "1,456/1,680" in output.read_text()
    assert json.loads(output.with_suffix(".manifest.json").read_text()) == manifest


def test_mechanism_report_renders_strict_dense_scope(tmp_path: Path):
    common, basis, spectra, bridge, retrieval, figures = _inputs(tmp_path / "inputs")
    amendment = _scope_amendment(tmp_path / "scope")
    output = tmp_path / "reports" / "dense-mechanism-summary.md"

    manifest = render_mechanism_report(
        common,
        spectra,
        bridge,
        retrieval,
        output,
        basis_dir=basis,
        spectrum_figure=figures[0],
        representation_figure=figures[1],
        late_token_figure=figures[2],
        families=("dense",),
        scope_amendment=amendment,
    )

    rendered = output.read_text(encoding="utf-8")
    assert manifest["families"] == ["dense"]
    assert manifest["scope_amendment"]["path"] == "configs/dense_scope_amendment.json"
    assert manifest["scope_amendment"]["sha256"] == _sha256(amendment)
    assert manifest["sources"]["common_state"]["anchors"] == 10
    assert manifest["sources"]["exact_spectra"]["spectra"] == 180
    assert manifest["sources"]["basis_sensitivity"]["records"] == 270
    assert manifest["sources"]["basis_sensitivity"]["summary_rows"] == 3
    assert manifest["sources"]["mechanism_bridge"]["checkpoints"] == 60
    assert manifest["sources"]["mechanism_bridge"]["within_run_transitions"] == 48
    assert manifest["sources"]["mechanism_bridge"]["correlations"] == 96
    assert manifest["sources"]["retrieval_dynamics"]["evaluation_units"] == 840
    assert set(manifest["figures"]) == {"retrieval_dynamics"}
    assert "DenseOn" in rendered
    assert "Late" not in rendered
    assert "MaxSim" not in rendered


def test_mechanism_report_rejects_changed_scope_binding(tmp_path: Path):
    common, basis, spectra, bridge, retrieval, figures = _inputs(tmp_path / "inputs")
    amendment = _scope_amendment(tmp_path / "scope", bad_hash=True)

    with pytest.raises(ValueError, match="Scope-amendment source differs"):
        render_mechanism_report(
            common,
            spectra,
            bridge,
            retrieval,
            tmp_path / "mechanism.md",
            basis_dir=basis,
            spectrum_figure=figures[0],
            representation_figure=figures[1],
            late_token_figure=figures[2],
            families=("dense",),
            scope_amendment=amendment,
        )


def test_mechanism_report_rejects_figure_hash_drift(tmp_path: Path):
    common, basis, spectra, bridge, retrieval, figures = _inputs(tmp_path)
    figures[1].write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="figure differs"):
        render_mechanism_report(
            common,
            spectra,
            bridge,
            retrieval,
            tmp_path / "mechanism.md",
            basis_dir=basis,
            spectrum_figure=figures[0],
            representation_figure=figures[1],
            late_token_figure=figures[2],
        )


def test_mechanism_report_rejects_partial_common_state(tmp_path: Path):
    common, basis, spectra, bridge, retrieval, figures = _inputs(tmp_path)
    manifest_path = common / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="complete frozen 20-anchor"):
        render_mechanism_report(
            common,
            spectra,
            bridge,
            retrieval,
            tmp_path / "mechanism.md",
            basis_dir=basis,
            spectrum_figure=figures[0],
            representation_figure=figures[1],
            late_token_figure=figures[2],
        )


def test_mechanism_report_materializes_missing_retrieval_summary(tmp_path: Path):
    retrieval = tmp_path / "retrieval"
    calls = []

    def build(*, output_dir):
        calls.append(output_dir)
        output_dir.mkdir(parents=True)
        (output_dir / "summary_manifest.json").write_text("{}\n", encoding="utf-8")

    manifest = ensure_retrieval_dynamics(retrieval, builder=build)

    assert manifest == retrieval / "summary_manifest.json"
    assert calls == [retrieval]
    ensure_retrieval_dynamics(retrieval, builder=lambda **kwargs: pytest.fail(str(kwargs)))
