from __future__ import annotations

import re
from pathlib import Path

import pytest

from embed_optim.dense_retrieval_dynamics_publication import DYNAMICS_EXTENSION_TEX
from embed_optim.paper_results import (
    PAPER_RESULT_TABLE_PATHS,
    IncompletePaperEvidenceError,
    _ci_classification,
    _latex_escape,
    _replace_headlines,
    build_headline_macros,
    build_result_tables,
    main,
    render_paper_results,
)


def _rendered_table_rows(text: str) -> dict[str, int]:
    result = {}
    for table in re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, flags=re.DOTALL):
        labels = re.findall(r"\\label\{(tab:[^{}]+)\}", table)
        assert len(labels) == 1
        body = table.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
        result[labels[0]] = sum(line.rstrip().endswith(r"\\") for line in body.splitlines())
    return result


def test_renderer_resolves_scope_amendment_against_repo_root_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "checkout"
    expected = repository / "configs/scope.json"
    captured: dict[str, Path] = {}

    def capture(_families, scope_path):
        captured["path"] = Path(scope_path)
        raise RuntimeError("captured")

    monkeypatch.setattr("embed_optim.paper_results.resolve_scope", capture)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="captured"):
        render_paper_results(
            repo_root=repository,
            families=("dense",),
            scope_amendment=Path("configs/scope.json"),
        )

    assert captured["path"] == expected.resolve()


def _rows():
    families = ("DenseOn", "LateOn")
    optimizers = ("AdamW", "Muon", "NorMuon")
    retrieval = [
        [
            family,
            optimizer,
            "0.4000",
            f"{index + 1}/4",
            "1.000",
            "2.000",
            str(3 - index),
        ]
        for family in families
        for index, optimizer in enumerate(optimizers)
    ]
    common = [
        [family, optimizer, "0.500", "0.600", "1.200", "0.900", "0.700"]
        for family in families
        for optimizer in ("Muon", "NorMuon")
    ]
    spectra = [
        [family, optimizer, f"0.{index + 4}", "0.6000", "2.00"]
        for family in families
        for index, optimizer in enumerate(optimizers)
    ]
    representation = [
        [
            family,
            optimizer,
            "0.1000",
            f"0.{index + 2}000",
            "0.5000",
            "0.8000",
            "0.3000",
            "0.4000",
        ]
        for family in families
        for index, optimizer in enumerate(optimizers)
    ]
    correlations = [
        [family, predictor, "mean BEIR nDCG@10", "48", rho]
        for family in families
        for predictor, rho in (
            ("unseen margin", "0.500"),
            ("trailing training loss (post-hoc)", "-0.250"),
        )
    ]
    functional = [
        [
            family,
            optimizer,
            "descent",
            "-0.0100",
            f"0.0{index + 1}00",
            "0.0200",
            "0.0300",
            "0.80",
        ]
        for family in families
        for index, optimizer in enumerate(optimizers)
    ]
    hybrid = [
        [family, f"{learning_rate:.0e}", "0.4000", "0.4100", "0.0100", "8/0/6"]
        for family in families
        for learning_rate in (1e-6, 3e-6, 1e-5, 3e-5)
    ]
    contrasts = ("Muon - AdamW", "NorMuon - AdamW", "NorMuon - Muon")
    short = [
        [
            family,
            contrast,
            "-0.0100 (3/0/0)",
            "0.0100 (3/0/0)",
            "0.0100 (3/0/0)",
            "0.0100 (3/0/0)",
        ]
        for family in families
        for contrast in contrasts
    ]
    intervals = {
        "Muon - AdamW": "[0.0010, 0.0190]",
        "NorMuon - AdamW": "[-0.0010, 0.0210]",
        "NorMuon - Muon": "[-0.0190, -0.0010]",
    }
    familywise_intervals = {
        "Muon - AdamW": "[0.0005, 0.0200]",
        "NorMuon - AdamW": "[-0.0050, 0.0250]",
        "NorMuon - Muon": "[-0.0200, -0.0005]",
    }
    confirmation = [
        [
            family,
            contrast,
            "0.0100",
            intervals[contrast],
            familywise_intervals[contrast],
            "3/0/0",
            "9/0/5",
        ]
        for family in families
        for contrast in contrasts
    ]
    return {
        "retrieval_rows": retrieval,
        "common_rows": common,
        "spectrum_rows": spectra,
        "representation_rows": representation,
        "correlation_rows": correlations,
        "functional_rows": functional,
        "hybrid_rows": hybrid,
        "short_rows": short,
        "confirmation_rows": confirmation,
    }


def _task_rows():
    return [
        [
            family,
            task,
            "0.4000",
            "0.4100",
            "0.4200",
            "+0.0100",
            "+0.0200",
        ]
        for family in ("DenseOn", "LateOn")
        for task in (
            "ClimateFEVER",
            "FEVER",
            "MSMARCO",
            "HotpotQA",
            "DBPedia",
            "QuoraRetrieval",
            "Touche2020",
            "NQ",
            "TRECCOVID",
            "FiQA2018",
            "ArguAna",
            "SCIDOCS",
            "NFCorpus",
            "SciFact",
        )
    ]


def _basis_result_rows():
    return [
        [family, optimizer, "0.99000", "0.01000", "0.00100", "0.00200", "0.00300"]
        for family in ("DenseOn", "LateOn")
        for optimizer in ("AdamW", "Muon", "NorMuon")
    ]


def _task_stability_rows():
    return [
        [
            family,
            f"{optimizer} - AdamW",
            f"{first * 20}--{(first + 1) * 20}%",
            "12/14",
            "0.900",
            "0.800",
        ]
        for family in ("DenseOn", "LateOn")
        for optimizer in ("Muon", "NorMuon")
        for first in range(1, 5)
    ]


def _tail_result_rows():
    discovery = [
        [family, optimizer, "-0.1000", "0.0100", "0.3000", "tail redistribution"]
        for family in ("DenseOn", "LateOn")
        for optimizer in ("Muon", "NorMuon")
    ]
    final = [
        [family, optimizer, "-0.0200", "3/3", "0.0100", "2/3", "supported"]
        for family in ("DenseOn", "LateOn")
        for optimizer in ("Muon", "NorMuon")
    ]
    return discovery, final


def _spectral_result_rows():
    factorial = [
        [family, metric, "-0.0200", "0.0100", "0.0010"]
        for family in ("DenseOn", "LateOn")
        for metric in ("contrastive loss", "positive margin")
    ]
    tail = [
        [family, condition, "-0.0200", "0.0100", "-0.1000", "-0.0500", "0.4000"]
        for family in ("DenseOn", "LateOn")
        for condition in (
            "Muon native",
            "Adam basis + Muon spectrum",
            "Muon basis + Adam spectrum",
        )
    ]
    return factorial, tail


def _system_result_rows():
    return [
        [family, optimizer, "3.52", "39.43", ratio, "7.41", "1.11", "1.67"]
        for family in ("DenseOn", "LateOn")
        for optimizer, ratio in (("AdamW", "1.00x"), ("Muon", "0.95x"), ("NorMuon", "0.93x"))
    ]


def test_headlines_report_every_frozen_evidence_tier_without_sign_overreach():
    final = {
        (family, optimizer): 0.4 + 0.01 * index
        for family in ("dense", "late")
        for index, optimizer in enumerate(("adamw", "muon", "normuon"))
    }

    headlines = build_headline_macros(final_medians=final, **_rows())

    assert set(headlines) == {
        "DiscoveryHeadline",
        "CommonStateHeadline",
        "RepresentationHeadline",
        "InterventionHeadline",
        "ConfirmationHeadline",
    }
    assert "0.4000/0.4100/0.4200" in headlines["DiscoveryHeadline"]
    assert "shared-gradient common states" in headlines["CommonStateHeadline"]
    assert "descriptive" in headlines["RepresentationHeadline"]
    assert "explicitly post-hoc loss diagnostic" in headlines["RepresentationHeadline"]
    assert "-0.250/-0.250" in headlines["RepresentationHeadline"]
    assert "hybrid-routing" in headlines["InterventionHeadline"]
    assert "positive" in headlines["ConfirmationHeadline"]
    assert "negative" in headlines["ConfirmationHeadline"]
    assert "inconclusive" in headlines["ConfirmationHeadline"]
    assert all("ResultPending" not in value for value in headlines.values())


def test_confirmation_headline_uses_familywise_not_nominal_interval():
    final = {
        (family, optimizer): 0.4
        for family in ("dense", "late")
        for optimizer in ("adamw", "muon", "normuon")
    }
    rows = _rows()
    muon_adamw = next(
        row for row in rows["confirmation_rows"] if row[0] == "DenseOn" and row[1] == "Muon - AdamW"
    )
    muon_adamw[3] = "[0.0010, 0.0190]"
    muon_adamw[4] = "[-0.0010, 0.0210]"

    headline = build_headline_macros(final_medians=final, **rows)["ConfirmationHeadline"]

    assert "Muon--AdamW 0.0100 [-0.0010, 0.0210] (inconclusive)" in headline


def test_headline_replacement_changes_only_the_five_declared_macros():
    headlines = {
        name: f"generated {name}"
        for name in (
            "DiscoveryHeadline",
            "CommonStateHeadline",
            "RepresentationHeadline",
            "InterventionHeadline",
            "ConfirmationHeadline",
        )
    }
    original = (
        "header\n"
        + "\n".join(f"\\newcommand{{\\{name}}}{{pending}}" for name in headlines)
        + "\nfooter\n"
    )

    rendered = _replace_headlines(original, headlines)

    assert rendered.startswith("header\n")
    assert rendered.endswith("footer\n")
    assert "pending" not in rendered
    assert all(f"{{{value}}}" in rendered for value in headlines.values())


def test_result_tables_cover_all_frozen_groups_and_contrasts():
    final = {
        (family, optimizer): 0.4 + 0.01 * index
        for family in ("dense", "late")
        for index, optimizer in enumerate(("adamw", "muon", "normuon"))
    }
    rows = _rows()
    rows.pop("correlation_rows")
    tail_discovery, tail_final = _tail_result_rows()
    spectral_factorial, spectral_tail = _spectral_result_rows()

    tables = build_result_tables(
        final_medians=final,
        task_rows=_task_rows(),
        task_stability_rows=_task_stability_rows(),
        basis_rows=_basis_result_rows(),
        tail_discovery_rows=tail_discovery,
        tail_final_rows=tail_final,
        spectral_factorial_rows=spectral_factorial,
        spectral_tail_rows=spectral_tail,
        system_rows=_system_result_rows(),
        **rows,
    )

    assert set(tables) == {
        "paper/generated/discovery.tex",
        "paper/generated/per-task.tex",
        "paper/generated/common-state.tex",
        "paper/generated/representation.tex",
        "paper/generated/intervention.tex",
        "paper/generated/confirmation.tex",
        "paper/generated/diagnostics.tex",
    }
    discovery = tables["paper/generated/discovery.tex"]
    per_task = tables["paper/generated/per-task.tex"]
    confirmation = tables["paper/generated/confirmation.tex"]
    common = tables["paper/generated/common-state.tex"]
    intervention = tables["paper/generated/intervention.tex"]
    diagnostics = tables["paper/generated/diagnostics.tex"]
    assert (
        sum(
            f"{family} & {optimizer}" in discovery
            for family in ("DenseOn", "LateOn")
            for optimizer in ("AdamW", "Muon", "NorMuon")
        )
        == 6
    )
    assert per_task.count("0.4000 & 0.4100 & 0.4200") == 28
    assert "test-selected comparisons" in per_task
    assert "Post-hoc adjacent-checkpoint stability" in per_task
    assert "12/14 & 0.900 & 0.800" in per_task
    assert (
        sum(
            f"{family} & {contrast}" in confirmation
            for family in ("DenseOn", "LateOn")
            for contrast in ("Muon - AdamW", "NorMuon - AdamW", "NorMuon - Muon")
        )
        == 6
    )
    assert "FWER 95\\% CI" in confirmation
    assert "Bonferroni correction" in confirmation
    assert r"\scriptsize" in confirmation
    assert r"\setlength{\tabcolsep}{2pt}" in confirmation
    assert "Function-preserving basis sensitivity" not in common
    assert "0.99000 & 0.01000 & 0.00100 & 0.00200 & 0.00300" in diagnostics
    assert "Post-hoc spectrum-versus-basis causal decomposition" not in intervention
    assert "Post-hoc spectrum-versus-basis causal decomposition" in diagnostics
    assert "tail redistribution" in diagnostics
    assert "cannot establish BEIR mediation" in diagnostics
    assert "DenseOn Muon/NorMuon throughput ratios were 0.95x/0.93x AdamW" in diagnostics
    assert "neither was faster for DenseOn" in diagnostics
    assert "dense-training-dynamics-by-run.png" in discovery
    assert "dense-lr-sensitivity.png" in discovery
    assert all("ResultPending" not in content for content in tables.values())


def test_dense_scope_headlines_and_tables_exclude_late_results():
    families = ("dense",)
    final = {
        ("dense", optimizer): 0.4 + 0.01 * index
        for index, optimizer in enumerate(("adamw", "muon", "normuon"))
    }
    rows = {
        name: [row for row in values if row[0] == "DenseOn"] for name, values in _rows().items()
    }
    headlines = build_headline_macros(
        final_medians=final,
        families=families,
        **rows,
    )
    table_rows = dict(rows)
    table_rows.pop("correlation_rows")
    tail_discovery, tail_final = _tail_result_rows()
    spectral_factorial, spectral_tail = _spectral_result_rows()
    tables = build_result_tables(
        final_medians=final,
        task_rows=[row for row in _task_rows() if row[0] == "DenseOn"],
        task_stability_rows=[row for row in _task_stability_rows() if row[0] == "DenseOn"],
        basis_rows=[row for row in _basis_result_rows() if row[0] == "DenseOn"],
        tail_discovery_rows=[row for row in tail_discovery if row[0] == "DenseOn"],
        tail_final_rows=[row for row in tail_final if row[0] == "DenseOn"],
        spectral_factorial_rows=[row for row in spectral_factorial if row[0] == "DenseOn"],
        spectral_tail_rows=[row for row in spectral_tail if row[0] == "DenseOn"],
        system_rows=[row for row in _system_result_rows() if row[0] == "DenseOn"],
        families=families,
        **table_rows,
    )

    assert all("LateOn" not in value for value in headlines.values())
    assert "for DenseOn" in headlines["RepresentationHeadline"]
    assert all("LateOn" not in value for value in tables.values())
    assert tables["paper/generated/discovery.tex"].count("DenseOn &") == 3
    assert tables["paper/generated/confirmation.tex"].count("DenseOn &") == 3
    assert "all six comparisons prespecified" in tables["paper/generated/confirmation.tex"]
    assert "Late token coverage" not in tables["paper/generated/representation.tex"]
    rendered = "\n".join(tables.values())
    assert set(re.findall(r"\\label\{((?:tab|fig):[^{}]+)\}", rendered)) == {
        "tab:discovery-results",
        "fig:dense-discovery-dynamics",
        "tab:denseon-per-task-results",
        "tab:task-delta-stability",
        "tab:common-state-results",
        "tab:representation-results",
        "tab:intervention-results",
        "tab:confirmation-results",
        "tab:training-systems-results",
        "tab:basis-sensitivity-results",
        "tab:tail-identity-results",
        "tab:tail-persistence-results",
        "tab:spectral-factorial-results",
        "tab:spectral-tail-results",
    }
    assert _rendered_table_rows(rendered) == {
        "tab:discovery-results": 3,
        "tab:denseon-per-task-results": 14,
        "tab:task-delta-stability": 8,
        "tab:common-state-results": 2,
        "tab:representation-results": 3,
        "tab:intervention-results": 1,
        "tab:confirmation-results": 3,
        "tab:training-systems-results": 3,
        "tab:basis-sensitivity-results": 3,
        "tab:tail-identity-results": 2,
        "tab:tail-persistence-results": 2,
        "tab:spectral-factorial-results": 2,
        "tab:spectral-tail-results": 3,
    }


def test_complete_renderer_routes_per_task_rows_only_to_result_tables(tmp_path, monkeypatch):
    headline_names = (
        "DiscoveryHeadline",
        "CommonStateHeadline",
        "RepresentationHeadline",
        "InterventionHeadline",
        "ConfirmationHeadline",
    )
    results = tmp_path / "paper/results.tex"
    results.parent.mkdir(parents=True)
    results.write_text(
        "\n".join(f"\\newcommand{{\\{name}}}{{pending}}" for name in headline_names)
        + "\n\\newcommand{\\ResultConclusion}{pending}\n"
    )
    (tmp_path / "reports").mkdir()
    dynamics_dir = tmp_path / "reports/dense-retrieval-dynamics"
    dynamics_dir.mkdir()
    for name in (
        "summary_manifest.json",
        "five_stage_retrieval_dynamics.csv",
        "five_stage_retrieval_dynamics.svg",
        "five_stage_retrieval_dynamics.pdf",
    ):
        (dynamics_dir / name).write_text(f"{name}\n", encoding="utf-8")
    claim = tmp_path / "claim.json"
    claim.write_text("{}")
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n")
    task_rows = _task_rows()
    captured = {}
    causal_evidence = {
        "complete": True,
        "source_table_records": [{"path": str(source)} for _ in range(5)],
        "temporal_short_branch": {
            "manifest": {"path": str(source)},
            "status": "supported",
            "claimable": True,
            "supported": True,
            "claim_boundary": "temporal boundary",
        },
        "dose_band": {
            "manifest": {"path": str(source)},
            "status": "negative",
            "claimable": True,
            "supported": False,
            "claim_boundary": "dose boundary",
        },
    }

    monkeypatch.setattr(
        "embed_optim.paper_results.load_causal_chain_evidence",
        lambda *_args, **_kwargs: causal_evidence,
    )
    monkeypatch.setattr(
        "embed_optim.paper_results.render_causal_chain_latex",
        lambda _evidence: "generated causal table\n",
    )
    monkeypatch.setattr(
        "embed_optim.paper_results.causal_chain_display_contract",
        lambda _evidence: {"complete": True, "source_tables": 5},
    )
    monkeypatch.setattr(
        "embed_optim.paper_results.render_causal_chain_headline_fragment",
        lambda _evidence: " causal headline",
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._discovery_figure_contract",
        lambda _root: {"source_manifest": {}, "panels": []},
    )
    monkeypatch.setattr("embed_optim.paper_results.load_publication_rows", lambda _root: ([], {}))
    monkeypatch.setattr(
        "embed_optim.paper_results.summarize_publication_rows",
        lambda _rows: [["Confirmatory AdamW", "3", "0.1", "0.2", "0.3", "0.4", "0.5"]],
    )
    monkeypatch.setattr(
        "embed_optim.paper_results.render_publication_latex",
        lambda _rows: "generated extension latex\n",
    )
    monkeypatch.setattr(
        "embed_optim.paper_results.PAPER_SOURCE_TABLE_PATHS",
        (
            *(source.relative_to(tmp_path) for _ in range(5)),
            (dynamics_dir / "five_stage_retrieval_dynamics.csv").relative_to(tmp_path),
            *(source.relative_to(tmp_path) for _ in range(18)),
        ),
    )

    monkeypatch.setattr(
        "embed_optim.paper_results.audit_paper",
        lambda **_kwargs: {"incomplete_evidence": [], "evidence": {}},
    )
    monkeypatch.setattr(
        "embed_optim.paper_results.load_paper_claim_protocol",
        lambda **_kwargs: (claim, {"status": "frozen", "frozen_at": "now"}, []),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results.expected_constant_macros",
        lambda *_args, **_kwargs: ({}, []),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._retrieval_rows",
        lambda *_args: ([], {}, source, source),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._discovery_final_medians",
        lambda *_args: ({}, source),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._discovery_task_rows",
        lambda *_args: (task_rows, source),
    )
    task_stability_rows = _task_stability_rows()
    monkeypatch.setattr(
        "embed_optim.paper_results._discovery_task_stability_rows",
        lambda *_args: (task_stability_rows, source),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._common_state_rows",
        lambda *_args: ([], {}, source),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._basis_rows",
        lambda *_args: (_basis_result_rows(), {}, source),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._spectrum_rows",
        lambda *_args: ([], {}, source),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._bridge_rows",
        lambda *_args: ([], [], {}, [source, source]),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._functional_rows",
        lambda *_args: ([], source, {}),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._hybrid_rows",
        lambda *_args: ([], source, {}),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._short_branch_rows",
        lambda *_args: ([], source, {}),
    )
    tail_discovery, tail_final = _tail_result_rows()
    monkeypatch.setattr(
        "embed_optim.paper_results._tail_stability_rows",
        lambda *_args: (tail_discovery, tail_final, (source, source), {}),
    )
    spectral_factorial, spectral_tail = _spectral_result_rows()
    monkeypatch.setattr(
        "embed_optim.paper_results._spectral_transplant_rows",
        lambda *_args: (spectral_factorial, spectral_tail, (source, source), {}),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._confirmation_rows",
        lambda *_args: ([], source, {}),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results._training_system_rows",
        lambda *_args: ([], source),
    )
    monkeypatch.setattr(
        "embed_optim.paper_results.build_final_conclusion_contract",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "plain": "final conclusion",
            "markdown": "final conclusion",
        },
    )

    def headlines(**kwargs):
        captured["headline_keys"] = set(kwargs)
        return {name: f"rendered {name}" for name in headline_names}

    def tables(**kwargs):
        captured["table_task_rows"] = kwargs["task_rows"]
        captured["table_task_stability_rows"] = kwargs["task_stability_rows"]
        return {
            path: f"generated {path}\n"
            for path in (
                "paper/generated/discovery.tex",
                "paper/generated/per-task.tex",
                "paper/generated/common-state.tex",
                "paper/generated/representation.tex",
                "paper/generated/intervention.tex",
                "paper/generated/confirmation.tex",
                "paper/generated/diagnostics.tex",
            )
        }

    monkeypatch.setattr("embed_optim.paper_results.build_headline_macros", headlines)
    monkeypatch.setattr("embed_optim.paper_results.build_result_tables", tables)

    manifest = render_paper_results(repo_root=tmp_path)

    assert "task_rows" not in captured["headline_keys"]
    assert captured["table_task_rows"] is task_rows
    assert captured["table_task_stability_rows"] is task_stability_rows
    assert len(manifest["source_tables"]) == 24
    assert manifest["dynamics_extension"]["role"] == "descriptive-only"
    assert len(manifest["result_tables"]) == 8
    assert manifest["causal_chain_display"] == {"complete": True, "source_tables": 5}

    mutation_targets = [results, tmp_path / DYNAMICS_EXTENSION_TEX]
    mutation_targets.extend(tmp_path / relative for relative in PAPER_RESULT_TABLE_PATHS)
    before = {path: path.read_bytes() for path in mutation_targets}
    render_paper_results(repo_root=tmp_path)
    assert {path: path.read_bytes() for path in mutation_targets} == before


def test_latex_escape_protects_generated_data_cells():
    assert _latex_escape("a_b&c%#${}~^\\") == (
        r"a\_b\&c\%\#\$\{\}\textasciitilde{}\textasciicircum{}\textbackslash{}"
    )


def test_interval_classification_uses_the_frozen_zero_boundary():
    assert _ci_classification("[0.0010, 0.1000]") == "positive"
    assert _ci_classification("[-0.1000, -0.0010]") == "negative"
    assert _ci_classification("[-0.0010, 0.1000]") == "inconclusive"
    assert _ci_classification("[0.0000, 0.1000]") == "inconclusive"
    with pytest.raises(ValueError, match="Reversed"):
        _ci_classification("[0.1000, -0.1000]")


def test_if_ready_only_suppresses_incomplete_evidence(monkeypatch, capsys):
    def incomplete(**_kwargs):
        raise IncompletePaperEvidenceError("missing frozen tier")

    monkeypatch.setattr("embed_optim.paper_results.render_paper_results", incomplete)

    main(["--if-ready"])

    assert "retaining audited draft headlines" in capsys.readouterr().out


def test_pending_causal_evidence_causes_no_paper_writes(tmp_path, monkeypatch):
    targets = [tmp_path / "paper/results.tex", tmp_path / "reports/paper-results.manifest.json"]
    targets.extend(tmp_path / relative for relative in PAPER_RESULT_TABLE_PATHS)
    before = {}
    for index, path in enumerate(targets):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sentinel {index}\n", encoding="utf-8")
        before[path] = path.read_bytes()
    monkeypatch.setattr(
        "embed_optim.paper_results.load_causal_chain_evidence",
        lambda *_args, **_kwargs: {"complete": False},
    )

    with pytest.raises(IncompletePaperEvidenceError, match="causal-chain evidence"):
        render_paper_results(repo_root=tmp_path)

    assert {path: path.read_bytes() for path in targets} == before


def test_malformed_complete_causal_evidence_causes_no_paper_writes(tmp_path, monkeypatch):
    targets = [tmp_path / "paper/results.tex", tmp_path / "reports/paper-results.manifest.json"]
    targets.extend(tmp_path / relative for relative in PAPER_RESULT_TABLE_PATHS)
    before = {}
    for index, path in enumerate(targets):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"sentinel {index}\n", encoding="utf-8")
        before[path] = path.read_bytes()

    def malformed(*_args, **_kwargs):
        raise ValueError("malformed complete causal evidence")

    monkeypatch.setattr("embed_optim.paper_results.load_causal_chain_evidence", malformed)

    with pytest.raises(ValueError, match="malformed complete causal evidence"):
        render_paper_results(repo_root=tmp_path)

    assert {path: path.read_bytes() for path in targets} == before


def test_default_cli_and_if_ready_do_not_suppress_other_failures(monkeypatch):
    def incomplete(**_kwargs):
        raise IncompletePaperEvidenceError("missing frozen tier")

    monkeypatch.setattr("embed_optim.paper_results.render_paper_results", incomplete)
    with pytest.raises(IncompletePaperEvidenceError, match="missing frozen tier"):
        main([])

    def malformed(**_kwargs):
        raise ValueError("stale complete evidence")

    monkeypatch.setattr("embed_optim.paper_results.render_paper_results", malformed)
    with pytest.raises(ValueError, match="stale complete evidence"):
        main(["--if-ready"])
