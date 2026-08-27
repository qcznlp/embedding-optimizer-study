from __future__ import annotations

import pytest

from embed_optim.paper_results import (
    IncompletePaperEvidenceError,
    _ci_classification,
    _latex_escape,
    _replace_headlines,
    build_headline_macros,
    build_result_tables,
    main,
    render_paper_results,
)


def _rows():
    families = ("DenseOn", "LateOn")
    optimizers = ("AdamW", "Muon", "NorMuon")
    retrieval = [
        [family, optimizer, "0.4000", f"{index + 1}/4", "1.000", "2.000", str(3 - index)]
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
        [family, optimizer, "0.1000", f"0.{index + 2}000", "0.5000", "0.8000", "0.3000", "0.4000"]
        for family in families
        for index, optimizer in enumerate(optimizers)
    ]
    correlations = [
        [family, "unseen margin", "mean BEIR nDCG@10", "48", "0.500"] for family in families
    ]
    functional = [
        [family, optimizer, "descent", "-0.0100", f"0.0{index + 1}00", "0.0200", "0.0300", "0.80"]
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
        [family, contrast, "-0.0100 (3/0/0)", "0.0100 (3/0/0)", "0.0100 (3/0/0)", "0.0100 (3/0/0)"]
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

    tables = build_result_tables(final_medians=final, task_rows=_task_rows(), **rows)

    assert set(tables) == {
        "paper/generated/discovery.tex",
        "paper/generated/per-task.tex",
        "paper/generated/common-state.tex",
        "paper/generated/representation.tex",
        "paper/generated/intervention.tex",
        "paper/generated/confirmation.tex",
    }
    discovery = tables["paper/generated/discovery.tex"]
    per_task = tables["paper/generated/per-task.tex"]
    confirmation = tables["paper/generated/confirmation.tex"]
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
    assert all("ResultPending" not in content for content in tables.values())


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
        "\n".join(f"\\newcommand{{\\{name}}}{{pending}}" for name in headline_names) + "\n"
    )
    (tmp_path / "reports").mkdir()
    claim = tmp_path / "claim.json"
    claim.write_text("{}")
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n")
    task_rows = _task_rows()
    captured = {}

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
    monkeypatch.setattr(
        "embed_optim.paper_results._common_state_rows",
        lambda *_args: ([], {}, source),
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
    monkeypatch.setattr(
        "embed_optim.paper_results._confirmation_rows",
        lambda *_args: ([], source, {}),
    )

    def headlines(**kwargs):
        captured["headline_keys"] = set(kwargs)
        return {name: f"rendered {name}" for name in headline_names}

    def tables(**kwargs):
        captured["table_task_rows"] = kwargs["task_rows"]
        return {
            path: f"generated {path}\n"
            for path in (
                "paper/generated/discovery.tex",
                "paper/generated/per-task.tex",
                "paper/generated/common-state.tex",
                "paper/generated/representation.tex",
                "paper/generated/intervention.tex",
                "paper/generated/confirmation.tex",
            )
        }

    monkeypatch.setattr("embed_optim.paper_results.build_headline_macros", headlines)
    monkeypatch.setattr("embed_optim.paper_results.build_result_tables", tables)

    manifest = render_paper_results(repo_root=tmp_path)

    assert "task_rows" not in captured["headline_keys"]
    assert captured["table_task_rows"] is task_rows
    assert len(manifest["source_tables"]) == 11
    assert len(manifest["result_tables"]) == 6


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
