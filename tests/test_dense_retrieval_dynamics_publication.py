from __future__ import annotations

from copy import deepcopy

import pytest

from embed_optim.dense_retrieval_dynamics_publication import (
    render_publication_latex,
    render_publication_markdown,
    summarize_publication_rows,
)


def _rows() -> list[dict[str, str]]:
    rows = []
    groups = (
        ("hybrid", "hybrid_adamw", [(42, f"hybrid-{index}") for index in range(4)]),
        (
            "confirmatory",
            "adamw",
            [(seed, "adamw-selected") for seed in (314159, 271828, 161803)],
        ),
        (
            "confirmatory",
            "muon",
            [(seed, "muon-selected") for seed in (314159, 271828, 161803)],
        ),
        (
            "confirmatory",
            "normuon",
            [(seed, "normuon-selected") for seed in (314159, 271828, 161803)],
        ),
    )
    for suite, optimizer, members in groups:
        for seed, run_id in members:
            for stage in range(1, 6):
                rows.append(
                    {
                        "suite": suite,
                        "model_family": "dense",
                        "optimizer": optimizer,
                        "run_id": run_id,
                        "training_seed": str(seed),
                        "stage": str(stage),
                        "fraction": str(stage / 5),
                        "tasks_completed": "14",
                        "mean_ndcg_at_10": str(0.5 + stage / 100),
                        "source_partition": (
                            "formal-stage5" if stage == 5 else "dynamics-stage1-4"
                        ),
                        "formal_source_stage5": "True" if stage == 5 else "False",
                        "joined_summary_role": "descriptive-only",
                        "joined_summary_used_for_formal_inference": "False",
                    }
                )
    return rows


def test_publication_accepts_same_run_id_across_three_distinct_confirmatory_seeds():
    summary = summarize_publication_rows(_rows())

    assert len(summary) == 4
    assert summary[0][:2] == ["Hybrid AdamW", "4"]
    assert summary[1][:2] == ["Confirmatory AdamW", "3"]
    assert all(row[2:] == ["0.5100", "0.5200", "0.5300", "0.5400", "0.5500"] for row in summary)
    markdown = render_publication_markdown(summary)
    latex = render_publication_latex(summary)
    assert "five_stage_retrieval_dynamics.csv" in markdown
    assert "five_stage_retrieval_dynamics.svg" in markdown
    assert "Descriptive time-to-quality" in markdown
    assert "displayed four-decimal fixed reference" in markdown
    assert "AdamW 100%; Muon 100%; NorMuon 100%" in markdown
    assert "AdamW 0.5300, Muon 0.5300, NorMuon 0.5300" in markdown
    assert "neither the CSV nor either figure is an inference input" in markdown
    assert "five_stage_retrieval_dynamics.pdf" in latex
    assert "not an inference input" in latex
    assert r"\paragraph{Descriptive time-to-quality.}" in latex
    assert latex.count(r"\label{fig:extended-retrieval-dynamics}") == 1
    assert latex.count(r"\label{tab:extended-retrieval-dynamics}") == 1
    assert all(line == line.rstrip() for line in latex.splitlines())
    body = latex.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    assert sum(line.rstrip().endswith(r"\\") for line in body.splitlines()) == 4


def test_publication_rejects_non_distinct_confirmatory_seed_or_boundary_drift():
    rows = _rows()
    forged = deepcopy(rows)
    for row in forged:
        if row["optimizer"] == "adamw" and row["training_seed"] == "271828":
            row["training_seed"] = "314159"
    with pytest.raises(ValueError, match="distinct runs/seeds|violates"):
        summarize_publication_rows(forged)

    drift = deepcopy(rows)
    drift[0]["joined_summary_used_for_formal_inference"] = "True"
    with pytest.raises(ValueError, match="descriptive boundary"):
        summarize_publication_rows(drift)


def test_publication_reports_unreached_adamw_final_reference_descriptively():
    summary = [
        ["Hybrid AdamW", "4", "0.5000", "0.5200", "0.5400", "0.5600", "0.5800"],
        [
            "Confirmatory AdamW",
            "3",
            "0.5500",
            "0.5800",
            "0.6000",
            "0.6100",
            "0.6050",
        ],
        ["Confirmatory Muon", "3", "0.5000", "0.5200", "0.5400", "0.5600", "0.5900"],
        ["Confirmatory NorMuon", "3", "0.5100", "0.5300", "0.5500", "0.5700", "0.6000"],
    ]

    markdown = render_publication_markdown(summary)
    latex = render_publication_latex(summary)

    assert "AdamW 80%; Muon not reached; NorMuon not reached" in markdown
    assert "stage-5 confirmatory AdamW nDCG@10 (0.6050)" in markdown
    assert r"AdamW 80\%; Muon not reached; NorMuon not reached" in latex
