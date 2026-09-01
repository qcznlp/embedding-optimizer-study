from __future__ import annotations

import json
from pathlib import Path

import pytest

from embed_optim import corpus_size_diagnostic
from embed_optim.corpus_size_diagnostic import (
    BLOG_MARKERS,
    _latex,
    _latex_appendix,
    _markdown,
    _permutation_pvalue,
    _render_blog,
    _spearman,
    build_diagnostic,
    load_protocol,
)


def test_protocol_discloses_post_hoc_timing_and_binds_sources() -> None:
    path, protocol = load_protocol("configs/corpus_size_diagnostic.json")

    assert path.name == "corpus_size_diagnostic.json"
    assert protocol["timing"] == {
        "discovery_beir_visible": True,
        "corpus_size_association_visible": True,
        "candidate_breadth_protocol_already_frozen": True,
        "candidate_breadth_data_or_scores_visible": False,
        "confirmatory_14_task_matrix_complete": False,
    }
    assert "cannot establish" in protocol["claim_boundary"]


def test_diagnostic_reconstructs_complete_selected_trajectories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        corpus_size_diagnostic,
        "_permutation_pvalue",
        lambda left, right, **_kwargs: (_spearman(left, right), 0.5),
    )

    task_rows, association_rows, summary = build_diagnostic("configs/corpus_size_diagnostic.json")

    assert len(task_rows) == 2 * 5 * 14
    assert len(association_rows) == 2 * 5
    assert {(row["optimizer"], row["stage"]) for row in association_rows} == {
        (optimizer, stage) for optimizer in ("muon", "normuon") for stage in range(1, 6)
    }
    assert summary["optimizer_summaries"]["muon"]["final_spearman_rho"] == pytest.approx(
        0.5868131868
    )
    assert summary["optimizer_summaries"]["normuon"]["final_spearman_rho"] == pytest.approx(
        0.5736263736
    )
    assert summary["optimizer_summaries"]["muon"]["large_corpus_half"]["positive_tasks"] == 7
    assert summary["optimizer_summaries"]["normuon"]["small_corpus_half"]["positive_tasks"] == 4


def test_permutation_test_is_deterministic_and_two_sided() -> None:
    left = [1.0, 2.0, 3.0, 4.0, 5.0]
    right = [1.0, 2.0, 3.0, 4.0, 5.0]

    first = _permutation_pvalue(left, right, permutations=5_000, seed=17)
    second = _permutation_pvalue(left, right, permutations=5_000, seed=17)

    assert first == second
    assert first[0] == pytest.approx(1.0)
    assert 0 < first[1] < 0.1


def test_publication_blocks_keep_exploratory_boundary_visible(tmp_path: Path) -> None:
    summary = json.loads(
        Path("reports/corpus-size-diagnostic/summary.json").read_text(encoding="utf-8")
    )
    markdown = _markdown(summary)
    latex = _latex(summary)
    appendix = _latex_appendix(summary)

    assert "post-hoc" in markdown.lower()
    assert "not evidence" in markdown
    assert "same-suite-selected" in latex
    assert "cannot distinguish corpus size" in latex
    assert "includegraphics" not in latex
    assert "includegraphics" in appendix

    begin, end = BLOG_MARKERS
    blog = tmp_path / "blog.md"
    blog.write_text(f"before\n{begin}\nold\n{end}\nafter\n", encoding="utf-8")
    _render_blog(blog, markdown, audit_only=False)
    _render_blog(blog, markdown, audit_only=True)
    assert blog.read_text(encoding="utf-8").count(begin) == 1


def test_protocol_rejects_changed_source_hash(tmp_path: Path) -> None:
    source = Path("configs/corpus_size_diagnostic.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["sources"]["evaluation_long"]["sha256"] = "0" * 64
    protocol = tmp_path / "configs" / source.name
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source differs"):
        load_protocol(protocol)


def test_tracked_text_outputs_use_canonical_line_endings() -> None:
    output = Path("reports/corpus-size-diagnostic")

    for name in ("stage_association.csv", "task_stage_deltas.csv"):
        payload = (output / name).read_bytes()
        assert b"\r" not in payload
    svg_lines = (output / "corpus_size_association.svg").read_text(encoding="utf-8").splitlines()
    assert all(line == line.rstrip() for line in svg_lines)
