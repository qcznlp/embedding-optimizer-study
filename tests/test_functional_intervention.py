from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from embed_optim.common_state_matrix import CommonStateJob
from embed_optim.functional_intervention import (
    group_scores,
    intervention_conditions,
    load_intervention_protocol,
    score_metrics,
)
from embed_optim.functional_intervention_matrix import (
    FunctionalInterventionJob,
    _job_cli,
    functional_intervention_job_complete,
    parse_args,
)
from embed_optim.functional_intervention_matrix import main as intervention_matrix_main
from embed_optim.functional_intervention_summary import (
    METRICS,
    _anchor_effects,
    _family_summary,
    _optimizer_contrasts,
    summarize_functional_interventions,
)
from embed_optim.functional_intervention_summary import parse_args as parse_summary_args
from embed_optim.geometry import _sha256


class _EmbeddingModel(nn.Module):
    def forward(self, features):
        return {"sentence_embedding": features["embedding"]}


def test_dense_group_scores_and_metrics_match_positive_first_contract():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    documents = [query.clone()]
    documents.extend(torch.tensor([[0.0, 1.0], [1.0, 0.0]]) for _ in range(7))
    scores = group_scores(
        _EmbeddingModel(),
        [{"embedding": query}, *({"embedding": value} for value in documents)],
        "dense",
    )
    metrics = score_metrics(scores, temperature=0.02)

    assert scores.shape == (2, 8)
    torch.testing.assert_close(metrics["positive_margin"], torch.ones(2))
    torch.testing.assert_close(metrics["reciprocal_rank"], torch.ones(2))
    torch.testing.assert_close(metrics["top1_accuracy"], torch.ones(2))
    assert torch.all(metrics["contrastive_loss"] < 1e-10)


def test_score_metrics_counts_ties_conservatively():
    scores = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    metrics = score_metrics(scores, temperature=1.0)

    assert metrics["reciprocal_rank"].item() == pytest.approx(0.5)
    assert metrics["top1_accuracy"].item() == 0.0
    assert metrics["positive_margin"].item() == 0.0


def test_frozen_functional_intervention_protocol_is_self_consistent():
    path, spec = load_intervention_protocol("configs/functional_intervention.json")
    conditions = intervention_conditions(spec)

    assert path.name == "functional_intervention.json"
    assert len(conditions) == 13
    assert conditions[0].condition == "baseline"
    assert sum(condition.direction == "descent" for condition in conditions) == 9
    assert sum(condition.direction == "sign_reversal" for condition in conditions) == 3
    assert spec["freeze_context"]["strict_beir_valid_units"] == 144
    assert spec["freeze_context"]["formal_common_state_outputs_visible"] is False
    assert "does not by itself" in spec["claim_boundary"]
    assert _sha256(Path(spec["common_state"]["spec"])) == spec["common_state"]["spec_sha256"]
    assert (
        _sha256(Path(spec["evaluation_probe"]["spec"])) == spec["evaluation_probe"]["spec_sha256"]
    )


def _declared(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_intervention_completion_audits_inputs_and_outputs(tmp_path: Path):
    spec_path, spec = load_intervention_protocol("configs/functional_intervention.json")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    common_root = tmp_path / "common"
    update_dir = common_root / "updates"
    gradient_dir = common_root / "gradients"
    update_dir.mkdir(parents=True)
    gradient_dir.mkdir()
    update_manifest = update_dir / "manifest.json"
    update_manifest.write_text('{"fixture":true}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sample = output_dir / "sample_metrics.jsonl"
    conditions = output_dir / "condition_metrics.jsonl"
    sample.write_text('{"sample":true}\n', encoding="utf-8")
    conditions.write_text('{"condition":true}\n', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "family": "dense",
        "checkpoint": {"path": str(checkpoint.resolve())},
        "intervention_spec": {"sha256": _sha256(spec_path)},
        "common_state_updates": {"sha256": _sha256(update_manifest)},
        "sample_records": spec["intervention"]["expected_sample_records_per_anchor"],
        "condition_records": spec["intervention"]["expected_conditions_per_anchor"],
        "outputs": {
            "sample_metrics": _declared(sample, output_dir),
            "condition_metrics": _declared(conditions, output_dir),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    common_job = CommonStateJob(
        family="dense",
        label="dense/pretrained",
        checkpoint=checkpoint.resolve(),
        gradient_dir=gradient_dir,
        update_dir=update_dir,
        gradient_steps=8,
        hidden_tensors=88,
        hidden_parameters=110_297_088,
    )
    job = FunctionalInterventionJob(common_job, output_dir)

    assert functional_intervention_job_complete(job, spec_path, verify_hashes=True)
    sample.write_text("changed\n", encoding="utf-8")
    assert not functional_intervention_job_complete(job, spec_path)


def test_worker_command_round_trips(tmp_path: Path):
    import sys
    from argparse import Namespace

    common = CommonStateJob(
        family="late",
        label="late/pretrained",
        checkpoint=(tmp_path / "checkpoint").resolve(),
        gradient_dir=(tmp_path / "gradients").resolve(),
        update_dir=(tmp_path / "updates").resolve(),
        gradient_steps=8,
        hidden_tensors=88,
        hidden_parameters=110_297_088,
    )
    job = FunctionalInterventionJob(common, (tmp_path / "output").resolve())
    command = _job_cli(
        job,
        Namespace(
            probe=tmp_path / "probe",
            intervention_spec=tmp_path / "intervention.json",
        ),
    )

    assert command[:3] == [sys.executable, "-m", "embed_optim.functional_intervention_matrix"]
    parsed = parse_args(command[3:])
    assert parsed.worker is True
    assert parsed.family == "late"
    assert parsed.label == "late/pretrained"
    assert parsed.output_dir == job.output_dir
    defaults = parse_args([])
    assert defaults.families == ["dense"]
    assert defaults.scope_amendment is None
    assert parse_args(["--families", "dense", "late"]).families == ["dense", "late"]
    summary_defaults = parse_summary_args([])
    assert summary_defaults.families == ["dense", "late"]
    assert summary_defaults.scope_amendment is None


def test_dense_intervention_matrix_builds_ten_authorized_jobs(monkeypatch):
    observed = {}

    def fake_common_jobs(configs, references, spec, output_root):
        observed["families"] = {config.model_family for config in configs}
        observed["references"] = set(references)
        return list(range(10))

    monkeypatch.setattr(
        "embed_optim.functional_intervention_matrix.build_common_state_jobs",
        fake_common_jobs,
    )
    monkeypatch.setattr(
        "embed_optim.functional_intervention_matrix.build_functional_intervention_jobs",
        lambda jobs, output_root: jobs,
    )

    def fake_run_matrix(jobs, args):
        observed["jobs"] = len(jobs)
        return 0

    monkeypatch.setattr("embed_optim.functional_intervention_matrix.run_matrix", fake_run_matrix)

    intervention_matrix_main(
        [
            "--families",
            "dense",
            "--scope-amendment",
            "configs/dense_scope_amendment.json",
            "--dry-run",
        ]
    )

    assert observed == {"families": {"dense"}, "references": {"dense"}, "jobs": 10}


def test_dense_intervention_matrix_requires_scope_amendment():
    with pytest.raises(ValueError, match="requires --scope-amendment"):
        intervention_matrix_main(["--dry-run"])


def _sample_record(condition, sample_id, loss, margin):
    return {
        "condition": condition.condition,
        "algorithm": condition.algorithm,
        "direction": condition.direction,
        "relative_scale": condition.relative_scale,
        "sample_id": sample_id,
        "group": "task",
        "contrastive_loss": loss,
        "positive_score": margin + 1.0,
        "hardest_negative_score": 1.0,
        "positive_margin": margin,
        "reciprocal_rank": 1.0 if margin > 0 else 0.5,
        "top1_accuracy": 1.0 if margin > 0 else 0.0,
    }


def test_summary_uses_paired_baseline_and_adamw_contrasts():
    _, spec = load_intervention_protocol("configs/functional_intervention.json")
    conditions = intervention_conditions(spec)
    records = []
    for condition in conditions:
        algorithm_bonus = {None: 0.0, "adamw": 0.1, "muon": 0.2, "normuon": 0.3}[
            condition.algorithm
        ]
        for sample_id in (10, 20):
            records.append(
                _sample_record(
                    condition,
                    sample_id,
                    loss=1.0 - algorithm_bonus,
                    margin=algorithm_bonus,
                )
            )
    effects, indexed = _anchor_effects(
        "dense/pretrained",
        "dense",
        records,
        [condition.condition for condition in conditions],
        2,
    )
    contrasts = _optimizer_contrasts("dense/pretrained", "dense", conditions, indexed)
    family = _family_summary(effects)

    muon = next(
        row for row in effects if row["algorithm"] == "muon" and row["direction"] == "descent"
    )
    assert muon["delta_contrastive_loss"] == pytest.approx(-0.2)
    assert muon["delta_positive_margin"] == pytest.approx(0.2)
    muon_vs_adam = next(
        row for row in contrasts if row["challenger"] == "muon" and row["direction"] == "descent"
    )
    assert muon_vs_adam["delta_delta_contrastive_loss"] == pytest.approx(-0.1)
    assert len(family) == 12
    assert all(metric in METRICS for metric in METRICS)


def test_dense_intervention_summary_has_half_frozen_anchor_matrix(tmp_path: Path, monkeypatch):
    spec = json.loads(Path("configs/functional_intervention.json").read_text(encoding="utf-8"))
    spec["evaluation_probe"]["count"] = 2
    spec["intervention"]["expected_sample_records_per_anchor"] = 26
    spec_path = tmp_path / "functional-intervention.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True) + "\n", encoding="utf-8")
    conditions = intervention_conditions(spec)
    records = [
        _sample_record(condition, sample_id, loss=1.0, margin=0.1)
        for condition in conditions
        for sample_id in (10, 20)
    ]
    jobs = []
    for index in range(10):
        label = f"dense/anchor-{index}"
        output_dir = tmp_path / "results" / label
        output_dir.mkdir(parents=True)
        sample_path = output_dir / "sample_metrics.jsonl"
        sample_path.write_text("{}\n", encoding="utf-8")
        (output_dir / "manifest.json").write_text(
            json.dumps({"outputs": {"sample_metrics": {"path": sample_path.name}}}) + "\n",
            encoding="utf-8",
        )
        common = CommonStateJob(
            family="dense",
            label=label,
            checkpoint=(tmp_path / "checkpoint" / str(index)).resolve(),
            gradient_dir=tmp_path / "common" / label / "gradients",
            update_dir=tmp_path / "common" / label / "updates",
            gradient_steps=8,
            hidden_tensors=88,
            hidden_parameters=110_297_088,
        )
        jobs.append(FunctionalInterventionJob(common, output_dir))
    monkeypatch.setattr(
        "embed_optim.functional_intervention_summary.functional_intervention_job_complete",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "embed_optim.functional_intervention_summary._read_jsonl",
        lambda path: records,
    )

    manifest = summarize_functional_interventions(
        jobs,
        tmp_path / "summary",
        intervention_spec=spec_path,
        families=("dense",),
        scope_amendment="configs/dense_scope_amendment.json",
    )

    assert manifest["families"] == ["dense"]
    assert manifest["anchors"] == 10
    assert manifest["anchor_effect_records"] == 120
    assert manifest["optimizer_contrast_records"] == 80
    assert manifest["family_summary_records"] == 12
    assert manifest["scope_amendment"]["status"] == ("user_directed_post_hoc_scope_amendment")


def test_dense_intervention_summary_requires_scope_amendment(tmp_path: Path):
    with pytest.raises(ValueError, match="requires --scope-amendment"):
        summarize_functional_interventions(
            [],
            tmp_path / "summary",
            intervention_spec="configs/functional_intervention.json",
            families=("dense",),
        )
