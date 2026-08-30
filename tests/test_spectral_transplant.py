from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from embed_optim.common_state_matrix import CommonStateJob
from embed_optim.geometry import _sha256
from embed_optim.spectral_transplant import (
    _band_slice,
    construct_spectral_transplants,
    load_spectral_transplant_protocol,
    spectral_conditions,
)
from embed_optim.spectral_transplant_matrix import (
    SpectralTransplantJob,
    _job_cli,
    parse_args,
    spectral_transplant_job_complete,
)
from embed_optim.spectral_transplant_matrix import main as spectral_matrix_main
from embed_optim.spectral_transplant_summary import (
    _anchor_effects,
    _anchor_tail_effects,
    _band_effects,
    _factorial_effects,
    _spectral_path_effects,
)


def test_spectral_matrix_defaults_dense_and_requires_scope_amendment():
    defaults = parse_args([])
    assert defaults.families == ["dense"]
    assert defaults.scope_amendment is None
    assert parse_args(["--families", "dense", "late"]).families == ["dense", "late"]
    with pytest.raises(ValueError, match="requires --scope-amendment"):
        spectral_matrix_main(["--dry-run"])


def test_frozen_spectral_transplant_protocol_is_self_consistent():
    path, spec = load_spectral_transplant_protocol("configs/spectral_transplant_intervention.json")
    conditions = spectral_conditions(spec)

    assert path.name == "spectral_transplant_intervention.json"
    assert len(conditions) == 8
    assert [
        condition.interpolation_lambda
        for condition in conditions
        if condition.spectrum_operation == "log_interpolation"
    ] == [0.25, 0.5, 0.75, 1.0]
    assert [condition.band for condition in conditions if condition.band] == [
        "head",
        "middle",
        "tail",
    ]
    assert spec["intervention"]["expected_conditions_per_anchor"] == 11
    assert spec["intervention"]["expected_sample_records_per_anchor"] == 11 * 224
    tail = spec["evaluation"]["tail_protocol"]
    assert tail["status"] == "frozen-before-spectral-transplant-output"
    assert tail["tail_count"] == 12
    assert spec["freeze_context"]["amendments"][0]["spectral_transplant_outputs_available"] is False
    root = path.parent.parent
    source = spec["source_inputs"]
    assert _sha256(root / source["common_state_spec"]) == source["common_state_spec_sha256"]
    assert (
        _sha256(root / source["functional_intervention_spec"])
        == source["functional_intervention_spec_sha256"]
    )
    assert spec["freeze_context"]["local_global_reversal_analysis_visible"] is True
    assert spec["freeze_context"]["short_branch_results_available"] is False


def test_quartile_bands_are_disjoint_and_cover_every_singular_value():
    selected = []
    for band in ("head", "middle", "tail"):
        interval = _band_slice(8, band)
        selected.extend(range(interval.start, interval.stop))

    assert selected == list(range(8))
    with pytest.raises(ValueError, match="Unknown spectral band"):
        _band_slice(8, "unknown")


def test_spectral_factorial_swaps_values_without_changing_requested_basis():
    adamw = torch.diag(torch.tensor([4.0, 2.0, 1.0], dtype=torch.float32))
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    muon = rotation @ torch.diag(torch.ones(3, dtype=torch.float32))
    _, spec = load_spectral_transplant_protocol("configs/spectral_transplant_intervention.json")
    conditions = spectral_conditions(spec)
    transformed, diagnostics = construct_spectral_transplants(
        adamw,
        muon,
        conditions,
        target_frobenius_norm=7.0,
        relative_floor=1e-7,
    )

    assert set(transformed) == {condition.name for condition in conditions}
    assert all(
        torch.linalg.vector_norm(direction).item() == pytest.approx(7.0, rel=1e-5)
        for direction in transformed.values()
    )
    adam_basis_muon_spectrum = transformed["adam-basis__muon-spectrum"]
    singular = torch.linalg.svdvals(adam_basis_muon_spectrum)
    torch.testing.assert_close(
        singular, torch.full_like(singular, 7 / 3**0.5), rtol=1e-5, atol=1e-5
    )
    assert torch.allclose(
        adam_basis_muon_spectrum, torch.diag(torch.diag(adam_basis_muon_spectrum))
    )

    muon_basis_adam_spectrum = transformed["muon-basis__adam-spectrum"]
    singular = torch.linalg.svdvals(muon_basis_adam_spectrum)
    torch.testing.assert_close(singular / singular[-1], torch.tensor([4.0, 2.0, 1.0]))
    assert not torch.allclose(
        muon_basis_adam_spectrum, torch.diag(torch.diag(muon_basis_adam_spectrum))
    )

    interpolation = [row for row in diagnostics if row["spectrum_operation"] == "log_interpolation"]
    assert [row["interpolation_lambda"] for row in interpolation] == [0.25, 0.5, 0.75, 1.0]
    assert [row["stable_rank"] for row in interpolation] == sorted(
        row["stable_rank"] for row in interpolation
    )


def test_spectral_factorial_rejects_mismatched_or_nonfinite_inputs():
    condition = spectral_conditions(
        load_spectral_transplant_protocol("configs/spectral_transplant_intervention.json")[1]
    )[:1]
    with pytest.raises(ValueError, match="same-shaped"):
        construct_spectral_transplants(
            torch.eye(2),
            torch.eye(3),
            condition,
            target_frobenius_norm=1.0,
            relative_floor=1e-7,
        )
    invalid = torch.eye(2)
    invalid[0, 0] = torch.nan
    with pytest.raises(ValueError, match="same-shaped"):
        construct_spectral_transplants(
            invalid,
            torch.eye(2),
            condition,
            target_frobenius_norm=1.0,
            relative_floor=1e-7,
        )


def test_spectral_worker_command_round_trips(tmp_path: Path):
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
    job = SpectralTransplantJob(common, (tmp_path / "output").resolve())
    command = _job_cli(
        job,
        Namespace(
            spectral_spec=Path("configs/spectral_transplant_intervention.json").resolve(),
            common_state_spec=Path("configs/common_state_probe.json").resolve(),
        ),
    )
    parsed = parse_args(command[3:])

    assert command[:3] == [
        __import__("sys").executable,
        "-m",
        "embed_optim.spectral_transplant_matrix",
    ]
    assert parsed.worker is True
    assert parsed.family == "late"
    assert parsed.label == "late/pretrained"
    assert parsed.output_dir == job.output_dir


def _declared(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_spectral_completion_audits_nested_direction_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec_path, spec = load_spectral_transplant_protocol(
        "configs/spectral_transplant_intervention.json"
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    update_dir = tmp_path / "common" / "updates"
    gradient_dir = tmp_path / "common" / "gradients"
    update_dir.mkdir(parents=True)
    gradient_dir.mkdir()
    update_manifest = update_dir / "manifest.json"
    update_manifest.write_text('{"fixture":true}\n', encoding="utf-8")
    source_identity = {
        "path": str(update_manifest.resolve()),
        "bytes": update_manifest.stat().st_size,
        "sha256": _sha256(update_manifest),
    }

    output_dir = tmp_path / "output"
    direction_dir = output_dir / "directions"
    direction_dir.mkdir(parents=True)
    direction_outputs = {}
    direction_names = [
        "direction_metrics",
        *(condition.name for condition in spectral_conditions(spec)),
    ]
    for name in direction_names:
        suffix = ".jsonl" if name == "direction_metrics" else ".safetensors"
        path = direction_dir / f"{name}{suffix}"
        path.write_bytes(name.encode())
        direction_outputs[name] = _declared(path, direction_dir)
    direction_manifest = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint": {"path": str(checkpoint.resolve())},
        "spectral_transplant_spec": {"sha256": _sha256(spec_path)},
        "source_update_manifest": source_identity,
        "tensors": spec["anchor_scope"]["expected_hidden_tensors_per_anchor"],
        "parameters": spec["anchor_scope"]["expected_hidden_parameters_per_anchor"],
        "condition_records": spec["anchor_scope"]["expected_hidden_tensors_per_anchor"]
        * len(spectral_conditions(spec)),
        "outputs": direction_outputs,
    }
    direction_manifest_path = direction_dir / "manifest.json"
    direction_manifest_path.write_text(json.dumps(direction_manifest), encoding="utf-8")

    sample = output_dir / "sample_metrics.jsonl"
    condition = output_dir / "condition_metrics.jsonl"
    sample.write_text('{"sample":true}\n', encoding="utf-8")
    condition.write_text('{"condition":true}\n', encoding="utf-8")
    transformed = spectral_conditions(spec)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "family": "dense",
        "checkpoint": {"path": str(checkpoint.resolve())},
        "spectral_transplant_spec": {"sha256": _sha256(spec_path)},
        "source_update_manifest": source_identity,
        "direction_manifest": {
            "path": str(direction_manifest_path.resolve()),
            "bytes": direction_manifest_path.stat().st_size,
            "sha256": _sha256(direction_manifest_path),
        },
        "conditions": [
            {"condition": name}
            for name in [
                "baseline",
                "adamw-native",
                "muon-native",
                *(item.name for item in transformed),
            ]
        ],
        "sample_records": spec["intervention"]["expected_sample_records_per_anchor"],
        "condition_records": spec["intervention"]["expected_conditions_per_anchor"],
        "outputs": {
            "sample_metrics": _declared(sample, output_dir),
            "condition_metrics": _declared(condition, output_dir),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
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
    job = SpectralTransplantJob(common_job, output_dir)
    monkeypatch.setattr(
        "embed_optim.spectral_transplant_matrix.common_state_job_complete",
        lambda *args, **kwargs: True,
    )

    assert spectral_transplant_job_complete(
        job, spec_path, Path("configs/common_state_probe.json"), verify_hashes=True
    )

    direction_manifest["outputs"].pop(transformed[-1].name)
    direction_manifest_path.write_text(json.dumps(direction_manifest), encoding="utf-8")
    manifest["direction_manifest"] = {
        "path": str(direction_manifest_path.resolve()),
        "bytes": direction_manifest_path.stat().st_size,
        "sha256": _sha256(direction_manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not spectral_transplant_job_complete(
        job, spec_path, Path("configs/common_state_probe.json"), verify_hashes=True
    )


def _sample(condition: str, sample_id: int, effect: float) -> dict:
    return {
        "condition": condition,
        "sample_id": sample_id,
        "contrastive_loss": 1.0 - effect,
        "positive_score": 1.0 + effect,
        "hardest_negative_score": 1.0,
        "positive_margin": effect,
        "reciprocal_rank": 0.5 + effect,
        "top1_accuracy": effect,
    }


def test_spectral_summary_builds_path_band_and_factorial_estimands():
    _, spec = load_spectral_transplant_protocol("configs/spectral_transplant_intervention.json")
    conditions = [
        "baseline",
        "adamw-native",
        "muon-native",
        *(condition.name for condition in spectral_conditions(spec)),
    ]
    effects = {
        "baseline": 0.0,
        "adamw-native": 0.10,
        "muon-native": 0.40,
        "adam-basis__spectrum-lambda-0.25": 0.15,
        "adam-basis__spectrum-lambda-0.50": 0.20,
        "adam-basis__spectrum-lambda-0.75": 0.25,
        "adam-basis__muon-spectrum": 0.30,
        "muon-basis__adam-spectrum": 0.20,
        "adam-basis__muon-head-spectrum": 0.11,
        "adam-basis__muon-middle-spectrum": 0.12,
        "adam-basis__muon-tail-spectrum": 0.13,
    }
    records = [
        _sample(condition, sample_id, effects[condition])
        for condition in conditions
        for sample_id in range(spec["evaluation"]["examples"])
    ]
    anchor = _anchor_effects("dense/test", "dense", records, spec)
    tail = _anchor_tail_effects("dense/test", "dense", records, spec)
    factorial = _factorial_effects(anchor)
    path = _spectral_path_effects(anchor)
    bands = _band_effects(anchor)

    assert len(anchor) == 10
    assert len(tail) == 9
    margin_factorial = next(row for row in factorial if row["metric"] == "positive_margin")
    assert margin_factorial["spectrum_main_effect"] == pytest.approx(0.2)
    assert margin_factorial["basis_main_effect"] == pytest.approx(0.1)
    assert margin_factorial["spectrum_basis_interaction"] == pytest.approx(0.0)
    margin_path = [row for row in path if row["metric"] == "positive_margin"]
    assert [row["interpolation_lambda"] for row in margin_path] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [row["contrast_vs_adamw_native"] for row in margin_path] == pytest.approx(
        [0.0, 0.05, 0.10, 0.15, 0.20]
    )
    margin_bands = [row for row in bands if row["metric"] == "positive_margin"]
    assert [row["contrast_vs_adamw_native"] for row in margin_bands] == pytest.approx(
        [0.01, 0.02, 0.03]
    )
    muon_tail = next(row for row in tail if row["condition"] == "muon-native")
    assert muon_tail["p95_pairwise_loss_contrast"] == pytest.approx(-0.30)
    assert muon_tail["p05_pairwise_margin_contrast"] == pytest.approx(0.30)
    assert muon_tail["mean_loss_contrast_on_adam_tail"] == pytest.approx(-0.30)
    assert muon_tail["mean_loss_contrast_on_condition_tail"] == pytest.approx(-0.30)
    assert muon_tail["worst_loss_tail_jaccard"] == pytest.approx(1.0)
