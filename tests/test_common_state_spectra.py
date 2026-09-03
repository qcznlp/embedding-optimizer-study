from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from embed_optim.common_state_matrix import CommonStateJob
from embed_optim.common_state_spectra import (
    CommonStateSpectrumJob,
    _job_cli,
    _read_spectrum_records,
    analyze_common_state_spectra,
    build_spectrum_jobs,
    load_spectrum_spec,
    parse_args,
    spectrum_job_complete,
    summarize_spectrum_matrix,
)
from embed_optim.common_state_summary import ExpectedCommonStateMetric
from embed_optim.geometry import _sha256
from embed_optim.update_geometry import ALGORITHMS


def _declared(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fixture(tmp_path: Path) -> tuple[CommonStateSpectrumJob, Path, Path]:
    common_state_spec = tmp_path / "common-state.json"
    common = json.loads(Path("configs/common_state_probe.json").read_text(encoding="utf-8"))
    common["anchor_protocol"]["expected_hidden_partition"] = {
        "tensors": 1,
        "parameters": 6,
    }
    common_state_spec.write_text(json.dumps(common, sort_keys=True) + "\n", encoding="utf-8")

    tensor_name = "encoder.layers.0.weight"
    spectrum_spec = tmp_path / "spectrum.json"
    spectrum = json.loads(
        Path("configs/common_state_spectrum_probe.json").read_text(encoding="utf-8")
    )
    spectrum["common_state_spec_sha256"] = _sha256(common_state_spec)
    spectrum["selection"]["tensor_names"] = [tensor_name]
    spectrum["selection"]["expected_tensors_per_anchor"] = 1
    spectrum["selection"]["expected_spectra"] = 20 * len(ALGORITHMS)
    spectrum["analysis"]["device"] = "cpu"
    spectrum_spec.write_text(json.dumps(spectrum, sort_keys=True) + "\n", encoding="utf-8")

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    gradient_dir = tmp_path / "common" / "dense" / "pretrained" / "gradients"
    update_dir = tmp_path / "common" / "dense" / "pretrained" / "updates"
    gradient_dir.mkdir(parents=True)
    update_dir.mkdir(parents=True)
    shards = []
    for index in range(8):
        shard = gradient_dir / f"gradient-{index:04d}.safetensors"
        shard.write_bytes(f"gradient-{index}".encode())
        shards.append(_declared(shard, gradient_dir))
    gradient_manifest = gradient_dir / "manifest.json"
    gradient_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "checkpoint": {"path": str(checkpoint.resolve())},
                "common_state_spec": {"sha256": _sha256(common_state_spec)},
                "partition_summary": {"hidden": {"tensors": 1, "parameters": 6}},
                "gradient_shards": shards,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = update_dir / "metrics.jsonl"
    metrics.write_text('{"fixture":true}\n', encoding="utf-8")
    outputs = {"metrics": _declared(metrics, update_dir)}
    matrices = {
        "adamw": torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float16),
        "muon": torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float16),
        "normuon": torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float16),
    }
    for algorithm, matrix in matrices.items():
        matched = update_dir / f"{algorithm}-matched.safetensors"
        save_file({tensor_name: matrix}, matched)
        outputs[f"{algorithm}_matched"] = _declared(matched, update_dir)
    update_manifest = update_dir / "manifest.json"
    update_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": {"path": str(checkpoint.resolve())},
                "common_state_spec": {"sha256": _sha256(common_state_spec)},
                "gradient_manifest": {
                    "path": str(gradient_manifest.resolve()),
                    "bytes": gradient_manifest.stat().st_size,
                    "sha256": _sha256(gradient_manifest),
                },
                "gradient_steps": 8,
                "tensors": 1,
                "parameters": 6,
                "outputs": outputs,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    common_job = CommonStateJob(
        family="dense",
        label="dense/pretrained",
        checkpoint=checkpoint.resolve(),
        gradient_dir=gradient_dir,
        update_dir=update_dir,
        gradient_steps=8,
        hidden_tensors=1,
        hidden_parameters=6,
    )
    return (
        CommonStateSpectrumJob(
            common_state=common_job,
            output_dir=tmp_path / "spectra" / "dense" / "pretrained",
        ),
        spectrum_spec,
        common_state_spec,
    )


def test_frozen_spectrum_protocol_is_bound_and_architecture_balanced():
    common_state_spec = Path("configs/common_state_probe.json")
    spectrum_spec = Path("configs/common_state_spectrum_probe.json")
    protocol = load_spectrum_spec(spectrum_spec, common_state_spec)

    assert protocol["common_state_spec_sha256"] == _sha256(common_state_spec)
    assert protocol["selection"]["expected_spectra"] == 360
    assert protocol["freeze_context"]["strict_beir_valid_units"] == 110
    tensors = protocol["selection"]["tensor_names"]
    assert {name.split(".")[2] for name in tensors} == {"0", "10", "21"}
    assert {"attn.Wqkv", "mlp.Wi"} == {".".join(name.split(".")[3:5]) for name in tensors}


def test_exact_spectrum_analysis_is_hashed_and_resumable(tmp_path: Path):
    job, spectrum_spec, common_state_spec = _fixture(tmp_path)

    manifest = analyze_common_state_spectra(
        job,
        spectrum_spec=spectrum_spec,
        common_state_spec=common_state_spec,
        device="cpu",
    )

    assert manifest["status"] == "complete"
    assert manifest["records"] == 3
    assert spectrum_job_complete(job, spectrum_spec, common_state_spec, verify_hashes=True)
    _, validated = _read_spectrum_records(
        job,
        spectrum_spec=spectrum_spec,
        common_state_spec=common_state_spec,
    )
    assert len(validated) == 3
    records = [
        json.loads(line)
        for line in (job.output_dir / "spectra.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    adamw = next(record for record in records if record["update_operator"] == "adamw")
    assert adamw["rank"] == 2
    assert adamw["singular_values"] == [3.0, 2.0]
    assert adamw["stable_rank"] == pytest.approx(13 / 9)
    before = (job.output_dir / "spectra.jsonl").stat().st_mtime_ns
    resumed = analyze_common_state_spectra(
        job,
        spectrum_spec=spectrum_spec,
        common_state_spec=common_state_spec,
        device="cpu",
    )
    assert resumed == manifest
    assert (job.output_dir / "spectra.jsonl").stat().st_mtime_ns == before


def test_spectrum_matrix_mapping_and_worker_command_round_trip(tmp_path: Path):
    job, spectrum_spec, common_state_spec = _fixture(tmp_path)
    rebuilt = build_spectrum_jobs([job.common_state], tmp_path / "new-root")
    assert rebuilt[0].output_dir == (tmp_path / "new-root" / "dense" / "pretrained").resolve()
    command = _job_cli(
        rebuilt[0],
        Namespace(common_state_spec=common_state_spec, spectrum_spec=spectrum_spec),
    )
    assert command[:3] == [sys.executable, "-m", "embed_optim.common_state_spectra"]
    parsed = parse_args(command[3:])
    assert parsed.worker is True
    assert parsed.label == "dense/pretrained"
    assert parsed.output_dir == rebuilt[0].output_dir
    defaults = parse_args([])
    assert defaults.families == ["dense", "late"]
    assert defaults.scope_amendment is None


def test_spectrum_summary_writes_exact_long_form_matrix(tmp_path: Path, monkeypatch):
    common_state_spec = tmp_path / "common.json"
    spectrum_spec = tmp_path / "spectrum.json"
    common_state_spec.write_text("common\n", encoding="utf-8")
    spectrum_spec.write_text("spectrum\n", encoding="utf-8")
    protocol = {
        "selection": {
            "expected_anchors": 20,
            "expected_spectra": 60,
            "families": ["dense", "late"],
            "tensor_names": ["encoder.layers.0.weight"],
        },
        "freeze_context": {"fixture": True},
    }
    monkeypatch.setattr(
        "embed_optim.common_state_spectra.load_spectrum_spec",
        lambda spectrum_path, common_path: protocol,
    )
    monkeypatch.setattr(
        "embed_optim.common_state_spectra.spectrum_job_complete",
        lambda *args, **kwargs: True,
    )

    jobs = []
    expected = []
    records_by_label = {}
    result_root = tmp_path / "results"
    for index in range(20):
        family = "dense" if index < 10 else "late"
        label = f"{family}/anchor-{index}"
        common = CommonStateJob(
            family=family,
            label=label,
            checkpoint=(tmp_path / "checkpoints" / str(index)).resolve(),
            gradient_dir=tmp_path / "common" / label / "gradients",
            update_dir=tmp_path / "common" / label / "updates",
            gradient_steps=8,
            hidden_tensors=1,
            hidden_parameters=6,
        )
        spectrum_job = CommonStateSpectrumJob(
            common_state=common,
            output_dir=result_root / label,
        )
        spectrum_job.output_dir.mkdir(parents=True)
        (spectrum_job.output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
        jobs.append(spectrum_job)
        expected.append(
            ExpectedCommonStateMetric(
                job=common,
                anchor_kind="pretrained" if index in {0, 10} else "checkpoint",
                source_optimizer="" if index in {0, 10} else "muon",
                learning_rate="" if index in {0, 10} else 1e-3,
                run_id="pretrained" if index in {0, 10} else f"anchor-{index}",
                stage=0 if index in {0, 10} else 5,
                fraction=0.0 if index in {0, 10} else 1.0,
                step=0 if index in {0, 10} else 10,
            )
        )
        records_by_label[label] = [
            {
                "schema_version": 1,
                "family": family,
                "label": label,
                "update_operator": algorithm,
                "tensor": "encoder.layers.0.weight",
                "shape": [2, 3],
                "source_dtype": "float16",
                "compute_dtype": "float32",
                "rank": 2,
                "frobenius_norm": 5**0.5,
                "spectral_norm": 2.0,
                "stable_rank": 1.25,
                "nuclear_norm": 3.0,
                "entropy_effective_rank": 1.8,
                "condition_number": 2.0,
                "singular_values": [2.0, 1.0],
            }
            for algorithm in ALGORITHMS
        ]

    monkeypatch.setattr(
        "embed_optim.common_state_spectra._read_spectrum_records",
        lambda job, **kwargs: (
            {"output": {"sha256": f"sha-{job.common_state.label}"}},
            records_by_label[job.common_state.label],
        ),
    )

    manifest = summarize_spectrum_matrix(
        jobs,
        expected,
        result_root,
        tmp_path / "summary",
        spectrum_spec=spectrum_spec,
        common_state_spec=common_state_spec,
    )

    assert manifest["complete"] is True
    assert manifest["valid_spectra"] == 60
    assert manifest["singular_values"] == 120
    assert manifest["outputs"]["spectrum_metrics"]["rows"] == 60
    assert manifest["outputs"]["singular_values"]["rows"] == 120

    dense_manifest = summarize_spectrum_matrix(
        jobs[:10],
        expected[:10],
        result_root,
        tmp_path / "dense-summary",
        spectrum_spec=spectrum_spec,
        common_state_spec=common_state_spec,
        families=("dense",),
        scope_amendment="configs/dense_scope_amendment.json",
    )

    assert dense_manifest["complete"] is True
    assert dense_manifest["families"] == ["dense"]
    assert dense_manifest["scope_amendment"]["status"] == ("user_directed_post_hoc_scope_amendment")
    assert dense_manifest["expected_anchors"] == dense_manifest["valid_anchors"] == 10
    assert dense_manifest["expected_spectra"] == dense_manifest["valid_spectra"] == 30
    assert dense_manifest["outputs"]["singular_values"]["rows"] == 60


def test_dense_spectrum_summary_requires_scope_amendment(tmp_path: Path, monkeypatch):
    common_state_spec = tmp_path / "common.json"
    spectrum_spec = tmp_path / "spectrum.json"
    common_state_spec.write_text("common\n", encoding="utf-8")
    spectrum_spec.write_text("spectrum\n", encoding="utf-8")
    monkeypatch.setattr(
        "embed_optim.common_state_spectra.load_spectrum_spec",
        lambda spectrum_path, common_path: {
            "selection": {
                "expected_anchors": 20,
                "expected_spectra": 360,
                "families": ["dense", "late"],
                "tensor_names": ["tensor"],
            }
        },
    )

    with pytest.raises(ValueError, match="requires --scope-amendment"):
        summarize_spectrum_matrix(
            [],
            [],
            tmp_path / "results",
            tmp_path / "summary",
            spectrum_spec=spectrum_spec,
            common_state_spec=common_state_spec,
            families=("dense",),
        )
