from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.common_state_matrix import CommonStateJob
from embed_optim.common_state_summary import (
    ExpectedCommonStateMetric,
    expected_common_state_metrics,
    summarize_common_state,
)
from embed_optim.config import OptimizerConfig, RunConfig
from embed_optim.geometry import _sha256
from embed_optim.update_geometry import ALGORITHMS


def _declared(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _matrix(scale: float) -> dict[str, object]:
    distribution = {"mean": scale, "cv": 0.5 * scale, "gini": 0.25, "max_to_median": 2.0}
    return {
        "frobenius_norm": scale,
        "row_norms": distribution,
        "column_norms": distribution,
        "top_1pct_row_energy": 0.2,
        "top_10pct_row_energy": 0.4,
        "algorithm": "exact",
        "rank": 2,
        "spectral_norm": scale,
        "approx_stable_rank": 1.5 * scale,
        "sketched_nuclear_norm": 1.7 * scale,
        "sketched_entropy_effective_rank": 1.8 * scale,
        "sketched_condition_number": 2.0,
        "captured_frobenius_energy": 1.0,
    }


def _spec(path: Path) -> None:
    payload = json.loads(Path("configs/common_state_probe.json").read_text(encoding="utf-8"))
    payload["anchor_protocol"]["expected_hidden_partition"] = {
        "tensors": 1,
        "parameters": 4,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _anchor(
    root: Path, spec: Path, index: int, *, corrupt_metrics: bool = False
) -> ExpectedCommonStateMetric:
    family = "dense" if index < 10 else "late"
    local = index % 10
    if local == 0:
        label = f"{family}/pretrained"
        anchor_kind = "pretrained"
        source_optimizer = ""
        learning_rate: float | str = ""
        run_id = "pretrained"
        stage = 0
        fraction = 0.0
        step = 0
    else:
        source_optimizer = ALGORITHMS[(local - 1) // 3]
        run_id = f"{source_optimizer}-anchor"
        stage = (1, 3, 5)[(local - 1) % 3]
        fraction = (0.2, 0.6, 1.0)[(local - 1) % 3]
        step = (2, 6, 10)[(local - 1) % 3]
        label = f"{family}/{run_id}/checkpoint-{step}"
        anchor_kind = "checkpoint"
        learning_rate = 1e-3
    checkpoint = root / "checkpoints" / label
    checkpoint.mkdir(parents=True)
    gradient_dir = root / label / "gradients"
    update_dir = root / label / "updates"
    gradient_dir.mkdir(parents=True)
    update_dir.mkdir(parents=True)

    shards = []
    for shard_index in range(8):
        shard = gradient_dir / f"gradient-{shard_index:04d}.safetensors"
        shard.write_bytes(f"gradient-{index}-{shard_index}".encode())
        shards.append(_declared(shard, gradient_dir))
    gradient_manifest = gradient_dir / "manifest.json"
    gradient_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "checkpoint": {"path": str(checkpoint.resolve())},
                "common_state_spec": {"sha256": _sha256(spec)},
                "partition_summary": {"hidden": {"tensors": 1, "parameters": 4}},
                "gradient_shards": shards,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    algorithms = {}
    for algorithm_index, algorithm in enumerate(ALGORITHMS, start=2):
        algorithms[algorithm] = {
            **_matrix(float(algorithm_index)),
            "cosine_with_final_gradient": 0.9 - algorithm_index * 0.1,
            "cosine_with_weight": 0.1 * algorithm_index,
            "per_unit_lr_update_to_weight": float(algorithm_index) / 2,
            "matched_frobenius_norm": 2.0,
        }
    record = {
        "schema_version": 1,
        "tensor": "encoder.layers.0.weight",
        "shape": [2, 2],
        "parameters": 4,
        "gradient_steps": 8,
        "weight_frobenius_norm": 2.0,
        "final_gradient": _matrix(1.0),
        "algorithms": algorithms,
        "pairwise_cosine": {
            "adamw__muon": 0.6,
            "adamw__normuon": 0.5,
            "muon__normuon": 0.8,
        },
    }
    metrics = update_dir / "metrics.jsonl"
    metrics.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    outputs = {"metrics": _declared(metrics, update_dir)}
    for algorithm in ALGORITHMS:
        matched = update_dir / f"{algorithm}-matched.safetensors"
        matched.write_bytes(f"{algorithm}-{index}".encode())
        outputs[f"{algorithm}_matched"] = _declared(matched, update_dir)
    update_manifest = update_dir / "manifest.json"
    update_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": {"path": str(checkpoint.resolve())},
                "common_state_spec": {"sha256": _sha256(spec)},
                "gradient_manifest": {
                    "path": str(gradient_manifest.resolve()),
                    "bytes": gradient_manifest.stat().st_size,
                    "sha256": _sha256(gradient_manifest),
                },
                "gradient_steps": 8,
                "tensors": 1,
                "parameters": 4,
                "outputs": outputs,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if corrupt_metrics:
        metrics.write_text("changed\n", encoding="utf-8")
    job = CommonStateJob(
        family=family,
        label=label,
        checkpoint=checkpoint.resolve(),
        gradient_dir=gradient_dir,
        update_dir=update_dir,
        gradient_steps=8,
        hidden_tensors=1,
        hidden_parameters=4,
    )
    return ExpectedCommonStateMetric(
        job=job,
        anchor_kind=anchor_kind,
        source_optimizer=source_optimizer,
        learning_rate=learning_rate,
        run_id=run_id,
        stage=stage,
        fraction=fraction,
        step=step,
    )


def test_strict_common_state_summary_writes_complete_paper_tables(tmp_path: Path):
    spec = tmp_path / "common-state.json"
    _spec(spec)
    result_root = tmp_path / "results"
    expected = [_anchor(result_root, spec, index) for index in range(20)]

    manifest = summarize_common_state(
        expected,
        result_root,
        tmp_path / "summary",
        common_state_spec=spec,
    )

    assert manifest["complete"] is True
    assert manifest["valid_anchors"] == 20
    assert {name: item["rows"] for name, item in manifest["outputs"].items()} == {
        "gradient_tensor_metrics": 20,
        "update_tensor_metrics": 60,
        "pairwise_tensor_cosines": 60,
        "gradient_anchor_metrics": 20,
        "anchor_metrics": 60,
        "pairwise_anchor_cosines": 60,
        "update_gradient_contrasts": 60,
        "anchor_contrasts": 40,
    }
    with (tmp_path / "summary" / "anchor_contrasts.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    dense_pretrained_muon = next(
        row
        for row in rows
        if row["label"] == "dense/pretrained" and row["update_operator"] == "muon"
    )
    assert float(dense_pretrained_muon["direction_frobenius_norm_to_adamw_ratio"]) == 1.5
    assert float(dense_pretrained_muon["cosine_with_adamw_parameter_weighted"]) == 0.6


def test_dense_common_state_summary_reuses_only_authorized_family(tmp_path: Path):
    spec = tmp_path / "common-state.json"
    _spec(spec)
    result_root = tmp_path / "results"
    all_expected = [_anchor(result_root, spec, index) for index in range(20)]

    manifest = summarize_common_state(
        all_expected[:10],
        result_root,
        tmp_path / "dense-summary",
        common_state_spec=spec,
        families=("dense",),
        scope_amendment="configs/dense_scope_amendment.json",
    )

    assert manifest["complete"] is True
    assert manifest["families"] == ["dense"]
    assert manifest["scope_amendment"]["status"] == ("user_directed_post_hoc_scope_amendment")
    assert manifest["expected_anchors"] == manifest["valid_anchors"] == 10
    assert {name: item["rows"] for name, item in manifest["outputs"].items()} == {
        "gradient_tensor_metrics": 10,
        "update_tensor_metrics": 30,
        "pairwise_tensor_cosines": 30,
        "gradient_anchor_metrics": 10,
        "anchor_metrics": 30,
        "pairwise_anchor_cosines": 30,
        "update_gradient_contrasts": 30,
        "anchor_contrasts": 20,
    }


def test_dense_common_state_summary_requires_scope_amendment(tmp_path: Path):
    spec = tmp_path / "common-state.json"
    _spec(spec)

    with pytest.raises(ValueError, match="requires --scope-amendment"):
        summarize_common_state(
            [],
            tmp_path / "results",
            tmp_path / "summary",
            common_state_spec=spec,
            families=("dense",),
        )


def test_dense_common_state_summary_rejects_changed_scope_binding(tmp_path: Path):
    spec = tmp_path / "common-state.json"
    _spec(spec)
    payload = json.loads(Path("configs/dense_scope_amendment.json").read_text(encoding="utf-8"))
    repository = tmp_path / "repository"
    (repository / "configs").mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    for binding in payload["source_bindings"]:
        source = Path(binding["path"])
        target = repository / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    changed = repository / payload["source_bindings"][0]["path"]
    changed.write_bytes(changed.read_bytes() + b"\n")
    amendment = repository / "configs" / "dense_scope_amendment.json"
    amendment.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Scope-amendment source differs"):
        summarize_common_state(
            [],
            tmp_path / "results",
            tmp_path / "summary",
            common_state_spec=spec,
            families=("dense",),
            scope_amendment=amendment,
        )


def test_expected_common_state_identity_preserves_full_checkpoint_stage(tmp_path: Path):
    config = RunConfig(
        run_id="muon-lr1e-3",
        model_family="dense",
        optimizer=OptimizerConfig(name="muon", lr=1e-3),
        model_name="fixture/dense",
        dataset_path="fixture",
        output_root=str(tmp_path / "outputs"),
    )
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "checkpoint_schedule.json").write_text(
        json.dumps(
            {
                "fractions": [0.2, 0.4, 0.6, 0.8, 1.0],
                "steps": [2, 4, 6, 8, 10],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = config.output_dir / "checkpoint-6"
    checkpoint.mkdir()
    job = CommonStateJob(
        family="dense",
        label="dense/muon-lr1e-3/checkpoint-6",
        checkpoint=checkpoint.resolve(),
        gradient_dir=tmp_path / "gradients",
        update_dir=tmp_path / "updates",
        gradient_steps=8,
        hidden_tensors=88,
        hidden_parameters=110_297_088,
    )

    observed = expected_common_state_metrics([job], [config])

    assert len(observed) == 1
    assert observed[0].stage == 3
    assert observed[0].fraction == 0.6
    assert observed[0].step == 6
    assert observed[0].source_optimizer == "muon"


def test_common_state_summary_rejects_hash_mismatch(tmp_path: Path):
    spec = tmp_path / "common-state.json"
    _spec(spec)
    result_root = tmp_path / "results"
    expected = [
        _anchor(result_root, spec, index, corrupt_metrics=index == 7) for index in range(20)
    ]

    with pytest.raises(ValueError, match="incomplete"):
        summarize_common_state(
            expected,
            result_root,
            tmp_path / "summary",
            common_state_spec=spec,
        )


def test_common_state_partial_summary_discloses_missing_anchors(tmp_path: Path):
    spec = tmp_path / "common-state.json"
    _spec(spec)
    result_root = tmp_path / "results"
    expected = [_anchor(result_root, spec, index) for index in range(20)]
    missing = expected[-1]
    (missing.job.update_dir / "manifest.json").unlink()

    manifest = summarize_common_state(
        expected,
        result_root,
        tmp_path / "summary",
        common_state_spec=spec,
        allow_partial=True,
    )

    assert manifest["complete"] is False
    assert manifest["valid_anchors"] == 19
    assert manifest["missing_labels"] == [missing.job.label]
