from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.candidate_breadth_evaluation import METRICS
from embed_optim.candidate_breadth_summary import (
    _candidate_breadth_figure,
    _candidate_breadth_outputs,
    _matrix_provenance,
    build_candidate_breadth_summary,
    candidate_breadth_decision,
    source_stratified_paired_bootstrap_ci,
    spearman,
)
from embed_optim.data import SPLITS
from embed_optim.geometry import _sha256


def test_spearman_uses_average_ranks() -> None:
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832981)
    with pytest.raises(ValueError, match="constant"):
        spearman([1, 1, 1], [1, 2, 3])


def test_source_stratified_paired_bootstrap_is_exact_deterministic_and_strict() -> None:
    constant = {source: [1.0] * 32 for source in SPLITS}
    assert source_stratified_paired_bootstrap_ci(
        constant,
        replicates=257,
        seed=20260902,
        confidence=0.95,
        label="constant",
    ) == pytest.approx((1.0, 1.0))

    varied = {
        source: [source_index + query_index / 32 for query_index in range(32)]
        for source_index, source in enumerate(SPLITS)
    }
    first = source_stratified_paired_bootstrap_ci(
        varied,
        replicates=2_000,
        seed=20260902,
        confidence=0.95,
        label="muon:7:contrastive_loss",
    )
    repeated = source_stratified_paired_bootstrap_ci(
        varied,
        replicates=2_000,
        seed=20260902,
        confidence=0.95,
        label="muon:7:contrastive_loss",
    )
    assert repeated == first
    assert first[0] < first[1]

    invalid = dict(varied)
    invalid[SPLITS[0]] = invalid[SPLITS[0]][:-1]
    with pytest.raises(ValueError, match="bootstrap contract"):
        source_stratified_paired_bootstrap_ci(
            invalid,
            replicates=2_000,
            seed=20260902,
            confidence=0.95,
            label="invalid",
        )


def _contrasts(*, broad_reversal: bool, broad_scale: float = 1.0) -> list[dict]:
    rows = []
    for optimizer in ("muon", "normuon"):
        rows.append(
            {
                "optimizer": optimizer,
                "negative_width": 7,
                "contrastive_loss_delta": -0.2,
                "positive_margin_delta": 0.1,
            }
        )
        rows.append(
            {
                "optimizer": optimizer,
                "negative_width": 2048,
                "contrastive_loss_delta": (0.2 if broad_reversal else -0.2) * broad_scale,
                "positive_margin_delta": (-0.1 if broad_reversal else 0.1) * broad_scale,
            }
        )
    return rows


def test_candidate_breadth_support_requires_endpoint_reversal_for_both_challengers() -> None:
    supported = candidate_breadth_decision(_contrasts(broad_reversal=True), baseline_pass=True)
    assert supported["decision"] == "supported"
    assert supported["width_2048_reversal_pass"] is True

    failed_baseline = candidate_breadth_decision(
        _contrasts(broad_reversal=True), baseline_pass=False
    )
    assert failed_baseline["decision"] == "not_supported"


def test_candidate_breadth_reports_attenuation_without_promoting_it_to_support() -> None:
    partial = candidate_breadth_decision(
        _contrasts(broad_reversal=False, broad_scale=0.4), baseline_pass=True
    )
    assert partial["decision"] == "partial_attenuation"
    assert partial["halfway_attenuation_pass"] is True

    unchanged = candidate_breadth_decision(_contrasts(broad_reversal=False), baseline_pass=True)
    assert unchanged["decision"] == "not_supported"


def test_candidate_breadth_publication_figure_requires_and_writes_complete_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    monkeypatch.setitem(matplotlib.rcParams, "pdf.fonttype", 3)
    monkeypatch.setitem(matplotlib.rcParams, "ps.fonttype", 3)
    widths = [7, 10, 32, 128, 512, 2048]
    calibration = [
        {
            "optimizer": optimizer,
            "negative_width": width,
            "loss_beir_spearman": -0.8 + index * 0.05,
            "margin_beir_spearman": 0.8 - index * 0.05,
        }
        for optimizer in ("adamw", "muon", "normuon")
        for index, width in enumerate(widths)
    ]
    contrasts = [
        {
            "optimizer": optimizer,
            "negative_width": width,
            "contrastive_loss_delta": -0.1 + index * 0.03,
            "contrastive_loss_delta_ci95_lower": -0.11 + index * 0.03,
            "contrastive_loss_delta_ci95_upper": -0.09 + index * 0.03,
            "positive_margin_delta": 0.1 - index * 0.03,
            "positive_margin_delta_ci95_lower": 0.09 - index * 0.03,
            "positive_margin_delta_ci95_upper": 0.11 - index * 0.03,
        }
        for optimizer in ("muon", "normuon")
        for index, width in enumerate(widths)
    ]

    outputs = _candidate_breadth_figure(calibration, contrasts, tmp_path)

    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42
    assert set(outputs) == {"svg", "pdf"}
    for suffix, record in outputs.items():
        path = tmp_path / record["path"]
        assert path.suffix == f".{suffix}"
        assert path.stat().st_size == record["bytes"] > 0
        assert len(record["sha256"]) == 64
        if suffix == "pdf":
            assert b"/Subtype /Type3" not in path.read_bytes()

    repeated = _candidate_breadth_figure(calibration, contrasts, tmp_path)
    assert {suffix: item["sha256"] for suffix, item in repeated.items()} == {
        suffix: item["sha256"] for suffix, item in outputs.items()
    }

    with pytest.raises(ValueError, match="complete frozen width coverage"):
        _candidate_breadth_figure(calibration[:-1], contrasts, tmp_path)


def test_candidate_breadth_output_audit_recomputes_bytes_instead_of_trusting_hashes(
    tmp_path,
) -> None:
    widths = [7, 10, 32, 128, 512, 2048]
    calibration = [
        {
            "optimizer": optimizer,
            "negative_width": width,
            "loss_beir_spearman": -0.8 + index * 0.05,
            "margin_beir_spearman": 0.8 - index * 0.05,
        }
        for optimizer in ("adamw", "muon", "normuon")
        for index, width in enumerate(widths)
    ]
    contrasts = [
        {
            "optimizer": optimizer,
            "negative_width": width,
            "contrastive_loss_delta": -0.1 + index * 0.03,
            "contrastive_loss_delta_ci95_lower": -0.11 + index * 0.03,
            "contrastive_loss_delta_ci95_upper": -0.09 + index * 0.03,
            "positive_margin_delta": 0.1 - index * 0.03,
            "positive_margin_delta_ci95_lower": 0.09 - index * 0.03,
            "positive_margin_delta_ci95_upper": 0.11 - index * 0.03,
        }
        for optimizer in ("muon", "normuon")
        for index, width in enumerate(widths)
    ]

    written = _candidate_breadth_outputs(calibration, contrasts, tmp_path, audit_only=False)
    audited = _candidate_breadth_outputs(calibration, contrasts, tmp_path, audit_only=True)
    assert audited == written

    calibration_path = tmp_path / written["calibration"]["path"]
    calibration_path.write_bytes(calibration_path.read_bytes() + b"self-signed-tamper\n")
    with pytest.raises(ValueError, match="calibration output differs from recomputation"):
        _candidate_breadth_outputs(calibration, contrasts, tmp_path, audit_only=True)


def test_summary_binds_matrix_jobs_to_the_full_source_audit(tmp_path) -> None:
    protocol = json.loads(Path("configs/candidate_breadth_probe.json").read_text(encoding="utf-8"))
    protocol_path = tmp_path / "configs" / "candidate_breadth_probe.json"
    protocol_path.parent.mkdir()
    protocol_path.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
    results_root = tmp_path / protocol["evaluation"]["results_root"]
    results_root.mkdir(parents=True)
    source_receipt = tmp_path / "reports" / "candidate-breadth" / "data-audit.json"
    source_receipt.parent.mkdir(parents=True, exist_ok=True)
    source_audit = {
        "schema_version": 1,
        "status": "complete",
        "upstream_reconstruction_verified": True,
        "protocol_sha256": _sha256(protocol_path),
        "manifest_sha256": "a" * 64,
    }
    source_receipt.write_text(json.dumps(source_audit) + "\n", encoding="utf-8")
    source_identity = {
        "path": str(source_receipt.relative_to(tmp_path)),
        "bytes": source_receipt.stat().st_size,
        "sha256": _sha256(source_receipt),
        "audit": source_audit,
    }
    jobs = []
    step = protocol["evaluation"]["checkpoint_step"]
    for index, run_id in enumerate(protocol["evaluation"]["run_ids"]):
        manifest_path = results_root / run_id / "manifest.json"
        manifest_path.parent.mkdir()
        manifest_path.write_text(
            json.dumps({"baseline_reproduction": {"maximum_absolute_error": 0.0}}) + "\n",
            encoding="utf-8",
        )
        jobs.append(
            {
                "run_id": run_id,
                "gpu": str(index % 8),
                "attempts": [{"attempt": 1, "returncode": 0}],
                "checkpoint": str(
                    tmp_path
                    / protocol["evaluation"]["checkpoint_root"]
                    / run_id
                    / f"checkpoint-{step}"
                ),
                "manifest": {
                    "path": str(manifest_path.relative_to(tmp_path)),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": _sha256(manifest_path),
                },
                "baseline_maximum_absolute_error": 0.0,
            }
        )
    matrix_receipt = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "path": str(protocol_path.relative_to(tmp_path)),
            "bytes": protocol_path.stat().st_size,
            "sha256": _sha256(protocol_path),
        },
        "data_audit": {
            **source_audit,
            "upstream_reconstruction_verified": False,
        },
        "source_audit": source_identity,
        "gpus": [str(index) for index in range(8)],
        "jobs": jobs,
    }
    receipt_path = results_root / "matrix-receipt.json"
    receipt_path.write_text(json.dumps(matrix_receipt) + "\n", encoding="utf-8")

    provenance = _matrix_provenance(
        root=tmp_path,
        protocol_path=protocol_path,
        protocol=protocol,
        results_root=results_root,
    )
    assert provenance["source_audit"]["sha256"] == _sha256(source_receipt)

    first_manifest = results_root / protocol["evaluation"]["run_ids"][0] / "manifest.json"
    first_manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matrix job changed"):
        _matrix_provenance(
            root=tmp_path,
            protocol_path=protocol_path,
            protocol=protocol,
            results_root=results_root,
        )


def test_summary_builds_and_audits_stratified_intervals_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("matplotlib")
    protocol = json.loads(Path("configs/candidate_breadth_probe.json").read_text(encoding="utf-8"))
    protocol_path = tmp_path / "configs" / "candidate_breadth_probe.json"
    protocol_path.parent.mkdir()
    protocol_path.write_text(json.dumps(protocol, sort_keys=True) + "\n", encoding="utf-8")
    widths = protocol["candidate_construction"]["negative_widths"]
    run_ids = protocol["evaluation"]["run_ids"]
    results_root = tmp_path / protocol["evaluation"]["results_root"]
    results_root.mkdir(parents=True)

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def record(path: Path) -> dict:
        return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}

    jobs = []
    beir_rows = []
    for run_index, run_id in enumerate(run_ids):
        optimizer, learning_rate = run_id.split("-lr", 1)
        optimizer_index = run_index % 4
        beir_rows.append(
            {
                "model_family": "dense",
                "stage": 5,
                "run_id": run_id,
                "optimizer": optimizer,
                "learning_rate": float(learning_rate),
                "mean_ndcg_at_10": 0.5 + optimizer_index * 0.01,
            }
        )
        run_root = results_root / run_id
        run_root.mkdir()
        group_path = run_root / "group_metrics.jsonl"
        sample_path = run_root / "sample_metrics.jsonl"
        write_jsonl(
            group_path,
            [
                {
                    "source": "__all__",
                    "negative_width": width,
                    "contrastive_loss": optimizer_index + width / 100_000,
                    "positive_margin": -optimizer_index - width / 100_000,
                }
                for width in widths
            ],
        )
        sample_rows = []
        for width_index, width in enumerate(widths):
            for sample_id in range(224):
                query_offset = sample_id / 100_000
                loss = 1.0 + optimizer_index / 100 + query_offset
                margin = 0.3 - optimizer_index / 100 - query_offset
                if optimizer in {"muon", "normuon"} and learning_rate == "3e-3":
                    progress = width_index / (len(widths) - 1)
                    loss += -0.2 + 0.4 * progress + query_offset / 10
                    margin += 0.1 - 0.2 * progress - query_offset / 10
                metric_values = {
                    "contrastive_loss": loss,
                    "positive_score": 0.7 + query_offset,
                    "hardest_negative_score": 0.4 + query_offset,
                    "positive_margin": margin,
                    "reciprocal_rank": 0.5,
                    "top1_accuracy": 0.0,
                }
                assert set(metric_values) == set(METRICS)
                sample_rows.append(
                    {
                        "negative_width": width,
                        "sample_id": sample_id,
                        "source": SPLITS[sample_id // 32],
                        **metric_values,
                    }
                )
        write_jsonl(sample_path, sample_rows)
        manifest = {
            "status": "complete",
            "protocol": {"sha256": _sha256(protocol_path)},
            "negative_widths": widths,
            "baseline_reproduction": {"maximum_absolute_error": 0.0},
            "outputs": {
                "group_metrics": record(group_path),
                "sample_metrics": record(sample_path),
            },
        }
        manifest_path = run_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
        jobs.append(
            {
                "run_id": run_id,
                "gpu": str(run_index % 8),
                "attempts": [{"attempt": 1, "returncode": 0}],
                "checkpoint": str(
                    tmp_path
                    / protocol["evaluation"]["checkpoint_root"]
                    / run_id
                    / f"checkpoint-{protocol['evaluation']['checkpoint_step']}"
                ),
                "manifest": {
                    "path": str(manifest_path.relative_to(tmp_path)),
                    "bytes": manifest_path.stat().st_size,
                    "sha256": _sha256(manifest_path),
                },
                "baseline_maximum_absolute_error": 0.0,
            }
        )

    beir_path = tmp_path / "reports" / "dense-discovery" / "checkpoint_summary.csv"
    beir_path.parent.mkdir(parents=True)
    with beir_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(beir_rows[0]))
        writer.writeheader()
        writer.writerows(beir_rows)

    source_receipt = tmp_path / "reports" / "candidate-breadth" / "data-audit.json"
    source_receipt.parent.mkdir(parents=True)
    source_audit = {
        "schema_version": 1,
        "status": "complete",
        "upstream_reconstruction_verified": True,
        "protocol_sha256": _sha256(protocol_path),
        "manifest_sha256": "a" * 64,
    }
    source_receipt.write_text(json.dumps(source_audit, sort_keys=True) + "\n", encoding="utf-8")
    source_identity = {
        "path": str(source_receipt.relative_to(tmp_path)),
        "bytes": source_receipt.stat().st_size,
        "sha256": _sha256(source_receipt),
        "audit": source_audit,
    }
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "path": str(protocol_path.relative_to(tmp_path)),
            "bytes": protocol_path.stat().st_size,
            "sha256": _sha256(protocol_path),
        },
        "data_audit": {**source_audit, "upstream_reconstruction_verified": False},
        "source_audit": source_identity,
        "gpus": [str(index) for index in range(8)],
        "jobs": jobs,
    }
    (results_root / "matrix-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    calls = []

    def bootstrap(deltas_by_source, **kwargs):
        assert list(deltas_by_source) == list(SPLITS)
        assert {source: len(values) for source, values in deltas_by_source.items()} == {
            source: 32 for source in SPLITS
        }
        assert kwargs["replicates"] == 50_000
        assert kwargs["seed"] == 20_260_902
        assert kwargs["confidence"] == 0.95
        calls.append(kwargs["label"])
        values = [value for source in SPLITS for value in deltas_by_source[source]]
        mean = sum(values) / len(values)
        return mean - 0.01, mean + 0.01

    monkeypatch.setattr(
        "embed_optim.candidate_breadth_summary.source_stratified_paired_bootstrap_ci",
        bootstrap,
    )
    summary = build_candidate_breadth_summary(protocol_path)
    assert summary["status"] == "complete"
    assert summary["decision"]["decision"] == "supported"
    assert len(calls) == 24
    contrast_path = source_receipt.parent / summary["outputs"]["contrasts"]["path"]
    with contrast_path.open(encoding="utf-8", newline="") as handle:
        contrasts = list(csv.DictReader(handle))
    assert len(contrasts) == 12
    assert all(row["contrastive_loss_delta_ci95_lower"] for row in contrasts)
    assert all(row["positive_margin_delta_ci95_upper"] for row in contrasts)

    audited = build_candidate_breadth_summary(protocol_path, audit_only=True)
    assert audited == summary
    assert len(calls) == 48
