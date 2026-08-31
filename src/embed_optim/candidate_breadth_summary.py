from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .candidate_breadth_data import load_candidate_breadth_protocol
from .candidate_breadth_evaluation import METRICS
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def spearman(values: list[float], outcomes: list[float]) -> float:
    if len(values) != len(outcomes) or len(values) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    if not all(math.isfinite(value) for value in (*values, *outcomes)):
        raise ValueError("Spearman inputs must be finite")
    left = _average_ranks(values)
    right = _average_ranks(outcomes)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    return numerator / (left_norm * right_norm)


def candidate_breadth_decision(
    contrasts: list[dict[str, Any]], *, baseline_pass: bool
) -> dict[str, Any]:
    indexed = {(str(row["optimizer"]), int(row["negative_width"])): row for row in contrasts}
    challengers = ("muon", "normuon")
    if any((optimizer, width) not in indexed for optimizer in challengers for width in (7, 2048)):
        raise ValueError("Candidate-breadth decision lacks challenger endpoint contrasts")
    width_7 = all(
        float(indexed[(optimizer, 7)]["contrastive_loss_delta"]) < 0
        and float(indexed[(optimizer, 7)]["positive_margin_delta"]) > 0
        for optimizer in challengers
    )
    width_2048 = all(
        float(indexed[(optimizer, 2048)]["contrastive_loss_delta"]) > 0
        and float(indexed[(optimizer, 2048)]["positive_margin_delta"]) < 0
        for optimizer in challengers
    )
    attenuation = True
    for optimizer in challengers:
        narrow = indexed[(optimizer, 7)]
        broad = indexed[(optimizer, 2048)]
        for metric in ("contrastive_loss_delta", "positive_margin_delta"):
            if abs(float(broad[metric])) > 0.5 * abs(float(narrow[metric])):
                attenuation = False
    supported = baseline_pass and width_7 and width_2048
    partial = baseline_pass and width_7 and not width_2048 and attenuation
    return {
        "baseline_reproduction_pass": baseline_pass,
        "width_7_ordering_pass": width_7,
        "width_2048_reversal_pass": width_2048,
        "halfway_attenuation_pass": attenuation,
        "decision": (
            "supported" if supported else "partial_attenuation" if partial else "not_supported"
        ),
    }


def _csv_payload(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = _csv_payload(rows, fields)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_figure(figure: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_format = path.suffix.removeprefix(".")
    if image_format not in {"svg", "pdf"}:
        raise ValueError(f"Unsupported candidate-breadth figure format: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{image_format}")
    metadata = (
        {"Date": None, "Creator": "embedding-optimizer-study"}
        if image_format == "svg"
        else {
            "CreationDate": None,
            "ModDate": None,
            "Creator": "embedding-optimizer-study",
        }
    )
    try:
        figure.savefig(temporary, format=image_format, bbox_inches="tight", metadata=metadata)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _candidate_breadth_figure(
    calibration_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "candidate-breadth-v1"
    import matplotlib.pyplot as plt

    widths = sorted({int(row["negative_width"]) for row in calibration_rows})
    contrast_widths = sorted({int(row["negative_width"]) for row in contrast_rows})
    optimizers = ("adamw", "muon", "normuon")
    challengers = ("muon", "normuon")
    if (
        widths != [7, 10, 32, 128, 512, 2048]
        or contrast_widths != widths
        or len(calibration_rows) != len(widths) * len(optimizers)
        or len(contrast_rows) != len(widths) * len(challengers)
    ):
        raise ValueError("Candidate-breadth figure requires complete frozen width coverage")

    colors = {"adamw": "#4C78A8", "muon": "#F58518", "normuon": "#54A24B"}
    labels = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}
    calibration = {
        (str(row["optimizer"]), int(row["negative_width"])): row for row in calibration_rows
    }
    contrasts = {(str(row["optimizer"]), int(row["negative_width"])): row for row in contrast_rows}
    if set(calibration) != {(optimizer, width) for optimizer in optimizers for width in widths}:
        raise ValueError("Candidate-breadth calibration rows are duplicated or incomplete")
    if set(contrasts) != {(optimizer, width) for optimizer in challengers for width in widths}:
        raise ValueError("Candidate-breadth contrast rows are duplicated or incomplete")

    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.1), sharex=True)
    top_panels = (
        (axes[0, 0], "loss_beir_spearman", "Validation loss vs. BEIR", "ideal sign: negative"),
        (axes[0, 1], "margin_beir_spearman", "Validation margin vs. BEIR", "ideal sign: positive"),
    )
    for axis, metric, title, subtitle in top_panels:
        for optimizer in optimizers:
            axis.plot(
                widths,
                [float(calibration[(optimizer, width)][metric]) for width in widths],
                color=colors[optimizer],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=labels[optimizer],
            )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9, alpha=0.75)
        axis.set_ylim(-1.05, 1.05)
        axis.set_title(f"{title}\n{subtitle}", fontsize=10.2)
        axis.set_ylabel("Spearman $\\rho$")
        axis.grid(alpha=0.20)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3, loc="lower center")

    bottom_panels = (
        (
            axes[1, 0],
            "contrastive_loss_delta",
            "High-dose loss contrast",
            "positive favors retrieval-optimal dose",
        ),
        (
            axes[1, 1],
            "positive_margin_delta",
            "High-dose margin contrast",
            "negative favors retrieval-optimal dose",
        ),
    )
    for axis, metric, title, subtitle in bottom_panels:
        for optimizer in challengers:
            axis.plot(
                widths,
                [float(contrasts[(optimizer, width)][metric]) for width in widths],
                color=colors[optimizer],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=labels[optimizer],
            )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9, alpha=0.75)
        axis.set_title(f"{title}\n{subtitle}", fontsize=10.2)
        axis.set_ylabel(r"$3\!\times\!10^{-3} - 3\!\times\!10^{-4}$")
        axis.grid(alpha=0.20)
    axes[1, 0].legend(frameon=False, fontsize=8, ncol=2, loc="best")

    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks(widths, [str(width) for width in widths])
        axis.minorticks_off()
    for axis in axes[1, :]:
        axis.set_xlabel("Number of negatives for the same query")
    figure.suptitle(
        "Does candidate breadth repair optimizer-specific validation calibration?", fontsize=12
    )
    figure.text(
        0.5,
        0.008,
        "Post-hoc mechanism diagnostic; candidate sets are strictly nested and width 7 must "
        "reproduce the frozen validation evaluator.",
        ha="center",
        fontsize=7.8,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    records = {
        suffix: _atomic_figure(
            figure,
            output_dir / f"candidate_breadth_calibration.{suffix}",
        )
        for suffix in ("svg", "pdf")
    }
    plt.close(figure)
    return records


def _candidate_breadth_outputs(
    calibration_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    audit_only: bool,
) -> dict[str, dict[str, Any]]:
    calibration_path = output_dir / "calibration_by_width.csv"
    contrasts_path = output_dir / "high_dose_contrasts.csv"
    payloads = {
        "calibration": _csv_payload(calibration_rows, list(calibration_rows[0])),
        "contrasts": _csv_payload(contrast_rows, list(contrast_rows[0])),
    }
    csv_paths = {"calibration": calibration_path, "contrasts": contrasts_path}
    if audit_only:
        for name, path in csv_paths.items():
            if not path.is_file() or path.read_bytes() != payloads[name]:
                raise ValueError(f"Candidate-breadth {name} output differs from recomputation")
    else:
        _atomic_csv(calibration_path, calibration_rows, list(calibration_rows[0]))
        _atomic_csv(contrasts_path, contrast_rows, list(contrast_rows[0]))

    csv_records = {
        name: {
            "path": path.name,
            "bytes": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
        for name, path in csv_paths.items()
    }
    if not audit_only:
        figures = _candidate_breadth_figure(calibration_rows, contrast_rows, output_dir)
        return {**csv_records, "figure_svg": figures["svg"], "figure_pdf": figures["pdf"]}

    with tempfile.TemporaryDirectory(prefix="candidate-breadth-summary-audit-") as directory:
        temporary_dir = Path(directory)
        expected_figures = _candidate_breadth_figure(calibration_rows, contrast_rows, temporary_dir)
        for suffix, record in expected_figures.items():
            expected = temporary_dir / record["path"]
            observed = output_dir / record["path"]
            if not observed.is_file() or observed.read_bytes() != expected.read_bytes():
                raise ValueError(
                    f"Candidate-breadth figure_{suffix} output differs from recomputation"
                )
        return {
            **csv_records,
            "figure_svg": expected_figures["svg"],
            "figure_pdf": expected_figures["pdf"],
        }


def _load_beir_scores(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model_family"] != "dense" or int(row["stage"]) != 5:
                continue
            rows[row["run_id"]] = {
                "optimizer": row["optimizer"],
                "learning_rate": float(row["learning_rate"]),
                "mean_ndcg_at_10": float(row["mean_ndcg_at_10"]),
            }
    if len(rows) != 12:
        raise ValueError(f"Expected 12 DenseOn final discovery scores, found {len(rows)}")
    return rows


def _matrix_provenance(
    *,
    root: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    results_root: Path,
) -> dict[str, Any]:
    receipt_path = results_root / "matrix-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("Candidate-breadth matrix receipt is not an object")
    expected_protocol = {
        "path": str(protocol_path.relative_to(root)),
        "bytes": protocol_path.stat().st_size,
        "sha256": _sha256(protocol_path),
    }
    source_audit = receipt.get("source_audit")
    if not isinstance(source_audit, dict) or set(source_audit) != {
        "path",
        "bytes",
        "sha256",
        "audit",
    }:
        raise ValueError("Candidate-breadth matrix lost its source-audit binding")
    if (
        not isinstance(source_audit["path"], str)
        or not isinstance(source_audit["bytes"], int)
        or not isinstance(source_audit["sha256"], str)
        or not isinstance(source_audit["audit"], dict)
    ):
        raise ValueError("Candidate-breadth source-audit identity is malformed")
    source_path = (root / source_audit["path"]).resolve()
    try:
        source_path.relative_to(root)
    except ValueError as error:
        raise ValueError("Candidate-breadth source-audit receipt escaped the study root") from error
    if (
        not source_path.is_file()
        or source_path.stat().st_size != source_audit["bytes"]
        or _sha256(source_path) != source_audit["sha256"]
        or json.loads(source_path.read_text(encoding="utf-8")) != source_audit["audit"]
        or source_audit["audit"].get("upstream_reconstruction_verified") is not True
        or source_audit["audit"].get("protocol_sha256") != _sha256(protocol_path)
    ):
        raise ValueError("Candidate-breadth source-audit receipt changed")
    expected_local_audit = {
        **source_audit["audit"],
        "upstream_reconstruction_verified": False,
    }
    gpus = receipt.get("gpus")
    jobs = receipt.get("jobs")
    if (
        set(receipt)
        != {
            "schema_version",
            "status",
            "protocol",
            "data_audit",
            "source_audit",
            "gpus",
            "jobs",
        }
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("protocol") != expected_protocol
        or receipt.get("data_audit") != expected_local_audit
        or not isinstance(gpus, list)
        or not gpus
        or len(set(gpus)) != len(gpus)
        or any(not isinstance(gpu, str) or not gpu.isdigit() for gpu in gpus)
        or not isinstance(jobs, list)
    ):
        raise ValueError("Candidate-breadth matrix receipt contract changed")

    evaluation = protocol["evaluation"]
    step = int(evaluation["checkpoint_step"])
    by_run = {str(job.get("run_id")): job for job in jobs if isinstance(job, dict)}
    if set(by_run) != set(evaluation["run_ids"]) or len(jobs) != len(by_run):
        raise ValueError("Candidate-breadth matrix job coverage changed")
    for run_id in evaluation["run_ids"]:
        job = by_run[run_id]
        manifest_path = results_root / run_id / "manifest.json"
        checkpoint = root / evaluation["checkpoint_root"] / run_id / f"checkpoint-{step}"
        expected_manifest = {
            "path": str(manifest_path.relative_to(root)),
            "bytes": manifest_path.stat().st_size if manifest_path.is_file() else -1,
            "sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        }
        attempts = job.get("attempts")
        if (
            set(job)
            != {
                "run_id",
                "gpu",
                "attempts",
                "checkpoint",
                "manifest",
                "baseline_maximum_absolute_error",
            }
            or job.get("gpu") not in gpus
            or job.get("checkpoint") != str(checkpoint)
            or job.get("manifest") != expected_manifest
            or not isinstance(attempts, list)
            or not attempts
            or any(
                not isinstance(attempt, dict)
                or set(attempt) != {"attempt", "returncode"}
                or int(attempt["attempt"]) != index
                for index, attempt in enumerate(attempts, start=1)
            )
            or attempts[-1]["returncode"] != 0
        ):
            raise ValueError(f"Candidate-breadth matrix job changed: {run_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_error = (manifest.get("baseline_reproduction") or {}).get("maximum_absolute_error")
        if job.get("baseline_maximum_absolute_error") != observed_error:
            raise ValueError(f"Candidate-breadth matrix baseline changed: {run_id}")
    return {
        "path": str(receipt_path.relative_to(root)),
        "bytes": receipt_path.stat().st_size,
        "sha256": _sha256(receipt_path),
        "source_audit": {
            "path": source_audit["path"],
            "sha256": source_audit["sha256"],
        },
    }


def build_candidate_breadth_summary(
    protocol_path: str | Path,
    *,
    output_dir: str | Path = "reports/candidate-breadth",
    audit_only: bool = False,
) -> dict[str, Any]:
    protocol_path, protocol = load_candidate_breadth_protocol(protocol_path)
    root = protocol_path.parent.parent.resolve()
    evaluation = protocol["evaluation"]
    results_root = (root / evaluation["results_root"]).resolve()
    beir_path = root / "reports" / "dense-discovery" / "checkpoint_summary.csv"
    beir = _load_beir_scores(beir_path)
    widths = protocol["candidate_construction"]["negative_widths"]
    matrix_provenance = _matrix_provenance(
        root=root,
        protocol_path=protocol_path,
        protocol=protocol,
        results_root=results_root,
    )
    group_by_run: dict[str, dict[int, dict[str, Any]]] = {}
    samples_by_run: dict[str, list[dict[str, Any]]] = {}
    inputs = []
    baseline_errors = []
    for run_id in evaluation["run_ids"]:
        run_root = results_root / run_id
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("protocol", {}).get("sha256") != _sha256(protocol_path)
            or manifest.get("negative_widths") != widths
        ):
            raise ValueError(f"Candidate-breadth manifest changed: {run_id}")
        baseline = manifest.get("baseline_reproduction") or {}
        baseline_errors.append(float(baseline.get("maximum_absolute_error", math.inf)))
        for item in manifest["outputs"].values():
            path = run_root / item["path"]
            if (
                not path.is_file()
                or path.stat().st_size != item["bytes"]
                or _sha256(path) != item["sha256"]
            ):
                raise ValueError(f"Candidate-breadth output changed: {path}")
        groups = {
            int(row["negative_width"]): row
            for row in _read_jsonl(run_root / manifest["outputs"]["group_metrics"]["path"])
            if row["source"] == "__all__"
        }
        if set(groups) != set(widths):
            raise ValueError(f"Candidate-breadth group coverage changed: {run_id}")
        group_by_run[run_id] = groups
        samples_by_run[run_id] = _read_jsonl(
            run_root / manifest["outputs"]["sample_metrics"]["path"]
        )
        inputs.append(
            {
                "run_id": run_id,
                "manifest_path": str(manifest_path.relative_to(root)),
                "manifest_sha256": _sha256(manifest_path),
            }
        )

    calibration_rows = []
    run_ids_by_optimizer: dict[str, list[str]] = defaultdict(list)
    for run_id, row in beir.items():
        run_ids_by_optimizer[row["optimizer"]].append(run_id)
    for optimizer, run_ids in sorted(run_ids_by_optimizer.items()):
        run_ids.sort(key=lambda run_id: beir[run_id]["learning_rate"])
        outcomes = [beir[run_id]["mean_ndcg_at_10"] for run_id in run_ids]
        for width in widths:
            calibration_rows.append(
                {
                    "optimizer": optimizer,
                    "negative_width": width,
                    "runs": len(run_ids),
                    "loss_beir_spearman": spearman(
                        [
                            float(group_by_run[run_id][width]["contrastive_loss"])
                            for run_id in run_ids
                        ],
                        outcomes,
                    ),
                    "margin_beir_spearman": spearman(
                        [
                            float(group_by_run[run_id][width]["positive_margin"])
                            for run_id in run_ids
                        ],
                        outcomes,
                    ),
                }
            )

    contrast_rows = []
    for optimizer in ("muon", "normuon"):
        optimal_id = f"{optimizer}-lr3e-4"
        high_id = f"{optimizer}-lr3e-3"
        indexed = {
            run_id: {
                (int(row["negative_width"]), int(row["sample_id"])): row
                for row in samples_by_run[run_id]
            }
            for run_id in (optimal_id, high_id)
        }
        if set(indexed[optimal_id]) != set(indexed[high_id]):
            raise ValueError(f"Candidate-breadth sample pairing changed for {optimizer}")
        for width in widths:
            keys = sorted(key for key in indexed[optimal_id] if key[0] == width)
            row = {
                "optimizer": optimizer,
                "negative_width": width,
                "samples": len(keys),
            }
            for metric in METRICS:
                deltas = [
                    float(indexed[high_id][key][metric]) - float(indexed[optimal_id][key][metric])
                    for key in keys
                ]
                row[f"{metric}_delta"] = sum(deltas) / len(deltas)
                lower_is_better = metric in {
                    "contrastive_loss",
                    "hardest_negative_score",
                }
                row[f"{metric}_high_dose_better_fraction"] = sum(
                    delta < 0 if lower_is_better else delta > 0 for delta in deltas
                ) / len(deltas)
            contrast_rows.append(row)

    baseline_pass = len(baseline_errors) == 12 and max(baseline_errors) <= 1e-5
    decision = candidate_breadth_decision(contrast_rows, baseline_pass=baseline_pass)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    summary_path = output_dir / "summary.json"
    summary_fields = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": {
            "path": str(protocol_path.relative_to(root)),
            "sha256": _sha256(protocol_path),
        },
        "inputs": inputs,
        "matrix_provenance": matrix_provenance,
        "beir_checkpoint_summary": {
            "path": str(beir_path.relative_to(root)),
            "bytes": beir_path.stat().st_size,
            "sha256": _sha256(beir_path),
        },
        "baseline_maximum_absolute_error": max(baseline_errors),
        "decision": decision,
        "calibration_rows": len(calibration_rows),
        "contrast_rows": len(contrast_rows),
        "claim_boundary": protocol["claim_boundary"],
    }
    if audit_only:
        if not summary_path.is_file():
            raise FileNotFoundError("Candidate-breadth summary outputs are incomplete")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        outputs = _candidate_breadth_outputs(
            calibration_rows,
            contrast_rows,
            output_dir,
            audit_only=True,
        )
        if existing != {**summary_fields, "outputs": outputs}:
            raise ValueError("Candidate-breadth summary differs from exact recomputation")
        return existing

    outputs = _candidate_breadth_outputs(
        calibration_rows,
        contrast_rows,
        output_dir,
        audit_only=False,
    )
    summary = {
        **summary_fields,
        "outputs": outputs,
    }
    _atomic_json(summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize optimizer calibration over nested candidate widths"
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/candidate_breadth_probe.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/candidate-breadth"))
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_candidate_breadth_summary(
        args.protocol, output_dir=args.output_dir, audit_only=args.audit_only
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
