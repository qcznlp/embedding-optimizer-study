from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .aggregate import audit_experiment_contract
from .config import RunConfig, load_matrix, resolve_matrix_path
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .representation_summary import (
    IDENTITY_FIELDS,
    REPRESENTATION_FIELDS,
    SCORE_FIELDS,
    TOKEN_METRICS,
)

OPTIMIZERS = ("adamw", "muon", "normuon")
COLORS = {"adamw": "#3B6FB6", "muon": "#D9792B", "normuon": "#3A9B62"}
LABELS = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}
TOKEN_FIELDS = [
    f"{prefix}_{stat}" for prefix in TOKEN_METRICS.values() for stat in ("mean", "median")
]
CHECKPOINT_FIELDS = [*IDENTITY_FIELDS, *SCORE_FIELDS, *TOKEN_FIELDS]
REPRESENTATION_TABLE_FIELDS = [
    *IDENTITY_FIELDS,
    "representation_role",
    *REPRESENTATION_FIELDS,
]


def _resolve_declared_path(summary_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (summary_dir / path).resolve()


def _read_declared_csv(
    summary_dir: Path,
    manifest: dict[str, Any],
    name: str,
    expected_fields: list[str] | None = None,
) -> tuple[list[dict[str, str]], Path]:
    declared = manifest.get("outputs", {}).get(name)
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        raise ValueError(f"Representation summary does not declare {name}.csv")
    path = _resolve_declared_path(summary_dir, declared["path"])
    if (
        not path.is_file()
        or path.stat().st_size != declared.get("bytes")
        or _sha256(path) != declared.get("sha256")
    ):
        raise ValueError(f"{name}.csv differs from its representation summary manifest")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if expected_fields is not None and list(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"{name}.csv schema changed")
        rows = list(reader)
    if len(rows) != declared.get("rows"):
        raise ValueError(f"{name}.csv row count differs from its manifest")
    return rows, path


def _expected_identities(configs: list[RunConfig]) -> set[tuple[str, str, str, int]]:
    identities = {
        (family, "reference", "pretrained", 0)
        for family in sorted({config.model_family for config in configs})
    }
    for config in configs:
        identities.update(
            (config.model_family, "checkpoint", config.run_id, stage) for stage in range(1, 6)
        )
    return identities


def _identity(row: dict[str, str]) -> tuple[str, str, str, int]:
    try:
        return row["family"], row["kind"], row["run_id"], int(row["stage"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid representation identity: {row}") from error


def _finite(row: dict[str, str], field: str, *, allow_empty: bool = False) -> float | None:
    value = row.get(field, "")
    if allow_empty and value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} value in representation summary") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field} value in representation summary")
    return parsed


def _validate_checkpoint_row(row: dict[str, str]) -> None:
    family, kind, run_id, stage = _identity(row)
    if family not in {"dense", "late"} or kind not in {"reference", "checkpoint"}:
        raise ValueError(f"Invalid checkpoint-metric family/kind: {row}")
    fraction = _finite(row, "fraction")
    step = _finite(row, "step")
    margin = _finite(row, "margin_mean")
    del margin
    if kind == "reference":
        if (
            run_id != "pretrained"
            or stage != 0
            or fraction != 0
            or step != 0
            or row["optimizer"] != ""
            or row["learning_rate"] != ""
            or row["reference_top1_agreement"] != ""
        ):
            raise ValueError("Malformed pretrained representation row")
    else:
        if (
            stage not in range(1, 6)
            or not math.isclose(float(fraction), stage / 5, rel_tol=0, abs_tol=1e-12)
            or row["optimizer"] not in OPTIMIZERS
            or not row["learning_rate"]
        ):
            raise ValueError("Malformed checkpoint representation row")
        agreement = _finite(row, "reference_top1_agreement")
        if agreement is None or not 0 <= agreement <= 1:
            raise ValueError("Reference top-1 agreement is outside [0, 1]")
    expected_scorer = "cosine" if family == "dense" else "mean_maxsim_cosine"
    if row["scorer"] != expected_scorer:
        raise ValueError("Representation scorer differs from the family contract")
    for field in TOKEN_FIELDS:
        value = _finite(row, field, allow_empty=family == "dense")
        if family == "late" and value is None:
            raise ValueError("LateOn token-utilization metric is missing")
        if family == "dense" and value is not None:
            raise ValueError("DenseOn unexpectedly contains token-utilization metrics")


def _validate_representation_row(row: dict[str, str]) -> None:
    family, _, _, _ = _identity(row)
    roles = {
        "dense": {"queries", "documents"},
        "late": {"query_tokens", "document_tokens", "pooled_queries", "pooled_documents"},
    }
    if family not in roles or row["representation_role"] not in roles[family]:
        raise ValueError("Representation role differs from the family contract")
    normalized_rank = _finite(row, "normalized_effective_rank")
    if normalized_rank is None or not 0 <= normalized_rank <= 1 + 1e-9:
        raise ValueError("Normalized representation effective rank is outside [0, 1]")


def load_representation_summary(
    summary_dir: Path,
    configs: list[RunConfig],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    summary_dir = summary_dir.resolve()
    manifest_path = summary_dir / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _expected_identities(configs)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("allow_partial") is not False
        or manifest.get("expected_jobs") != len(expected)
        or manifest.get("valid_jobs") != len(expected)
        or manifest.get("missing_labels") != []
    ):
        raise ValueError("Representation summary is incomplete or noncanonical")
    probe = manifest.get("probe") or {}
    for field in ("manifest_path", "spec_path"):
        path = Path(str(probe.get(field, "")))
        digest_field = field.replace("path", "sha256")
        if not path.is_file() or _sha256(path) != probe.get(digest_field):
            raise ValueError("Representation probe identity no longer matches its summary")

    checkpoint_rows, checkpoint_path = _read_declared_csv(
        summary_dir, manifest, "checkpoint_metrics", CHECKPOINT_FIELDS
    )
    representation_rows, representation_path = _read_declared_csv(
        summary_dir, manifest, "representation_metrics", REPRESENTATION_TABLE_FIELDS
    )
    _, group_path = _read_declared_csv(summary_dir, manifest, "group_metrics")

    checkpoint_identities = [_identity(row) for row in checkpoint_rows]
    if (
        len(checkpoint_identities) != len(set(checkpoint_identities))
        or set(checkpoint_identities) != expected
    ):
        raise ValueError("Checkpoint-metric identities do not cover the frozen matrix")
    for row in checkpoint_rows:
        _validate_checkpoint_row(row)

    representations: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for row in representation_rows:
        _validate_representation_row(row)
        identity = _identity(row)
        if row["representation_role"] in representations[identity]:
            raise ValueError("Duplicate representation role for one checkpoint")
        representations[identity].add(row["representation_role"])
    expected_roles = {
        "dense": {"queries", "documents"},
        "late": {"query_tokens", "document_tokens", "pooled_queries", "pooled_documents"},
    }
    if set(representations) != expected or any(
        roles != expected_roles[identity[0]] for identity, roles in representations.items()
    ):
        raise ValueError("Representation-role coverage does not match the frozen matrix")

    provenance = {
        "summary_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "probe": probe,
        "checkpoint_metrics": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
        "representation_metrics": {
            "path": str(representation_path),
            "sha256": _sha256(representation_path),
        },
        "group_metrics": {"path": str(group_path), "sha256": _sha256(group_path)},
    }
    return checkpoint_rows, representation_rows, provenance


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _metric_values(
    checkpoint_rows: list[dict[str, str]],
    representation_rows: list[dict[str, str]],
    *,
    family: str,
    optimizer: str,
    fraction: float,
    metric: str,
) -> list[float]:
    if metric == "normalized_effective_rank":
        role = "queries" if family == "dense" else "query_tokens"
        selected = [
            row
            for row in representation_rows
            if row["family"] == family
            and row["kind"] == "checkpoint"
            and row["optimizer"] == optimizer
            and math.isclose(float(row["fraction"]), fraction, rel_tol=0, abs_tol=1e-12)
            and row["representation_role"] == role
        ]
    else:
        selected = [
            row
            for row in checkpoint_rows
            if row["family"] == family
            and row["kind"] == "checkpoint"
            and row["optimizer"] == optimizer
            and math.isclose(float(row["fraction"]), fraction, rel_tol=0, abs_tol=1e-12)
        ]
    if len(selected) != 4:
        raise ValueError(
            f"Expected four learning-rate values for {family}/{optimizer}/{fraction}/{metric}"
        )
    return [float(row[metric]) for row in selected]


def _reference_value(
    checkpoint_rows: list[dict[str, str]],
    representation_rows: list[dict[str, str]],
    *,
    family: str,
    metric: str,
) -> float:
    if metric == "reference_top1_agreement":
        return 1.0
    if metric == "normalized_effective_rank":
        role = "queries" if family == "dense" else "query_tokens"
        selected = [
            row
            for row in representation_rows
            if row["family"] == family
            and row["kind"] == "reference"
            and row["representation_role"] == role
        ]
    else:
        selected = [
            row for row in checkpoint_rows if row["family"] == family and row["kind"] == "reference"
        ]
    if len(selected) != 1:
        raise ValueError(f"Expected one pretrained value for {family}/{metric}")
    return float(selected[0][metric])


def plot_representation_dynamics(
    matrix_path: Path,
    training_summary: Path,
    unseen_summary: Path,
    output_path: Path,
) -> dict[str, Any]:
    matrix_path = resolve_matrix_path(matrix_path).resolve()
    configs = load_matrix(matrix_path)
    audit = audit_experiment_contract(configs)
    if audit["complete"] is not True:
        raise ValueError("Experiment matrix differs from the frozen 24-run contract")
    tiers = {
        "training": load_representation_summary(training_summary, configs),
        "unseen": load_representation_summary(unseen_summary, configs),
    }
    if tiers["training"][2]["probe"]["spec_sha256"] == tiers["unseen"][2]["probe"]["spec_sha256"]:
        raise ValueError("Training and unseen summaries point to the same probe specification")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the analysis extra to plot representation dynamics") from error

    row_grid = [
        ("training", "dense", "DenseOn · training probe"),
        ("unseen", "dense", "DenseOn · unseen BEIR probe"),
        ("training", "late", "LateOn · training probe"),
        ("unseen", "late", "LateOn · unseen BEIR probe"),
    ]
    metrics = [
        ("margin_mean", "Positive − hardest-negative margin"),
        ("normalized_effective_rank", "Query normalized effective rank"),
        ("reference_top1_agreement", "Top-1 agreement with pretrained"),
    ]
    fractions = [stage / 5 for stage in range(1, 6)]
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "embedding-optimizer-study-representation-dynamics",
        }
    ):
        figure, axes = plt.subplots(
            len(row_grid),
            len(metrics),
            figsize=(11.2, 9.4),
            squeeze=False,
            sharex=True,
        )
        for row_index, (tier, family, row_label) in enumerate(row_grid):
            checkpoint_rows, representation_rows, _ = tiers[tier]
            for column_index, (metric, title) in enumerate(metrics):
                axis = axes[row_index][column_index]
                reference = _reference_value(
                    checkpoint_rows,
                    representation_rows,
                    family=family,
                    metric=metric,
                )
                for optimizer in OPTIMIZERS:
                    medians = [reference]
                    lowers = [reference]
                    uppers = [reference]
                    for fraction in fractions:
                        values = _metric_values(
                            checkpoint_rows,
                            representation_rows,
                            family=family,
                            optimizer=optimizer,
                            fraction=fraction,
                            metric=metric,
                        )
                        medians.append(_percentile(values, 0.5))
                        lowers.append(_percentile(values, 0.25))
                        uppers.append(_percentile(values, 0.75))
                    x_values = [0.0, *fractions]
                    axis.plot(
                        x_values,
                        medians,
                        color=COLORS[optimizer],
                        marker="o",
                        markersize=3,
                        linewidth=1.45,
                        label=LABELS[optimizer],
                    )
                    axis.fill_between(
                        x_values,
                        lowers,
                        uppers,
                        color=COLORS[optimizer],
                        alpha=0.13,
                        linewidth=0,
                    )
                axis.set_xlim(-0.025, 1.025)
                axis.set_xticks([0, *fractions])
                axis.grid(color="#cccccc", linewidth=0.55, alpha=0.6)
                if row_index == 0:
                    axis.set_title(title, fontweight="bold")
                if column_index == 0:
                    axis.set_ylabel(row_label)
                if row_index == len(row_grid) - 1:
                    axis.set_xlabel("Training fraction (0 = pretrained)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.966),
            ncol=3,
            frameon=False,
        )
        figure.suptitle(
            "Representation and score geometry across optimizer trajectories",
            fontsize=13,
            fontweight="bold",
            y=0.997,
        )
        figure.text(
            0.5,
            0.008,
            "Lines are medians and bands are interquartile ranges across the four learning-rate "
            "runs. The shared x=0 point is the pretrained reference; training-probe examples were "
            "seen during fine-tuning, while the BEIR probe is held out.",
            ha="center",
            fontsize=7.4,
            color="#444444",
        )
        figure.tight_layout(rect=(0.025, 0.04, 0.995, 0.935))
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.stem}.tmp.{os.getpid()}{output_path.suffix}"
        )
        figure.savefig(
            temporary,
            format=output_path.suffix.lstrip("."),
            metadata={"Date": None, "Creator": "embedding-optimizer-study"},
        )
        plt.close(figure)
        os.replace(temporary, output_path)

    result = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
        "sources": {tier: values[2] for tier, values in tiers.items()},
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "jobs": sum(len(values[0]) for values in tiers.values()),
        "families": ["dense", "late"],
        "optimizers": list(OPTIMIZERS),
        "metrics": [metric for metric, _ in metrics],
        "aggregation": "median-and-interquartile-range-over-four-learning-rates",
        "pretrained_reference_fraction": 0.0,
    }
    _atomic_json(output_path.with_suffix(".manifest.json"), result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot strict training and unseen representation-probe dynamics"
    )
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=Path("results/representation-space/training/summary"),
    )
    parser.add_argument(
        "--unseen-summary",
        type=Path,
        default=Path("results/representation-space/decontaminated-beir/summary"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/representation-space/representation-dynamics.svg"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            plot_representation_dynamics(
                args.matrix,
                args.training_summary,
                args.unseen_summary,
                args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
