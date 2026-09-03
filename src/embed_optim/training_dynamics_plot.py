from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

FAMILIES = ("dense", "late")
OPTIMIZERS = ("adamw", "muon", "normuon")
FAMILY_LABELS = {"dense": "DenseOn", "late": "LateOn"}
OPTIMIZER_LABELS = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}
OPTIMIZER_COLORS = {"adamw": "#4C78A8", "muon": "#F58518", "normuon": "#54A24B"}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_declared(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _portable_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root.resolve()))
    except ValueError:
        return str(resolved)


def _read_declared_csv(
    repository_root: Path,
    manifest: dict[str, Any],
    output_key: str,
    *,
    required_fields: set[str],
) -> tuple[list[dict[str, str]], Path]:
    declared = manifest.get("outputs", {}).get(output_key)
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        raise ValueError(f"Training summary does not declare {output_key}")
    path = _resolve_declared(repository_root, declared["path"])
    if (
        not path.is_file()
        or path.stat().st_size != declared.get("bytes")
        or _sha256(path) != declared.get("sha256")
    ):
        raise ValueError(f"Declared training table differs from its manifest: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not required_fields.issubset(fields):
            raise ValueError(f"Required fields are absent from {path}")
        rows = list(reader)
    if len(rows) != declared.get("rows"):
        raise ValueError(f"Declared row count differs for {path}")
    return rows, path


def _finite(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} value in training plot source: {row}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} value in training plot source: {row}")
    return value


def _validate_stage_rows(rows: list[dict[str, str]]) -> None:
    indexed: dict[tuple[str, str, str, int], dict[str, str]] = {}
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        family = row.get("model_family", "")
        optimizer = row.get("optimizer", "")
        run_id = row.get("run_id", "")
        try:
            stage = int(row.get("stage", ""))
        except ValueError as error:
            raise ValueError(f"Invalid stage in training plot source: {row}") from error
        identity = (family, optimizer, run_id, stage)
        if (
            family not in FAMILIES
            or optimizer not in OPTIMIZERS
            or not run_id
            or not 1 <= stage <= 5
            or identity in indexed
        ):
            raise ValueError(f"Invalid or duplicate training-stage identity: {identity}")
        fraction = _finite(row, "fraction")
        loss = _finite(row, "mean_loss")
        loss_sd = _finite(row, "loss_standard_deviation")
        learning_rate = _finite(row, "learning_rate")
        if fraction != stage / 5 or loss <= 0 or loss_sd < 0 or learning_rate <= 0:
            raise ValueError(f"Invalid training-stage measurement: {identity}")
        indexed[identity] = row
        grouped[(family, optimizer)].add(run_id)
    expected_groups = {(family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS}
    if (
        len(rows) != 120
        or set(grouped) != expected_groups
        or any(len(run_ids) != 4 for run_ids in grouped.values())
        or any(
            {identity[3] for identity in indexed if identity[:3] == (family, optimizer, run_id)}
            != {1, 2, 3, 4, 5}
            for (family, optimizer), run_ids in grouped.items()
            for run_id in run_ids
        )
    ):
        raise ValueError("Training plot requires 24 complete five-stage learning-rate trajectories")


def _validate_system_rows(rows: list[dict[str, str]]) -> None:
    indexed = {}
    for row in rows:
        identity = (row.get("model_family", ""), row.get("optimizer", ""))
        if (
            identity[0] not in FAMILIES
            or identity[1] not in OPTIMIZERS
            or identity in indexed
            or int(row.get("learning_rate_points", 0)) != 4
        ):
            raise ValueError(f"Invalid or duplicate system-summary identity: {identity}")
        for field in (
            "throughput_to_adamw_ratio",
            "optimizer_state_to_adamw_ratio",
            "wall_time_hours_median",
            "samples_per_second_median",
            "optimizer_state_gib_median",
        ):
            if _finite(row, field) <= 0:
                raise ValueError(f"Invalid {field} for {identity}")
        indexed[identity] = row
    expected = {(family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS}
    if len(rows) != 6 or set(indexed) != expected:
        raise ValueError("System plot requires all six family/optimizer summaries")
    for family in FAMILIES:
        baseline = indexed[(family, "adamw")]
        if (
            abs(_finite(baseline, "throughput_to_adamw_ratio") - 1) > 1e-12
            or abs(_finite(baseline, "optimizer_state_to_adamw_ratio") - 1) > 1e-12
        ):
            raise ValueError(f"AdamW system baseline differs for {family}")


def _atomic_figure(figure: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_format = path.suffix.removeprefix(".")
    if image_format not in {"png", "svg"}:
        raise ValueError(f"Unsupported training figure format: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{image_format}")
    figure.savefig(
        temporary,
        format=image_format,
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "embedding-optimizer-study"},
    )
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _loss_figure(rows: list[dict[str, str]], path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "embedding-optimizer-training-loss-v1"
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharex=True, sharey="row")
    for row_index, family in enumerate(FAMILIES):
        for column_index, optimizer in enumerate(OPTIMIZERS):
            axis = axes[row_index][column_index]
            subset = [
                row
                for row in rows
                if row["model_family"] == family and row["optimizer"] == optimizer
            ]
            by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in subset:
                by_run[row["run_id"]].append(row)
            ordered_runs = sorted(
                by_run.items(), key=lambda item: _finite(item[1][0], "learning_rate")
            )
            colors = plt.get_cmap("viridis")([0.12, 0.38, 0.64, 0.90])
            for color, (_run_id, values) in zip(colors, ordered_runs, strict=True):
                ordered = sorted(values, key=lambda row: int(row["stage"]))
                fractions = [_finite(row, "fraction") for row in ordered]
                losses = [_finite(row, "mean_loss") for row in ordered]
                deviations = [_finite(row, "loss_standard_deviation") for row in ordered]
                learning_rate = _finite(ordered[0], "learning_rate")
                axis.plot(
                    fractions,
                    losses,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    color=color,
                    label=f"{learning_rate:.0e}",
                )
                axis.fill_between(
                    fractions,
                    [loss - deviation for loss, deviation in zip(losses, deviations, strict=True)],
                    [loss + deviation for loss, deviation in zip(losses, deviations, strict=True)],
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
            axis.set_title(f"{FAMILY_LABELS[family]} — {OPTIMIZER_LABELS[optimizer]}")
            axis.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
            axis.grid(alpha=0.22)
            if row_index == 1:
                axis.set_xlabel("Training fraction")
            if column_index == 0:
                axis.set_ylabel("Trailing-window contrastive loss")
            axis.legend(title="Configured LR", fontsize=7.5, title_fontsize=8, frameon=False)
    figure.suptitle(
        "Discovery training dynamics: all four learning-rate trajectories",
        fontsize=13,
    )
    figure.tight_layout()
    record = _atomic_figure(figure, path)
    plt.close(figure)
    return record


def _system_figure(rows: list[dict[str, str]], path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "embedding-optimizer-system-tradeoffs-v1"
    import matplotlib.pyplot as plt
    import numpy as np

    indexed = {(row["model_family"], row["optimizer"]): row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    x = np.arange(len(FAMILIES), dtype=float)
    width = 0.24
    metrics = (
        ("throughput_to_adamw_ratio", "Training throughput / AdamW"),
        ("optimizer_state_to_adamw_ratio", "Optimizer-state size / AdamW"),
    )
    for axis, (field, title) in zip(axes, metrics, strict=True):
        for optimizer_index, optimizer in enumerate(OPTIMIZERS):
            values = [_finite(indexed[(family, optimizer)], field) for family in FAMILIES]
            axis.bar(
                x + (optimizer_index - 1) * width,
                values,
                width,
                color=OPTIMIZER_COLORS[optimizer],
                label=OPTIMIZER_LABELS[optimizer],
            )
            for position, value in zip(x + (optimizer_index - 1) * width, values, strict=True):
                axis.text(position, value + 0.018, f"{value:.2f}", ha="center", fontsize=8)
        axis.axhline(1, color="#444444", linewidth=1, linestyle="--", alpha=0.6)
        axis.set_xticks(x, [FAMILY_LABELS[family] for family in FAMILIES])
        axis.set_title(title)
        axis.set_ylim(0, 1.12)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Ratio (AdamW = 1.0)")
    axes[1].legend(frameon=False, loc="lower right")
    figure.suptitle("Native recipe systems trade-offs across four LR sweep points", fontsize=13)
    figure.tight_layout()
    record = _atomic_figure(figure, path)
    plt.close(figure)
    return record


def plot_training_dynamics(
    summary_dir: str | Path = "reports/training-dynamics",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    summary = Path(summary_dir).resolve()
    repository_root = summary.parent.parent
    manifest_path = summary / "summary_manifest.json"
    manifest = _load_json(manifest_path)
    coverage = manifest.get("coverage", {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or coverage.get("runs") != 24
        or coverage.get("checkpoints") != 120
        or coverage.get("optimizer_family_groups") != 6
        or coverage.get("history_rows") != 9_384
    ):
        raise ValueError("Training plot source is not the strict complete discovery summary")
    stage_rows, stage_path = _read_declared_csv(
        repository_root,
        manifest,
        "stages",
        required_fields={
            "model_family",
            "optimizer",
            "learning_rate",
            "run_id",
            "stage",
            "fraction",
            "mean_loss",
            "loss_standard_deviation",
        },
    )
    system_rows, system_path = _read_declared_csv(
        repository_root,
        manifest,
        "optimizer_systems",
        required_fields={
            "model_family",
            "optimizer",
            "learning_rate_points",
            "wall_time_hours_median",
            "samples_per_second_median",
            "throughput_to_adamw_ratio",
            "optimizer_state_gib_median",
            "optimizer_state_to_adamw_ratio",
        },
    )
    _validate_stage_rows(stage_rows)
    _validate_system_rows(system_rows)
    output = Path(output_dir).resolve() if output_dir is not None else summary
    outputs = {
        "training_loss_dynamics.svg": _loss_figure(
            stage_rows, output / "training_loss_dynamics.svg"
        ),
        "system_tradeoffs.svg": _system_figure(system_rows, output / "system_tradeoffs.svg"),
    }
    for record in outputs.values():
        record["path"] = _portable_path(Path(record["path"]), repository_root)
    plot_manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "source_summary": {
            "path": _portable_path(manifest_path, repository_root),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "sources": {
            "stages": {
                "path": _portable_path(stage_path, repository_root),
                "bytes": stage_path.stat().st_size,
                "sha256": _sha256(stage_path),
                "rows": len(stage_rows),
            },
            "optimizer_systems": {
                "path": _portable_path(system_path, repository_root),
                "bytes": system_path.stat().st_size,
                "sha256": _sha256(system_path),
                "rows": len(system_rows),
            },
        },
        "outputs": outputs,
        "interpretation": (
            "Loss bands summarize variation across ten trailing logged observations, not random "
            "seeds. System bars are medians across four learning-rate sweep points on matched "
            "hardware; they do not establish time-to-retrieval-quality."
        ),
    }
    _atomic_json(output / "plot_manifest.json", plot_manifest)
    return plot_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render strictly sourced discovery training and systems figures"
    )
    parser.add_argument("--summary-dir", type=Path, default=Path("reports/training-dynamics"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = plot_training_dynamics(args.summary_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
