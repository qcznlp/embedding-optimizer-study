from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    (
        "Displacement scale",
        "normuon_to_muon_displacement_ratio",
        (0.994, 1.006),
    ),
    ("Row-norm balance", "normuon_to_muon_row_cv_ratio", (0.15, 0.48)),
    (
        "Top-row concentration",
        "normuon_to_muon_top_1pct_row_energy_ratio",
        (0.55, 0.75),
    ),
)
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
PHASE_METRICS = (
    ("Row-norm CV", "reference_delta_row_cv_parameter_weighted"),
    ("Top-1% row energy share", "reference_delta_top_1pct_row_energy_parameter_weighted"),
)
OPTIMIZER_STYLES = {
    "adamw": ("#0072B2", "o", "AdamW"),
    "muon": ("#E69F00", "s", "Muon"),
    "normuon": ("#009E73", "^", "NorMuon"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_contrasts(path: Path) -> tuple[list[dict[str, Any]], list[str], list[float], list[int]]:
    required = {
        "model_family",
        "learning_rate",
        "stage",
        "step",
        *(field for _, field, _ in METRICS),
    }
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing contrast columns in {path}: {sorted(missing)}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"No optimizer contrast rows in {path}")

    rows: list[dict[str, Any]] = []
    series_stages: dict[tuple[str, float], set[int]] = defaultdict(set)
    for raw in raw_rows:
        family = raw["model_family"]
        learning_rate = float(raw["learning_rate"])
        stage = int(raw["stage"])
        step = int(raw["step"])
        metrics = {field: float(raw[field]) for _, field, _ in METRICS}
        if not family or learning_rate <= 0 or stage <= 0 or step <= 0:
            raise ValueError(f"Invalid optimizer contrast identity: {raw}")
        if not all(math.isfinite(value) and value > 0 for value in metrics.values()):
            raise ValueError(f"Invalid optimizer contrast metric: {raw}")
        key = (family, learning_rate)
        if stage in series_stages[key]:
            raise ValueError(f"Duplicate contrast stage for {key}: {stage}")
        series_stages[key].add(stage)
        rows.append(
            {
                "model_family": family,
                "learning_rate": learning_rate,
                "stage": stage,
                "step": step,
                **metrics,
            }
        )

    stage_sets = set(map(frozenset, series_stages.values()))
    if len(stage_sets) != 1:
        raise ValueError("Optimizer contrast series have different stage coverage")
    families = sorted(
        {row["model_family"] for row in rows}, key=lambda name: (name != "dense", name)
    )
    learning_rates = sorted({row["learning_rate"] for row in rows})
    stages = sorted(next(iter(stage_sets)))
    return rows, families, learning_rates, stages


def plot_pair_contrasts(input_path: Path, output_path: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    rows, families, learning_rates, stages = _read_contrasts(input_path)
    if len(learning_rates) > len(COLORS):
        raise ValueError(f"At most {len(COLORS)} learning rates can be plotted")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the analysis extra to plot geometry") from error

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_family"], row["learning_rate"])].append(row)
    for series in grouped.values():
        series.sort(key=lambda row: row["stage"])

    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "embedding-optimizer-study-geometry",
        }
    ):
        figure, axes = plt.subplots(
            len(families),
            len(METRICS),
            figsize=(11.2, 3.15 * len(families)),
            squeeze=False,
            sharex=True,
        )
        for row_index, family in enumerate(families):
            for metric_index, (title, field, default_limits) in enumerate(METRICS):
                axis = axes[row_index][metric_index]
                values: list[float] = []
                for color, learning_rate in zip(COLORS, learning_rates, strict=False):
                    series = grouped.get((family, learning_rate))
                    if series is None:
                        continue
                    y_values = [item[field] for item in series]
                    values.extend(y_values)
                    axis.plot(
                        stages,
                        y_values,
                        color=color,
                        marker="o",
                        linewidth=1.7,
                        markersize=4,
                        label=f"{learning_rate:g}",
                    )
                if not values:
                    raise ValueError(f"No values for {family}/{field}")
                span = max(values) - min(values)
                padding = max(span * 0.08, 0.0005)
                axis.set_ylim(
                    min(default_limits[0], min(values) - padding),
                    max(default_limits[1], max(values) + padding),
                )
                if metric_index == 0:
                    axis.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--")
                    axis.set_ylabel(f"{family.capitalize()}On\nNorMuon / Muon")
                if row_index == 0:
                    axis.set_title(title, fontweight="bold")
                axis.set_xticks(stages)
                axis.grid(axis="y", color="#cccccc", linewidth=0.6, alpha=0.65)
                if row_index == len(families) - 1:
                    if stages == [1, 2, 3, 4, 5]:
                        axis.set_xticklabels(["20%", "40%", "60%", "80%", "100%"])
                    axis.set_xlabel("Training progress")

        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            title="Learning rate",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=max(1, len(labels)),
            frameon=False,
        )
        figure.suptitle(
            "Checkpoint trajectory signature: NorMuon redistributes row energy at nearly "
            "unchanged scale",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        figure.text(
            0.5,
            0.012,
            "All panels show NorMuon / Muon ratios from exact hidden-matrix checkpoint "
            "trajectories; these are not individual optimizer updates.",
            ha="center",
            fontsize=8,
            color="#444444",
        )
        figure.tight_layout(rect=(0.02, 0.055, 0.99, 0.91))
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

    return {
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "rows": len(rows),
        "families": families,
        "learning_rates": learning_rates,
        "stages": stages,
    }


def _read_phase_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[int]]:
    required = {
        "model_family",
        "optimizer",
        "learning_rate",
        "stage",
        "step",
        "reference_displacement_to_weight_ratio",
        *(field for _, field in PHASE_METRICS),
    }
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing phase columns in {path}: {sorted(missing)}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"No checkpoint trajectory rows in {path}")

    rows: list[dict[str, Any]] = []
    series_stages: dict[tuple[str, str, float], set[int]] = defaultdict(set)
    for raw in raw_rows:
        family = raw["model_family"]
        optimizer = raw["optimizer"]
        learning_rate = float(raw["learning_rate"])
        stage = int(raw["stage"])
        step = int(raw["step"])
        displacement = float(raw["reference_displacement_to_weight_ratio"])
        metrics = {field: float(raw[field]) for _, field in PHASE_METRICS}
        if optimizer not in OPTIMIZER_STYLES:
            raise ValueError(f"Unsupported optimizer in {path}: {optimizer!r}")
        if not family or learning_rate <= 0 or stage <= 0 or step <= 0 or displacement <= 0:
            raise ValueError(f"Invalid checkpoint phase identity: {raw}")
        if not math.isfinite(displacement) or not all(
            math.isfinite(value) and value > 0 for value in metrics.values()
        ):
            raise ValueError(f"Invalid checkpoint phase metric: {raw}")
        key = (family, optimizer, learning_rate)
        if stage in series_stages[key]:
            raise ValueError(f"Duplicate checkpoint phase stage for {key}: {stage}")
        series_stages[key].add(stage)
        rows.append(
            {
                "model_family": family,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "stage": stage,
                "step": step,
                "reference_displacement_to_weight_ratio": displacement,
                **metrics,
            }
        )

    stage_sets = set(map(frozenset, series_stages.values()))
    if len(stage_sets) != 1:
        raise ValueError("Checkpoint phase series have different stage coverage")
    families = sorted(
        {row["model_family"] for row in rows}, key=lambda name: (name != "dense", name)
    )
    optimizers = [name for name in OPTIMIZER_STYLES if any(r["optimizer"] == name for r in rows)]
    stages = sorted(next(iter(stage_sets)))
    return rows, families, optimizers, stages


def plot_optimizer_phase(input_path: Path, output_path: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    rows, families, optimizers, stages = _read_phase_rows(input_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the analysis extra to plot geometry") from error

    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_family"], row["optimizer"], row["learning_rate"])].append(row)
    for series in grouped.values():
        series.sort(key=lambda row: row["stage"])

    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "embedding-optimizer-study-phase",
        }
    ):
        figure, axes = plt.subplots(
            len(families),
            len(PHASE_METRICS),
            figsize=(9.6, 3.35 * len(families)),
            squeeze=False,
            sharex=True,
        )
        for row_index, family in enumerate(families):
            family_rows = [row for row in rows if row["model_family"] == family]
            ranges = {
                optimizer: (
                    min(
                        row["reference_displacement_to_weight_ratio"]
                        for row in family_rows
                        if row["optimizer"] == optimizer
                    ),
                    max(
                        row["reference_displacement_to_weight_ratio"]
                        for row in family_rows
                        if row["optimizer"] == optimizer
                    ),
                )
                for optimizer in optimizers
            }
            overlap_lower = max(bounds[0] for bounds in ranges.values())
            overlap_upper = min(bounds[1] for bounds in ranges.values())
            for metric_index, (title, field) in enumerate(PHASE_METRICS):
                axis = axes[row_index][metric_index]
                if overlap_lower < overlap_upper:
                    axis.axvspan(
                        overlap_lower,
                        overlap_upper,
                        color="#999999",
                        alpha=0.12,
                        linewidth=0,
                    )
                values: list[float] = []
                for optimizer in optimizers:
                    color, marker, _ = OPTIMIZER_STYLES[optimizer]
                    learning_rates = sorted(
                        {key[2] for key in grouped if key[0] == family and key[1] == optimizer}
                    )
                    for rate_index, learning_rate in enumerate(learning_rates):
                        series = grouped[(family, optimizer, learning_rate)]
                        x_values = [
                            item["reference_displacement_to_weight_ratio"] for item in series
                        ]
                        y_values = [item[field] for item in series]
                        values.extend(y_values)
                        alpha = 0.5 + 0.5 * (rate_index + 1) / len(learning_rates)
                        axis.plot(
                            x_values,
                            y_values,
                            color=color,
                            marker=marker,
                            linewidth=1.25,
                            markersize=3.7,
                            alpha=alpha,
                        )
                span = max(values) - min(values)
                padding = max(span * 0.08, max(values) * 0.02)
                axis.set_ylim(max(0.0, min(values) - padding), max(values) + padding)
                axis.set_xscale("log")
                axis.grid(color="#cccccc", linewidth=0.6, alpha=0.65)
                if metric_index == 0:
                    axis.set_ylabel(f"{family.capitalize()}On")
                if row_index == 0:
                    axis.set_title(title, fontweight="bold")
                if row_index == len(families) - 1:
                    axis.set_xlabel(r"Relative displacement  $||W_t-W_0||_F / ||W_t||_F$")

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=OPTIMIZER_STYLES[name][0],
                marker=OPTIMIZER_STYLES[name][1],
                linewidth=1.5,
                label=OPTIMIZER_STYLES[name][2],
            )
            for name in optimizers
        ]
        legend_handles.append(
            Line2D([0], [0], color="#999999", linewidth=7, alpha=0.2, label="shared scale range")
        )
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.958),
            ncol=len(legend_handles),
            frameon=False,
        )
        figure.suptitle(
            "Optimizer trajectories occupy different row-geometry regimes",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        figure.text(
            0.5,
            0.012,
            "Lines join five saved checkpoints for one learning rate. Shading marks only the "
            "observed shared displacement range; comparisons remain descriptive.",
            ha="center",
            fontsize=8,
            color="#444444",
        )
        figure.tight_layout(rect=(0.025, 0.055, 0.99, 0.91))
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

    return {
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "rows": len(rows),
        "families": families,
        "optimizers": optimizers,
        "stages": stages,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot matched Muon/NorMuon geometry trajectories")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/weight-space/optimizer_pair_contrast_trajectory.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/weight-space/optimizer_pair_contrast_trajectory.svg"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(json.dumps(plot_pair_contrasts(args.input, args.output), sort_keys=True))


def _phase_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot optimizer checkpoint geometry by scale")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/weight-space/checkpoint_trajectory.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/weight-space/optimizer_geometry_phase.svg"),
    )
    return parser


def phase_main(argv: list[str] | None = None) -> None:
    args = _phase_parser().parse_args(argv)
    print(json.dumps(plot_optimizer_phase(args.input, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
