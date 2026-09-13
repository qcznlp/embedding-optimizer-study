"""Render the paper's optimizer-to-retrieval evidence map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#3569A8"
ORANGE = "#D97721"
GREEN = "#2E7D61"
PURPLE = "#7356A8"
INK = "#17212B"
MUTED = "#52606D"
LIGHT = "#F4F7FA"


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edge: str = INK,
    face: str = "white",
    size: float = 10,
    weight: str = "normal",
    linestyle: str = "-",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.35,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        fontweight=weight,
        linespacing=1.25,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    style: str = "-",
    width: float = 1.4,
    connection: str = "arc3",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=width,
            linestyle=style,
            color=color,
            connectionstyle=connection,
        )
    )


def render(output_pdf: Path, output_png: Path) -> None:
    figure, axis = plt.subplots(figsize=(14.2, 6.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.patch.set_facecolor("white")

    axis.text(
        0.02,
        0.965,
        "The object of explanation is the reached retriever, not the shape of one update",
        fontsize=17,
        fontweight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.02,
        0.915,
        "Same pretrained DenseOn, examples, and schedule; optimizer identity changes the path through weight space.",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )

    _box(
        axis,
        0.025,
        0.635,
        0.13,
        0.15,
        "Common start\nDenseOn-unsupervised\n$\\theta_0$",
        face=LIGHT,
        weight="bold",
    )
    _box(
        axis,
        0.215,
        0.735,
        0.145,
        0.105,
        "AdamW trajectory\n$\\theta_{A,1} \\rightarrow \\cdots \\rightarrow \\theta_{A,5}$",
        edge=BLUE,
        face="#EDF4FC",
        weight="bold",
    )
    _box(
        axis,
        0.215,
        0.575,
        0.145,
        0.105,
        "Muon trajectory\n$\\theta_{M,1} \\rightarrow \\cdots \\rightarrow \\theta_{M,5}$",
        edge=ORANGE,
        face="#FFF3E8",
        weight="bold",
    )
    _arrow(axis, (0.155, 0.71), (0.215, 0.787), color=BLUE)
    _arrow(axis, (0.155, 0.71), (0.215, 0.627), color=ORANGE)

    axis.text(
        0.287,
        0.535,
        "same data and schedule",
        ha="center",
        fontsize=8.8,
        color=MUTED,
    )
    axis.text(
        0.287,
        0.505,
        "operator fingerprint  ≠  mechanism",
        ha="center",
        fontsize=9.1,
        color=INK,
        fontweight="bold",
    )

    _box(
        axis,
        0.445,
        0.625,
        0.22,
        0.19,
        "Reached retriever $\\theta_t \\mapsto Z_t$\n\nWeight state: distance · subspace\nRepresentation: rank · random removal\nUtility: helpful / degrading coordinates",
        edge=GREEN,
        face="#EDF8F3",
        size=9.2,
        weight="bold",
    )
    _arrow(axis, (0.36, 0.787), (0.445, 0.75), color=BLUE)
    _arrow(axis, (0.36, 0.627), (0.445, 0.69), color=ORANGE)

    _box(
        axis,
        0.77,
        0.65,
        0.19,
        0.14,
        "Full-corpus retrieval\n14 decontaminated BEIR tasks\nmean nDCG@10",
        edge=PURPLE,
        face="#F4F0FA",
        size=9.3,
        weight="bold",
    )
    _arrow(axis, (0.665, 0.72), (0.77, 0.72), color=PURPLE, style="--")
    axis.text(
        0.718,
        0.75,
        "out-of-dose prediction",
        ha="center",
        fontsize=8.6,
        color=PURPLE,
    )

    _box(
        axis,
        0.025,
        0.09,
        0.285,
        0.28,
        "Distinct evidence questions\n\n1. how do reached states differ?\n2. do measurements predict unseen doses?\n3. what do state and continuation contribute?",
        edge=GREEN,
        face="#EDF8F3",
        size=9.1,
        weight="bold",
    )

    _box(
        axis,
        0.365,
        0.065,
        0.595,
        0.325,
        "",
        edge=PURPLE,
        face="#FBF9FD",
    )
    axis.text(
        0.385,
        0.36,
        "Mechanism intervention: cross reached state × reset next optimizer",
        fontsize=11.0,
        fontweight="bold",
        color=PURPLE,
        va="top",
    )
    axis.text(0.52, 0.292, "reset AdamW", fontsize=9.2, ha="center", color=BLUE)
    axis.text(0.655, 0.292, "reset Muon", fontsize=9.2, ha="center", color=ORANGE)
    axis.text(0.445, 0.232, "AdamW state", fontsize=8.8, ha="right", color=BLUE)
    axis.text(0.445, 0.153, "Muon state", fontsize=8.8, ha="right", color=ORANGE)
    for x, y, label in (
        (0.465, 0.197, "$A\\rightarrow A$"),
        (0.60, 0.197, "$A\\rightarrow M$"),
        (0.465, 0.118, "$M\\rightarrow A$"),
        (0.60, 0.118, "$M\\rightarrow M$"),
    ):
        _box(axis, x, y, 0.11, 0.06, label, edge=PURPLE, face="white", size=9.3)
    axis.text(
        0.755,
        0.225,
        "What carries retrieval value?\n\n1. reached-state effect\n2. continuation-operator effect\n3. state × operator interaction",
        fontsize=9.0,
        color=INK,
        va="center",
        linespacing=1.35,
    )
    axis.text(
        0.755,
        0.095,
        "matched 50K continuation → fixed probe + BEIR",
        fontsize=8.2,
        color=MUTED,
        va="center",
    )
    _arrow(axis, (0.287, 0.575), (0.49, 0.39), color=ORANGE, style="--")
    _arrow(axis, (0.287, 0.735), (0.54, 0.39), color=BLUE, style="--")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(output_pdf, bbox_inches="tight", dpi=300)
    figure.savefig(output_png, bbox_inches="tight", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("paper/figures/optimizer-weight-dimension-map.pdf"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("paper/figures/optimizer-weight-dimension-map.png"),
    )
    args = parser.parse_args()
    render(args.output_pdf, args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
