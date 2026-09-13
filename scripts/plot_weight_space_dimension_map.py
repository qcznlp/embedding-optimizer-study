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
    size: float = 8.2,
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
    figure, axis = plt.subplots(figsize=(7.1, 4.3))
    figure.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.025)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    figure.patch.set_facecolor("white")

    axis.text(
        0.02,
        0.965,
        "From optimizer trajectories to retrieval function",
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.02,
        0.915,
        "Shared start and training data; compare reached weights, functional dimensions, and retrieval.",
        fontsize=7.5,
        color=MUTED,
        va="top",
    )

    _box(
        axis,
        0.025,
        0.635,
        0.13,
        0.15,
        "Common start\nDenseOn $\\theta_0$\npretrained",
        face=LIGHT,
        weight="bold",
    )
    _box(
        axis,
        0.215,
        0.735,
        0.145,
        0.105,
        "AdamW\n$\\theta_{A,1} \\rightarrow \\cdots \\rightarrow \\theta_{A,5}$",
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
        "Muon\n$\\theta_{M,1} \\rightarrow \\cdots \\rightarrow \\theta_{M,5}$",
        edge=ORANGE,
        face="#FFF3E8",
        weight="bold",
    )
    _arrow(axis, (0.155, 0.71), (0.215, 0.787), color=BLUE)
    _arrow(axis, (0.155, 0.71), (0.215, 0.627), color=ORANGE)

    axis.text(
        0.287,
        0.535,
        "five retained stages",
        ha="center",
        fontsize=7.3,
        color=MUTED,
    )
    axis.text(
        0.287,
        0.505,
        "all learning rates reported",
        ha="center",
        fontsize=7.3,
        color=INK,
        fontweight="bold",
    )

    _box(
        axis,
        0.445,
        0.60,
        0.22,
        0.23,
        "Reached retriever\n$\\theta_t \\mapsto Z_t$\n\nWeight geometry\n(distance · subspace)\nCoordinate utility · rank",
        edge=GREEN,
        face="#EDF8F3",
        size=7.7,
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
        "Full-corpus retrieval\n14 BEIR tasks\nnDCG@10",
        edge=PURPLE,
        face="#F4F0FA",
        size=8.0,
        weight="bold",
    )
    _arrow(axis, (0.665, 0.72), (0.77, 0.72), color=PURPLE, style="--")
    axis.text(
        0.718,
        0.75,
        "held-out\nprediction",
        ha="center",
        fontsize=7.0,
        color=PURPLE,
    )

    _box(
        axis,
        0.025,
        0.09,
        0.285,
        0.28,
        "Evidence questions\n\n1. how do reached states differ?\n2. predict retrieval\nat unseen rates?\n3. response to continuation?",
        edge=GREEN,
        face="#EDF8F3",
        size=7.7,
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
        "State × continuation (optimizer history reset)",
        fontsize=9.0,
        fontweight="bold",
        color=PURPLE,
        va="top",
    )
    axis.text(0.52, 0.292, "AdamW", fontsize=8.0, ha="center", color=BLUE)
    axis.text(0.655, 0.292, "Muon", fontsize=8.0, ha="center", color=ORANGE)
    axis.text(0.445, 0.232, "A state", fontsize=7.6, ha="right", color=BLUE)
    axis.text(0.445, 0.153, "M state", fontsize=7.6, ha="right", color=ORANGE)
    for x, y, label in (
        (0.465, 0.197, "$A\\rightarrow A$"),
        (0.60, 0.197, "$A\\rightarrow M$"),
        (0.465, 0.118, "$M\\rightarrow A$"),
        (0.60, 0.118, "$M\\rightarrow M$"),
    ):
        _box(axis, x, y, 0.11, 0.06, label, edge=PURPLE, face="white", size=8.2)
    axis.text(
        0.755,
        0.225,
        "Endpoint contrasts\n\nsource state (averaged)\ncontinuation (averaged)\nstate × operator",
        fontsize=7.8,
        color=INK,
        va="center",
        linespacing=1.35,
    )
    axis.text(
        0.755,
        0.095,
        "50K · probe-calibrated scale",
        fontsize=7.0,
        color=MUTED,
        va="center",
    )
    _arrow(axis, (0.55, 0.60), (0.55, 0.39), color=PURPLE, style="--")
    axis.text(0.575, 0.475, "fixed 60% states", fontsize=7.5, color=PURPLE)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(
            output_pdf,
            bbox_inches="tight",
            dpi=300,
            metadata={"CreationDate": None, "ModDate": None},
        )
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
