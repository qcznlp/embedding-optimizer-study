from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common_state_spectra import resolve_spectrum_spec
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .update_geometry import ALGORITHMS

COLORS = {"adamw": "#3B6FB6", "muon": "#D9792B", "normuon": "#3A9B62"}
LABELS = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}
TENSOR_PATTERN = re.compile(r"^0\.layers\.(?P<layer>\d+)\.(?P<module>attn\.Wqkv|mlp\.Wi)\.weight$")
REQUIRED_FIELDS = {
    "family",
    "anchor_kind",
    "source_optimizer",
    "learning_rate",
    "run_id",
    "stage",
    "fraction",
    "step",
    "label",
    "update_operator",
    "tensor",
    "rows",
    "columns",
    "rank",
    "singular_index",
    "normalized_index",
    "singular_value",
    "frobenius_normalized_value",
    "spectral_normalized_value",
    "energy_fraction",
    "cumulative_energy_fraction",
}


def _resolve_declared_path(summary_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (summary_dir / path).resolve()


def _load_rows(
    summary_dir: Path, spectrum_spec: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[int], list[str]]:
    summary_dir = summary_dir.resolve()
    spectrum_spec = spectrum_spec.resolve()
    manifest_path = summary_dir / "summary_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("allow_partial") is not False
        or manifest.get("valid_anchors") != manifest.get("expected_anchors")
        or manifest.get("valid_spectra") != manifest.get("expected_spectra")
        or manifest.get("spectrum_spec", {}).get("sha256") != _sha256(spectrum_spec)
    ):
        raise ValueError("Spectrum summary is incomplete or bound to a different frozen protocol")
    declared = manifest.get("outputs", {}).get("singular_values")
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        raise ValueError("Spectrum summary does not declare singular_values.csv")
    input_path = _resolve_declared_path(summary_dir, declared["path"])
    if (
        not input_path.is_file()
        or input_path.stat().st_size != declared.get("bytes")
        or _sha256(input_path) != declared.get("sha256")
    ):
        raise ValueError("Long-form singular values differ from their summary manifest")
    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError("Long-form singular-value schema changed")
        raw_rows = list(reader)
    if len(raw_rows) != declared.get("rows") or len(raw_rows) != manifest.get("singular_values"):
        raise ValueError("Long-form singular-value row count differs from its manifest")

    protocol = json.loads(spectrum_spec.read_text(encoding="utf-8"))
    selected = protocol.get("selection", {}).get("tensor_names")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Frozen spectrum protocol has no selected tensors")
    tensor_metadata = {}
    for tensor in selected:
        match = TENSOR_PATTERN.fullmatch(tensor)
        if match is None:
            raise ValueError(f"Unsupported selected tensor name: {tensor}")
        tensor_metadata[tensor] = (int(match.group("layer")), match.group("module"))
    layers = sorted({value[0] for value in tensor_metadata.values()})
    modules = sorted({value[1] for value in tensor_metadata.values()})
    expected_cross = {(layer, module) for layer in layers for module in modules}
    if set(tensor_metadata.values()) != expected_cross:
        raise ValueError("Frozen spectrum tensors do not form a complete layer/module grid")

    rows: list[dict[str, Any]] = []
    series: dict[tuple[str, str, str, str], list[tuple[int, int, float]]] = defaultdict(list)
    labels_by_family: dict[str, set[str]] = defaultdict(set)
    for raw in raw_rows:
        try:
            family = raw["family"]
            operator = raw["update_operator"]
            tensor = raw["tensor"]
            rank = int(raw["rank"])
            index = int(raw["singular_index"])
            normalized_index = float(raw["normalized_index"])
            value = float(raw["spectral_normalized_value"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid singular-value row: {raw}") from error
        if (
            family not in {"dense", "late"}
            or operator not in ALGORITHMS
            or tensor not in tensor_metadata
            or rank <= 0
            or not 1 <= index <= rank
            or not math.isfinite(normalized_index)
            or not math.isclose(normalized_index, index / rank, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid plotted singular-value row: {raw}")
        layer, module = tensor_metadata[tensor]
        row = {
            "family": family,
            "label": raw["label"],
            "operator": operator,
            "tensor": tensor,
            "layer": layer,
            "module": module,
            "rank": rank,
            "index": index,
            "normalized_index": normalized_index,
            "value": value,
        }
        rows.append(row)
        series[(family, raw["label"], operator, tensor)].append((index, rank, value))
        labels_by_family[family].add(raw["label"])
    if set(labels_by_family) != {"dense", "late"} or any(
        len(labels) != 10 for labels in labels_by_family.values()
    ):
        raise ValueError("Exact-spectrum plot requires ten frozen anchors per model family")
    expected_series = 20 * len(ALGORITHMS) * len(selected)
    if len(series) != expected_series:
        raise ValueError(
            f"Expected {expected_series} complete spectrum series, found {len(series)}"
        )
    for key, entries in series.items():
        ranks = {rank for _, rank, _ in entries}
        if len(ranks) != 1:
            raise ValueError(f"Spectrum rank changes within series {key}")
        rank = next(iter(ranks))
        ordered = sorted(entries)
        if [index for index, _, _ in ordered] != list(range(1, rank + 1)):
            raise ValueError(f"Spectrum indices are incomplete or duplicated for {key}")
        values = [value for _, _, value in ordered]
        if not math.isclose(values[0], 1.0, rel_tol=1e-6, abs_tol=1e-6) or any(
            values[index] < values[index + 1] for index in range(len(values) - 1)
        ):
            raise ValueError(f"Spectrum is not spectrally normalized and descending for {key}")
    return rows, manifest, layers, modules


def plot_common_state_spectra(
    summary_dir: Path,
    output_path: Path,
    *,
    spectrum_spec: Path = Path("configs/common_state_spectrum_probe.json"),
) -> dict[str, Any]:
    summary_dir = summary_dir.resolve()
    output_path = output_path.resolve()
    spectrum_spec = resolve_spectrum_spec(spectrum_spec).resolve()
    rows, source_manifest, layers, modules = _load_rows(summary_dir, spectrum_spec)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the analysis extra to plot common-state spectra") from error

    tensor_lookup = {
        (int(match.group("layer")), match.group("module")): tensor
        for tensor in json.loads(spectrum_spec.read_text(encoding="utf-8"))["selection"][
            "tensor_names"
        ]
        if (match := TENSOR_PATTERN.fullmatch(tensor)) is not None
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["operator"], row["tensor"])].append(row)

    family_labels = {"dense": "DenseOn", "late": "LateOn"}
    module_labels = {"attn.Wqkv": "Attention Wqkv", "mlp.Wi": "MLP Wi"}
    families = ["dense", "late"]
    row_grid = [(family, module) for family in families for module in modules]
    epsilon = 1e-7
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "embedding-optimizer-study-common-state-spectra",
        }
    ):
        figure, axes = plt.subplots(
            len(row_grid),
            len(layers),
            figsize=(10.8, 2.55 * len(row_grid)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for row_index, (family, module) in enumerate(row_grid):
            for column_index, layer in enumerate(layers):
                axis = axes[row_index][column_index]
                tensor = tensor_lookup[(layer, module)]
                for operator in ALGORITHMS:
                    series_rows = grouped[(family, operator, tensor)]
                    by_index: dict[int, list[float]] = defaultdict(list)
                    ranks = set()
                    for row in series_rows:
                        by_index[row["index"]].append(row["value"])
                        ranks.add(row["rank"])
                    if len(ranks) != 1 or any(len(values) != 10 for values in by_index.values()):
                        raise ValueError(
                            f"Incomplete anchor coverage for {family}/{operator}/{tensor}"
                        )
                    rank = next(iter(ranks))
                    indices = np.arange(1, rank + 1)
                    matrix = np.asarray([by_index[index] for index in indices], dtype=np.float64)
                    median = np.median(matrix, axis=1)
                    lower = np.quantile(matrix, 0.25, axis=1)
                    upper = np.quantile(matrix, 0.75, axis=1)
                    x_values = indices / rank
                    axis.plot(
                        x_values,
                        np.maximum(median, epsilon),
                        color=COLORS[operator],
                        linewidth=1.45,
                        label=LABELS[operator],
                    )
                    axis.fill_between(
                        x_values,
                        np.maximum(lower, epsilon),
                        np.maximum(upper, epsilon),
                        color=COLORS[operator],
                        alpha=0.13,
                        linewidth=0,
                    )
                axis.set_yscale("log")
                axis.set_xlim(0, 1)
                axis.grid(color="#cccccc", linewidth=0.55, alpha=0.6)
                if row_index == 0:
                    axis.set_title(f"Layer {layer}", fontweight="bold")
                if column_index == 0:
                    axis.set_ylabel(f"{family_labels[family]}\n{module_labels[module]}")
                if row_index == len(row_grid) - 1:
                    axis.set_xlabel("Normalized singular-value index")
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.968),
            ncol=3,
            frameon=False,
        )
        figure.suptitle(
            "Exact common-state update spectra across frozen anchors",
            fontsize=13,
            fontweight="bold",
            y=0.997,
        )
        figure.text(
            0.5,
            0.009,
            "Lines are medians and bands are interquartile ranges across 10 frozen anchors per "
            "family. Values are normalized by each matched direction's spectral norm; values "
            f"below {epsilon:g} are clipped for display.",
            ha="center",
            fontsize=7.5,
            color="#444444",
        )
        figure.tight_layout(rect=(0.025, 0.038, 0.99, 0.935))
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
        "source_summary_manifest": {
            "path": str(summary_dir / "summary_manifest.json"),
            "sha256": _sha256(summary_dir / "summary_manifest.json"),
        },
        "spectrum_spec": {"path": str(spectrum_spec), "sha256": _sha256(spectrum_spec)},
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "input_rows": len(rows),
        "anchors": source_manifest["valid_anchors"],
        "spectra": source_manifest["valid_spectra"],
        "families": families,
        "layers": layers,
        "modules": modules,
        "operators": list(ALGORITHMS),
        "aggregation": "median-and-interquartile-range-over-ten-anchors-per-family",
        "display_floor": epsilon,
    }
    _atomic_json(output_path.with_suffix(".manifest.json"), result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exact common-state update spectra across all frozen anchors"
    )
    parser.add_argument(
        "--summary-dir", type=Path, default=Path("results/common-state-spectra/summary")
    )
    parser.add_argument(
        "--spectrum-spec", type=Path, default=Path("configs/common_state_spectrum_probe.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/common-state/exact-update-spectra.svg")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            plot_common_state_spectra(
                args.summary_dir,
                args.output,
                spectrum_spec=args.spectrum_spec,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
