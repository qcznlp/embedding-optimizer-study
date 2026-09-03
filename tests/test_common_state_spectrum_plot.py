from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from embed_optim.common_state_spectrum_plot import REQUIRED_FIELDS, plot_common_state_spectra
from embed_optim.geometry import _sha256
from embed_optim.update_geometry import ALGORITHMS


def _summary(tmp_path: Path, *, omit_label: str | None = None) -> tuple[Path, Path]:
    summary = tmp_path / "summary"
    summary.mkdir()
    spectrum_spec = Path("configs/common_state_spectrum_probe.json").resolve()
    tensors = json.loads(spectrum_spec.read_text(encoding="utf-8"))["selection"]["tensor_names"]
    path = summary / "singular_values.csv"
    rows = []
    for family in ("dense", "late"):
        for anchor in range(10):
            label = f"{family}/anchor-{anchor}"
            if label == omit_label:
                continue
            for operator_index, operator in enumerate(ALGORITHMS):
                for tensor in tensors:
                    for index, value in enumerate((1.0, 0.5, 0.25), start=1):
                        adjusted = (
                            value
                            if index == 1
                            else value * (1 - operator_index * 0.03) * (1 - anchor * 0.001)
                        )
                        rows.append(
                            {
                                "family": family,
                                "anchor_kind": "pretrained" if anchor == 0 else "checkpoint",
                                "source_optimizer": "" if anchor == 0 else "muon",
                                "learning_rate": "" if anchor == 0 else 1e-3,
                                "run_id": "pretrained" if anchor == 0 else f"anchor-{anchor}",
                                "stage": 0 if anchor == 0 else 5,
                                "fraction": 0 if anchor == 0 else 1,
                                "step": 0 if anchor == 0 else 3907,
                                "label": label,
                                "update_operator": operator,
                                "tensor": tensor,
                                "rows": 6,
                                "columns": 3,
                                "rank": 3,
                                "singular_index": index,
                                "normalized_index": index / 3,
                                "singular_value": adjusted,
                                "frobenius_normalized_value": adjusted / 1.2,
                                "spectral_normalized_value": adjusted,
                                "energy_fraction": adjusted**2 / 1.3125,
                                "cumulative_energy_fraction": min(1.0, index / 3),
                            }
                        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (summary / "summary_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "allow_partial": False,
                "expected_anchors": 20,
                "valid_anchors": 20,
                "expected_spectra": 360,
                "valid_spectra": 360,
                "singular_values": len(rows),
                "spectrum_spec": {"sha256": _sha256(spectrum_spec)},
                "outputs": {
                    "singular_values": {
                        "path": str(path),
                        "rows": len(rows),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary, spectrum_spec


def test_common_state_spectrum_plot_is_complete_and_deterministic(tmp_path: Path):
    summary, spectrum_spec = _summary(tmp_path)
    output = tmp_path / "spectra.svg"

    first = plot_common_state_spectra(summary, output, spectrum_spec=spectrum_spec)
    first_bytes = output.read_bytes()
    second = plot_common_state_spectra(summary, output, spectrum_spec=spectrum_spec)

    assert first == second
    assert output.read_bytes() == first_bytes
    assert first["anchors"] == 20
    assert first["spectra"] == 360
    assert first["layers"] == [0, 10, 21]
    assert first["modules"] == ["attn.Wqkv", "mlp.Wi"]
    assert first_bytes.startswith(b"<?xml")
    sidecar = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert sidecar == first
    assert sidecar["output"]["sha256"] == _sha256(output)


def test_common_state_spectrum_plot_rejects_missing_anchor(tmp_path: Path):
    summary, spectrum_spec = _summary(tmp_path, omit_label="late/anchor-9")

    with pytest.raises(ValueError, match="ten frozen anchors"):
        plot_common_state_spectra(
            summary,
            tmp_path / "spectra.svg",
            spectrum_spec=spectrum_spec,
        )
