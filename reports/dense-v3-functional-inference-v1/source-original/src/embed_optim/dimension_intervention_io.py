"""Portable new-only state feature bundles with fresh numerical readback.

This low-level serializer never authenticates a primary checkpoint. Its caller
must supply an independently admitted input plan and freshly computed features.
"""

import csv
import io
from pathlib import Path

import numpy as np

from .primary_contract import canonical, file_identity, read_json, require_same, verify_file
from .primary_v3_outcomes import csv_bytes
from .primary_v3_validation_io import write_new


def _contents(result):
    if set(result["tables"]) != {
        "checkpoint_summary",
        "task_summary",
        "random_removal",
        "rotation_summary",
    }:
        raise ValueError("Incomplete dimension table family")
    files = {}
    for name, rows in result["tables"].items():
        if rows:
            files[name + ".csv"] = csv_bytes(rows)
        elif name == "rotation_summary" and not result["rotation_checks"]:
            stream = io.StringIO(newline="")
            csv.writer(stream, lineterminator="\n").writerow(
                sorted(result["tables"]["task_summary"][0])
            )
            files[name + ".csv"] = stream.getvalue().encode()
        else:
            raise ValueError("A required dimension table is empty")
    files["numerical_evidence.json"] = (
        canonical(
            {
                "numerical_policy": result["numerical_policy"],
                "rotation_checks": result["rotation_checks"],
                "scientific_completion": False,
            }
        )
        + b"\n"
    )
    arrays = dict(result["attributions"])
    for item in result["rotated_attributions"]:
        if not np.array_equal(item["task_groups"], arrays["task_groups"]):
            raise ValueError("Rotated task attributions are unpaired")
        for key in ("ndcg_removal_gain", "margin_removal_gain"):
            name = f"rotation_{item['seed']}_{key}"
            if name in arrays:
                raise ValueError("Repeated rotation seed")
            arrays[name] = item[key]
    return files, arrays


def inspect_state(output, plan, result):
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Require an ordinary retained state bundle")
    manifest = read_json(output / "manifest.json")
    if set(manifest) != {"status", "plan", "outputs"} or manifest["status"] != "complete":
        raise ValueError("Incomplete state feature manifest")
    require_same(manifest["plan"], plan)
    files, arrays = _contents(result)
    names = {*files, "coordinate_attribution.npz"}
    if set(manifest["outputs"]) != names or {p.name for p in output.iterdir()} != {
        *names,
        "manifest.json",
    }:
        raise ValueError("State feature inventory differs")
    for name, binding in manifest["outputs"].items():
        if binding["path"] != name:
            raise ValueError("State output path differs")
        verify_file(output / name, binding)
    for name, expected in files.items():
        if (output / name).read_bytes() != expected:
            raise ValueError(f"State table differs from fresh numerical computation: {name}")
    with np.load(output / "coordinate_attribution.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("State attribution array family differs")
        for name, expected in arrays.items():
            actual = saved[name]
            if (
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not np.array_equal(actual, expected)
            ):
                raise ValueError(
                    f"State attribution differs from fresh numerical computation: {name}"
                )
    for name, binding in manifest["outputs"].items():
        verify_file(output / name, binding)
    return {
        "plan": plan,
        "outputs": manifest["outputs"],
        "fresh_numerics_verified": True,
        "scientific_completion": False,
    }


def save_state(output, plan, result):
    output = Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("Use a new state bundle under an existing explicit parent")
    files, arrays = _contents(result)
    output.mkdir()
    for name, data in files.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    with (output / "coordinate_attribution.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    names = {*files, "coordinate_attribution.npz"}
    outputs = {name: {"path": name, **file_identity(output / name)} for name in sorted(names)}
    write_new(output / "manifest.json", {"status": "complete", "plan": plan, "outputs": outputs})
    return inspect_state(output, plan, result)
