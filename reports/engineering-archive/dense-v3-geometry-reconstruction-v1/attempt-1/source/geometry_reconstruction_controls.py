"""Fully rehashed semantic controls on explicitly named synthetic copies only.

Every affected provenance envelope receives a new *test* anchor. A valid transport
checksum must not excuse inconsistent schemas, zero/rank support or wrong derived
aggregates. This module must never be used to repair primary evidence.
"""

import csv
import json
import math
from pathlib import Path, PurePosixPath

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import digest, file_identity, read_json
from embed_optim.primary_v3_outcomes import csv_bytes

CASES = (
    "stage_anchor",
    "record_missing",
    "record_duplicate",
    "record_shape",
    "entry_count",
    "zero_presence",
    "metric_schema",
    "sketch_rank",
    "health_rank",
    "health_signal",
    "health_gap",
    "health_missing",
    "basis_missing",
    "basis_extra",
    "basis_dtype",
    "basis_shape",
    "basis_nonfinite",
    "basis_identity",
    "basis_values",
    "manifest_settings",
    "manifest_path",
    "geometry_summary",
    "geometry_table",
    "checkpoint_row",
    "provenance_duplicate",
    "provenance_path_escape",
    "geometry_extra",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def stage_change(value, case):
    record, entry = value["records"][0], value["entry_records"][0]
    health = value["subspace_health"][0]
    if case == "stage_anchor":
        value["previous_step"] = 0
    elif case == "record_missing":
        value["records"].pop()
    elif case == "record_duplicate":
        value["records"][1] = record
    elif case == "record_shape":
        record["shape"] = [16, 16]
    elif case == "entry_count":
        entry["saved_segment_nonzero_parameters"] = entry["parameters"] + 1
    elif case == "zero_presence":
        entry["saved_segment_nonzero_parameters"] = 0
    elif case == "metric_schema":
        record["weight"].pop("row_norms")
    elif case == "sketch_rank":
        record["delta_from_previous"]["rank"] = 32
    elif case == "health_rank":
        health["retained_rank"] = 15
    elif case == "health_signal":
        health["retained_signal_supported"] = False
    elif case == "health_gap":
        health["relative_boundary_gap"] = 0.25
    elif case == "health_missing":
        value["subspace_health"].pop()
    elif case == "checkpoint_row":
        key = "saved_segment_to_weight_ratio"
        value["checkpoint_row"][key] = math.nextafter(value["checkpoint_row"][key], math.inf)
    else:
        raise ValueError("Unknown stage control")


def basis_change(path, case):
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        values = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    name = next(iter(values))
    if case == "basis_missing":
        values.pop(name)
    elif case == "basis_extra":
        values["unconsumed|tensor|left"] = torch.zeros((768, 16), dtype=torch.float32)
    elif case == "basis_dtype":
        values[name] = values[name].double()
    elif case == "basis_shape":
        values[name] = values[name][:16].contiguous()
    elif case == "basis_nonfinite":
        values[name][0, 0] = float("nan")
    elif case == "basis_identity":
        metadata["run_identity_sha256"] = "0" * 64
    elif case == "basis_values":
        values[name] = values[name].roll(31, dims=0).contiguous()
    else:
        raise ValueError("Unknown basis control")
    save_file(values, path, metadata=metadata)


def mutate(root, case):
    root = Path(root).absolute()
    if not root.name.startswith("rehashed-") or case not in CASES:
        raise ValueError("Use a new explicitly named rehashed synthetic geometry copy")
    outer = read_json(root / "manifest.json")
    inference = read_json(root / "inference/evidence.json")
    if (
        inference.get("upstream_primary_admission_simulated") is not True
        or inference.get("native_axis_coordinate_bases_synthetic") is not True
    ):
        raise ValueError("Refuse to alter nonsynthetic geometry evidence")
    geometry = root / "geometry"
    run = sorted((geometry / "runs").iterdir())[0]
    manifest = read_json(run / "manifest.json")
    stage = manifest["outputs"][1]
    if case.startswith("basis_"):
        basis_change(run / stage["bases"]["path"], case)
    elif case == "manifest_settings":
        manifest["plan"]["settings"]["subspace_rank"] = 8
    elif case == "manifest_path":
        stage["records"]["path"] = "../" + stage["records"]["path"]
    elif case == "geometry_table":
        path = geometry / "run_pair_subspace_overlap.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        row = next(r for r in rows if r["mean_subspace_overlap"] != "")
        row["mean_subspace_overlap"] = math.nextafter(float(row["mean_subspace_overlap"]), math.inf)
        path.write_bytes(csv_bytes(rows))
    elif case == "geometry_extra":
        write_json(geometry / "unused.json", {"explicit_synthetic_extra": True})
    elif case not in {"geometry_summary", "provenance_duplicate", "provenance_path_escape"}:
        path = run / stage["records"]["path"]
        value = read_json(path)
        stage_change(value, case)
        write_json(path, value)
    for output in manifest["outputs"]:
        for key in ("records", "bases"):
            # Do not resolve a deliberately invalid path, even in a synthetic control.
            name = PurePosixPath(output[key]["path"]).name
            output[key].update(file_identity(run / name))
    write_json(run / "manifest.json", manifest)
    summary = read_json(geometry / "summary.json")
    for row in summary["raw_bindings"]:
        row.update(file_identity(geometry / row["path"]))
    if case == "geometry_summary":
        summary["boundary"] = "Incorrectly claimed complete publication"
    write_json(geometry / "summary.json", summary)
    original = inference["original_bridge_evidence"]
    original["geometry_reader"] = summary
    original_root = next(
        PurePosixPath(r["path"]).parent
        for r in original["source_bindings"]
        if PurePosixPath(r["path"]).name == "summary.json"
    )
    for row in original["source_bindings"]:
        relative = PurePosixPath(row["path"]).relative_to(original_root)
        row.update(file_identity(geometry / str(relative)))
    if case == "provenance_duplicate":
        original["source_bindings"].append(original["source_bindings"][0])
    elif case == "provenance_path_escape":
        row = original["source_bindings"][0]
        path = PurePosixPath(row["path"])
        row["path"] = str(path.parent / ".." / path.name)
    write_json(root / "inference/evidence.json", inference)
    plan = read_json(root / "inference/manifest.json")
    plan["plan"]["evidence_sha256"] = digest(inference)
    write_json(root / "inference/manifest.json", plan)
    outer["metadata"]["authoring_evidence_sha256"] = digest(inference)
    outer["metadata"]["authoring_plan"] = plan["plan"]
    names = [name for name in files.inventory(root) if name != "manifest.json"]
    outer["files"] = {name: file_identity(root / name) for name in names}
    outer["summary"] = {
        "files": len(names),
        "bytes": sum(r["bytes"] for r in outer["files"].values()),
    }
    write_json(root / "manifest.json", outer)
    anchor = file_identity(root / "manifest.json")["sha256"]
    files.inspect(root, anchor)
    return anchor
