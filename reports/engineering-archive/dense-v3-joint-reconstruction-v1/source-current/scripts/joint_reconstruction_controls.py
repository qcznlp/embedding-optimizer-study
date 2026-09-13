"""Fully rehashed contradictions, restricted to named copies of the joint simulation.

Never apply these mutators to primary artifacts or use them to repair evidence.
Late-table controls can be tested at their own consumer after raw branches pass;
they do not claim to repeat the costly raw-vector calculation for every mutation.
"""

import copy
import csv
import json
import math
from pathlib import Path

from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import canonical, digest, file_identity, read_json
from embed_optim.primary_v3_joint_reconstruction import provenance
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_reconstruction_inputs import ReconstructionInput

CASES = {
    "provenance_missing": "full",
    "provenance_duplicate": "full",
    "provenance_reordered": "full",
    "provenance_escape": "full",
    "validation_identity": "full",
    "vector_identity": "full",
    "outcome_value": "full",
    "geometry_value": "full",
    "feature_value": "full",
    "bridge_prediction": "bridge",
    "bridge_panel": "bridge",
    "bridge_diagnostic": "bridge",
    "bridge_reader": "bridge",
    "functional_prediction": "functional",
    "functional_feature_name": "functional",
    "functional_contrast": "functional",
    "functional_figure": "functional",
    "functional_stage": "functional",
    "functional_decision": "functional",
    "functional_rotation": "functional",
    "functional_latex": "functional",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def change_csv(path, key, *, replacement=None):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if replacement is None:
        row = next(row for row in rows if row[key] != "")
        row[key] = math.nextafter(float(row[key]), math.inf)
    else:
        rows[0][key] = replacement
    path.write_bytes(csv_bytes(rows))


def bundle(root, evidence=None):
    manifest = read_json(root / "manifest.json")
    if evidence is not None:
        (root / "evidence.json").write_bytes(canonical(evidence) + b"\n")
        manifest["plan"]["evidence_sha256"] = digest(evidence)
    for name, record in manifest["outputs"].items():
        record.update(file_identity(root / name))
    write_json(root / "manifest.json", manifest)
    return {
        "plan": manifest["plan"],
        "outputs": manifest["outputs"],
        "recomputed_bytes_verified": True,
        "scientific_completion": False,
    }


def mutate(root, case):
    root = Path(root).absolute()
    if not root.name.startswith("rehashed-joint-") or case not in CASES:
        raise ValueError("Require a named rehashed-joint synthetic copy")
    outer = read_json(root / "manifest.json")
    evidence = read_json(root / "inference/evidence.json")
    if (
        outer["metadata"]
        .get("joint_fixture", {})
        .get("all_upstream_measurements_and_primary_admissions_simulated")
        is not True
        or evidence.get("upstream_primary_admission_simulated") is not True
    ):
        raise ValueError("Refuse to mutate nonsynthetic joint evidence")
    inputs = ReconstructionInput.load(root, file_identity(root / "manifest.json")["sha256"])
    locations = {
        r["original"]: r["local"]
        for r in provenance(inputs, evidence["original_bridge_evidence"]["geometry_reader"])
    }
    old = evidence["original_bridge_evidence"]
    if case == "validation_identity":
        path = sorted((root / "validation").glob("*/admission.json"))[0]
        value = read_json(path)
        value["checkpoint"]["step"] = 782
        write_json(path, value)
    elif case == "vector_identity":
        path = root / "vectors/admission.json"
        value = read_json(path)
        run = value["complete_runs"][sorted(value["complete_runs"])[0]]
        run["checkpoints"].pop()
        write_json(path, value)
        manifest = read_json(root / "vectors/manifest.json")
        manifest["admission"].update(file_identity(path))
        write_json(root / "vectors/manifest.json", manifest)
    elif case == "outcome_value":
        change_csv(root / "outcomes/run_stage_scores.csv", "mean_ndcg_at_10")
        old["outcome_reader"] = bundle(root / "outcomes")
    elif case == "geometry_value":
        run = sorted((root / "geometry/runs").iterdir())[0]
        manifest = read_json(run / "manifest.json")
        record = manifest["outputs"][1]["records"]
        path = run / record["path"]
        value = read_json(path)
        value["checkpoint_row"]["saved_segment_to_weight_ratio"] = math.nextafter(
            value["checkpoint_row"]["saved_segment_to_weight_ratio"], math.inf
        )
        write_json(path, value)
        record.update(file_identity(path))
        write_json(run / "manifest.json", manifest)
        summary = read_json(root / "geometry/summary.json")
        for row in summary["raw_bindings"]:
            row.update(file_identity(root / "geometry" / row["path"]))
        write_json(root / "geometry/summary.json", summary)
        old["geometry_reader"] = summary
    elif case == "feature_value":
        state = root / "features/states/pretrained"
        path = state / "task_summary.csv"
        change_csv(path, "margin_helpful_mass_share")
        manifest = read_json(state / "manifest.json")
        manifest["outputs"][path.name].update(file_identity(path))
        write_json(state / "manifest.json", manifest)
        top = read_json(root / "features/manifest.json")
        top["states"]["pretrained"].update(file_identity(state / "manifest.json"))
        write_json(root / "features/manifest.json", top)
        evidence["dimension_feature_reader"]["manifest_sha256"] = file_identity(
            root / "features/manifest.json"
        )["sha256"]
    elif case == "bridge_prediction":
        change_csv(root / "bridge/held_out_predictions.csv", "feature_prediction")
    elif case == "bridge_panel":
        change_csv(root / "bridge/bridge_rows.csv", "saved_segment_row_norm_cv")
    elif case == "bridge_diagnostic":
        old["numerical_diagnostics"]["feature_designs"][0]["augmented_numeric_rank"] = 8
    elif case == "functional_prediction":
        change_csv(root / "inference/held_out_predictions.csv", "feature_prediction")
    elif case == "functional_feature_name":
        change_csv(
            root / "inference/feature_prediction_summary.csv",
            "feature",
            replacement="log_saved_segment_to_weight_ratio",
        )
    elif case == "functional_contrast":
        change_csv(root / "inference/primary_contrasts.csv", "mean_difference")
    elif case == "functional_figure":
        change_csv(root / "inference/figure_points.csv", "mean")
    elif case == "functional_stage":
        change_csv(root / "inference/optimizer_stage_metrics.csv", "stage", replacement=2)
    elif case == "functional_decision":
        decisions = evidence["decisions"]["constructive_native_coordinate_use"]
        decisions["muon"] = not decisions["muon"]
    elif case == "functional_rotation":
        decisions = evidence["decisions"]["direction_stable_across_tested_rotations"]
        decisions["muon"] = not decisions["muon"]
    elif case == "functional_latex":
        evidence["rendered_latex"] += "\nIncorrect synthetic conclusion.\n"
    for row in old["source_bindings"]:
        row.update(file_identity(root / locations[row["path"]]))
    evidence["original_bridge_reader"] = bundle(root / "bridge", old)
    if case == "bridge_reader":
        evidence["original_bridge_reader"]["recomputed_bytes_verified"] = False
    prefix = evidence["source_bindings"][: -len(old["source_bindings"])]
    for row in prefix:
        row.update(file_identity(root / locations[row["path"]]))
    evidence["source_bindings"] = prefix + copy.deepcopy(old["source_bindings"])
    if case == "provenance_missing":
        evidence["source_bindings"].pop(0)
    elif case == "provenance_duplicate":
        evidence["source_bindings"].insert(0, copy.deepcopy(evidence["source_bindings"][0]))
    elif case == "provenance_reordered":
        rows = evidence["source_bindings"]
        rows[0], rows[1] = rows[1], rows[0]
    elif case == "provenance_escape":
        row = evidence["source_bindings"][0]
        path = Path(row["path"])
        row["path"] = str(path.parent / ".." / path.name)
    current = bundle(root / "inference", evidence)
    outer["metadata"]["authoring_plan"] = current["plan"]
    outer["metadata"]["authoring_evidence_sha256"] = digest(evidence)
    outer["files"] = {
        name: file_identity(root / name)
        for name in files.inventory(root)
        if name != "manifest.json"
    }
    outer["summary"] = {
        "files": len(outer["files"]),
        "bytes": sum(row["bytes"] for row in outer["files"].values()),
    }
    write_json(root / "manifest.json", outer)
    anchor = file_identity(root / "manifest.json")["sha256"]
    files.inspect(root, anchor)
    return anchor
