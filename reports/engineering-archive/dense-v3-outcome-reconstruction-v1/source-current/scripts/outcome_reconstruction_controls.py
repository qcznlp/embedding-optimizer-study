"""Deliberately inconsistent, fully rehashed *synthetic copy* controls.

Never repair a real archive or use the new test anchor as scientific admission.
All affected provenance envelopes are updated so numerical/schema checks, rather
than a stale file checksum, must refuse the contradictory input.
"""

import csv
import json
from pathlib import Path, PurePosixPath

import numpy as np

from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import digest, file_identity, read_json
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_validation_io import read_record_file

CASES = (
    "validation_scores",
    "validation_metrics",
    "validation_row",
    "validation_missing",
    "validation_duplicate",
    "validation_summary",
    "validation_selection",
    "validation_extra",
    "beir_revision",
    "beir_split",
    "beir_subset",
    "beir_primary_score",
    "beir_model",
    "beir_settings",
    "beir_checkpoint_wrapper",
    "beir_missing_job",
    "beir_duplicate_task",
    "beir_path_escape",
    "beir_missing_file",
    "beir_extra",
    "outcome_table",
)


def write_json(path, value, *, allow_nan=False):
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=allow_nan) + "\n")


def validation_change(root, selection, case):
    if case == "validation_selection":
        optimizer = next(iter(selection["selected"]))
        old = selection["selected"][optimizer]
        selection["selected"][optimizer] = next(
            row["run_id"]
            for row in selection["run_metrics"]
            if row["optimizer"] == optimizer and row["run_id"] != old
        )
        return
    if case == "validation_extra":
        write_json(root / "validation/unused.json", {"explicit_synthetic_extra": True})
        return
    stored = selection["source_receipts"][0]
    directory = root / "validation" / digest(stored["plan"])
    scores, summary = directory / "sample_scores.jsonl", directory / "summary.json"
    manifest = read_json(directory / "manifest.json")
    if case == "validation_summary":
        value = read_json(summary)
        value["groups"][0]["contrastive_loss"] += 0.1
        write_json(summary, value)
        stored["summary"] = value
    else:
        rows = read_record_file(scores)
        if case == "validation_scores":
            rows[0]["scores"][0] = float(np.float32(rows[0]["scores"][0] + 0.25))
        elif case == "validation_metrics":
            rows[0]["metrics"]["contrastive_loss"] = float(
                np.float32(rows[0]["metrics"]["contrastive_loss"] + 0.25)
            )
        elif case == "validation_row":
            rows[0]["row"]["query_id"] += 1
        elif case == "validation_missing":
            rows.pop()
        elif case == "validation_duplicate":
            rows[1] = rows[0]
        else:
            raise ValueError("Unknown validation mutation")
        with scores.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    for key, path in (("sample_scores", scores), ("summary", summary)):
        manifest["outputs"][key].update(file_identity(path))
    write_json(directory / "manifest.json", manifest)
    stored["outputs"] = manifest["outputs"]
    stored["manifest"].update(file_identity(directory / "manifest.json"))


def beir_change(root, grid, case):
    if case == "beir_missing_job":
        grid["evaluations"].pop()
        return
    if case == "beir_extra":
        write_json(root / "beir/unused.json", {"explicit_synthetic_extra": True})
        return
    job = grid["evaluations"][0]
    plan = job["plan"]
    directory = root / "beir" / plan["cache_key"]
    if case == "beir_checkpoint_wrapper":
        plan["checkpoint"]["scope"] = "dense_primary_correctness_v2"
        write_json(directory / "primary_admission.json", plan)
        return
    if case == "beir_duplicate_task":
        job["tasks"][1] = job["tasks"][0]
        return
    task = job["tasks"][0]
    original = PurePosixPath(task["files"][0]["path"])
    relative = original.relative_to(plan["results_root"])
    path = directory / relative
    if case == "beir_path_escape":
        task["files"][0]["path"] = str(original.parent / ".." / original.name)
        return
    if case == "beir_missing_file":
        path.unlink()  # An exact ordinary task file in a caller-owned synthetic copy only.
        return
    if case in {"beir_model", "beir_settings"}:
        if case == "beir_model":
            changed = path.parent / "model_meta.json"
            value = read_json(changed)
            value["name"] = "wrong-run/checkpoint-782"
            write_json(changed, value)
        else:
            changed = path.parent / "run_settings.jsonl"
            rows = [json.loads(line) for line in changed.read_text().splitlines()]
            rows[0]["version"]["torch"] = "wrong-version"
            with changed.open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
        # A shared settings/meta file is referenced by all fourteen task receipts.
        for item in job["tasks"]:
            for row in item["files"]:
                if PurePosixPath(row["path"]).name == changed.name:
                    row.update(file_identity(changed))
        return
    value = json.loads(path.read_text())  # Preserve allowed unused auxiliary NaN fields.
    split = next(iter(value["scores"]))
    score = value["scores"][split][0]
    if case == "beir_revision":
        value["dataset_revision"] = "0" * 40
    elif case == "beir_split":
        value["scores"] = {"train": value["scores"][split]}
    elif case == "beir_subset":
        score["hf_subset"] = "wrong-subset"
    elif case == "beir_primary_score":
        score["main_score"] += 0.1
    else:
        raise ValueError("Unknown BEIR mutation")
    write_json(path, value, allow_nan=True)
    task["files"][0].update(file_identity(path))


def mutate(root, case):
    root = Path(root).absolute()
    if not root.name.startswith("rehashed-") or case not in CASES:
        raise ValueError("Use a new explicitly named rehashed synthetic copy")
    outer = read_json(root / "manifest.json")
    inference = read_json(root / "inference/evidence.json")
    if inference.get("upstream_primary_admission_simulated") is not True:
        raise ValueError("Refuse to alter nonsynthetic primary evidence")
    evidence = read_json(root / "outcomes/evidence.json")
    if case.startswith("validation_"):
        validation_change(root, evidence["validation_selection"], case)
    elif case.startswith("beir_"):
        beir_change(root, evidence["grid"], case)
    else:
        path = root / "outcomes/primary_summary.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        key = "mean_delta_ndcg_at_10"
        rows[0][key] = float(np.nextafter(float(rows[0][key]), np.inf))
        path.write_bytes(csv_bytes(rows))
    write_json(root / "outcomes/evidence.json", evidence)
    outcome = read_json(root / "outcomes/manifest.json")
    outcome["plan"]["evidence_sha256"] = digest(evidence)
    for name, row in outcome["outputs"].items():
        row.update(file_identity(root / "outcomes" / name))
    write_json(root / "outcomes/manifest.json", outcome)
    inference["original_bridge_evidence"]["outcome_evidence"] = evidence
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
        "bytes": sum(row["bytes"] for row in outer["files"].values()),
    }
    write_json(root / "manifest.json", outer)
    anchor = file_identity(root / "manifest.json")["sha256"]
    files.inspect(root, anchor)
    return anchor
