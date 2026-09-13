"""Explicit preparation-only data amendment; never weaken an old execution contract."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .data import SPLITS, _seed_for
from .primary_contract import digest, read_json, require_same, verify_file

SCOPE = "dense_primary_data_amendment_v3"
STATUS = "prepared_data_amendment_not_execution_authorized"
INPUT_SHA = "40b7d3e63b1c4a17b995507d5d96c042980acca96baa08f6c0c52dda66b1f1f9"
ACCEPTANCE_SHA = "3b2cd1902e02a43242b16fee41127cea3c8671e389606d9363c8776a0ef6a52c"
SEED = 20260906
QUOTAS = {source: 42 if source == "fiqa" else 41 for source in SPLITS}
TEXT_COLUMNS = ("query", "positive", *(f"negative_{i}" for i in range(7)))
SOURCES = (
    "src/embed_optim/primary_data_revision.py",
    "scripts/prepare_dense_revised_inputs.py",
    "scripts/prepare_dense_data_amendment.py",
    "scripts/audit_dense_revised_natural.py",
    "scripts/audit_dense_natural_data.py",
    "scripts/run_dense_natural_diagnostic.py",
    "scripts/audit_dense_identity_entrypoint.py",
    "scripts/audit_prepared_dense_gpu.py",
    "scripts/pause_dense_evaluation_for_audit.py",
    "src/embed_optim/primary_completion.py",
    "src/embed_optim/primary_contract.py",
    "src/embed_optim/primary_io.py",
    "src/embed_optim/gpu_lease.py",
    "configs/formal_runtime.json",
)


def load_amendment(path, root):
    root, path = Path(root).resolve(), Path(path).resolve()
    value = read_json(path)
    if (value.get("scope"), value.get("status"), value.get("schema_version")) != (SCOPE, STATUS, 1):
        raise ValueError("Not the explicit preparation-only data amendment")
    if value.get("execution_authorized") is not False:
        raise ValueError("This data amendment cannot authorize formal execution")
    if type(value["schema_version"]) is not int or set(value["source_bindings"]) != set(SOURCES):
        raise ValueError("Incomplete exact data-amendment source closure")
    if set(value["parents"]) != {"primary", "validation", "inputs"}:
        raise ValueError("Incomplete data-amendment parent closure")
    if (
        value["parents"]["validation"]["sha256"]
        != "a7e20fd3ebb3477a7104bae322bf2fb5340ad47fe1377bcf20e923b64697d2d1"
        or value["parents"]["inputs"]["sha256"]
        != "02561536a7ac691290f127dc07ddef41cede4b15e9bfbabb97eee0b569a0fa59"
    ):
        raise ValueError("The original validation/input parent must not be replaced")
    for binding in value["parents"].values():
        verify_file(root / binding["path"], binding)
    for name, binding in value["source_bindings"].items():
        verify_file(root / name, binding)
    if (
        value["inputs"]["sha256"] != INPUT_SHA
        or value["data_acceptance"]["sha256"] != ACCEPTANCE_SHA
    ):
        raise ValueError("Revised inputs lack the declared independent parent acceptance")
    for name in ("inputs", "data_acceptance"):
        verify_file(root / value[name]["path"], value[name])
    inputs = read_json(root / value["inputs"]["path"])
    if (
        inputs["scope"] != "prepared_dense_revised_primary_actual_inputs_v3"
        or inputs["preparation_passed"] is not True
    ):
        raise ValueError("Require actual full-horizon input preparation")
    verify_file(Path(inputs["data_producer"]["path"]), inputs["data_producer"])
    producer = read_json(Path(inputs["data_producer"]["path"]))
    if (
        value["parents"]["primary"]["sha256"]
        != "e21a7226c09740d38d49855738267c080f615f8d4f60c2abf9dbf3ccd874edd2"
    ):
        raise ValueError("The original primary parent must not be replaced")
    parent = read_json(root / value["parents"]["primary"]["path"])
    require_same(value["unchanged_evaluation"], parent["evaluation"])
    require_same(value["unchanged_analysis"], parent["analysis"])
    require_same(
        value["diagnostic"],
        {
            "rows": 288,
            "source_quotas": QUOTAS,
            "seed": SEED,
            "include_all_training_replacement_groups": True,
            "steps_per_optimizer": 3,
            "global_query_groups": [128, 128, 32],
            "algorithms": ["adamw", "muon", "normuon"],
            "gpu_pool": ["4", "5", "6", "7"],
            "untrained_base_required": True,
            "no_outcome_selection": True,
        },
    )
    if len(inputs["runs"]) != 12 or inputs["common_identity"]["data"]["rows"] != 500000:
        raise ValueError("Incomplete revised primary input grid")
    for partition, count in (("training", 500000), ("validation", 4096)):
        binding = value["datasets"][partition]
        if binding["rows"] != count:
            raise ValueError("A revised data total differs")
        require_same(binding["files"], producer["partitions"][partition]["files"])
        require_same(
            binding["changed_sample_ids"], producer["partitions"][partition]["changed_sample_ids"]
        )
        if str(Path(binding["root"]) / "manifest.json") not in {
            r["path"] for r in binding["files"]
        }:
            raise ValueError("The amended data root differs from its authenticated manifest")
        for row in binding["files"]:
            verify_file(Path(row["path"]), row)
    for subset in inputs["derived_subset_revised_parent_checks"]:
        if subset["every_value_matches_revised_parent"] is not True:
            raise ValueError("A derived subset is not admitted to its revised parent")
        for row in subset["files"]:
            verify_file(Path(row["path"]), row)
    return value, inputs


def select_natural_coverage(dataset, replacement_ids):
    """Predeclared source-balanced coverage, including all amended training groups."""
    source_values = np.array(dataset["source"])
    replacements = set(replacement_ids)
    if len(replacements) != 6 or any(
        type(i) is not int or not 0 <= i < len(dataset) for i in replacements
    ):
        raise ValueError("Require exactly six declared training replacement positions")
    if set(source_values) != set(SPLITS):
        raise ValueError("Natural coverage requires all seven sources")
    chosen = {}
    for source in SPLITS:
        positions = np.flatnonzero(source_values == source).tolist()
        mandatory = sorted(replacements & set(positions))
        remaining = [i for i in positions if i not in replacements]
        required = QUOTAS[source] - len(mandatory)
        if required < 0 or len(remaining) < required:
            raise ValueError("A source cannot fill its declared diagnostic quota")
        rng = np.random.default_rng(_seed_for(SEED, source, "revised-natural-coverage"))
        chosen[source] = mandatory + [
            remaining[int(i)] for i in rng.permutation(len(remaining))[:required]
        ]
    indices = [
        chosen[source][i]
        for i in range(max(QUOTAS.values()))
        for source in SPLITS
        if i < len(chosen[source])
    ]
    if len(indices) != 288 or len(set(indices)) != 288 or not replacements.issubset(indices):
        raise ValueError("Natural coverage is incomplete or duplicated")
    selected = dataset.select(indices)
    rows = list(selected)
    for row in rows:
        if any(not isinstance(row.get(c), str) or not row[c].strip() for c in TEXT_COLUMNS):
            raise ValueError("A selected coverage group has an invalid original text")
        if type(row.get("length")) is not int or row["length"] <= 0:
            raise ValueError("A selected coverage group has invalid length metadata")
    require_same(dict(Counter(r["source"] for r in rows)), QUOTAS)
    return selected, {
        "selection": "seven_source_balanced_including_all_six_replacements_then_seeded_remaining_positions_interleaved_by_source",
        "seed": SEED,
        "source_quotas": QUOTAS,
        "original_row_count": len(dataset),
        "selected_row_count": len(rows),
        "selected_positions": indices,
        "mandatory_replacement_positions": sorted(replacements),
        "selected_rows_sha256": digest(rows),
        "row_sha256": [digest(r) for r in rows],
        "selected_columns": list(selected.column_names),
        "expected_global_query_groups": [128, 128, 32],
        "boundary": "Diagnostic coverage, not a proportional training-distribution estimate or outcome-based sample; preserve all selected fields without token/text edits.",
    }
