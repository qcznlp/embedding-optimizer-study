"""Complete synthetic raw-outcome reconstruction; never model or retrieval findings."""

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embed_optim import primary_v3_outcome_reconstruction as replay
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import digest, file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_outcome_primitives import (
    producer_path,
    recorded_runs,
    validation_identity,
)
from embed_optim.primary_v3_outcomes import TABLE_COUNTS, csv_bytes
from embed_optim.primary_v3_reconstruction_authoring import build
from embed_optim.primary_v3_reconstruction_inputs import ReconstructionInput
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from scripts.outcome_reconstruction_fixture import make, unit_raw_source

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(autouse=True)
def isolate_old_archival_test_aliases(monkeypatch):
    # Isolate only the two known frozen test aliases; production still rejects mixed sources.
    for side in ("before", "after"):
        name = "embed_optim._main_resume_" + side
        module = sys.modules.get(name)
        if module is not None:
            assert (
                Path(module.__file__)
                == ROOT
                / f"reports/engineering-archive/main-resume-v1/{side}/corrected_completion_pipeline.py"
            )
            monkeypatch.delitem(sys.modules, name)


@pytest.fixture(scope="module")
def contract():
    primary = PrimaryV3Contract.load(
        ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE
    )
    return FunctionalInferenceContract.load(ROOT / INFERENCE_PROTOCOL, primary)


@pytest.fixture(scope="module")
def archive(contract, tmp_path_factory):
    root = tmp_path_factory.mktemp("explicit-synthetic-outcome-reconstruction")
    raw = unit_raw_source(root / "all-synthetic-raw", contract)
    return make(root / "fixture", contract, raw=raw)


def load(archive):
    return ReconstructionInput.load(*archive)


def test_complete_original_outcome_reconstruction(archive, tmp_path):
    inputs = load(archive)
    result = replay.reconstruct(inputs, tmp_path / "recomputed")
    assert result["original_outcome_numerics_verified"] is True
    assert result["table_counts"] == TABLE_COUNTS
    assert result["validation_records_replayed"] == 49152
    assert result["beir_task_results_reparsed"] == 840
    assert result["upstream_admission_simulated"] is True
    for key in (
        "checkpoint_tensors_revalidated",
        "model_encoding_repeated",
        "retrieval_repeated",
        "raw_validation_text_revalidated",
        "timing_remeasured",
        "raw_vector_features_recomputed",
        "geometry_primitives_recomputed",
        "functional_inference_recomputed",
        "primary_scientific_admission",
        "manuscript_installed",
        "scientific_completion",
    ):
        assert result[key] is False
    for name in ("evidence.json", *(n + ".csv" for n in TABLE_COUNTS)):
        assert (tmp_path / "recomputed/outcomes" / name).read_bytes() == (
            inputs.root / "outcomes" / name
        ).read_bytes()
    mappings = read_json(tmp_path / "recomputed/local_addresses.json")
    assert len(mappings) == 48 + 60 + 840 * 3
    assert len({r["local"] for r in mappings}) == 48 + 60 * 17


@pytest.mark.parametrize(
    "case",
    [
        "scope",
        "run_identity",
        "fingerprint",
        "run_count",
        "stage_count",
        "seal_schema",
        "checkpoint_identity",
        "step_type",
        "missing_payload",
        "duplicate_payload",
        "payload_order",
        "seal_identity",
        "bad_file_hash",
        "bad_file_size",
        "logical_sizes",
        "world_size",
        "timing",
    ],
)
def test_recorded_metadata_must_match_complete_primary_schema(contract, archive, case):
    admitted = copy.deepcopy(read_json(archive[0] / "vectors/admission.json"))
    runs = admitted["complete_runs"]
    run = next(iter(runs.values()))
    checkpoint = run["checkpoints"][0]
    if case == "scope":
        run["scope"] = "dense_primary_correctness_v2"
    elif case == "run_identity":
        run["run_identity_sha256"] = "0" * 64
    elif case == "fingerprint":
        run["dataset_fingerprint"] = "different"
    elif case == "run_count":
        runs.pop(next(iter(runs)))
    elif case == "stage_count":
        run["checkpoints"].pop()
    elif case == "seal_schema":
        checkpoint.pop("checkpoint_seal")
    elif case == "checkpoint_identity":
        checkpoint["run_identity_sha256"] = "0" * 64
    elif case == "step_type":
        checkpoint["step"] = float(checkpoint["step"])
    elif case == "missing_payload":
        checkpoint["files"] = [r for r in checkpoint["files"] if r["path"] != "optimizer.pt"]
    elif case == "duplicate_payload":
        checkpoint["files"].insert(0, checkpoint["files"][0])
    elif case == "payload_order":
        checkpoint["files"] = list(reversed(checkpoint["files"]))
    elif case == "seal_identity":
        checkpoint["checkpoint_seal"]["sha256"] = "0" * 64
    elif case == "bad_file_hash":
        checkpoint["files"][0]["sha256"] = "bad"
    elif case == "bad_file_size":
        checkpoint["files"][0]["bytes"] = True
    elif case == "logical_sizes":
        run["system_metrics"]["checkpoint_bytes"]["checkpoint-782"] += 1
    elif case == "world_size":
        run["system_metrics"]["world_size"] = 1
    elif case == "timing":
        run["accepted_timing"]["segments"] = 4
    with pytest.raises((ValueError, KeyError)):
        recorded_runs(contract.primary, admitted)


@pytest.mark.parametrize(
    "case",
    [
        "count",
        "content",
        "order",
        "sample",
        "negative_count",
        "negative_type",
        "duplicate_query",
        "row_hash",
        "extra_field",
    ],
)
def test_validation_identity_is_complete_and_typed(contract, archive, case):
    value = copy.deepcopy(
        read_json(archive[0] / "manifest.json")["metadata"]["validation_identity"]
    )
    rows = value["row_identities"]
    if case == "count":
        rows.pop()
    elif case == "content":
        value["content"]["content_sha256"] = "0" * 64
    elif case == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif case == "sample":
        rows[0]["sample_id"] = True
    elif case == "negative_count":
        rows[0]["negative_ids"].pop()
    elif case == "negative_type":
        rows[0]["negative_ids"][0] = 1.0
    elif case == "duplicate_query":
        rows[1]["query_id"] = rows[0]["query_id"]
    elif case == "row_hash":
        rows[0]["row_sha256"] = "bad"
    elif case == "extra_field":
        rows[0]["unexpected"] = 1
    value["row_identities_sha256"] = digest(rows)
    with pytest.raises(ValueError):
        validation_identity(contract.primary, value)


@pytest.mark.parametrize(
    "path", ["", "../results", "/old/../results", "/old//results", "old\\results", "/old/\x00"]
)
def test_producer_paths_are_lexical_not_resolution_requests(path):
    with pytest.raises(ValueError):
        producer_path(path)


def rehash(root):
    path = root / "manifest.json"
    value = read_json(path)
    value["files"] = {name: file_identity(root / name) for name in value["files"]}
    value["summary"]["bytes"] = sum(r["bytes"] for r in value["files"].values())
    path.write_text(json.dumps(value))
    anchor = file_identity(path)["sha256"]
    files.inspect(root, anchor)
    return anchor


@pytest.mark.parametrize(
    "name", ["primary_summary", "secondary_summary", "run_stage_scores", "system_metrics"]
)
def test_rehashed_numeric_output_is_rejected_with_fresh_output_retained(archive, tmp_path, name):
    target = tmp_path / "changed"
    shutil.copytree(archive[0], target)
    path = target / "outcomes" / (name + ".csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    key = next(
        k
        for k, value in rows[0].items()
        if k
        not in {
            "run_id",
            "optimizer",
            "learning_rate",
            "stage",
            "step",
            "world_size",
            "contrast",
            "decision",
        }
        and _is_number(value)
    )
    rows[0][key] = float(np.nextafter(float(rows[0][key]), np.inf))
    path.write_bytes(csv_bytes(rows))
    manifest_path = target / "outcomes/manifest.json"
    manifest = read_json(manifest_path)
    manifest["outputs"][path.name].update(file_identity(path))
    manifest_path.write_text(json.dumps(manifest))
    inputs = load((target, rehash(target)))
    output = tmp_path / "fresh-recomputed"
    with pytest.raises(ValueError, match="differs from fresh evidence/statistical recomputation"):
        replay.reconstruct(inputs, output)
    assert (output / "outcomes" / path.name).exists()
    assert not (output / "reconstruction.json").exists()


def _is_number(value):
    try:
        float(value)
    except ValueError:
        return False
    return bool(value)


def test_selection_finishes_before_task_score_parsing(archive, monkeypatch):
    inputs = load(archive)
    calls = []

    def refusal(*args):
        calls.append("validation")
        raise ValueError("incomplete validation")

    monkeypatch.setattr(replay, "selection_from_records", refusal)
    monkeypatch.setattr(replay, "grid_from_tasks", lambda *a: calls.append("BEIR"))
    with pytest.raises(ValueError, match="incomplete validation"):
        replay.expected_outcomes(inputs)
    assert calls == ["validation"]


def test_actual_missing_primary_authoring_still_refuses(contract, tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(experiment_root=tmp_path / "absent", output=tmp_path / "forbidden")
    with pytest.raises(ValueError, match="ordinary retained run"):
        build(contract, args)
    assert not args.output.exists()


def test_cold_cli_uses_only_archived_sources(archive, tmp_path):
    root, anchor = archive
    code = """
import json, os, runpy, sys
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[3])
forbidden = [sys.argv[4], str(root.parent/'producer')]
def guard(event, args):
    if event in {'socket.connect','socket.bind','socket.getaddrinfo'}:
        raise RuntimeError('Network forbidden')
    if event in {'open','os.listdir','os.scandir'} and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path = os.path.abspath(os.fsdecode(args[0]))
        if any(path == p or path.startswith(p+os.sep) for p in forbidden):
            raise RuntimeError('Original source/producer forbidden')
sys.addaudithook(guard)
sys.argv = ['outcome-reconstruction','--archive-root',str(root),'--expected-manifest-sha256',sys.argv[2],'--output',str(output)]
runpy.run_module('embed_optim.primary_v3_outcome_reconstruction',run_name='__main__')
modules = [m for n,m in sys.modules.items() if n.startswith('embed_optim') and getattr(m,'__file__',None)]
assert modules and all(Path(m.__file__).is_relative_to(root/'source/src') for m in modules)
result=json.loads((output/'reconstruction.json').read_text())
assert result['validation_records_replayed']==49152 and result['beir_task_results_reparsed']==840
assert result['upstream_admission_simulated'] is True and result['scientific_completion'] is False
print('ARCHIVED_SOURCE_COUNT='+str(len(modules)))
"""
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(root / "source/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(root), anchor, str(tmp_path / "cold"), str(ROOT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "ARCHIVED_SOURCE_COUNT=" in result.stdout


def test_new_output_and_external_anchor_are_required(archive, tmp_path):
    with pytest.raises(ValueError):
        load((archive[0], "0" * 64))
    inputs = load(archive)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="new outcome reconstruction"):
        replay.reconstruct(inputs, output)
