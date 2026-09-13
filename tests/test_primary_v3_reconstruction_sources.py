"""Real contract relocation and bounded source-selection fixtures, not primary results."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_v3_reconstruction_authoring as authoring
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_outcomes import save_bundle
from embed_optim.primary_v3_reconstruction_authoring import build, require_parent
from embed_optim.primary_v3_reconstruction_sources import (
    INFERENCE_PROTOCOL,
    analysis_selection,
    source_selection,
)
from embed_optim.primary_v3_validation import ValidationContract

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(scope="module")
def contract():
    primary = PrimaryV3Contract.load(
        ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE
    )
    return FunctionalInferenceContract.load(ROOT / INFERENCE_PROTOCOL, primary)


def test_real_source_closure_loads_without_model_payloads(contract, tmp_path):
    selected = source_selection(contract, {})
    output = tmp_path / "source-payload"
    receipt = files.write(
        output,
        selected,
        {"scope": "engineering_actual_source_contract_closure", "primary_results_present": False},
    )
    source, training = output / "source", output / "training-source"
    primary = PrimaryV3Contract.load(
        source / "configs/dense_primary_v3_protocol.json", source, training
    )
    loaded = FunctionalInferenceContract.load(source / INFERENCE_PROTOCOL, primary)
    assert loaded.sha256 == contract.sha256
    assert [loaded.primary.expected_identity(r["run_id"]) for r in primary.inputs["runs"]] == [
        contract.primary.expected_identity(r["run_id"]) for r in contract.primary.inputs["runs"]
    ]
    files.inspect(output, receipt["manifest_sha256"])


def test_fresh_python_imports_only_archived_package(contract, tmp_path):
    selected = source_selection(contract, {})
    original = tmp_path / "original-payload"
    receipt = files.write(
        original,
        selected,
        {"scope": "engineering_actual_source_contract_closure", "primary_results_present": False},
    )
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    original.rename(tmp_path / "preserved-original")
    source, training = relocated / "source", relocated / "training-source"
    code = "from pathlib import Path; import sys; from embed_optim.primary_v3_contract import PrimaryV3Contract; from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract; root=Path(sys.argv[1]); p=PrimaryV3Contract.load(root/'configs/dense_primary_v3_protocol.json',root,Path(sys.argv[2])); c=FunctionalInferenceContract.load(root/'configs/dense_primary_v3_dimension_inference_protocol.json',p); assert c.sha256==sys.argv[3]; mods=[m for n,m in sys.modules.items() if n.startswith('embed_optim') and getattr(m,'__file__',None)]; assert mods and all(Path(m.__file__).is_relative_to(root/'src') for m in mods); print(len(mods))"
    env = {
        **os.environ,
        "PYTHONPATH": str(source / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(source), str(training), contract.sha256],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 20
    files.inspect(relocated, receipt["manifest_sha256"])


def test_real_missing_primary_stops_before_archive_creation(contract, tmp_path):
    assert require_parent(contract).is_file()
    args = SimpleNamespace(experiment_root=tmp_path / "absent", output=tmp_path / "forbidden")
    with pytest.raises(ValueError, match="ordinary retained run"):
        build(contract, args)
    assert not args.output.exists()


def selection_fixture(tmp_path):
    args = SimpleNamespace()
    evidence = {"source_bindings": []}
    for role, attr in (
        ("vectors", "dimension_vectors"),
        ("features", "dimension_features"),
        ("bridge", "bridge_root"),
        ("geometry", "geometry_root"),
        ("outcomes", "outcomes_root"),
    ):
        directory = tmp_path / role
        directory.mkdir()
        path = directory / "control.json"
        path.write_text(json.dumps({"scope": "engineering_source_selection_fixture", "role": role}))
        setattr(args, attr, directory)
        evidence["source_bindings"].append({"path": str(path), **file_identity(path)})
    args.inference_root = tmp_path / "inference"
    args.inference_root.mkdir()
    for name in ("manifest.json", "evidence.json"):
        (args.inference_root / name).write_text("{}")
    reader = {
        "outputs": {
            "evidence.json": {
                "path": "evidence.json",
                **file_identity(args.inference_root / "evidence.json"),
            }
        }
    }
    args.results_root = tmp_path / "beir"
    args.results_root.mkdir()
    plan = {"scope": "engineering_mock_task", "results_root": str(args.results_root)}
    (args.results_root / "primary_admission.json").write_text(json.dumps(plan))
    (args.results_root / "task.json").write_text("{}")
    args.validation_root = tmp_path / "validation"
    args.validation_root.mkdir()
    for name in ("manifest.json", "admission.json", "sample_scores.jsonl"):
        (args.validation_root / name).write_text("{}")
    validation = {
        "plan": {},
        "manifest": {
            "path": str(args.validation_root / "manifest.json"),
            **file_identity(args.validation_root / "manifest.json"),
        },
        "outputs": {
            "sample_scores": {
                "path": "sample_scores.jsonl",
                **file_identity(args.validation_root / "sample_scores.jsonl"),
            }
        },
    }
    evidence["original_bridge_evidence"] = {
        "outcome_evidence": {
            "grid": {
                "evaluations": [
                    {
                        "plan": plan,
                        "tasks": [
                            {
                                "files": [
                                    {
                                        "path": str(args.results_root / "task.json"),
                                        **file_identity(args.results_root / "task.json"),
                                    }
                                ]
                            }
                        ],
                    }
                ]
            },
            "validation_selection": {"source_receipts": [validation]},
        }
    }
    return args, evidence, reader


def test_full_local_builder_with_explicitly_simulated_upstream_admission(
    contract, tmp_path, monkeypatch
):
    args, evidence, _ = selection_fixture(tmp_path)
    evidence["upstream_primary_admission_simulated"] = True
    plan = {"scope": "engineering_synthetic_authoring_wiring", "scientific_completion": False}
    tables = {"control": [{"scope": "engineering_only", "value": 1}]}
    args.inference_root.rename(tmp_path / "preserved-inference-placeholders")
    save_bundle(args.inference_root, plan, evidence, tables)
    args.validation_data = tmp_path / "no-real-validation-in-this-fixture"
    args.vector_manifest_sha256 = "0" * 64
    args.output = tmp_path / "explicit-synthetic-reconstruction"
    monkeypatch.setattr(authoring.inference, "gather", lambda *a: (plan, evidence, tables))
    monkeypatch.setattr(
        ValidationContract,
        "data",
        lambda *a: (
            None,
            {"scope": "engineering_simulated_validation_admission", "raw_text_included": False},
        ),
    )
    result = build(contract, args)
    payload = files.inspect(args.output, result["manifest_sha256"])
    assert payload["metadata"]["scientific_completion"] is False
    assert (
        read_json(args.output / "inference/evidence.json")["upstream_primary_admission_simulated"]
        is True
    )
    assert {
        "source",
        "training-source",
        "vectors",
        "features",
        "geometry",
        "outcomes",
        "bridge",
        "inference",
        "validation",
        "beir",
    } == {name.split("/", 1)[0] for name in payload["files"]}
    assert result["files"] > 100


@pytest.mark.parametrize(
    "changed", [None, "unbound_file", "changed_binding", "outside_beir", "changed_admission"]
)
def test_complete_explicit_selection_never_scans_other_results(tmp_path, changed):
    args, evidence, reader = selection_fixture(tmp_path)
    if changed == "unbound_file":
        (args.geometry_root / "extra").write_text("not admitted")
    elif changed == "changed_binding":
        (args.geometry_root / "control.json").write_text("changed")
    elif changed == "outside_beir":
        path = tmp_path / "outside.json"
        path.write_text("{}")
        evidence["original_bridge_evidence"]["outcome_evidence"]["grid"]["evaluations"][0]["tasks"][
            0
        ]["files"] = [{"path": str(path), **file_identity(path)}]
    elif changed == "changed_admission":
        (args.results_root / "primary_admission.json").write_text("{}")
    if changed:
        with pytest.raises(ValueError):
            analysis_selection(args, evidence, reader)
    else:
        selected = analysis_selection(args, evidence, reader)
        assert len(selected) == 12
        output = tmp_path / "selected"
        receipt = files.write(output, selected, {"scope": "engineering_synthetic_source_selection"})
        assert read_json(output / "beir/primary_admission.json")["results_root"] == str(
            args.results_root
        )
        files.inspect(output, receipt["manifest_sha256"])
