"""Bounded publication transport checks, never simulated primary admission."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_v3_publication as publication
from embed_optim import primary_v3_publication_archive as archive
from embed_optim import primary_v3_publication_reconstruction as reconstruction
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import canonical, file_identity
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_publication_contract import PublicationContract
from embed_optim.primary_v3_reconstruction_sources import source_selection

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(scope="module")
def contract():
    primary = PrimaryV3Contract.load(ROOT / publication.PARENTS["primary"][0], ROOT, CANDIDATE)
    return PublicationContract.load(ROOT / archive.PROTOCOL, primary)


@pytest.mark.parametrize("operation", [archive.prepare, archive.build])
def test_real_absent_primary_refuses_before_archive_inputs_or_output(contract, tmp_path, operation):
    with pytest.raises(ValueError, match="ordinary retained run"):
        operation(contract, SimpleNamespace(experiment_root=tmp_path / "absent"))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("marker", sorted(publication.SIMULATION_KEYS))
def test_simulated_gather_cannot_author_an_archive(contract, tmp_path, monkeypatch, marker):
    monkeypatch.setattr(
        publication, "gather", lambda *a: ({}, {"evidence.json": canonical({marker: True})}, [])
    )
    with pytest.raises(ValueError, match="Synthetic admission"):
        archive.prepare(contract, SimpleNamespace())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("marker", sorted(publication.SIMULATION_KEYS))
def test_numerical_reader_retains_every_known_simulation_annotation(marker):
    assert reconstruction.simulation({"nested": [{marker: True}]}) is True
    assert reconstruction.simulation({"nested": [{marker: False}]}) is False


@pytest.mark.parametrize("value", [None, 0, 1, "false", []])
def test_simulation_annotations_have_no_truthiness_coercion(value):
    with pytest.raises(ValueError, match="explicit Boolean"):
        reconstruction.simulation({"upstream_admission_simulated": value})


def test_earlier_true_marker_does_not_skip_a_later_invalid_annotation():
    with pytest.raises(ValueError, match="explicit Boolean"):
        reconstruction.simulation(
            [{"upstream_admission_simulated": True}, {"simulated_admission": None}]
        )


def test_source_only_closure_loads_actual_publication_parents_after_relocation(contract, tmp_path):
    selected = source_selection(contract.inference, {})
    acceptance = archive.authoring.require_parent(contract.inference)
    files.select(
        selected, "source", archive.authoring.ACCEPTANCE[0], acceptance, file_identity(acceptance)
    )
    archive.add_publication_sources(contract, selected)
    root = tmp_path / "source-only"
    result = files.write(root, selected, {"source_only": True, "primary_admission_supplied": False})
    manifest = files.inspect(root, result["manifest_sha256"])
    for name in ("primary_v3_publication_archive.py", "primary_v3_publication_reconstruction.py"):
        assert "source/src/embed_optim/" + name in manifest["files"]
    primary = PrimaryV3Contract.load(
        root / "source" / publication.PARENTS["primary"][0],
        root / "source",
        root / "training-source",
    )
    restored = PublicationContract.load(root / "source" / archive.PROTOCOL, primary)
    archive.require_foundation(restored)
    assert restored.sha256 == contract.sha256
    assert restored.payload["formal_execution_authorized"] is False


def test_actual_archive_cli_refuses_missing_primary(contract, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    argv = [
        "--repository",
        str(ROOT),
        "--training-root",
        str(CANDIDATE),
        "--vector-manifest-sha256",
        "a" * 64,
    ]
    for name in (
        "experiment-root",
        "results-root",
        "validation-data",
        "validation-root",
        "outcomes-root",
        "geometry-root",
        "reference",
        "bridge-root",
        "dimension-vectors",
        "dimension-features",
        "probe-root",
        "inference-root",
        "publication-root",
        "output",
    ):
        argv.extend(["--" + name, str(tmp_path / name)])
    with pytest.raises(ValueError, match="ordinary retained run"):
        archive.main(argv)
    assert not list(tmp_path.iterdir())


@pytest.fixture
def provenance_fixture(tmp_path, monkeypatch):
    names = reconstruction.joint.bundle_names(publication.inference.TABLE_COUNTS)
    role = tmp_path / "inference"
    role.mkdir()
    bound = []
    for name in names:
        (role / name).write_text(name)
        bound.append(
            {"path": "/unavailable/producer/inference/" + name, **file_identity(role / name)}
        )
    inputs = SimpleNamespace(
        root=tmp_path,
        evidence={"source_bindings": [], "original_bridge_evidence": {"geometry_reader": {}}},
    )
    previous = [
        {
            "original": "/unavailable/producer/outcomes/evidence.json",
            "local": "outcomes/evidence.json",
        }
    ]
    monkeypatch.setattr(reconstruction.joint, "provenance", lambda *a: previous)
    return inputs, {"source_bindings": bound}, previous


def test_complete_inference_role_uses_lexical_provenance_not_producer_files(provenance_fixture):
    inputs, evidence, _ = provenance_fixture
    result = reconstruction.provenance(inputs, evidence)
    assert len(result) == len(evidence["source_bindings"])
    assert all(row["local"].startswith("inference/") for row in result)


@pytest.mark.parametrize("kind", ["missing", "extra", "order", "subtree", "overlap", "schema"])
def test_inference_provenance_rejects_rehashed_semantic_changes(provenance_fixture, kind):
    inputs, original, previous = provenance_fixture
    evidence = copy.deepcopy(original)
    rows = evidence["source_bindings"]
    if kind == "missing":
        rows.pop()
    elif kind == "extra":
        rows.append(rows[0])
    elif kind == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif kind == "subtree":
        rows[-1]["path"] = "/elsewhere/" + Path(rows[-1]["path"]).name
    elif kind == "overlap":
        previous[0]["original"] = "/unavailable/producer/inference/outcomes/evidence.json"
    else:
        rows[0]["unexpected"] = True
    with pytest.raises(ValueError):
        reconstruction.provenance(inputs, evidence)
