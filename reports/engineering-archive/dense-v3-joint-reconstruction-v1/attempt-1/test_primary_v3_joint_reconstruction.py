"""Joint raw-to-inference checks; no test fixture is primary model evidence."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_v3_bridge as bridge
from embed_optim import primary_v3_joint_reconstruction as joint
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_outcomes import TABLE_COUNTS as OUTCOME_COUNTS
from embed_optim.primary_v3_reconstruction_authoring import build
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(autouse=True)
def isolate_old_archival_test_aliases(monkeypatch):
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


@pytest.fixture
def minimal_roles(contract, tmp_path):
    """Only mapping syntax, not a fake full ReconstructionInput admission."""
    root = tmp_path / "explicit-synthetic-minimal-role-mapping"
    old, full = [], []
    groups = [
        ("outcomes", joint.bundle_names(OUTCOME_COUNTS), old),
        (
            "geometry",
            [
                "summary.json",
                *(name + ".csv" for name in contract.bridge.geometry.payload["table_counts"]),
            ],
            old,
        ),
        ("bridge", joint.bundle_names(bridge.TABLE_COUNTS), full),
        ("vectors", ["admission.json"], full),
        ("features", ["admission.json"], full),
    ]
    for role, names, destination in groups:
        for name in names:
            path = root / role / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"explicit synthetic mapping-only placeholder\n")
            destination.append(
                {"path": f"/unavailable-original/{role}/{name}", **file_identity(path)}
            )
    full.extend(copy.deepcopy(old))
    evidence = {"source_bindings": full, "original_bridge_evidence": {"source_bindings": old}}
    return SimpleNamespace(root=root, evidence=evidence, contract=contract)


def test_geometry_binding_order_matches_actual_frozen_authoring(minimal_roles):
    # gather iterates the loaded protocol, not the Python TABLE_COUNTS insertion order.
    result = joint.provenance(minimal_roles, {"raw_bindings": []})
    assert len(result) == 26
    assert len({row["local"] for row in result}) == 26
    assert all(not Path(row["original"]).exists() for row in result)


@pytest.mark.parametrize(
    "case", ["missing", "duplicate", "reordered", "escape", "suffix", "extra_file"]
)
def test_joint_provenance_rejects_mapping_contradictions(minimal_roles, case):
    old = minimal_roles.evidence["original_bridge_evidence"]["source_bindings"]
    full = minimal_roles.evidence["source_bindings"]
    if case == "missing":
        old.pop()
    elif case == "duplicate":
        old.append(copy.deepcopy(old[0]))
    elif case == "reordered":
        full[0], full[1] = full[1], full[0]
    elif case == "escape":
        full[0]["path"] = "/unavailable-original/bridge/../manifest.json"
    elif case == "suffix":
        full[-1]["sha256"] = "0" * 64
    else:
        (minimal_roles.root / "bridge/unused.csv").write_bytes(b"extra synthetic file\n")
    with pytest.raises(ValueError):
        joint.provenance(minimal_roles, {"raw_bindings": []})


def test_local_role_requires_complete_ordered_inventory(tmp_path):
    root = tmp_path / "archive"
    (root / "bridge").mkdir(parents=True)
    for name in ("manifest.json", "evidence.json"):
        (root / "bridge" / name).write_bytes(name.encode())
    inputs = SimpleNamespace(root=root)
    rows = [
        {"path": "/original/" + name, **file_identity(root / "bridge" / name)}
        for name in ("manifest.json", "evidence.json")
    ]
    _, addresses = joint.local_bindings(inputs, rows, "bridge", ["manifest.json", "evidence.json"])
    assert [row["local"] for row in addresses] == ["bridge/manifest.json", "bridge/evidence.json"]
    with pytest.raises(ValueError):
        joint.local_bindings(inputs, rows[::-1], "bridge", ["manifest.json", "evidence.json"])


def test_actual_authoring_still_refuses_absent_primary(contract, tmp_path):
    args = SimpleNamespace(
        experiment_root=tmp_path / "missing-primary", output=tmp_path / "forbidden"
    )
    with pytest.raises(ValueError, match="ordinary retained run"):
        build(contract, args)
    assert not args.output.exists()


def test_output_must_be_new_and_outside_input(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    inputs = SimpleNamespace(root=root)
    for path in (tmp_path, root, root / "nested"):
        with pytest.raises(ValueError, match="new joint reconstruction"):
            joint.reconstruct(inputs, path)


def test_joint_geometry_first_stage_uses_one_displacement_anchor(contract):
    from scripts.joint_reconstruction_geometry import joint_stage
    from scripts.vector_reconstruction_fixture import admitted_fixture

    admitted, _ = admitted_fixture(contract)
    primary = contract.primary
    recipe = primary.expected_identity(primary.inputs["runs"][0]["run_id"])["recipe"]
    stage, bases = joint_stage(
        recipe, 0, primary.payload["checkpoint_steps"], 1, admitted["reference"]
    )
    assert stage["checkpoint_row"]["saved_segment_to_weight_ratio"] > 0
    assert stage["previous_step"] == stage["cumulative_anchor_step"] == 0
    for row in stage["entry_records"]:
        assert row["saved_segment_nonzero_parameters"] == row["cumulative_nonzero_parameters"] > 0
    for key, value in bases.items():
        if key.startswith("saved_segment|"):
            assert value.equal(bases[key.replace("saved_segment|", "cumulative|", 1)])


def test_minimal_mapping_fixture_is_not_full_admission(minimal_roles):
    with pytest.raises((ValueError, FileNotFoundError)):
        files.inspect(minimal_roles.root, "0" * 64)
