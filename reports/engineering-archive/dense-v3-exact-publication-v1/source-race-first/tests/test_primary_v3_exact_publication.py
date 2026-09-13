"""Synthetic presentation/IO checks, distinct from actual primary-admission refusals."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_v3_exact_bridge as exact
from embed_optim import primary_v3_exact_publication as publication
from embed_optim import primary_v3_exact_publication_render as rendering
from embed_optim.bridge_numerics import evaluate
from embed_optim.primary_contract import file_identity, read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_exact_publication_contract import (
    OUTPUTS,
    PARENTS,
    ExactPublicationContract,
)
from embed_optim.primary_v3_outcomes import csv_bytes
from scripts.prepare_dense_v3_exact_publication import payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(scope="module")
def primary():
    return PrimaryV3Contract.load(ROOT / PARENTS["primary"][0], ROOT, CANDIDATE)


@pytest.fixture(scope="module")
def cases(primary):
    base = runpy.run_path(str(ROOT / "tests/test_exact_bridge_measurements.py"))["fixture"](primary)
    result = []
    for kind in ("varied", "undefined", "baseline_equivalent"):
        data = copy.deepcopy(base)
        if kind == "undefined":
            row = data[1][0]
            for displacement in ("saved_segment", "cumulative"):
                prefix = f"exact_{displacement}"
                row[prefix + "_nonzero_parameters"] = row[prefix + "_resolved_parameters"] = 0
                row[prefix + "_zero_parameters"] = 100
                row[prefix + "_nonzero_parameter_fraction"] = 0.0
                pair = next(
                    r
                    for r in data[2]
                    if r["displacement_kind"] == displacement
                    and r["first_optimizer"] == "adamw"
                    and r["second_optimizer"] == "muon"
                )
                pair.update(
                    mean_subspace_overlap=None,
                    defined_parameters=90,
                    defined_parameter_fraction=0.9,
                    undefined_boundary_parameters=10,
                    all_nonzero_pair_subspaces_resolved=False,
                )
            for column, _, _ in exact.CHECKPOINT_FEATURES.values():
                row[column] = None
        elif kind == "baseline_equivalent":
            for row in data[1]:
                for column, _, _ in exact.CHECKPOINT_FEATURES.values():
                    row[column] = 0.25
            for row in data[2]:
                row["mean_subspace_overlap"] = row["resolved_only_mean_subspace_overlap"] = 0.3
        old, _ = evaluate(data[0])
        tables, _ = exact.sensitivity_tables(primary, old, data[1], data[2])
        result.append((kind, old, data[1], data[2], tables))
    return result


@pytest.mark.parametrize("index", range(3))
def test_every_correspondence_kept_with_correct_undefined_and_equivalence(primary, cases, index):
    kind, old, cp, pairs, tables = cases[index]
    summary = rendering.checked_sensitivity(primary, old, cp, pairs, tables)
    assert [r["feature"] for r in summary["rows"]] == list(exact.FEATURES)
    assert len(old["feature_prediction_summary"]) == 9
    latex = rendering.render(summary)
    assert all(row["label"] in latex for row in summary["rows"])
    assert "full singular spectrum" in latex and "nonzero-matrix denominators" in latex
    assert summary["scientific_completion"] is summary["primary_admission_supplied"] is False
    if kind == "undefined":
        assert all(row["exact"]["predictively_useful"] is None for row in summary["rows"])
        assert all(row["exact"]["folds_defined"] == 0 for row in summary["rows"])
        assert "undefined for all five" in rendering.finding(summary["rows"])
    if kind == "baseline_equivalent":
        assert all(row["exact"]["criterion"] == "baseline equivalent" for row in summary["rows"])
        assert all(row["exact"]["predictively_useful"] is False for row in summary["rows"])
        assert all(row["exact"]["folds_defined"] == 4 for row in summary["rows"])


@pytest.mark.parametrize("name", exact.TABLE_COUNTS)
def test_every_exact_table_is_compared_with_fresh_calculation(primary, cases, name):
    _, old, cp, pairs, tables = cases[0]
    changed = copy.deepcopy(tables)
    changed[name][0]["untrusted_extra_column"] = 1
    with pytest.raises(ValueError):
        rendering.checked_sensitivity(primary, old, cp, pairs, changed)


@pytest.mark.parametrize("value", ["0", "1", "-1", "1/7", ""])
def test_exact_difference_wire_type_keeps_zero_integer_rational_and_null(tmp_path, cases, value):
    # Decoder-only control: these altered differences are not numerically admitted evidence.
    tables = copy.deepcopy(cases[0][-1])
    tables["predictive_sensitivity_summary"][0][publication.DIFFERENCE_FIELD] = value or None
    for name, rows in tables.items():
        (tmp_path / (name + ".csv")).write_bytes(csv_bytes(rows))
    assert publication.read_exact_tables(tmp_path) == tables


@pytest.mark.parametrize("value", ["0.0", "2/2", "1/0", "nan"])
def test_noncanonical_difference_is_not_silently_retyped(tmp_path, cases, value):
    tables = copy.deepcopy(cases[0][-1])
    tables["predictive_sensitivity_summary"][0][publication.DIFFERENCE_FIELD] = value
    for name, rows in tables.items():
        (tmp_path / (name + ".csv")).write_bytes(csv_bytes(rows))
    with pytest.raises(ValueError, match="rational difference"):
        publication.read_exact_tables(tmp_path)


def test_combination_preserves_every_original_byte_and_adds_both_invoked_macros(primary, cases):
    original = (
        b"% synthetic macro-only presentation control\n"
        b"\\newcommand{\\CorrectedGeometryBridgeFinding}{Original finding}\n"
        b"\\newcommand{\\CorrectedGeometryBridgeTable}{Original table}\n"
    )
    _, old, cp, pairs, tables = cases[0]
    exact_latex = rendering.render(
        rendering.checked_sensitivity(primary, old, cp, pairs, tables)
    ).encode()
    combined = rendering.combine(original, exact_latex)
    assert combined == original + b"\n" + exact_latex + rendering.WRAPPER.encode()
    assert combined.startswith(original)
    with pytest.raises(ValueError, match="already extended"):
        rendering.combine(combined, exact_latex)


@pytest.mark.parametrize("kind", ["pending", "missing", "duplicate", "wrong_exact"])
def test_incomplete_or_pending_original_text_cannot_be_extended(kind):
    finding = b"\\newcommand{\\CorrectedGeometryBridgeFinding}{x}\n"
    original = finding + b"\\newcommand{\\CorrectedGeometryBridgeTable}{y}\n"
    exact_latex = b"% Generated exact-measurement sensitivity; control only\n"
    if kind == "pending":
        original += b"\\ResultPending{x}"
    elif kind == "missing":
        original = finding
    elif kind == "duplicate":
        original += finding
    else:
        exact_latex = b"not the declared formatter"
    with pytest.raises(ValueError):
        rendering.combine(original, exact_latex)


def contract_for(primary, tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload(ROOT)))
    return ExactPublicationContract.load(path, primary)


@pytest.mark.parametrize(
    "field",
    [
        "scope",
        "schema_version",
        "status",
        "scientific_completion",
        "formal_execution_authorized",
        "manuscript_installation_authorized",
        "parents",
        "sources",
        "rules",
        "outputs",
    ],
)
def test_changed_preparation_declaration_is_rejected(primary, tmp_path, field):
    value = payload(ROOT)
    value[field] = {
        "scope": "legacy",
        "schema_version": True,
        "status": "released",
        "scientific_completion": True,
        "formal_execution_authorized": True,
        "manuscript_installation_authorized": True,
        "parents": {},
        "sources": {},
        "rules": {},
        "outputs": [],
    }[field]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        ExactPublicationContract.load(path, primary)


def test_actual_entry_rejects_missing_primary_before_exact_inputs(primary, tmp_path):
    contract = contract_for(primary, tmp_path)
    with pytest.raises(ValueError, match="ordinary retained run"):
        publication.gather(contract, SimpleNamespace(experiment_root=tmp_path / "absent"))


@pytest.mark.parametrize("mode", ["build", "inspect"])
def test_actual_cli_rejects_missing_primary_without_output(primary, tmp_path, monkeypatch, mode):
    contract = contract_for(primary, tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fixed = {"repository": ROOT, "training-root": CANDIDATE, "protocol": contract.path}
    args = [mode, "--vector-manifest-sha256", "not-an-admission"]
    for name in publication.ARGUMENT_PATHS:
        args += ["--" + name, str(fixed.get(name, tmp_path / name))]
    with pytest.raises(ValueError, match="ordinary retained run"):
        publication.main(args)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("marker", sorted(publication.original.SIMULATION_KEYS))
def test_simulated_original_success_cannot_enter_later_inputs(
    primary, tmp_path, monkeypatch, marker
):
    contract = contract_for(primary, tmp_path)
    contents = {"evidence.json": json.dumps({marker: True}).encode()}
    monkeypatch.setattr(publication.original, "gather", lambda *args: ({}, contents, []))
    with pytest.raises(ValueError, match="Synthetic admission"):
        publication.gather(contract, SimpleNamespace())


@pytest.mark.parametrize("marker", sorted(publication.original.SIMULATION_KEYS))
def test_simulated_exact_success_is_rejected_before_shared_evidence(
    primary, tmp_path, monkeypatch, marker
):
    contract = contract_for(primary, tmp_path)
    contents = {"evidence.json": b"{}"}
    monkeypatch.setattr(publication.original, "gather", lambda *args: ({}, contents, []))
    monkeypatch.setattr(publication.original, "inspect", lambda *args: {})
    monkeypatch.setattr(publication.exact.original, "snapshot", lambda *args: [])
    monkeypatch.setattr(publication.exact, "gather", lambda *args: ({}, {marker: True}, {}))
    with pytest.raises(ValueError, match="Synthetic admission"):
        publication.gather(contract, SimpleNamespace(publication_root=tmp_path))


def test_two_branch_evidence_must_match_not_only_primary_count(primary, tmp_path, monkeypatch):
    contract = contract_for(primary, tmp_path)
    contents = {
        "evidence.json": json.dumps(
            {"functional_evidence": {"original_bridge_evidence": {"source": "a"}}}
        ).encode()
    }
    monkeypatch.setattr(publication.original, "gather", lambda *args: ({}, contents, []))
    monkeypatch.setattr(publication.original, "inspect", lambda *args: {})
    monkeypatch.setattr(publication.exact.original, "snapshot", lambda *args: [])
    monkeypatch.setattr(
        publication.exact,
        "gather",
        lambda *args: ({}, {"original_bridge_evidence": {"source": "b"}}, {}),
    )
    with pytest.raises(ValueError):
        publication.gather(contract, SimpleNamespace(publication_root=tmp_path))


@pytest.mark.parametrize("target", ["original_publication", "exact_bundle"])
def test_input_changed_after_inspection_cannot_be_authenticated_as_new_input(
    primary, tmp_path, monkeypatch, target
):
    # Explicitly mocked upstream admission isolates the real adapter's snapshot timing.
    # No output from this control is a model admission or numerical generation proof.
    contract = contract_for(primary, tmp_path)
    common = {"scope": "synthetic_timing_control"}
    old = {name: b"synthetic timing control\n" for name in publication.original.OUTPUTS}
    old["evidence.json"] = json.dumps(
        {"functional_evidence": {"original_bridge_evidence": common}}
    ).encode()
    original_root, exact_root = tmp_path / "original", tmp_path / "exact"
    publication.original.save(original_root, {}, old)
    exact_root.mkdir()
    for name in ["manifest.json", "evidence.json", *(name + ".csv" for name in exact.TABLE_COUNTS)]:
        (exact_root / name).write_bytes(b"synthetic source timing control\n")
    diagnostics = {"numerical_diagnostics": {}, "measurement_coverage": []}
    exact_evidence = {"original_bridge_evidence": common, "source_bindings": [], **diagnostics}

    def second_gather(*args):
        if target == "original_publication":
            path = original_root / "findings.json"
            path.write_bytes(path.read_bytes() + b"changed after first inspection\n")
            manifest = read_json(original_root / "manifest.json")
            manifest["outputs"]["findings.json"] = {"path": "findings.json", **file_identity(path)}
            (original_root / "manifest.json").write_text(json.dumps(manifest))
        return {}, exact_evidence, {}

    def decode(*args):
        if target == "exact_bundle":
            path = exact_root / "bridge_rows.csv"
            path.write_bytes(path.read_bytes() + b"changed after exact inspection\n")
        return {}

    monkeypatch.setattr(publication.original, "gather", lambda *args: ({}, old, []))
    monkeypatch.setattr(publication.exact, "gather", second_gather)
    monkeypatch.setattr(publication, "inspect_bundle", lambda *args: {})
    monkeypatch.setattr(publication, "read_exact_tables", decode)
    monkeypatch.setattr(publication.original, "read_tables", lambda *args: {})
    monkeypatch.setattr(
        publication,
        "generate",
        lambda *args: {"exact-summary.json": json.dumps({"diagnostics": diagnostics}).encode()},
    )
    args = SimpleNamespace(
        publication_root=original_root,
        exact_bridge_root=exact_root,
        exact_geometry_root=tmp_path / "geometry",
    )
    with pytest.raises(ValueError):
        publication.gather(contract, args)


@pytest.mark.parametrize("name", OUTPUTS)
def test_rehashed_output_mutation_does_not_pass_fresh_evidence_comparison(tmp_path, name):
    contents = {key: b"synthetic IO control; no primary evidence\n" for key in OUTPUTS}
    plan, output = {"scope": "engineering_io_control"}, tmp_path / "result"
    assert publication.save(output, plan, contents)["recomputed_bytes_verified"] is True
    (output / name).write_bytes(contents[name] + b"changed\n")
    manifest = read_json(output / "manifest.json")
    manifest["outputs"][name] = {"path": name, **file_identity(output / name)}
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="freshly reconstructed"):
        publication.inspect(output, plan, contents)


@pytest.mark.parametrize("kind", ["extra", "manifest", "symlink", "overwrite"])
def test_inventory_and_ordinary_new_output_requirement(tmp_path, kind):
    contents = {key: b"synthetic IO control\n" for key in OUTPUTS}
    plan, output = {"scope": "engineering_io_control"}, tmp_path / "result"
    publication.save(output, plan, contents)
    if kind == "overwrite":
        with pytest.raises(ValueError, match="new ordinary"):
            publication.save(output, plan, contents)
        return
    if kind == "extra":
        (output / "extra.txt").write_text("unexpected")
    elif kind == "manifest":
        manifest = read_json(output / "manifest.json")
        manifest["status"] = "complete"
        (output / "manifest.json").write_text(json.dumps(manifest))
    else:
        path = output / "manifest.json"
        path.rename(tmp_path / "preserved-manifest.json")
        path.symlink_to(tmp_path / "preserved-manifest.json")
    with pytest.raises(ValueError):
        publication.inspect(output, plan, contents)


def test_manuscript_destination_is_never_an_authoring_target(tmp_path, monkeypatch):
    contract = SimpleNamespace(primary=SimpleNamespace(repository=tmp_path))
    monkeypatch.setattr(publication, "gather", lambda *args: ({}, {}, []))
    args = SimpleNamespace(output=tmp_path / "paper/generated/exact")
    with pytest.raises(ValueError, match="manuscript destinations"):
        publication.operate(contract, args, write=True)
    assert not args.output.exists()
