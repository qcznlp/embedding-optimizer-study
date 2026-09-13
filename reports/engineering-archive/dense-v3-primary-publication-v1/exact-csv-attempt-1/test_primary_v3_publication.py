"""Complete-table synthetic checks; no fixture is a checkpoint-backed publication."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_v3_publication as publication
from embed_optim import primary_v3_publication_bridge as bridge_render
from embed_optim import primary_v3_publication_tables as display
from embed_optim.primary_contract import file_identity, read_json
from embed_optim.primary_v3_bridge import bridge_tables
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import csv_bytes, outcome_tables, system_rows
from embed_optim.primary_v3_publication_contract import OUTPUTS, PARENTS, PublicationContract
from scripts.prepare_dense_v3_publication import payload

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(scope="module")
def primary():
    return PrimaryV3Contract.load(ROOT / PARENTS["primary"][0], ROOT, CANDIDATE)


@pytest.fixture(scope="module")
def complete(primary):
    producers = runpy.run_path(str(ROOT / "tests/test_primary_v3_outcomes.py"))
    scores, selection = producers["fixture"].__wrapped__(primary)
    runs = producers["synthetic_runs"].__wrapped__(primary)
    outcomes = outcome_tables(primary, scores, selection)
    outcomes["system_metrics"] = system_rows(primary, runs)
    checkpoints, pairs, _ = runpy.run_path(str(ROOT / "tests/test_primary_v3_bridge.py"))["inputs"](
        primary
    )
    bridge, _ = bridge_tables(primary, checkpoints, pairs, outcomes["run_stage_scores"])
    return outcomes, checkpoints, pairs, bridge, selection, runs


def test_complete_primary_display_keeps_true_v3_selection_and_systems(primary, complete):
    value = display.summarize(primary, *complete)
    assert len(value["primary"]) == len(value["secondary"]) == 3
    assert len(value["optimizer_stage"]) == 15 and len(value["bridge"]) == 9
    assert value["outcome_table_counts"]["all_task_scores"] == 840
    assert value["all_geometry_states_retained"] == 60
    assert value["all_geometry_pairs_retained"] == 660
    for row in value["validation_selected"]:
        assert row["run_id"] == complete[4]["selected"][row["optimizer"]]
    for row in value["systems"]:
        members = [r for r in complete[0]["system_metrics"] if r["optimizer"] == row["optimizer"]]
        assert row["mean_wall_time_seconds"] == sum(r["wall_time_seconds"] for r in members) / 4
        assert (
            row["mean_maximum_checkpoint_gib"]
            == sum(r["maximum_checkpoint_gib"] for r in members) / 4
        )
        assert "mean_checkpoint_gib" not in row and "mean_wall_time_hours" not in row
    assert value["primary_admission_supplied_by_this_helper"] is False


@pytest.mark.parametrize("table", list(publication.TABLE_COUNTS))
def test_every_outcome_table_requires_its_complete_population(primary, complete, table):
    changed = copy.deepcopy(complete)
    changed[0][table].pop()
    with pytest.raises(ValueError):
        display.summarize(primary, *changed)


@pytest.mark.parametrize(
    "kind", ["summary", "secondary", "system", "identity", "duplicate", "bridge"]
)
def test_rehashed_or_complete_count_changes_do_not_supply_evidence(primary, complete, kind):
    changed = copy.deepcopy(complete)
    if kind == "summary":
        changed[0]["primary_summary"][0]["mean_delta_ndcg_at_10"] += 0.001
    elif kind == "secondary":
        changed[4]["selected"]["muon"] = changed[4]["selected"]["adamw"]
    elif kind == "system":
        changed[0]["system_metrics"][0]["maximum_checkpoint_gib"] += 0.01
    elif kind == "identity":
        changed[1][0]["learning_rate"] *= 2
    elif kind == "duplicate":
        changed[1][0] = dict(changed[1][1])
    else:
        changed[3]["feature_prediction_summary"][0]["folds_defined"] = 3
    with pytest.raises(ValueError):
        display.summarize(primary, *changed)


@pytest.mark.parametrize("index", range(4))
def test_promoted_bridge_retains_every_candidate_row_and_exact_latex(index):
    archive = ROOT / "reports/engineering-archive/dense-v3-primary-renderer-v1"
    candidate = runpy.run_path(str(archive / "candidate/primary_v3_publication_render.py"))
    tables = read_json(archive / "probe-result.json")["cases"][index]["complete_kernel_tables"]
    expected, actual = candidate["render_bridge"](tables), bridge_render.render_bridge(tables)
    assert actual["rows"] == expected["rows"]
    assert actual["latex"].splitlines()[1:] == expected["latex"].splitlines()[1:]
    assert actual["primary_admission_supplied"] is False


@pytest.mark.parametrize("index", range(4))
def test_complete_csv_roundtrip_preserves_exact_rational_strings(tmp_path, index):
    archive = ROOT / "reports/engineering-archive/dense-v3-primary-renderer-v1"
    tables = read_json(archive / "probe-result.json")["cases"][index]["complete_kernel_tables"]
    for name, rows in tables.items():
        (tmp_path / (name + ".csv")).write_bytes(csv_bytes(rows))
    restored = publication.read_tables(tmp_path, publication.inference.original.TABLE_COUNTS)
    assert restored == tables
    assert bridge_render.render_bridge(restored) == bridge_render.render_bridge(tables)


def contract_for(primary, tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload(ROOT)))
    return PublicationContract.load(path, primary)


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
def test_changed_preparation_declaration_refuses(primary, tmp_path, field):
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
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        PublicationContract.load(path, primary)


def test_actual_missing_primary_refuses_before_later_inputs(primary, tmp_path):
    contract = contract_for(primary, tmp_path)
    with pytest.raises(ValueError, match="ordinary retained run"):
        publication.gather(contract, SimpleNamespace(experiment_root=tmp_path / "absent"))


@pytest.mark.parametrize("marker", sorted(publication.SIMULATION_KEYS))
def test_simulated_upstream_success_cannot_author_primary_evidence(
    primary, tmp_path, monkeypatch, marker
):
    contract = contract_for(primary, tmp_path)
    monkeypatch.setattr(publication.inference, "gather", lambda *args: ({}, {marker: True}, {}))
    with pytest.raises(ValueError, match="Synthetic admission"):
        publication.gather(contract, SimpleNamespace())


@pytest.mark.parametrize("mode", ["build", "inspect"])
def test_actual_cli_refuses_missing_primary_without_output(primary, tmp_path, monkeypatch, mode):
    contract = contract_for(primary, tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    args = [
        mode,
        "--repository",
        str(ROOT),
        "--training-root",
        str(CANDIDATE),
        "--protocol",
        str(contract.path),
        "--vector-manifest-sha256",
        "not-an-admission",
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
        "output",
    ):
        args += ["--" + name, str(tmp_path / name)]
    with pytest.raises(ValueError, match="ordinary retained run"):
        publication.main(args)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name", OUTPUTS)
def test_io_primitive_rejects_rehashed_output_mutation(tmp_path, name):
    # Explicit non-scientific bytes test only the IO primitive, not public authoring.
    contents = {key: b"synthetic transport control\n" for key in OUTPUTS}
    plan = {"scope": "engineering_io_control", "scientific_completion": False}
    output = tmp_path / "saved"
    assert publication.save(output, plan, contents)["recomputed_bytes_verified"] is True
    (output / name).write_bytes(contents[name] + b"changed\n")
    manifest = read_json(output / "manifest.json")
    manifest["outputs"][name] = {"path": name, **file_identity(output / name)}
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="freshly reconstructed"):
        publication.inspect(output, plan, contents)


@pytest.mark.parametrize("kind", ["extra", "manifest", "symlink", "overwrite"])
def test_io_inventory_manifest_and_no_overwrite(tmp_path, kind):
    contents = {name: b"synthetic IO only\n" for name in OUTPUTS}
    plan, output = {"scope": "engineering_io_control"}, tmp_path / "saved"
    publication.save(output, plan, contents)
    if kind == "overwrite":
        with pytest.raises(ValueError, match="new ordinary"):
            publication.save(output, plan, contents)
        return
    if kind == "extra":
        (output / "extra.txt").write_text("extra")
    elif kind == "manifest":
        value = read_json(output / "manifest.json")
        value["status"] = "complete"
        (output / "manifest.json").write_text(json.dumps(value))
    else:
        path = output / "manifest.json"
        path.rename(tmp_path / "moved-manifest.json")
        path.symlink_to(tmp_path / "moved-manifest.json")
    with pytest.raises(ValueError):
        publication.inspect(output, plan, contents)


def test_preparation_does_not_write_inside_manuscript(tmp_path, monkeypatch):
    contract = SimpleNamespace(primary=SimpleNamespace(repository=tmp_path))
    monkeypatch.setattr(publication, "gather", lambda *args: ({}, {}, []))
    args = SimpleNamespace(output=tmp_path / "paper/generated/new-evidence")
    with pytest.raises(ValueError, match="manuscript destinations"):
        publication.operate(contract, args, write=True)
    assert not args.output.exists()
