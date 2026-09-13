import ast
import copy
import json
import subprocess
from pathlib import Path

import pytest

from embed_optim import primary_v3_completion as completion
from embed_optim import primary_v3_io as io
from embed_optim import primary_v3_training as training
from embed_optim.primary_contract import PrimaryContract, digest, read_json
from embed_optim.primary_v3_contract import PARENTS, PATHS, SCOPE, PrimaryV3Contract

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/dense_primary_v3_protocol.json"
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


def actual():
    return PrimaryV3Contract.load(PROTOCOL, ROOT, CANDIDATE)


def test_twelve_actual_revised_recipes_preserve_full_horizon_and_rules():
    contract = actual()
    assert contract.payload["scope"] == SCOPE
    assert contract.payload["status"] == "prepared_not_execution_authorized"
    assert len(contract.inputs["runs"]) == 12
    assert (
        contract.payload["evaluation"]["validation_selection"]["dataset"]
        == PATHS["validation_path"]
    )
    assert (
        "ties choose the lower learning rate"
        in contract.payload["evaluation"]["validation_selection"]["rule"]
    )
    for row in contract.inputs["runs"]:
        expected = contract.expected_identity(row["run_id"])
        assert digest(expected) == row["expected_identity_sha256"]
        assert expected["data"]["rows"] == 500000
        assert expected["recipe"]["epochs"] == 1
        assert expected["execution"]["training_arguments"]["max_steps"] == -1


@pytest.mark.parametrize("action", ["train", "backup", "evaluate"])
def test_draft_fails_before_any_output_or_model_import(tmp_path, action):
    contract = actual()
    with pytest.raises(ValueError, match="not execution authorized"):
        if action == "train":
            training.execute(contract, "verified-v3-muon-3e-4", tmp_path)
        elif action == "backup":
            io.backup(
                contract,
                tmp_path / "checkpoint-782",
                "verified-v3-muon-3e-4",
                782,
                tmp_path / "receipt.json",
            )
        else:
            io.evaluate(
                contract,
                tmp_path / "checkpoint-782",
                "verified-v3-muon-3e-4",
                782,
                tmp_path / "results",
                "0",
            )
    assert not list(tmp_path.iterdir())


def test_old_and_new_contract_loaders_do_not_accept_each_others_versions():
    with pytest.raises(ValueError):
        PrimaryContract.load(PROTOCOL, ROOT, CANDIDATE)
    with pytest.raises(ValueError):
        PrimaryV3Contract.load(ROOT / "configs/dense_primary_v2_protocol.json", ROOT, CANDIDATE)


@pytest.mark.parametrize(
    "bad_id",
    [
        "padded-muon-3e-4",
        "verified-muon-3e-4",
        "diagnostic-natural-muon",
        "verified-v3-muon-9e-4",
        "",
    ],
)
def test_historical_diagnostic_or_unplanned_run_ids_are_not_primary(bad_id):
    with pytest.raises(ValueError, match="Unknown or ambiguous"):
        actual().expected_identity(bad_id)


@pytest.mark.parametrize("step", [1, 2, 3, 1564, 3906, 3908, True, 782.0])
def test_short_diagnostic_and_unscheduled_steps_are_not_primary(tmp_path, step):
    with pytest.raises(ValueError, match="five v3 primary stages"):
        actual().checkpoint(tmp_path / "missing", "verified-v3-muon-3e-4", step)


@pytest.mark.parametrize(
    "keys,value",
    [
        (("scope",), "dense_primary_correctness_v2"),
        (("schema_version",), True),
        (("scientific_completion",), True),
        (("data_path",), "data/denseon-sft-500k-seed42"),
        (("validation_path",), "data/validation-4096-seed20260826"),
        (("output_root",), "outputs/dense-no-packing-v1"),
        (("checkpoint_prefix",), "old"),
        (("world_size",), 8),
        (("checkpoint_steps",), [1, 2, 3]),
        (("input_bindings", "sha256"), "0" * 64),
        (("natural_acceptance", "sha256"), "0" * 64),
        (("data_amendment", "sha256"), "0" * 64),
        (("primary_parent", "sha256"), "0" * 64),
        (("evaluation", "tasks"), ["SciFact"]),
        (("evaluation", "validation_selection", "rule"), "choose the best BEIR"),
        (("evaluation", "validation_selection", "dataset"), "old"),
        (("analysis", "primary_estimand"), "best learning rate on BEIR"),
        (("datasets", "training", "rows"), 499999),
        (("datasets", "validation", "rows"), 4095),
        (("datasets", "training", "files"), []),
        (("datasets", "validation", "files"), []),
        (("training_sources",), {}),
        (("consumer_sources",), {}),
        (("beir_task_revisions", "SciFact", "revision"), "main"),
        (("worker_python",), "unreviewed-python"),
        (("wandb", "entity"), "another-owner"),
    ],
)
def test_changed_protocol_rejected(tmp_path, keys, value):
    payload = read_json(PROTOCOL)
    node = payload
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        PrimaryV3Contract.load(changed, ROOT, CANDIDATE)


def test_status_edit_does_not_authorize_a_split_or_uncommitted_checkout(tmp_path):
    payload = read_json(PROTOCOL)
    payload["status"] = "reviewed_primary_execution_lock"
    path = tmp_path / "false-release.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="one assembled committed"):
        PrimaryV3Contract.load(path, ROOT, CANDIDATE, require_released=True)
    # Make the uncommitted condition explicit. The developer's actual checkout
    # can now be clean and committed; that must not change this negative test.
    uncommitted = tmp_path / "uncommitted-repository"
    uncommitted.mkdir()
    subprocess.run(["git", "init", "-q", str(uncommitted)], check=True)
    for name, _ in PARENTS.values():
        target = uncommitted / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    inside = uncommitted / "false-release.json"
    inside.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="exact committed"):
        PrimaryV3Contract.load(inside, uncommitted, uncommitted, require_released=True)


def test_v3_io_retains_every_existing_transfer_and_admission_operation():
    old = (ROOT / "src/embed_optim/primary_io.py").read_text()
    new = (ROOT / "src/embed_optim/primary_v3_io.py").read_text()
    old = old.replace("PrimaryContract.load", "PrimaryV3Contract.load")
    old = old.replace("primary_v2_", "primary_v3_").replace("primary-v2-", "primary-v3-")

    def functions(source):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    assert functions(old) == functions(new)


def test_matrix_collects_nothing_until_every_run_is_complete(tmp_path, monkeypatch):
    contract = actual()
    calls = []

    def incomplete(*args):
        raise ValueError("fixture missing full run")

    monkeypatch.setattr(PrimaryV3Contract, "complete_run", incomplete)
    monkeypatch.setattr(completion, "inspect_evaluation", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="missing full run"):
        completion.inspect_matrix(contract, tmp_path, tmp_path / "results")
    assert not calls


def test_complete_grid_requires_all_840_distinct_cells(tmp_path, monkeypatch):
    contract = actual()
    runs_seen = []

    def complete(_self, _directory, run_id):
        runs_seen.append(run_id)
        return {
            "dataset_fingerprint": contract.inputs["data_linkage_audit"][
                "training_view_fingerprint"
            ]
        }

    def evaluated(_contract, _checkpoint, run_id, step, _root):
        assert len(runs_seen) == 12
        return {
            "tasks": [
                {"task": t, "ndcg_at_10": 0.5} for t in contract.payload["evaluation"]["tasks"]
            ]
        }

    monkeypatch.setattr(PrimaryV3Contract, "complete_run", complete)
    monkeypatch.setattr(completion, "inspect_evaluation", evaluated)
    value = completion.inspect_matrix(contract, tmp_path, tmp_path / "results")
    assert value["scope"] == "dense_primary_v3_complete_grid"
    assert len(value["score_rows"]) == 840
    assert value["scientific_completion"] is False
    bad_contract = copy.deepcopy(contract)
    bad_contract.payload["evaluation"]["tasks"][0] = bad_contract.payload["evaluation"]["tasks"][1]
    runs_seen.clear()
    with pytest.raises(ValueError, match="grid is incomplete"):
        completion.inspect_matrix(bad_contract, tmp_path, tmp_path / "results")
