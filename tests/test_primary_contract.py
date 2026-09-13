import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import primary_contract as core
from embed_optim import primary_io, primary_training

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/dense_primary_v2_protocol.json"
TRAINING = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


def actual():
    return core.PrimaryContract.load(PROTOCOL, ROOT, TRAINING)


def test_actual_twelve_recipe_contract_is_read_only_and_source_bound():
    value = actual()
    assert value.payload["status"] == core.DRAFT
    assert len(value.inputs["runs"]) == 12
    for row in value.inputs["runs"]:
        identity = value.expected_identity(row["run_id"])
        assert identity["data"]["rows"] == 500000
        assert identity["recipe"]["seed"] == 42
        assert identity["recipe"]["global_batch_size"] == 128
        assert identity["execution"]["training_arguments"]["max_steps"] == -1


@pytest.mark.parametrize("action", ["train", "backup", "evaluate"])
def test_draft_cannot_reach_a_mutating_consumer(tmp_path, action):
    contract = actual()
    with pytest.raises(ValueError, match="not execution authorized"):
        if action == "train":
            primary_training.execute(contract, "verified-muon-3e-4", tmp_path)
        elif action == "backup":
            primary_io.backup(
                contract, tmp_path / "missing", "verified-muon-3e-4", 782, tmp_path / "receipt.json"
            )
        else:
            primary_io.evaluate(
                contract, tmp_path / "missing", "verified-muon-3e-4", 782, tmp_path / "results", "0"
            )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad_id", ["padded-muon-3e-4", "verified-muon-9e-4", "late-muon-3e-4", ""])
def test_old_or_unplanned_run_is_not_a_primary_input(bad_id):
    with pytest.raises(ValueError, match="Unknown or ambiguous"):
        actual().expected_identity(bad_id)


@pytest.mark.parametrize("step", [1, 2, 1564, 3906, 3908, True, 782.0])
def test_unscheduled_or_nonintegral_stage_rejected_without_payload_read(tmp_path, step):
    with pytest.raises(ValueError, match="five primary stages"):
        actual().checkpoint(tmp_path / "missing", "verified-muon-3e-4", step)


@pytest.mark.parametrize(
    "change", ["tasks", "analysis", "source", "data_receipt", "world", "steps", "closure"]
)
def test_protocol_drift_is_not_accepted(tmp_path, change):
    payload = json.loads(PROTOCOL.read_text())
    if change == "tasks":
        payload["evaluation"]["tasks"].pop()
    elif change == "analysis":
        payload["analysis"]["primary_estimand"] = "pick the best BEIR learning rate"
    elif change == "source":
        payload["training_sources"]["src/embed_optim/train.py"]["sha256"] = "0" * 64
    elif change == "data_receipt":
        payload["input_bindings"]["sha256"] = "0" * 64
    elif change == "world":
        payload["world_size"] = 8
    elif change == "steps":
        payload["checkpoint_steps"][0] = 783
    elif change == "closure":
        payload["consumer_sources"].pop("src/embed_optim/primary_io.py")
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        core.PrimaryContract.load(path, ROOT, TRAINING)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../b", "a//b", "a\\b", ".", ""])
def test_only_canonical_inside_root_paths_allowed(path):
    with pytest.raises(ValueError):
        core.relative_path(path)


@pytest.fixture
def diagnostic(tmp_path):
    # Deliberately not a primary identity. This isolates the shared byte verifier
    # and mocked network interface below; PrimaryContract.load never accepts it.
    expected = {
        "scope": "engineering-only-fixture",
        "execution": {"world_size": 4},
        "numerical_contract": {"diagnostic": True},
        "recipe": {"run_id": "fixture"},
    }
    checkpoint = tmp_path / "checkpoint-782"
    checkpoint.mkdir()
    names = (
        "config.json",
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        *(f"rng_state_{r}.pth" for r in range(4)),
    )
    for name in names:
        (checkpoint / name).write_bytes(b"inert-payload-no-pickle-or-model-loading")
    (checkpoint / "trainer_state.json").write_text('{"global_step":782}')
    (checkpoint / "dense_numerical_contract.json").write_text(
        json.dumps(expected["numerical_contract"])
    )
    (checkpoint / core.RUN_RECEIPT).write_text(json.dumps(expected))
    files = [{"path": p.name, **core.file_identity(p)} for p in sorted(checkpoint.iterdir())]
    seal = {
        "schema_version": 1,
        "step": 782,
        "run_identity_sha256": core.digest(expected),
        "files": files,
    }
    (checkpoint / core.SEAL).write_text(json.dumps(seal))
    return checkpoint, expected


def test_shared_verifier_accepts_matching_complete_diagnostic_fixture(diagnostic):
    path, expected = diagnostic
    checked = core.inspect_sealed_checkpoint(path, expected, 782)
    assert checked["run_identity_sha256"] == core.digest(expected)


@pytest.mark.parametrize(
    "corruption",
    [
        "model",
        "optimizer",
        "rng",
        "missing",
        "extra",
        "wrong_identity",
        "symlink",
        "duplicate",
        "wrong_step",
    ],
)
def test_shared_verifier_rejects_invalid_payload(diagnostic, tmp_path, corruption):
    path, expected = diagnostic
    if corruption in {"model", "optimizer", "rng"}:
        name = {
            "model": "model.safetensors",
            "optimizer": "optimizer.pt",
            "rng": "rng_state_3.pth",
        }[corruption]
        (path / name).write_bytes(b"corrupted")
    elif corruption == "missing":
        (path / "scheduler.pt").rename(tmp_path / "retained-scheduler.pt")
    elif corruption == "extra":
        (path / "unsealed-file").write_text("new")
    elif corruption == "wrong_identity":
        expected = copy.deepcopy(expected)
        expected["recipe"]["run_id"] = "different"
    elif corruption == "symlink":
        original = path / "optimizer.pt"
        original.rename(tmp_path / "retained-optimizer.pt")
        original.symlink_to(tmp_path / "retained-optimizer.pt")
    else:
        seal = json.loads((path / core.SEAL).read_text())
        if corruption == "duplicate":
            seal["files"].append(seal["files"][0])
        else:
            seal["step"] = 1563
        (path / core.SEAL).write_text(json.dumps(seal))
    with pytest.raises(ValueError):
        core.inspect_sealed_checkpoint(path, expected, 782)


class MockReleasedFixture:
    """Explicit unit-only backend fixture, never a PrimaryContract release bypass."""

    sha256 = "d" * 64
    payload = {"checkpoint_repository": "fixture/repo", "checkpoint_prefix": "new-fixture"}

    def __init__(self, path, expected):
        self.path, self.expected = path, expected

    def require_execution(self):
        return self

    def checkpoint(self, path, run_id, step):
        return {
            "scope": "engineering-only-fixture",
            "run_id": run_id,
            "protocol_sha256": self.sha256,
            **core.inspect_sealed_checkpoint(path, self.expected, step),
        }


class MockApi:
    def __init__(self):
        self.calls = []

    def repo_info(self, *args, **kwargs):
        self.calls.append(("head", kwargs))
        return SimpleNamespace(sha="a" * 40)

    def create_commit(self, *args, **kwargs):
        self.calls.append(("upload", kwargs))
        return SimpleNamespace(oid="b" * 40)


def test_backup_and_audit_bind_original_commit_without_head_substitution(
    diagnostic, tmp_path, monkeypatch
):
    checkpoint, expected = diagnostic
    contract, api = MockReleasedFixture(checkpoint, expected), MockApi()
    files = contract.checkpoint(checkpoint, "fixture", 782)["files"]
    remote = {
        r["path"]: {"bytes": r["bytes"], "kind": "sha256", "digest": r["sha256"]} for r in files
    }
    revisions = []

    def inventory(api, repo, prefix, revision):
        revisions.append(revision)
        return {} if revision == "a" * 40 else remote

    monkeypatch.setattr(primary_io, "remote_inventory", inventory)
    receipt_path = tmp_path / "backup.json"
    receipt = primary_io.backup(contract, checkpoint, "fixture", 782, receipt_path, api=api)
    before = receipt_path.read_bytes()
    assert receipt["scientific_completion"] is False
    assert revisions == ["a" * 40, "b" * 40]
    assert api.calls[1][1]["parent_commit"] == "a" * 40
    assert len(api.calls[1][1]["operations"]) == len(files)
    result = primary_io.audit_backup(contract, checkpoint, "fixture", 782, receipt_path, api=api)
    assert result["commit_oid"] == "b" * 40 and revisions[-1] == "b" * 40
    assert receipt_path.read_bytes() == before and len(api.calls) == 2
    receipt["commit_oid"] = "main"
    alternate = tmp_path / "mutable-revision.json"
    alternate.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="immutable upload commit"):
        primary_io.audit_backup(contract, checkpoint, "fixture", 782, alternate, api=api)


def test_backup_rejects_existing_remote_prefix_without_overwriting(
    diagnostic, tmp_path, monkeypatch
):
    path, expected = diagnostic
    contract, api = MockReleasedFixture(path, expected), MockApi()
    monkeypatch.setattr(primary_io, "remote_inventory", lambda *args: {"already-exists": {}})
    with pytest.raises(ValueError, match="already exists"):
        primary_io.backup(contract, path, "fixture", 782, tmp_path / "receipt.json", api=api)
    assert [x[0] for x in api.calls] == ["head"]
    assert not (tmp_path / "receipt.json").exists()


def test_corrupt_payload_rejected_before_any_upload_call(diagnostic, tmp_path):
    path, expected = diagnostic
    contract, api = MockReleasedFixture(path, expected), MockApi()
    (path / "optimizer.pt").write_bytes(b"wrong")
    with pytest.raises(ValueError):
        primary_io.backup(contract, path, "fixture", 782, tmp_path / "receipt.json", api=api)
    assert api.calls == [] and not (tmp_path / "receipt.json").exists()


def test_evaluation_plan_is_content_and_protocol_addressed(diagnostic, tmp_path):
    path, expected = diagnostic
    contract = MockReleasedFixture(path, expected)
    contract.payload = {**contract.payload, "evaluation": {"tasks": ["diagnostic-only"]}}
    first = primary_io.evaluation_plan(contract, path, "fixture", 782, tmp_path / "results")
    assert first["scientific_completion"] is False
    assert not (tmp_path / "results").exists()
    contract.sha256 = "e" * 64
    other = primary_io.evaluation_plan(contract, path, "fixture", 782, tmp_path / "results")
    assert other["cache_key"] != first["cache_key"]


def test_released_flag_alone_cannot_authorize_mixed_source_checkout(tmp_path):
    payload = json.loads(PROTOCOL.read_text())
    payload["status"] = core.RELEASED
    path = tmp_path / "unreviewed-release.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="assembled committed"):
        core.PrimaryContract.load(path, ROOT, TRAINING, require_released=True)


def rehearsal():
    return json.loads(
        (
            ROOT / "reports/engineering-archive/dense-primary-contract-v1/rehearsal/result.json"
        ).read_text()
    )


def test_real_read_only_rehearsal_has_complete_non_scientific_coverage():
    from scripts.validate_dense_primary_contract import check_rehearsal

    assert check_rehearsal(rehearsal())["old_checkpoints_rejected"] == 60


@pytest.mark.parametrize("change", ["protocol", "science", "write", "old", "duplicate", "corrupt"])
def test_read_only_rehearsal_cannot_lose_coverage_or_expand_claims(change):
    from scripts.validate_dense_primary_contract import check_rehearsal

    payload = rehearsal()
    if change == "protocol":
        payload["protocol"]["sha256"] = "0" * 64
    elif change == "science":
        payload["scientific_completion"] = True
    elif change == "write":
        payload["hf_writes_executed"] = 1
    elif change == "old":
        payload["old_primary_checkpoints_rejected"].pop()
    elif change == "duplicate":
        payload["real_diagnostic_checkpoints"][0] = payload["real_diagnostic_checkpoints"][1]
    else:
        payload["real_corrupt_optimizer_copies_rejected"][0]["rejected"] = "different failure"
    with pytest.raises(AssertionError):
        check_rehearsal(payload)
