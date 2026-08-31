from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

import embed_optim.wandb_dense_provenance_audit as provenance_audit
from embed_optim.distribution_audit import _wandb_source_receipt_distribution_problems
from embed_optim.wandb_dense_provenance_audit import (
    EXPECTED_COUNTS,
    EXPECTED_GIT_REMOTE,
    EXPECTED_TOTAL_RUNS,
    SourceRunSpec,
    _git_commit_exists,
    _merge_source_wandb_config,
    audit_dense_source_provenance,
    audit_remote_source_run,
    load_frozen_dense_study,
    normalize_remote_source_history,
    receipt_envelope,
    write_receipt,
)

ROOT = Path(__file__).parents[1]


def _study():
    return load_frozen_dense_study(
        ROOT,
        scope_amendment="configs/dense_scope_amendment.json",
        experiment_matrix="configs/experiment.yaml",
        hybrid_matrix="configs/hybrid_adamw.yaml",
        training_plan="configs/dense_training_queue.json",
    )


def _expected_history() -> tuple[dict[str, int | float], ...]:
    return (
        {
            "global_step": 1,
            "train/epoch": 0.1,
            "train/grad_norm": 2.0,
            "train/learning_rate": 0.0,
            "train/loss": 1.5,
        },
        {"global_step": 2, "train/epoch": 1.0},
    )


def _remote_rows(
    history: tuple[dict[str, int | float], ...] | None = None,
) -> list[dict[str, int | float]]:
    result = []
    for index, row in enumerate(history or _expected_history()):
        remote = {"_step": index, "train/global_step": row["global_step"]}
        remote.update({key: value for key, value in row.items() if key != "global_step"})
        result.append(remote)
    return result


class FakeRun(SimpleNamespace):
    def scan_history(self, *, page_size: int):
        assert page_size == 1000
        return iter(self.history_rows)


class FakeApi:
    def __init__(self, runs):
        self.remote_runs = list(runs)
        self.paths = []

    def runs(self, path: str, *, per_page: int):
        self.paths.append((path, per_page))
        return list(self.remote_runs)


def _fake_run(spec: SourceRunSpec) -> FakeRun:
    return FakeRun(
        id=spec.source_run_id,
        state="finished",
        group="dense",
        tags=["dense", spec.config.optimizer.name, f"seed-{spec.config.seed}"],
        config=json.loads(json.dumps(spec.config.as_dict())),
        metadata={
            "git": {
                "remote": EXPECTED_GIT_REMOTE,
                "commit": "a" * 40,
            }
        },
        history_rows=_remote_rows(),
    )


def test_frozen_dense_source_study_has_exact_34_unique_runs():
    study = _study()

    assert len(study.runs) == EXPECTED_TOTAL_RUNS == 34
    assert Counter(spec.phase for spec in study.runs) == EXPECTED_COUNTS
    assert len({spec.source_run_id for spec in study.runs}) == 34
    assert (
        len(
            {
                (spec.config.output_root, spec.config.model_family, spec.config.run_id)
                for spec in study.runs
            }
        )
        == 34
    )
    assert study.entity == "stevezenguom"
    assert study.project == "embedding-optimizer-study"
    assert len(study.matrix_paths) == 8


def test_remote_history_last_write_normalization_is_deterministic():
    rows = [
        {
            "train/global_step": 1,
            "train/loss": 2.0,
            "train/epoch": 0.1,
        },
        {
            "train/global_step": 1,
            "train/loss": 1.5,
            "train/grad_norm": 2.0,
            "train/learning_rate": 0.0,
        },
        {"train/global_step": 2, "train/epoch": 1.0},
    ]

    normalized = normalize_remote_source_history(rows)

    assert normalized.history == _expected_history()
    assert normalized.raw_rows == 3
    assert normalized.duplicate_rows == 1
    assert normalized.overwritten_values == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda run: setattr(run, "state", "running"), "not finished"),
        (lambda run: setattr(run, "group", "other"), "group is not dense"),
        (lambda run: setattr(run, "tags", ["dense"]), "missing required tags"),
        (
            lambda run: run.config.update(seed=-1),
            "changed=seed",
        ),
        (
            lambda run: run.config.pop("seed"),
            "missing=seed",
        ),
        (
            lambda run: run.config.update(unexpected_remote_key=True),
            "extra=unexpected_remote_key",
        ),
        (
            lambda run: run.metadata["git"].update(remote="https://example.invalid/repo.git"),
            "Git origin differs",
        ),
        (
            lambda run: setattr(
                run,
                "history_rows",
                _remote_rows(({"global_step": 1, "train/loss": 99.0},)),
            ),
            "history differs",
        ),
    ],
)
def test_remote_source_audit_fails_closed_on_provenance_drift(mutation, message):
    spec = _study().runs[0]
    run = _fake_run(spec)
    mutation(run)

    with pytest.raises(RuntimeError, match=message):
        audit_remote_source_run(
            spec,
            run,
            repository=ROOT,
            expected_git_remote=EXPECTED_GIT_REMOTE,
            expected_remote_config=json.loads(json.dumps(spec.config.as_dict())),
            expected_history=_expected_history(),
            commit_exists=lambda _repository, _commit: True,
        )


def test_remote_source_audit_requires_a_locally_reachable_git_commit():
    spec = _study().runs[0]
    run = _fake_run(spec)

    with pytest.raises(RuntimeError, match="not reachable from a local Git ref"):
        audit_remote_source_run(
            spec,
            run,
            repository=ROOT,
            expected_git_remote=EXPECTED_GIT_REMOTE,
            expected_remote_config=json.loads(json.dumps(spec.config.as_dict())),
            expected_history=_expected_history(),
            commit_exists=lambda _repository, _commit: False,
        )


def test_git_commit_validation_rejects_a_dangling_commit(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked.txt").write_text("reachable\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Audit Test",
            "-c",
            "user.email=audit@example.invalid",
            "commit",
            "-q",
            "-m",
            "reachable",
        ],
        check=True,
    )
    reachable = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dangling = subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Audit Test",
            "-c",
            "user.email=audit@example.invalid",
            "commit-tree",
            tree,
        ],
        input="dangling\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert _git_commit_exists(tmp_path, reachable) is True
    assert _git_commit_exists(tmp_path, dangling) is False


def test_full_audit_covers_all_34_source_runs_without_remote_mutation():
    study = _study()
    api = FakeApi(_fake_run(spec) for spec in study.runs)

    audit = audit_dense_source_provenance(
        study,
        expected_git_remote=EXPECTED_GIT_REMOTE,
        api=api,
        history_loader=lambda _repository, _config: _expected_history(),
        config_loader=lambda _repository, config: json.loads(json.dumps(config.as_dict())),
        commit_exists=lambda _repository, _commit: True,
        verify_local_origin=False,
    )

    assert api.paths == [("stevezenguom/embedding-optimizer-study", 200)]
    assert audit["status"] == "passed"
    assert audit["remote_access"] == "read-only"
    assert audit["verified_runs"] == 34
    assert Counter(record["phase"] for record in audit["runs"]) == EXPECTED_COUNTS
    assert len({record["source_wandb_run_id"] for record in audit["runs"]}) == 34
    assert all(record["history_sha256"] for record in audit["runs"])
    assert all(record["config_sha256"] for record in audit["runs"])
    assert all(record["run_config_sha256"] for record in audit["runs"])


def test_full_audit_rejects_a_duplicate_or_missing_remote_source_identity():
    study = _study()
    runs = [_fake_run(spec) for spec in study.runs]
    runs.append(copy.deepcopy(runs[0]))
    api = FakeApi(runs)

    with pytest.raises(RuntimeError, match="exactly one remote W&B source run"):
        audit_dense_source_provenance(
            study,
            expected_git_remote=EXPECTED_GIT_REMOTE,
            api=api,
            history_loader=lambda _repository, _config: _expected_history(),
            config_loader=lambda _repository, config: json.loads(json.dumps(config.as_dict())),
            commit_exists=lambda _repository, _commit: True,
            verify_local_origin=False,
        )


def test_receipt_is_secret_free_and_self_hashing(tmp_path: Path):
    audit = {
        "schema_version": 1,
        "status": "passed",
        "verified_runs": 34,
        "runs": [{"source_wandb_run_id": "safe-run", "history_sha256": "b" * 64}],
    }
    envelope = receipt_envelope(audit)
    receipt = tmp_path / "receipt.json"
    write_receipt(receipt, envelope)

    observed = json.loads(receipt.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert observed["audit_sha256"] == expected_hash
    serialized = receipt.read_text(encoding="utf-8").lower()
    assert "wandb" + "_v1_" not in serialized
    assert "api_key" not in serialized
    assert "authorization" not in serialized


def test_receipt_refuses_secret_markers():
    token_prefix = "wandb" + "_v1_"
    with pytest.raises(RuntimeError, match="secret marker"):
        receipt_envelope({"status": "passed", "note": token_prefix + "not-a-real-secret"})


def _complete_receipt_audit() -> dict[str, object]:
    runs = []
    for phase, count in EXPECTED_COUNTS.items():
        for index in range(count):
            runs.append(
                {
                    "phase": phase,
                    "source_wandb_run_id": f"{phase}-{index}",
                    "optimizer": "muon",
                    "seed": index,
                    "group": "dense",
                    "required_tags": ["dense", "muon", f"seed-{index}"],
                    "config_sha256": "a" * 64,
                    "run_config_sha256": "b" * 64,
                    "config_keys": 179,
                    "history_sha256": "c" * 64,
                    "git_commit": "d" * 40,
                    "git_commit_validation": "reachable-local-ref",
                    "state": "finished",
                }
            )
    return {
        "schema_version": 1,
        "status": "passed",
        "remote_access": "read-only",
        "entity": "stevezenguom",
        "project": "embedding-optimizer-study",
        "expected_git_remote": EXPECTED_GIT_REMOTE,
        "expected_counts": EXPECTED_COUNTS,
        "verified_runs": EXPECTED_TOTAL_RUNS,
        "runs": runs,
    }


def test_distribution_contract_packages_and_revalidates_present_receipt(tmp_path: Path):
    source = "reports/wandb/dense_source_provenance_audit.json"
    write_receipt(tmp_path / source, receipt_envelope(_complete_receipt_audit()))
    data_files = {
        source: PurePosixPath(
            "share/embedding-optimizer-study/reports/wandb/dense_source_provenance_audit.json"
        )
    }

    assert _wandb_source_receipt_distribution_problems(tmp_path, data_files) == []

    tampered = json.loads((tmp_path / source).read_text(encoding="utf-8"))
    tampered["audit"]["runs"][0]["state"] = "running"
    (tmp_path / source).write_text(json.dumps(tampered), encoding="utf-8")
    problems = _wandb_source_receipt_distribution_problems(tmp_path, data_files)
    assert "W&B source-audit receipt envelope/hash differs" in problems
    assert "W&B source-audit receipt does not prove the frozen 34-run contract" in problems


def test_publication_contract_requires_wandb_before_distribution_and_tracks_receipt():
    gates = (ROOT / "docs/completion-gates.md").read_text(encoding="utf-8")
    source_audit = "python -m embed_optim.wandb_dense_provenance_audit"
    canonical_sync = "python -m embed_optim.wandb_sync"

    assert "--include-wandb" in gates
    assert "all 34 frozen Dense source runs" in gates
    assert "reports/wandb/dense_source_provenance_audit.json" in gates
    assert gates.index(source_audit) < gates.index(canonical_sync) < gates.index("uv build")
    assert '"reports/wandb/*.json"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!reports/wandb/dense_source_provenance_audit.json" in gitignore

    for path in (ROOT / "README.md", ROOT / "docs/blog.md"):
        document = path.read_text(encoding="utf-8")
        source_command = "embed-optim-audit-wandb-dense-sources"
        sync_command = "embed-optim-sync-wandb"
        assert "--include-wandb" in document
        assert "34 frozen Dense source runs" in " ".join(document.split())
        assert (
            document.index(source_command)
            < document.index(sync_command)
            < document.index("uv build")
        )


def test_exact_source_config_merge_only_strips_two_observed_empty_mappings():
    config = _study().runs[0].config

    merged = _merge_source_wandb_config(
        model_config={"_name_or_path": "/local/checkpoint", "model_only": 1},
        training_arguments={
            "learning_rate_mapping": {},
            "router_mapping": {},
            "trainer_only": 2,
        },
        run_config=config,
    )

    assert "learning_rate_mapping" not in merged
    assert "router_mapping" not in merged
    assert merged["_name_or_path"] == config.model_name
    assert merged["model_only"] == 1
    assert merged["trainer_only"] == 2
    normalized_run_config = json.loads(json.dumps(config.as_dict()))
    assert all(merged[key] == value for key, value in normalized_run_config.items())


@pytest.mark.parametrize(
    "mapping",
    [
        {"router_mapping": {}},
        {"learning_rate_mapping": {}, "router_mapping": {"unexpected": "value"}},
    ],
)
def test_exact_source_config_merge_rejects_unmodeled_mapping_state(mapping):
    with pytest.raises(RuntimeError, match="cannot reconstruct the exact W&B config"):
        _merge_source_wandb_config(
            model_config={},
            training_arguments=mapping,
            run_config=_study().runs[0].config,
        )


def test_remote_audit_implementation_has_no_wandb_mutation_path():
    source = inspect.getsource(provenance_audit)

    assert "wandb.init" not in source
    assert "run.update" not in source
    assert "run.finish" not in source
    assert "run.log" not in source
    assert "run.tags =" not in source


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"train/global_step": 0, "train/loss": 1.0}],
        [{"train/global_step": 1, "train/loss": float("nan")}],
        [{"train/loss": 1.0}],
    ],
)
def test_remote_history_rejects_incomplete_or_nonfinite_rows(rows):
    with pytest.raises(RuntimeError):
        normalize_remote_source_history(rows)
