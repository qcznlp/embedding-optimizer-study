import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embed_optim import primary_completion as c

ROOT = Path(__file__).resolve().parents[1]
INPUTS = json.loads((ROOT / "reports/dense-primary-v2/input-bindings.json").read_text())
PROTOCOL = json.loads((ROOT / "configs/dense_primary_v2_protocol.json").read_text())
VERSIONS = json.loads((ROOT / "configs/formal_runtime.json").read_text())["packages"]


def expected(name="muon"):
    row = next(
        r for r in INPUTS["runs"] if r["identity_fields"]["recipe"]["optimizer"]["name"] == name
    )
    return copy.deepcopy({**INPUTS["common_identity"], **row["identity_fields"]})


def optimizer_fixture(name):
    identity = expected(name)
    shapes = [(2, 2), (3, 2), (2,)]
    groups = c.group_contract(identity)
    ids = [[0, 1], [2]] if name == "adamw" else [[0], [1], [2]]
    state = {}
    for group, members in zip(groups, ids, strict=True):
        group.update(params=members, lr=group["initial_lr"])
        for member in members:
            value = torch.zeros(shapes[member])
            if group["algorithm"] == "adamw":
                state[member] = {
                    "step": torch.tensor(1.0),
                    "exp_avg": value,
                    "exp_avg_sq": value.clone(),
                }
            else:
                state[member] = {"momentum_buffer": value}
                if name == "normuon":
                    state[member]["second_moment"] = torch.zeros((shapes[member][0], 1))
    scheduler = {
        "last_epoch": 1,
        "_step_count": 2,
        "base_lrs": [g["initial_lr"] for g in groups],
        "_last_lr": [g["initial_lr"] for g in groups],
        "lr_lambdas": [{} for _ in groups],
    }
    return {"state": state, "param_groups": groups}, scheduler, identity, shapes


@pytest.mark.parametrize("name", ["adamw", "muon", "normuon"])
def test_actual_optimizer_tensor_reader_accepts_matching_small_state_fixture(name):
    optimizer, scheduler, identity, shapes = optimizer_fixture(name)
    result = c.inspect_optimizer_state(optimizer, scheduler, identity, 1, 10, shapes)
    assert result["parameter_states"] == 3


@pytest.mark.parametrize("name", ["adamw", "muon", "normuon"])
@pytest.mark.parametrize(
    "change",
    [
        "missing_state",
        "duplicate_id",
        "nonfinite",
        "negative_moment",
        "counter",
        "scheduler_step",
        "scheduler_lr",
        "model_shape",
    ],
)
def test_tensor_reader_rejects_semantically_bad_state_even_if_readable(name, change):
    optimizer, scheduler, identity, shapes = optimizer_fixture(name)
    if change == "missing_state":
        optimizer["state"].pop(0)
    elif change == "duplicate_id":
        optimizer["param_groups"][-1]["params"].append(0)
    elif change == "nonfinite":
        values = optimizer["state"][0]
        values["exp_avg" if name == "adamw" else "momentum_buffer"][0, 0] = float("nan")
    elif change == "negative_moment":
        optimizer["state"][2]["exp_avg_sq"][0] = -1
    elif change == "counter":
        optimizer["state"][2]["step"] = torch.tensor(2.0)
    elif change == "scheduler_step":
        scheduler["_step_count"] = 1
    elif change == "scheduler_lr":
        scheduler["_last_lr"][0] *= 2
    else:
        shapes[0] = (1, 4)
    with pytest.raises(ValueError):
        c.inspect_optimizer_state(optimizer, scheduler, identity, 1, 10, shapes)


@pytest.mark.parametrize("name", ["muon", "normuon"])
def test_new_normuon_label_is_required_without_relabeling_muon(name):
    optimizer, scheduler, identity, shapes = optimizer_fixture(name)
    original = optimizer["param_groups"][0]["ns_implementation"]
    optimizer["param_groups"][0]["ns_implementation"] = (
        "unfused-bfloat16-v1" if name == "normuon" else "unfused-bfloat16-additive-eps-v2"
    )
    assert original != optimizer["param_groups"][0]["ns_implementation"]
    with pytest.raises(ValueError, match="ns_implementation"):
        c.inspect_optimizer_state(optimizer, scheduler, identity, 1, 10, shapes)


def timing():
    return {
        "schema_version": 1,
        "total_wall_time_seconds_max_rank": 3.0,
        "segments": [
            {
                "start_step_exclusive": i,
                "end_step_inclusive": i + 1,
                "started_at_utc": f"2026-09-06T11:00:0{i}Z",
                "checkpoint_completed_at_utc": f"2026-09-06T11:00:0{i + 1}Z",
                "wall_time_seconds_max_rank": 1.0,
            }
            for i in range(3)
        ],
    }


def test_timing_covers_full_epoch_without_overlaps():
    assert c.timing_summary(timing(), [1, 2, 3])["segments"] == 3


@pytest.mark.parametrize(
    "change", ["gap", "overlap", "missing_tail", "total", "bad_timestamp", "nan", "bool"]
)
def test_incomplete_or_invalid_timing_is_not_a_whole_run(change):
    value = timing()
    if change in {"gap", "overlap"}:
        value["segments"][1]["start_step_exclusive"] = 2 if change == "gap" else 0
    elif change == "missing_tail":
        value["segments"].pop()
    elif change == "total":
        value["total_wall_time_seconds_max_rank"] = 4.0
    elif change == "bad_timestamp":
        value["segments"][0]["started_at_utc"] = "2026-09-06T11:00:03Z"
    elif change == "nan":
        value["segments"][0]["wall_time_seconds_max_rank"] = float("nan")
    else:
        value["segments"][0]["start_step_exclusive"] = False
    with pytest.raises(ValueError):
        c.timing_summary(value, [1, 2, 3])


def task_fixture(directory, task="SciFact", *, run_id="fixture", step=782):
    directory.mkdir(parents=True, exist_ok=True)
    split, name = ("dev" if task == "MSMARCO" else "test"), f"{task}Decontaminated"
    result = {
        "task_name": name,
        "dataset_revision": PROTOCOL["beir_task_revisions"][task]["revision"],
        "mteb_version": VERSIONS["mteb"],
        "evaluation_time": 1.0,
        "scores": {
            split: [
                {
                    "hf_subset": "default",
                    "mteb_version": VERSIONS["mteb"],
                    "ndcg_at_10": 0.5,
                    "main_score": 0.5,
                    "nauc_recall_at_100_max": float("nan"),
                }
            ]
        },
    }
    meta = {
        "name": f"{run_id}/checkpoint-{step}",
        "revision": "local",
        "max_tokens": 8192,
        "embed_dim": 768,
        "similarity_fn_name": "cosine",
        "framework": ["Sentence Transformers"],
    }
    settings = {
        "task": name,
        "split": split,
        "subset": "default",
        "version": {k: VERSIONS[k] for k in c.RUNTIME_PACKAGES},
    }
    path = directory / f"{name}.json"
    path.write_text(json.dumps(result))
    (directory / "model_meta.json").write_text(json.dumps(meta))
    (directory / "run_settings.jsonl").write_text(json.dumps(settings) + "\n")
    return path, result, meta, settings


def inspect_task(path, task="SciFact"):
    return c.task_score(
        path,
        task,
        "fixture",
        782,
        8192,
        PROTOCOL["beir_task_revisions"][task]["revision"],
        VERSIONS,
    )


def test_undefined_auxiliary_metric_is_preserved_not_imputed_into_primary(tmp_path):
    path, _, _, _ = task_fixture(tmp_path)
    original = path.read_bytes()
    result = inspect_task(path)
    assert result["ndcg_at_10"] == 0.5
    assert len(result["undefined_auxiliary_metrics_not_used"]) == 1
    assert path.read_bytes() == original
    assert math.isnan(json.loads(path.read_text())["scores"]["test"][0]["nauc_recall_at_100_max"])


@pytest.mark.parametrize(
    "change",
    [
        "revision",
        "task",
        "main",
        "nan_primary",
        "inf_aux",
        "nan_other",
        "range",
        "bool",
        "missing_subset",
        "duplicate_subset",
        "model",
        "runtime",
        "settings_duplicate",
        "duplicate_key",
    ],
)
def test_bad_primary_score_or_metadata_is_still_rejected(tmp_path, change):
    path, value, meta, settings = task_fixture(tmp_path)
    row = value["scores"]["test"][0]
    if change == "revision":
        value["dataset_revision"] = "wrong"
    elif change == "task":
        value["task_name"] = "DifferentDecontaminated"
    elif change == "main":
        row["main_score"] = 0.6
    elif change == "nan_primary":
        row["ndcg_at_10"] = float("nan")
    elif change == "inf_aux":
        row["nauc_recall_at_100_max"] = float("inf")
    elif change == "nan_other":
        row["not_a_known_auxiliary_field"] = float("nan")
    elif change == "range":
        row["ndcg_at_10"] = row["main_score"] = 1.1
    elif change == "bool":
        row["ndcg_at_10"] = True
    elif change == "missing_subset":
        row["hf_subset"] = "different"
    elif change == "duplicate_subset":
        value["scores"]["test"].append(copy.deepcopy(row))
    elif change == "model":
        meta["name"] = "legacy/checkpoint-782"
    elif change == "runtime":
        settings["version"]["torch"] = "different"
    path.write_text(json.dumps(value))
    (path.parent / "model_meta.json").write_text(json.dumps(meta))
    (path.parent / "run_settings.jsonl").write_text(
        (json.dumps(settings) + "\n") * (2 if change == "settings_duplicate" else 1)
    )
    if change == "duplicate_key":
        path.write_text(path.read_text()[:-1] + ', "task_name":"SciFactDecontaminated"}')
    with pytest.raises(ValueError):
        inspect_task(path)


@pytest.mark.parametrize("coverage", ["complete", "missing", "duplicate"])
def test_task_coverage_requires_fourteen_distinct_valid_files(tmp_path, monkeypatch, coverage):
    # Explicit schema-only fixture: no genuine checkpoint or primary result is invented.
    plan = {"results_root": str(tmp_path), "tasks": PROTOCOL["evaluation"]["tasks"]}
    contract = SimpleNamespace(
        repository=ROOT, payload=PROTOCOL, expected_identity=lambda _: expected()
    )
    monkeypatch.setattr(c, "evaluation_plan", lambda *args: plan)
    (tmp_path / "primary_admission.json").write_text(json.dumps(plan))
    for task in plan["tasks"]:
        if coverage == "missing" and task == "SciFact":
            continue
        task_fixture(tmp_path / "dense" / task, task, run_id="fixture")
    if coverage == "duplicate":
        task_fixture(tmp_path / "dense" / "duplicated", "SciFact", run_id="fixture")
    if coverage == "complete":
        result = c.inspect_evaluation(
            contract, tmp_path / "not-a-real-checkpoint", "fixture", 782, tmp_path
        )
        assert (
            result["complete_task_files_verified"] is True
            and result["scientific_completion"] is False
        )
        assert len(result["tasks"]) == 14
    else:
        with pytest.raises(ValueError, match="fourteen distinct"):
            c.inspect_evaluation(
                contract, tmp_path / "not-a-real-checkpoint", "fixture", 782, tmp_path
            )


def actual_contract():
    from embed_optim.primary_contract import PrimaryContract

    return PrimaryContract.load(
        ROOT / "configs/dense_primary_v2_protocol.json",
        ROOT,
        ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source",
    )


def test_checked_in_completion_lock_binds_this_source_and_its_parent():
    assert (
        c.check_completion_lock(
            ROOT / "configs/dense_primary_v2_completion_protocol.json", actual_contract()
        )["bytes"]
        > 0
    )


@pytest.mark.parametrize("change", ["parent", "reader", "grid", "status"])
def test_completion_lock_cannot_be_rebound_silently(tmp_path, change):
    lock = json.loads((ROOT / "configs/dense_primary_v2_completion_protocol.json").read_text())
    if change in {"parent", "reader"}:
        lock["parent_protocol" if change == "parent" else "reader_source"]["sha256"] = "0" * 64
    elif change == "grid":
        lock["required_grid"]["cells"] = 839
    else:
        lock["status"] = "released"
    path = tmp_path / "wrong-lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        c.check_completion_lock(path, actual_contract())


@pytest.mark.parametrize("coverage", ["complete", "missing_run", "missing_task"])
def test_full_grid_adapter_requires_all_runs_before_collecting_840_cells(
    tmp_path, monkeypatch, coverage
):
    # The numerical/task readers have separate tests and real-file probes. This
    # tests only the matrix adapter's ordering and exact coverage, with mock rows.
    contract = actual_contract()
    calls = []

    def run_reader(directory, identity, steps):
        calls.append("run")
        if coverage == "missing_run" and len(calls) == 12:
            raise ValueError("missing final run")
        return {
            "whole_run_artifacts_verified": True,
            "dataset_fingerprint": contract.inputs["data_linkage_audit"][
                "training_view_fingerprint"
            ],
        }

    def eval_reader(*args):
        assert calls[:12] == ["run"] * 12
        calls.append("evaluation")
        tasks = contract.payload["evaluation"]["tasks"]
        if coverage == "missing_task":
            tasks = tasks[:-1]
        return {"tasks": [{"task": task, "ndcg_at_10": 0.5} for task in tasks]}

    monkeypatch.setattr(c, "inspect_complete_run", run_reader)
    monkeypatch.setattr(c, "inspect_evaluation", eval_reader)
    if coverage == "complete":
        result = c.inspect_matrix(contract, tmp_path, tmp_path)
        assert len(result["score_rows"]) == 840
        assert result["scientific_completion"] is False
        assert calls == ["run"] * 12 + ["evaluation"] * 60
    else:
        with pytest.raises(ValueError):
            c.inspect_matrix(contract, tmp_path, tmp_path)
        if coverage == "missing_run":
            assert "evaluation" not in calls
