"""Worker lifecycle controls, NOT real calibration, GPU or training experiments.

The orchestration fixtures explicitly mock GPU context, distributed collectives,
factory construction, training, saves and native reading. Genuine archived source
metadata is used only to test bindings. No synthetic payload is a formal result.
"""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_factorial_v3_run_contract import component, metadata_identity
from transformers import TrainerState
from transformers.trainer_utils import TrainOutput

from embed_optim import factorial_v3_worker as worker
from embed_optim.primary_contract import digest, file_identity, read_json

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = Path(worker.__file__).resolve().parents[2]


def parameters(tmp_path, state="adamw_state", operator="adamw", seed=314159):
    locations = worker.Locations(
        repository=REPOSITORY, primary_source=tmp_path / "primary",
        experiment=tmp_path / "experiment", data_store=tmp_path / "data",
        evidence=tmp_path / "input-evidence",
    )
    calibrations = {
        label: {
            "path": str(tmp_path / "calibration" / label),
            "calibration_binding": {"bytes": 1, "sha256": "a" * 64},
        }
        for label in worker.STATES
    }
    return (
        locations, calibrations, state, operator, seed, SOURCE,
        tmp_path / "runs", tmp_path / "worker-records",
    ), {
        "project": "worker-synthetic-fixture", "entity": "local-test-only",
        "expected_worker_source": file_identity(worker.__file__),
    }


def created_fixture(declared):
    """Uninitialized shell of the real class; no constructor or model is run."""
    identity = metadata_identity(declared["state"], declared["operator"], declared["seed"])
    observed = component(identity)
    trainer = object.__new__(worker.RunBoundFactorialTrainer)
    trainer._bound_run = identity
    trainer._require_identity = lambda: copy.deepcopy(observed)
    trainer.args = SimpleNamespace(output_dir=declared["run_root"])
    trainer.state = TrainerState()
    trainer.optimizer = trainer.lr_scheduler = None
    trainer._resume_binding = trainer._resume_step = None
    trainer.model = SimpleNamespace(parameters=lambda: [])
    config, arguments = worker.factory.recipe(
        identity, Path(declared["locations"]["data_store"]) / worker.factory.BRANCH,
        declared["output_root"], project=declared["project"], entity=declared["entity"],
    )
    creation = {
        "scope": worker.factory.SCOPE, "scientific_admission": False,
        "execution_authorized": False, "run_identity_sha256": digest(identity),
        "factory_source": file_identity(worker.factory.__file__),
        "recipe": config.as_dict(), "requested_arguments": arguments,
        "loaded_model": {
            "semantic_config_sha256": "a" * 64, "pooling": {}, "transformer": {},
            "prompts": {}, "tokenizer_backend_sha256": "b" * 64,
            "embedding_dimensions": 768, "maximum_context": 8192,
            "backend": "flash_attention_2", "gradient_checkpointing": True,
            "training_mode": True, "diagnostic_cpu": False,
            "forward_or_backward_executed": False,
        },
        "training_executed": False, "source_release_or_resource_gate_waived": False,
    }
    return trainer, creation, observed


def mocked_worker(monkeypatch, tmp_path, state="adamw_state", operator="adamw", seed=314159):
    args, kwargs = parameters(tmp_path, state, operator, seed)
    declared = worker.request(*args, **kwargs)
    trainer, creation, observed = created_fixture(declared)
    calls = []
    monkeypatch.setattr(worker.factory, "require_gpu_context", lambda: None)
    monkeypatch.setattr(worker, "collective_phase", lambda action: action())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def gather(destination, value):
        for rank in range(4):
            destination[rank] = copy.deepcopy(value)
            if isinstance(value, dict) and "rank" in value:
                destination[rank]["rank"] = rank

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *a, **k: None)

    def prepare(*actual_args, **actual_kwargs):
        calls.append(("prepare", actual_args, actual_kwargs))
        Path(declared["run_root"]).mkdir(parents=True, exist_ok=False)
        return trainer, creation

    def train(**options):
        calls.append(("train", options))
        trainer.state = TrainerState(
            global_step=391, max_steps=391, num_train_epochs=1,
            train_batch_size=8, epoch=1.0,
        )
        return TrainOutput(391, 0.2, {"train_loss": 0.2, "train_runtime": 1.0})

    def save():
        calls.append(("save",))
        path = Path(declared["run_root"]) / worker.contract.RUN_COMPLETE
        worker._write_new(path, {"synthetic_placeholder_not_a_native_run": True})
        return file_identity(path)

    def native(locations, request, binding):
        calls.append(("native_read", request, binding))
        return {"run_identity": trainer._bound_run, "explicitly_mocked_native_read": True}

    monkeypatch.setattr(worker.factory, "prepare_trainer", prepare)
    trainer.train = train
    trainer.save_run_completion = save
    monkeypatch.setattr(worker, "_read_native", native)
    return args, kwargs, declared, trainer, creation, observed, calls


@pytest.mark.parametrize("state", ("adamw_state", "muon_state"))
@pytest.mark.parametrize("operator", ("adamw", "muon"))
@pytest.mark.parametrize("seed", (314159, 271828, 161803))
def test_mocked_complete_lifecycle_and_fresh_reader(monkeypatch, tmp_path, state, operator, seed):
    args, kwargs, declared, _, _, _, calls = mocked_worker(monkeypatch, tmp_path, state, operator, seed)
    binding = worker.run_branch(*args, **kwargs)
    assert [v[0] for v in calls] == ["prepare", "train", "save", "native_read"]
    assert calls[1][1] == {"resume_from_checkpoint": None}
    assert calls[0][1] == (*args[:7],)
    assert calls[0][2] == {"project": kwargs["project"], "entity": kwargs["entity"]}
    complete = read_json(Path(declared["record_root"]) / worker.COMPLETE)
    assert [r["rank"] for r in complete["rank_results"]] == list(range(4))
    assert complete["process_exit_observed"] is False
    assert complete["scientific_admission"] is False
    proof = worker.read_execution(args[0], declared, binding)
    assert proof["worker_and_native_artifacts_verified"] is True
    assert proof["training_reexecuted"] is False
    assert [v[0] for v in calls].count("train") == 1
    assert [v[0] for v in calls].count("native_read") == 2
    assert not torch.cuda.is_initialized()


def test_actual_cpu_call_refuses_before_any_write_or_factory(monkeypatch, tmp_path):
    args, kwargs = parameters(tmp_path)
    monkeypatch.setattr(worker.factory, "prepare_trainer", lambda *a, **k: pytest.fail("early factory"))
    with pytest.raises(ValueError, match="already admitted four-rank NCCL"):
        worker.run_branch(*args, **kwargs)
    assert not args[6].exists() and not args[7].exists()
    assert not torch.cuda.is_initialized()


def test_actual_assembled_source_closure(tmp_path):
    args, kwargs = parameters(tmp_path)
    sources = worker.require_sources(args[0], args[5], kwargs["expected_worker_source"])
    assert len(sources) == 70
    assert sources["src/embed_optim/factorial_v3_factory.py"]["sha256"] == worker.FACTORY_SHA
    assert sources["src/embed_optim/factorial_v3_worker.py"] == file_identity(worker.__file__)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("problem", (
    "missing_calibration", "same_calibration", "hand_entered_rate", "bad_binding",
    "wrong_state", "wrong_operator", "wrong_seed", "relative_output", "record_in_model",
    "record_in_source", "record_in_data", "symlink_record", "wrong_worker_source",
))
def test_request_and_source_controls(tmp_path, problem):
    args, kwargs = parameters(tmp_path)
    args = list(args)
    if problem == "missing_calibration":
        del args[1]["muon_state"]
    elif problem == "same_calibration":
        args[1]["muon_state"]["path"] = args[1]["adamw_state"]["path"]
    elif problem == "hand_entered_rate":
        args[1]["muon_state"]["hidden_lr"] = 1e-3
    elif problem == "bad_binding":
        args[1]["adamw_state"]["calibration_binding"]["bytes"] = True
    elif problem == "wrong_state":
        args[2] = "normuon_state"
    elif problem == "wrong_operator":
        args[3] = "normuon"
    elif problem == "wrong_seed":
        args[4] = 42
    elif problem == "relative_output":
        args[6] = Path("relative")
    elif problem == "record_in_model":
        args[7] = args[6] / "records"
    elif problem == "record_in_source":
        args[7] = SOURCE / "records"
    elif problem == "record_in_data":
        args[7] = args[0].data_store / "records"
    elif problem == "symlink_record":
        args[7].symlink_to(tmp_path / "missing-target", target_is_directory=True)
    else:
        kwargs["expected_worker_source"] = {"bytes": 1, "sha256": "f" * 64}
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.request(*args, **kwargs)
        worker.require_sources(args[0], args[5], kwargs["expected_worker_source"])


@pytest.mark.parametrize("problem", (
    "step", "epoch", "optimizer", "scheduler", "resume_binding", "resume_step",
    "gradient", "creation_recipe", "creation_authority", "cpu_loading", "short_context",
))
def test_fresh_factory_controls(tmp_path, problem):
    args, kwargs = parameters(tmp_path)
    declared = worker.request(*args, **kwargs)
    sources = worker.require_sources(args[0], args[5], kwargs["expected_worker_source"])
    trainer, creation, _ = created_fixture(declared)
    if problem == "step":
        trainer.state.global_step = 1
    elif problem == "epoch":
        trainer.state.epoch = 0.2
    elif problem == "optimizer":
        trainer.optimizer = object()
    elif problem == "scheduler":
        trainer.lr_scheduler = object()
    elif problem == "resume_binding":
        trainer._resume_binding = {}
    elif problem == "resume_step":
        trainer._resume_step = 79
    elif problem == "gradient":
        trainer.model.parameters = lambda: [SimpleNamespace(grad=1)]
    elif problem == "creation_recipe":
        creation["recipe"]["max_length"] = 512
    elif problem == "creation_authority":
        creation["execution_authorized"] = True
    elif problem == "cpu_loading":
        creation["loaded_model"]["diagnostic_cpu"] = True
    else:
        creation["loaded_model"]["maximum_context"] = 512
    with pytest.raises(ValueError):
        worker._require_created(trainer, creation, declared, sources)


@pytest.mark.parametrize("phase", ("train", "save", "native_read"))
def test_failure_preserves_attempt_without_completion_or_retry(monkeypatch, tmp_path, phase):
    args, kwargs, declared, trainer, _, _, calls = mocked_worker(monkeypatch, tmp_path)

    def fail(*a, **k):
        raise RuntimeError("synthetic private exception text must not be serialized")

    if phase == "train":
        trainer.train = fail
    elif phase == "save":
        trainer.save_run_completion = fail
    else:
        monkeypatch.setattr(worker, "_read_native", fail)
    with pytest.raises(RuntimeError):
        worker.run_branch(*args, **kwargs)
    records = Path(declared["record_root"])
    assert (records / worker.STARTED).is_file()
    assert not (records / worker.COMPLETE).exists()
    failed = records / "rank-0-failed.json"
    before = file_identity(failed)
    assert "private exception text" not in failed.read_text()
    count = len(calls)
    with pytest.raises(ValueError, match="existing attempt"):
        worker.run_branch(*args, **kwargs)
    assert len(calls) == count and file_identity(failed) == before


@pytest.mark.parametrize("problem", ("partial", "nan", "duplicate_rank", "missing_rank"))
def test_rank_completion_controls(problem):
    rows = [{"rank": i, "global_step": 391, "training_loss": 0.2, "metrics": {}} for i in range(4)]
    if problem == "partial":
        rows[2]["global_step"] = 390
    elif problem == "nan":
        rows[1]["metrics"]["train_loss"] = float("nan")
    elif problem == "duplicate_rank":
        rows[3]["rank"] = 2
    else:
        rows.pop()
    with pytest.raises(ValueError):
        worker._require_rank_results(rows)


def replace_json(path, value):
    import json

    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize("problem", (
    "changed_factory_recipe", "changed_factory_arguments", "extra_failure", "claimed_exit",
    "changed_native", "changed_rank", "missing_calibration_on_real_reader",
))
def test_fresh_reader_controls_even_after_rehash(monkeypatch, tmp_path, problem):
    original_native = worker._read_native
    args, kwargs, declared, _, _, _, _ = mocked_worker(monkeypatch, tmp_path)
    binding = worker.run_branch(*args, **kwargs)
    records = Path(declared["record_root"])
    complete = read_json(records / worker.COMPLETE)
    if problem.startswith("changed_factory_"):
        value = read_json(records / "factory.json")
        if problem == "changed_factory_recipe":
            value["creation"]["recipe"]["max_length"] = 512
        else:
            value["creation"]["requested_arguments"]["max_steps"] = 10
        replace_json(records / "factory.json", value)
        complete["factory_record"] = file_identity(records / "factory.json")
    elif problem == "extra_failure":
        worker._write_new(records / "rank-1-failed.json", {"synthetic_failure": True})
    elif problem == "claimed_exit":
        complete["process_exit_observed"] = True
    elif problem == "changed_native":
        complete["native_readback"]["explicitly_mocked_native_read"] = False
    elif problem == "changed_rank":
        complete["rank_results"][0]["global_step"] = 300
    else:
        monkeypatch.setattr(worker, "_read_native", original_native)
    replace_json(records / worker.COMPLETE, complete)
    binding = file_identity(records / worker.COMPLETE)
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.read_execution(args[0], declared, binding)
    assert not torch.cuda.is_initialized()


def test_module_has_no_launcher_or_process_control():
    import ast

    tree = ast.parse(Path(worker.__file__).read_text())
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"init_process_group", "destroy_process_group", "acquire", "Popen", "kill", "system"}
    assert not torch.cuda.is_initialized()
