from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from embed_optim import evaluate_matrix
from embed_optim.evaluation_utils import (
    FAST_PLAID_INDEX_KWARGS,
    late_ipc_result_path,
    task_result_remaining,
)


class _FakeProcess:
    def __init__(self, polls):
        self.polls = list(polls)
        self.returncode = None

    def poll(self):
        value = self.polls.pop(0) if self.polls else 0
        if value is not None:
            self.returncode = value
        return value


def test_late_evaluation_uses_second_pool_after_dense_finishes(tmp_path, monkeypatch):
    dense = tmp_path / "dense" / "checkpoint-1"
    late_one = tmp_path / "late-one" / "checkpoint-1"
    late_two = tmp_path / "late-two" / "checkpoint-1"
    monkeypatch.setattr(
        evaluate_matrix,
        "_selected_models",
        lambda args: {"dense": [dense], "late": [late_one, late_two]},
    )
    monkeypatch.setattr(evaluate_matrix.time, "sleep", lambda seconds: None)

    launches = []

    def popen(command, **kwargs):
        launches.append((command, kwargs.get("env", {}).get("CUDA_VISIBLE_DEVICES")))
        return (
            _FakeProcess([None, 0])
            if "dense_parallel.py" in " ".join(command)
            else _FakeProcess([0])
        )

    monkeypatch.setattr(evaluate_matrix.subprocess, "Popen", popen)
    args = Namespace(
        matrix="unused.yaml",
        families=["dense", "late"],
        run_ids=[],
        stages=None,
        tasks=["SciFact"],
        gpus_a="0,1,2,3",
        gpus_b="4,5,6,7",
        late_port_a=29610,
        late_port=29620,
        results_root=str(tmp_path / "results"),
        log_dir=str(tmp_path / "logs"),
    )

    assert evaluate_matrix.run_evaluation(args) == 0
    late_launches = [(command, gpus) for command, gpus in launches if command[0] == "accelerate"]
    assert [gpus for _, gpus in late_launches] == ["4,5,6,7", "0,1,2,3"]
    assert [command[command.index("--models") + 1] for command, _ in late_launches] == [
        str(late_one),
        str(late_two),
    ]
    assert all(command[command.index("--num_processes") + 1] == "4" for command, _ in late_launches)


def test_late_command_keeps_checkpoint_scoped_to_one_worker(tmp_path):
    args = Namespace(tasks=["SciFact", "FiQA2018"])
    model = tmp_path / "run" / "checkpoint-782"
    worker = tmp_path / "scripts" / "eval" / "late_interaction.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# worker\n")
    command = evaluate_matrix._late_command(
        tmp_path, model, args, tmp_path / "results", 29620, num_processes=4
    )
    model_index = command.index("--models")
    tasks_index = command.index("--tasks")
    assert command[model_index + 1 : tasks_index] == [str(model)]
    assert command[tasks_index + 1 : command.index("--results_folder")] == [
        "FiQA2018",
        "SciFact",
    ]


def test_evaluation_worker_falls_back_to_wheel_data_files(tmp_path):
    repo = tmp_path / "source-without-workers"
    prefix = tmp_path / "venv"
    worker = (
        prefix / "share" / "embedding-optimizer-study" / "scripts" / "eval" / "dense_parallel.py"
    )
    worker.parent.mkdir(parents=True)
    worker.write_text("# installed worker\n")

    assert evaluate_matrix._evaluation_script(repo, "dense_parallel.py", prefix) == worker


def test_late_ipc_result_paths_are_rank_stable_and_job_unique():
    base = late_ipc_result_path("model-a", "SciFact", "default", "test")
    assert base == late_ipc_result_path("model-a", "SciFact", "default", "test")
    variants = {
        late_ipc_result_path("model-b", "SciFact", "default", "test"),
        late_ipc_result_path("model-a", "FiQA2018", "default", "test"),
        late_ipc_result_path("model-a", "SciFact", "en", "test"),
        late_ipc_result_path("model-a", "SciFact", "default", "dev"),
    }
    assert base not in variants
    assert len(variants) == 4
    assert Path(base).name.startswith("mteb_plaid_results_")


def test_fast_plaid_settings_match_the_pinned_paper_protocol():
    assert FAST_PLAID_INDEX_KWARGS == {
        "override": True,
        "nbits": 4,
        "n_ivf_probe": 8,
        "n_full_scores": 8192,
        "seed": 42,
    }


def test_task_result_remaining_requires_every_split_and_subset(tmp_path):
    result = tmp_path / "revision" / "SciFactDecontaminated.json"
    result.parent.mkdir()
    result.write_text('{"scores":{"test":[{"hf_subset":"default"}]}}')

    assert not task_result_remaining(tmp_path, "SciFactDecontaminated", ["default"], ["test"])
    assert task_result_remaining(tmp_path, "SciFactDecontaminated", ["default"], ["dev"])
    assert task_result_remaining(tmp_path, "SciFactDecontaminated", ["default", "other"], ["test"])

    result.write_text("not-json")
    assert task_result_remaining(tmp_path, "SciFactDecontaminated", ["default"], ["test"])


def test_dense_decontaminated_cache_uses_result_task_name(monkeypatch):
    import importlib
    import sys

    eval_dir = Path(__file__).resolve().parents[1] / "scripts" / "eval"
    monkeypatch.syspath_prepend(str(eval_dir))
    dense_parallel = importlib.import_module("dense_parallel")
    task = SimpleNamespace(
        metadata=SimpleNamespace(name="SciFactDecontaminated", eval_splits=["test"]),
        hf_subsets=["default"],
    )
    monkeypatch.setattr(dense_parallel, "get_decontaminated_task", lambda name: task)

    assert dense_parallel.task_cache_requirements("SciFact", True) == (
        "SciFactDecontaminated",
        ["default"],
        ["test"],
    )
    jobs = [
        ("model-a", "SciFact", "results-a"),
        ("model-b", "ClimateFEVER", "results-b"),
        ("model-c", "MSMARCO", "results-c"),
    ]
    assert [job[1] for job in dense_parallel.order_jobs(jobs, True)] == [
        "ClimateFEVER",
        "MSMARCO",
        "SciFact",
    ]
    sys.modules.pop("dense_parallel", None)
