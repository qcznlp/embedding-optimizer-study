from argparse import Namespace

from embed_optim import evaluate_matrix


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
    command = evaluate_matrix._late_command(
        tmp_path, model, args, tmp_path / "results", 29620, num_processes=4
    )
    model_index = command.index("--models")
    tasks_index = command.index("--tasks")
    assert command[model_index + 1 : tasks_index] == [str(model)]
    assert command[tasks_index + 1 : command.index("--results_folder")] == args.tasks
