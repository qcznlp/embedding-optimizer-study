import importlib
import json
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import evaluate_matrix
from embed_optim.evaluation_utils import (
    FAST_PLAID_INDEX_KWARGS,
    configure_atomic_mteb_results,
    disable_mteb_cache_writes,
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
    monkeypatch.setattr(evaluate_matrix, "_validate_training_inputs", lambda args: None)
    monkeypatch.setattr(evaluate_matrix.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(evaluate_matrix, "_worker_python", lambda executable=None: "/system/python")
    monkeypatch.setattr(evaluate_matrix, "_validate_formal_runtime", lambda python, matrix: None)
    monkeypatch.setattr(evaluate_matrix, "_validate_worker_runtime", lambda python, models: {})
    monkeypatch.setattr(evaluate_matrix, "_validate_worker_sources", lambda python, sources: None)
    monkeypatch.setattr(evaluate_matrix, "_record_evaluation_inputs", lambda results, models: None)

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
        worker_python="/system/python",
    )

    assert evaluate_matrix.run_evaluation(args) == 0
    late_launches = [
        (command, gpus) for command, gpus in launches if "accelerate.commands.launch" in command
    ]
    dense_launch = next(command for command, _ in launches if "dense_parallel.py" in command[1])
    assert dense_launch[0] == "/system/python"
    assert [gpus for _, gpus in late_launches] == ["4,5,6,7", "0,1,2,3"]
    assert [command[command.index("--models") + 1] for command, _ in late_launches] == [
        str(late_one),
        str(late_two),
    ]
    assert all(command[command.index("--num_processes") + 1] == "4" for command, _ in late_launches)


def test_evaluation_cli_defaults_dense_and_validates_scope_before_gpu_work(monkeypatch):
    args = evaluate_matrix.parse_args([])
    assert args.families == ["dense"]
    assert evaluate_matrix.parse_args(["--families", "dense", "late"]).families == [
        "dense",
        "late",
    ]

    monkeypatch.setattr(
        evaluate_matrix,
        "_selected_models",
        lambda args: pytest.fail("scope must be validated before selecting checkpoints"),
    )
    with pytest.raises(ValueError, match="requires --scope-amendment"):
        evaluate_matrix.run_evaluation(args)


def test_evaluation_preflight_requires_deep_validated_training_inputs(monkeypatch):
    config = SimpleNamespace(model_family="dense", run_id="adamw-test")
    monkeypatch.setattr(evaluate_matrix, "_selected_configs", lambda args: [config])
    monkeypatch.setattr(
        evaluate_matrix,
        "audit_dataset_artifacts",
        lambda configs: {
            "complete": True,
            "training_view_fingerprint": "expected-view",
            "errors": [],
        },
    )
    observed = {}

    def audit_training(configs, *, deep, expected_dataset_fingerprint):
        observed.update(
            configs=configs,
            deep=deep,
            expected_dataset_fingerprint=expected_dataset_fingerprint,
        )
        return {
            "complete": True,
            "verified_runs": 1,
            "verified_checkpoints": 5,
            "errors": [],
        }

    monkeypatch.setattr(evaluate_matrix, "audit_training_artifacts", audit_training)
    evaluate_matrix._validate_training_inputs(SimpleNamespace())
    assert observed == {
        "configs": [config],
        "deep": True,
        "expected_dataset_fingerprint": "expected-view",
    }

    monkeypatch.setattr(
        evaluate_matrix,
        "audit_training_artifacts",
        lambda *args, **kwargs: {
            "complete": False,
            "verified_runs": 0,
            "verified_checkpoints": 4,
            "errors": ["checkpoint payload is corrupt"],
        },
    )
    with pytest.raises(RuntimeError, match="checkpoint payload is corrupt"):
        evaluate_matrix._validate_training_inputs(SimpleNamespace())


def test_evaluation_formal_runtime_uses_worker_interpreter(tmp_path, monkeypatch):
    matrix = tmp_path / "experiment.yaml"
    spec = tmp_path / "formal_runtime.json"
    matrix.write_text("formal_runtime: formal_runtime.json\n")
    spec.write_text("{}")
    observed = {}

    def run(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return SimpleNamespace(returncode=0, stdout='{"valid": true}', stderr="")

    monkeypatch.setattr(evaluate_matrix.subprocess, "run", run)
    evaluate_matrix._validate_formal_runtime("/formal/python", matrix)

    assert observed["command"] == [
        "/formal/python",
        "-m",
        "embed_optim.runtime",
        "--spec",
        str(spec),
    ]
    assert observed["kwargs"]["timeout"] == 60

    monkeypatch.setattr(
        evaluate_matrix.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="mismatch", stderr=""),
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        evaluate_matrix._validate_formal_runtime("/wrong/python", matrix)


def test_evaluation_selection_rejects_unknown_and_empty_run_sets(monkeypatch):
    config = SimpleNamespace(model_family="dense", run_id="adamw-known")
    monkeypatch.setattr(evaluate_matrix, "load_matrix", lambda path: [config])

    with pytest.raises(ValueError, match="adamw-typo"):
        evaluate_matrix._selected_configs(
            SimpleNamespace(
                matrix="matrix.yaml",
                families=["dense"],
                run_ids=["adamw-known", "adamw-typo"],
            )
        )
    with pytest.raises(ValueError, match="contains no training runs"):
        evaluate_matrix._selected_configs(
            SimpleNamespace(matrix="matrix.yaml", families=["late"], run_ids=[])
        )


def test_late_command_keeps_checkpoint_scoped_to_one_worker(tmp_path):
    args = Namespace(tasks=["SciFact", "FiQA2018"])
    model = tmp_path / "run" / "checkpoint-782"
    worker = tmp_path / "scripts" / "eval" / "late_interaction.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# worker\n")
    command = evaluate_matrix._late_command(
        tmp_path,
        model,
        args,
        tmp_path / "results",
        29620,
        num_processes=4,
        worker_python="/system/python",
    )
    assert command[:3] == ["/system/python", "-m", "accelerate.commands.launch"]
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


def test_evaluation_python_is_explicit_and_resolved(tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text("")

    assert evaluate_matrix._worker_python(str(interpreter)) == str(interpreter.resolve())


def test_worker_runtime_must_match_every_training_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "outputs" / "dense" / "run" / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "completed.json").write_text(
        '{"versions":{"torch":"2","transformers":"5","sentence-transformers":"5",'
        '"pylate":"1","late-interaction-kernels":"1"}}'
    )
    monkeypatch.setattr(
        evaluate_matrix,
        "_runtime_versions",
        lambda python: {
            "mteb": "2",
            "torch": "2",
            "sentence-transformers": "5",
            "flash-attn": "2",
            "transformers": "5",
            "pylate": "1",
            "fast-plaid": "1",
            "late-interaction-kernels": "1",
        },
    )

    versions = evaluate_matrix._validate_worker_runtime("/system/python", {"dense": [checkpoint]})
    assert versions["flash-attn"] == "2"

    sources = evaluate_matrix._evaluation_source_manifest(Path.cwd())
    evaluate_matrix._record_runtime(tmp_path, "/system/python", versions, sources)
    evaluate_matrix._record_runtime(tmp_path, "/same-stack/python", versions, sources)
    assert json.loads((tmp_path / "evaluation_runtime.json").read_text())["versions"] == versions

    monkeypatch.setattr(
        evaluate_matrix,
        "_runtime_versions",
        lambda python: {
            **versions,
            "torch": "different",
        },
    )
    import pytest

    with pytest.raises(RuntimeError, match="differs from training"):
        evaluate_matrix._validate_worker_runtime("/system/python", {"dense": [checkpoint]})

    with pytest.raises(RuntimeError, match="changed across a resumed results directory"):
        evaluate_matrix._record_runtime(
            tmp_path,
            "/system/python",
            {**versions, "mteb": "different"},
            sources,
        )

    changed_sources = json.loads(json.dumps(sources))
    changed_sources["scripts/eval/late_interaction.py"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="changed across a resumed results directory"):
        evaluate_matrix._record_runtime(tmp_path, "/system/python", versions, changed_sources)


def test_worker_interpreter_must_import_the_recorded_evaluation_sources(monkeypatch):
    sources = evaluate_matrix._evaluation_source_manifest(Path.cwd())
    observed = evaluate_matrix._worker_package_source_manifest(sys.executable)
    expected = {label: sources[label] for label in evaluate_matrix.EVALUATION_SOURCE_MODULES}
    assert observed == expected
    evaluate_matrix._validate_worker_sources(sys.executable, sources)

    changed = json.loads(json.dumps(observed))
    changed["src/embed_optim/evaluation_utils.py"]["sha256"] = "0" * 64
    monkeypatch.setattr(evaluate_matrix, "_worker_package_source_manifest", lambda python: changed)
    with pytest.raises(RuntimeError, match="imports different package source files"):
        evaluate_matrix._validate_worker_sources(sys.executable, sources)


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
    model = SimpleNamespace(
        model_card_data=SimpleNamespace(model_name=None, base_model_revision=None)
    )
    dense_sequential = importlib.import_module("dense_sequential")
    dense_sequential.set_local_model_identity(
        model, "/study/outputs/dense/adamw-lr1e-6/checkpoint-782"
    )
    assert model.model_card_data.model_name == "adamw-lr1e-6/checkpoint-782"
    assert model.model_card_data.base_model_revision == "local"
    sys.modules.pop("dense_parallel", None)


def _late_evaluation_module(monkeypatch):
    eval_dir = Path(__file__).resolve().parents[1] / "scripts" / "eval"
    monkeypatch.syspath_prepend(str(eval_dir))
    return importlib.import_module("late_interaction")


def test_late_temporary_file_cleanup_is_idempotent(tmp_path, monkeypatch):
    late = _late_evaluation_module(monkeypatch)
    files = [tmp_path / "result.pkl", tmp_path / "result.pkl.ready"]
    for path in files:
        path.write_text("temporary")

    late.remove_temporary_files([*files, tmp_path / "already-missing"])
    late.remove_temporary_files(files)
    assert not any(path.exists() for path in files)


def test_late_corpus_embedding_release_clears_caller_reference(monkeypatch):
    late = _late_evaluation_module(monkeypatch)
    embeddings = [object(), object()]
    caller_reference = embeddings
    calls = []
    monkeypatch.setattr(late.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(late.torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(late.torch.cuda, "is_available", lambda: False)

    late.release_corpus_embeddings(embeddings)

    assert caller_reference == []
    assert calls == ["gc", "empty_cache"]


def test_late_adaptive_encode_releases_each_microbatch_and_preserves_order(monkeypatch):
    late = _late_evaluation_module(monkeypatch)
    calls = []
    cache_calls = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append((list(texts), kwargs))
            if len(texts) > 2:
                raise late.torch.OutOfMemoryError("injected batch OOM")
            return [late.torch.full((int(text) + 1, 3), int(text)) for text in texts]

    monkeypatch.setattr(late.torch.cuda, "empty_cache", lambda: cache_calls.append(True))
    embeddings, splits = late.encode_batch_to_fp16_numpy(
        FakeModel(), ["0", "1", "2", "3", "4"], prompt="p", is_query=False
    )

    assert [texts for texts, _ in calls] == [
        ["0", "1", "2", "3", "4"],
        ["0", "1"],
        ["2", "3", "4"],
        ["2"],
        ["3", "4"],
    ]
    assert splits == 2
    assert len(cache_calls) == 2
    assert all(isinstance(embedding, late.np.ndarray) for embedding in embeddings)
    assert all(embedding.dtype == late.np.float16 for embedding in embeddings)
    assert [embedding.shape for embedding in embeddings] == [(i + 1, 3) for i in range(5)]
    assert [float(embedding[0, 0]) for embedding in embeddings] == list(range(5))
    assert all(call["batch_size"] == len(texts) for texts, call in calls)
    assert all(call["prompt"] == "p" and not call["is_query"] for _, call in calls)


def test_late_adaptive_encode_reraises_single_text_oom(monkeypatch):
    late = _late_evaluation_module(monkeypatch)
    cache_calls = []

    class AlwaysOomModel:
        def encode(self, texts, **kwargs):
            raise late.torch.OutOfMemoryError("single text cannot fit")

    monkeypatch.setattr(late.torch.cuda, "empty_cache", lambda: cache_calls.append(True))
    with pytest.raises(late.torch.OutOfMemoryError, match="single text cannot fit"):
        late.encode_batch_to_fp16_numpy(AlwaysOomModel(), ["only"], prompt=None, is_query=True)
    assert cache_calls == [True]


def test_late_auto_index_cleanup_runs_when_retrieval_fails(tmp_path, monkeypatch):
    late = _late_evaluation_module(monkeypatch)
    model = object.__new__(late.AccelerateMultiVectorModel)
    model._index_autodelete = True
    model._index_dir = tmp_path / "mteb-index-failed"
    model._index_name = "index"
    model._index_dir.mkdir()
    (model._index_dir / "partial-index").write_bytes(b"partial")

    def fail(*args, **kwargs):
        raise RuntimeError("injected retrieval failure")

    monkeypatch.setattr(model, "_index_and_retrieve_impl", fail)
    with pytest.raises(RuntimeError, match="injected retrieval failure"):
        model._index_and_retrieve({}, [], [], 10)

    assert not (tmp_path / "mteb-index-failed").exists()
    assert model._index_dir is None
    assert model._index_name is None


def test_mteb_result_writes_are_atomic_and_preserve_previous_result_on_failure(
    tmp_path, monkeypatch
):
    import mteb.results.task_result as task_result_module

    class FakeTaskResult:
        def __init__(self, payload, fail=False):
            self.payload = payload
            self.fail = fail

        def to_disk(self, path):
            path.write_text(self.payload)
            if self.fail:
                raise RuntimeError("injected interrupted write")

    monkeypatch.setattr(task_result_module, "TaskResult", FakeTaskResult)
    configure_atomic_mteb_results()
    configure_atomic_mteb_results()
    result = tmp_path / "SciFactDecontaminated.json"
    result.write_text("old-complete-result")

    FakeTaskResult("new-complete-result").to_disk(result)
    assert result.read_text() == "new-complete-result"

    with pytest.raises(RuntimeError, match="injected interrupted write"):
        FakeTaskResult("partial", fail=True).to_disk(result)
    assert result.read_text() == "new-complete-result"
    assert not list(tmp_path.glob(".*.tmp.json"))


def test_mteb_run_settings_merge_is_locked_atomic_and_uses_current_schema(tmp_path):
    import mteb
    import mteb.cache.result_cache as result_cache_module

    configure_atomic_mteb_results()
    settings = tmp_path / "run_settings.jsonl"
    minor = int(mteb.__version__.split(".")[1])

    def write_task(index):
        scope = (
            {"splits": ["test"], "subsets": ["default"]}
            if minor >= 19
            else {"split": "test", "subset": "default"}
        )
        result_cache_module._write_and_merge_keyed_json(
            settings,
            [
                {
                    "task": f"Task{index}",
                    **scope,
                    "version": {"mteb": "test"},
                    "encode_kwargs": {},
                }
            ],
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_task, range(32)))

    rows = [json.loads(line) for line in settings.read_text().splitlines()]
    assert {row["task"] for row in rows} == {f"Task{index}" for index in range(32)}
    if minor >= 19:
        assert all(row["splits"] == ["test"] for row in rows)
        assert all(row["subsets"] == ["default"] for row in rows)
    else:
        assert all(row["split"] == "test" for row in rows)
        assert all(row["subset"] == "default" for row in rows)
    assert not list(tmp_path.glob(".*.tmp.jsonl"))


def test_mteb_model_metadata_is_atomic_and_matches_result_path(tmp_path, monkeypatch):
    import mteb
    from mteb.cache import ResultCache
    from mteb.models.model_meta import ModelMeta, ScoringFunction

    from embed_optim import evaluation_utils

    class FakeTaskResult:
        task_name = "SciFactDecontaminated"
        scores = {"test": [{"hf_subset": "default"}]}

        def to_disk(self, path):
            path.write_text("{}")

    meta = ModelMeta(
        loader=None,
        name="run/checkpoint-1",
        revision="local",
        release_date=None,
        languages=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=8192,
        embed_dim=768,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["Sentence Transformers", "PyTorch"],
        similarity_fn_name=ScoringFunction.COSINE,
        use_instructions=False,
        training_datasets=None,
    )
    configure_atomic_mteb_results()
    cache = ResultCache(tmp_path)
    cache.save_to_cache(FakeTaskResult(), meta)

    result_dir = tmp_path / "results" / "run__checkpoint-1" / "local"
    meta_path = result_dir / "model_meta.json"
    assert (result_dir / "SciFactDecontaminated.json").is_file()
    assert json.loads(meta_path.read_text())["name"] == "run/checkpoint-1"
    settings = [
        json.loads(line) for line in (result_dir / "run_settings.jsonl").read_text().splitlines()
    ]
    if int(mteb.__version__.split(".")[1]) >= 19:
        assert settings[0]["splits"] == ["test"]
        assert settings[0]["subsets"] == ["default"]
    else:
        assert settings[0]["split"] == "test"
        assert settings[0]["subset"] == "default"

    previous_meta = meta_path.read_text()

    def fail_json_dump(payload, handle, **kwargs):
        handle.write("{")
        raise RuntimeError("injected interrupted metadata write")

    monkeypatch.setattr(evaluation_utils.json, "dump", fail_json_dump)
    with pytest.raises(RuntimeError, match="injected interrupted metadata write"):
        cache.save_to_cache(FakeTaskResult(), meta)
    assert meta_path.read_text() == previous_meta
    assert not list(result_dir.glob(".model_meta.*.tmp.json"))


def test_non_main_mteb_cache_is_read_only():
    calls = []
    cache = SimpleNamespace(save_to_cache=lambda *args, **kwargs: calls.append((args, kwargs)))

    disable_mteb_cache_writes(cache)
    cache.save_to_cache("result", "model")

    assert calls == []
