"""Evaluate all five checkpoints on the pinned decontaminated BEIR suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig, load_matrix
from .decontamination import DECONTAMINATED_TASK_NAMES, decontaminated_corpus_size

EVALUATION_PACKAGES = (
    "mteb",
    "torch",
    "sentence-transformers",
    "flash-attn",
    "transformers",
    "pylate",
    "fast-plaid",
    "late-interaction-kernels",
)
TRAINING_RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "sentence-transformers",
    "pylate",
    "late-interaction-kernels",
)
EVALUATION_SOURCE_MODULES = {
    "src/embed_optim/evaluate_matrix.py": "embed_optim.evaluate_matrix",
    "src/embed_optim/evaluation_utils.py": "embed_optim.evaluation_utils",
    "src/embed_optim/decontamination.py": "embed_optim.decontamination",
    "src/embed_optim/pylate_compat.py": "embed_optim.pylate_compat",
    "src/embed_optim/aggregate.py": "embed_optim.aggregate",
}


@dataclass
class EvaluationProcess:
    family: str
    process: subprocess.Popen
    handle: object
    model: Path | None = None


def _worker_python(executable: str | None = None) -> str:
    """Resolve the one interpreter used by dense and distributed late workers."""

    requested = executable or sys.executable
    candidate = shutil.which(requested)
    if candidate is None:
        path = Path(requested).expanduser()
        candidate = str(path.resolve()) if path.is_file() else None
    if candidate is None:
        raise FileNotFoundError(f"Cannot locate evaluation interpreter {requested!r}")
    return str(Path(candidate).resolve())


def _runtime_versions(python: str) -> dict[str, str]:
    """Read the packages that affect evaluation from an isolated interpreter."""

    script = (
        "import importlib.metadata as m,json; import flash_attn; "
        f"print(json.dumps({{p:m.version(p) for p in {EVALUATION_PACKAGES!r}}},sort_keys=True))"
    )
    result = subprocess.run(
        [python, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        versions = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not parse evaluation runtime from {python}") from error
    if set(versions) != set(EVALUATION_PACKAGES) or any(not value for value in versions.values()):
        raise RuntimeError(f"Incomplete evaluation runtime from {python}: {versions}")
    return versions


def _worker_package_source_manifest(python: str) -> dict[str, dict]:
    """Fingerprint package modules imported by the actual worker interpreter."""

    script = f"""
import hashlib
import importlib
import json
from pathlib import Path

modules = {EVALUATION_SOURCE_MODULES!r}
output = {{}}
for label, module_name in modules.items():
    path = Path(importlib.import_module(module_name).__file__).resolve()
    content = path.read_bytes()
    output[label] = {{"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}}
print(json.dumps(output, sort_keys=True))
"""
    result = subprocess.run(
        [python, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not parse evaluation sources from {python}") from error


def _validate_worker_sources(python: str, source_files: dict[str, dict]) -> None:
    expected = {label: source_files[label] for label in EVALUATION_SOURCE_MODULES}
    observed = _worker_package_source_manifest(python)
    if observed != expected:
        raise RuntimeError(
            f"Evaluation worker {python} imports different package source files: "
            f"expected {expected}, got {observed}"
        )


def _validate_worker_runtime(python: str, models: dict[str, list[Path]]) -> dict[str, str]:
    """Fail before GPU work if training and evaluation libraries are not identical."""

    versions = _runtime_versions(python)
    training_versions: set[tuple[str, ...]] = set()
    for model in (path for paths in models.values() for path in paths):
        completed_path = model.parent / "completed.json"
        try:
            recorded = json.loads(completed_path.read_text())["versions"]
            training_versions.add(tuple(recorded[package] for package in TRAINING_RUNTIME_PACKAGES))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Missing training runtime in {completed_path}") from error
    if len(training_versions) != 1:
        raise RuntimeError(f"Training runs used inconsistent runtimes: {training_versions}")
    expected = next(iter(training_versions))
    actual = tuple(versions[package] for package in TRAINING_RUNTIME_PACKAGES)
    if actual != expected:
        raise RuntimeError(
            "Evaluation runtime differs from training for "
            f"{TRAINING_RUNTIME_PACKAGES}: expected {expected}, got {actual}"
        )
    print(f"evaluation interpreter: {python} | versions={json.dumps(versions, sort_keys=True)}")
    return versions


def _record_runtime(
    results: Path,
    python: str,
    versions: dict[str, str],
    source_files: dict[str, dict],
) -> None:
    """Persist one immutable runtime identity for all resumable evaluation jobs."""

    path = results / "evaluation_runtime.json"
    payload = {
        "schema_version": 2,
        "python": python,
        "versions": versions,
        "source_files": source_files,
    }
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid evaluation runtime manifest: {path}") from error
        if (
            existing.get("schema_version") != 2
            or existing.get("versions") != versions
            or existing.get("source_files") != source_files
        ):
            raise RuntimeError(
                f"Evaluation runtime changed across a resumed results directory: {path}"
            )
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _evaluation_script(repo: Path, name: str, prefix: Path | None = None) -> Path:
    """Locate an evaluation worker in a source checkout or an installed wheel."""

    prefix = Path(sys.prefix) if prefix is None else prefix
    candidates = (
        repo / "scripts" / "eval" / name,
        prefix / "share" / "embedding-optimizer-study" / "scripts" / "eval" / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate evaluation worker {name!r}; checked {candidates}")


def _evaluation_source_manifest(repo: Path, prefix: Path | None = None) -> dict[str, dict]:
    """Fingerprint every in-repository source file that can affect reported scores."""

    package = Path(__file__).resolve().parent
    paths = {
        "src/embed_optim/evaluate_matrix.py": Path(__file__).resolve(),
        "src/embed_optim/evaluation_utils.py": package / "evaluation_utils.py",
        "src/embed_optim/decontamination.py": package / "decontamination.py",
        "src/embed_optim/pylate_compat.py": package / "pylate_compat.py",
        "src/embed_optim/aggregate.py": package / "aggregate.py",
        **{
            f"scripts/eval/{name}": _evaluation_script(repo, name, prefix)
            for name in ("dense_parallel.py", "dense_sequential.py", "late_interaction.py")
        },
    }
    manifest = {}
    for label, path in paths.items():
        content = path.read_bytes()
        manifest[label] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    return manifest


def checkpoint_paths(config: RunConfig, stages: list[int] | None = None) -> list[Path]:
    schedule_path = config.output_dir / "checkpoint_schedule.json"
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint schedule: {schedule_path}")
    steps = sorted(json.loads(schedule_path.read_text())["steps"])
    if len(steps) != 5:
        raise RuntimeError(f"Expected five checkpoint steps in {schedule_path}, got {steps}")
    selected = range(1, 6) if not stages else stages
    paths = [config.output_dir / f"checkpoint-{steps[stage - 1]}" for stage in selected]
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    return [path.resolve() for path in paths]


def _selected_models(args: argparse.Namespace) -> dict[str, list[Path]]:
    configs = [
        config
        for config in load_matrix(args.matrix)
        if config.model_family in args.families
        and (not args.run_ids or config.run_id in args.run_ids)
    ]
    return {
        family: [
            checkpoint
            for config in configs
            if config.model_family == family
            for checkpoint in checkpoint_paths(config, args.stages)
        ]
        for family in args.families
    }


def _late_command(
    repo: Path,
    model: Path,
    args: argparse.Namespace,
    results: Path,
    port: int,
    num_processes: int,
    worker_python: str,
) -> list[str]:
    ordered_tasks = sorted(args.tasks, key=decontaminated_corpus_size, reverse=True)
    return [
        worker_python,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(num_processes),
        "--main_process_port",
        str(port),
        str(_evaluation_script(repo, "late_interaction.py")),
        "--models",
        str(model),
        "--tasks",
        *ordered_tasks,
        "--results_folder",
        str(results / "late"),
        "--fa2",
        "--decontaminated",
    ]


def _launch_late(
    repo: Path,
    model: Path,
    args: argparse.Namespace,
    results: Path,
    log_dir: Path,
    pool: str,
    gpus: str,
    port: int,
    worker_python: str,
) -> EvaluationProcess:
    handle = (log_dir / f"late-evaluation-{pool}.log").open("a")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus}
    process = subprocess.Popen(
        _late_command(
            repo,
            model,
            args,
            results,
            port,
            len([gpu for gpu in gpus.split(",") if gpu.strip()]),
            worker_python,
        ),
        cwd=repo,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    print(f"late pool-{pool} started {model}", flush=True)
    return EvaluationProcess("late", process, handle, model)


def run_evaluation(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    models = _selected_models(args)
    worker_python = _worker_python(getattr(args, "worker_python", None))
    results = Path(args.results_root).resolve()
    log_dir = Path(args.log_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    versions = _validate_worker_runtime(worker_python, models)
    source_files = _evaluation_source_manifest(repo)
    _validate_worker_sources(worker_python, source_files)
    _record_runtime(results, worker_python, versions, source_files)
    dense_job: EvaluationProcess | None = None

    if dense_models := models.get("dense"):
        ordered_tasks = sorted(args.tasks, key=decontaminated_corpus_size, reverse=True)
        command = [
            worker_python,
            str(_evaluation_script(repo, "dense_parallel.py")),
            "--gpus",
            args.gpus_a,
            "--results_folder",
            str(results / "dense"),
            "--models",
            *(str(path) for path in dense_models),
            "--tasks",
            *ordered_tasks,
            "--log_dir",
            str(log_dir / "dense-tasks"),
            "--bf16",
            "--fa2",
            "--local",
            "--decontaminated",
        ]
        handle = (log_dir / "dense-evaluation.log").open("a")
        dense_job = EvaluationProcess(
            "dense",
            subprocess.Popen(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT),
            handle,
        )

    late_queue = list(models.get("late", []))
    late_jobs: dict[str, EvaluationProcess] = {}
    pools = {
        "a": (args.gpus_a, args.late_port_a),
        "b": (args.gpus_b, args.late_port),
    }
    failures = 0
    while dense_job is not None or late_queue or late_jobs:
        if dense_job is not None and dense_job.process.poll() is not None:
            return_code = dense_job.process.returncode
            dense_job.handle.close()
            print(f"dense evaluation exited {return_code}", flush=True)
            failures += return_code != 0
            dense_job = None

        for pool, job in list(late_jobs.items()):
            if job.process.poll() is None:
                continue
            return_code = job.process.returncode
            job.handle.close()
            print(f"late pool-{pool} {job.model} exited {return_code}", flush=True)
            failures += return_code != 0
            del late_jobs[pool]

        for pool, (gpus, port) in pools.items():
            if not late_queue or pool in late_jobs or (pool == "a" and dense_job is not None):
                continue
            late_jobs[pool] = _launch_late(
                repo,
                late_queue.pop(0),
                args,
                results,
                log_dir,
                pool,
                gpus,
                port,
                worker_python,
            )
        if dense_job is not None or late_queue or late_jobs:
            time.sleep(1)
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument(
        "--families", nargs="+", choices=["dense", "late"], default=["dense", "late"]
    )
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--stages", nargs="*", type=int, choices=range(1, 6))
    parser.add_argument("--tasks", nargs="+", default=list(DECONTAMINATED_TASK_NAMES))
    parser.add_argument("--gpus-a", default="0,1,2,3")
    parser.add_argument("--gpus-b", default="4,5,6,7")
    parser.add_argument("--late-port-a", type=int, default=29610)
    parser.add_argument("--late-port", type=int, default=29620)
    parser.add_argument("--results-root", default="results/decontaminated-beir")
    parser.add_argument("--log-dir", default="logs/evaluation")
    parser.add_argument(
        "--worker-python",
        default=None,
        help="Python executable for every evaluator (default: this command's interpreter)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    failures = run_evaluation(parse_args(argv))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
