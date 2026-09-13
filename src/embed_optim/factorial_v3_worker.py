"""Fresh-run lifecycle inside a separately admitted four-rank factorial worker.

This component does not launch a process, initialize a process group, acquire a
GPU, authorize execution, resume a run, or publish. The outer source/runtime,
primary-completion, resource and real-GPU admission handoff is still required.
It connects the unchanged factory, actual Trainer and deep whole-run reader.
No component success below is itself an authorization or scientific finding.
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import factorial_v3_factory as factory
from . import factorial_v3_run_contract as contract
from .factorial_v3_bound_trainer import RunBoundFactorialTrainer, collective_phase
from .factorial_v3_inputs import ORDER_SEEDS, STATES, Locations
from .primary_contract import digest, file_identity, read_json, require_same, verify_file

SCOPE = "dense-v3-factorial-fresh-worker-component-v1"
FACTORY_SHA = "bddb443318481d3d7dbddd7b5eb5f5a0f651939537dd281486ad8c305721209b"
STARTED = "worker-started.json"
COMPLETE = "worker-complete.json"


def _path(value):
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError("Worker paths must be explicit absolute paths without symlinks")
    return path


def _binding(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"bytes", "sha256"}
        or type(value["bytes"]) is not int
        or value["bytes"] <= 0
    ):
        raise ValueError("Require an externally supplied exact file binding")
    contract._require_hex(value["sha256"], 64)
    return dict(value)


def require_sources(locations, source_root, expected_worker_source):
    """Authenticate all 69 unchanged parent files and this separately bound module."""
    source_root = _path(source_root)
    actual = Path(__file__).resolve()
    if actual != source_root / "src/embed_optim/factorial_v3_worker.py":
        raise ValueError("Worker imported from outside its supplied source assembly")
    verify_file(actual, _binding(expected_worker_source))
    sources = contract.source_identity(locations.repository, source_root)
    factory_path = source_root / "src/embed_optim/factorial_v3_factory.py"
    factory_source = file_identity(factory_path)
    if Path(factory.__file__).resolve() != factory_path or factory_source["sha256"] != FACTORY_SHA:
        raise ValueError("Worker changed the accepted factory source")
    sources["src/embed_optim/factorial_v3_factory.py"] = factory_source
    sources["src/embed_optim/factorial_v3_worker.py"] = dict(expected_worker_source)
    return sources


def request(
    locations,
    calibration_directories,
    state,
    operator,
    seed,
    source_root,
    output_root,
    record_root,
    *,
    project,
    entity,
    expected_worker_source,
):
    """Structural request only: neither fabricated rates nor permission are accepted."""
    if type(locations) is not Locations:
        raise ValueError("Require the explicit genuine-input location roles")
    roles = {name: str(_path(value)) for name, value in dataclasses.asdict(locations).items()}
    if state not in STATES or operator not in ("adamw", "muon"):
        raise ValueError("Worker cell is outside the fixed crossed design")
    if type(seed) is not int or seed not in ORDER_SEEDS:
        raise ValueError("Worker order seed is outside the fixed crossed design")
    if not isinstance(calibration_directories, dict) or set(calibration_directories) != set(STATES):
        raise ValueError("Worker requires both externally bound calibration chains")
    calibrated = {}
    for label in STATES:
        item = calibration_directories[label]
        if not isinstance(item, dict) or set(item) != {"path", "calibration_binding"}:
            raise ValueError("A calibration directory is not an exact external binding")
        calibrated[label] = {
            "path": str(_path(item["path"])),
            "calibration_binding": _binding(item["calibration_binding"]),
        }
    if len({v["path"] for v in calibrated.values()}) != 2:
        raise ValueError("The two fixed sources require distinct calibration directories")
    if any(not isinstance(v, str) or not v.strip() for v in (project, entity)):
        raise ValueError("Worker requires explicit logging identities, never credentials")
    source_root, output_root, record_root = map(_path, (source_root, output_root, record_root))
    run_id = f"factorial-v3-{state}-{operator}-seed{seed}"
    run_root = output_root / "dense" / run_id
    protected = [
        source_root,
        output_root,
        locations.primary_source,
        locations.data_store,
        locations.evidence,
        *(Path(v["path"]) for v in calibrated.values()),
        locations.experiment / "outputs",
    ]
    if any(record_root.is_relative_to(p) or p.is_relative_to(record_root) for p in protected):
        raise ValueError("Worker records must be separate from source, model and data trees")
    return {
        "scope": SCOPE,
        "locations": roles,
        "calibrations": calibrated,
        "state": state,
        "operator": operator,
        "seed": seed,
        "run_id": run_id,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "run_root": str(run_root),
        "record_root": str(record_root),
        "project": project,
        "entity": entity,
        "worker_source": _binding(expected_worker_source),
        "resume_from_checkpoint": None,
        "source_release_or_resource_gate_waived": False,
        "execution_authorized": False,
        "scientific_admission": False,
    }


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _consensus(value):
    import torch.distributed as dist

    observed = [None] * 4
    dist.all_gather_object(observed, digest(value))
    if len(set(observed)) != 1:
        raise ValueError("Ranks disagree on the exact worker request or created run")


def _require_factory_record(creation, component, declared, sources):
    """Apply the same semantic checks at construction and fresh saved-record reading."""
    identity = contract.require_identity(component["bound_factorial_run"])
    require_same(
        (identity["state"], identity["operator"], identity["seed"], identity["run_id"]),
        tuple(declared[k] for k in ("state", "operator", "seed", "run_id")),
    )
    require_same(
        identity["sources"],
        {
            k: v
            for k, v in sources.items()
            if not k.endswith(("factorial_v3_factory.py", "factorial_v3_worker.py"))
        },
    )
    config, arguments = factory.recipe(
        identity,
        Path(declared["locations"]["data_store"]) / factory.BRANCH,
        declared["output_root"],
        project=declared["project"],
        entity=declared["entity"],
    )
    require_same(creation["recipe"], config.as_dict())
    require_same(creation["requested_arguments"], arguments)
    require_same(creation["factory_source"], sources["src/embed_optim/factorial_v3_factory.py"])
    if set(creation) != {
        "scope",
        "scientific_admission",
        "execution_authorized",
        "run_identity_sha256",
        "factory_source",
        "recipe",
        "requested_arguments",
        "loaded_model",
        "training_executed",
        "source_release_or_resource_gate_waived",
    }:
        raise ValueError("Incomplete or extended factory creation record")
    if (
        creation["scope"] != factory.SCOPE
        or creation["run_identity_sha256"] != digest(identity)
        or any(
            creation[k] is not False
            for k in (
                "scientific_admission",
                "execution_authorized",
                "training_executed",
                "source_release_or_resource_gate_waived",
            )
        )
    ):
        raise ValueError("Created Trainer does not match its genuine factory record")
    loaded = creation["loaded_model"]
    if set(loaded) != {
        "semantic_config_sha256",
        "pooling",
        "transformer",
        "prompts",
        "tokenizer_backend_sha256",
        "embedding_dimensions",
        "maximum_context",
        "backend",
        "gradient_checkpointing",
        "training_mode",
        "diagnostic_cpu",
        "forward_or_backward_executed",
    }:
        raise ValueError("Incomplete or extended loaded-model record")
    for key, value in {
        "embedding_dimensions": 768,
        "maximum_context": 8192,
        "backend": "flash_attention_2",
        "gradient_checkpointing": True,
        "training_mode": True,
        "diagnostic_cpu": False,
        "forward_or_backward_executed": False,
    }.items():
        require_same(loaded[key], value)
    for key in ("semantic_config_sha256", "tokenizer_backend_sha256"):
        contract._require_hex(loaded[key], 64)
    contract.require_component(component, identity)
    return identity


def _require_created(trainer, creation, declared, sources):
    if type(trainer) is not RunBoundFactorialTrainer:
        raise ValueError("Worker requires the actual source-bound factorial Trainer")
    component = trainer._require_identity()
    identity = _require_factory_record(creation, component, declared, sources)
    require_same(identity, trainer._bound_run)
    if (
        Path(trainer.args.output_dir) != Path(declared["run_root"])
        or trainer.state.global_step != 0
        or trainer.state.epoch not in (None, 0, 0.0)
        or trainer.optimizer is not None
        or trainer.lr_scheduler is not None
        or trainer._resume_binding is not None
        or trainer._resume_step is not None
        or any(p.grad is not None for p in trainer.model.parameters())
    ):
        raise ValueError("Fresh worker inherited training, optimizer, scheduler or gradient state")
    return component


def _rank_result(rank, result, trainer):
    contract.require_final_state(dataclasses.asdict(trainer.state))
    if type(result.global_step) is not int or result.global_step != 391:
        raise ValueError("Trainer returned before the fixed complete horizon")
    values = [result.training_loss, *result.metrics.values()]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Actual Trainer returned a nonfinite or nonnumeric metric")
    return {
        "rank": rank,
        "global_step": result.global_step,
        "training_loss": result.training_loss,
        "metrics": dict(result.metrics),
    }


def _require_rank_results(rows):
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Require actual completion observations from all four ranks")
    for rank, row in enumerate(rows):
        if (
            set(row) != {"rank", "global_step", "training_loss", "metrics"}
            or type(row["rank"]) is not int
            or row["rank"] != rank
            or type(row["global_step"]) is not int
            or row["global_step"] != 391
            or not isinstance(row["metrics"], dict)
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (row["training_loss"], *row["metrics"].values())
            )
        ):
            raise ValueError("Incomplete, duplicated or invalid rank-level Trainer completion")


def _read_native(locations, declared, binding):
    return contract.read_complete_run(
        locations,
        declared["calibrations"],
        state=declared["state"],
        operator=declared["operator"],
        seed=declared["seed"],
        source_root=declared["source_root"],
        root=declared["run_root"],
        binding=binding,
    )


def _require_native(native, identity):
    require_same(native["run_identity"], identity)
    if (
        native.get("scope") != contract.SCOPE
        or native.get("input_calibration_and_run_verified") is not True
        or native.get("scientific_admission") is not False
        or native.get("execution_authorized") is not False
    ):
        raise ValueError("Native reader did not verify the complete input/calibration/run chain")
    artifacts = native["artifacts"]
    if (
        artifacts.get("scope") != contract.SCOPE
        or artifacts.get("whole_run_artifacts_verified") is not True
        or artifacts.get("scientific_admission") is not False
        or artifacts.get("run_identity_sha256") != digest(identity)
        or artifacts.get("optimizer_steps") != 391
        or artifacts.get("dataset_rows") != 50000
        or [r["step"] for r in artifacts.get("checkpoints", [])] != list(contract.STEPS)
    ):
        raise ValueError("Native reader did not verify all five full-horizon stages")


def run_branch(
    locations,
    calibration_directories,
    state,
    operator,
    seed,
    source_root,
    output_root,
    record_root,
    *,
    project,
    entity,
    expected_worker_source,
):
    """Call once on every rank of a separately admitted, exclusively owned GPU worker.

    No launch or authority is inferred here. An outer launcher must handle killed
    ranks/broken collectives. Any failure preserves this attempt and all payloads;
    there is no automatic retry, partial-run acceptance or implicit latest resume.
    """
    import torch.distributed as dist

    factory.require_gpu_context()
    declared = collective_phase(
        lambda: request(
            locations,
            calibration_directories,
            state,
            operator,
            seed,
            source_root,
            output_root,
            record_root,
            project=project,
            entity=entity,
            expected_worker_source=expected_worker_source,
        )
    )
    sources = collective_phase(
        lambda: require_sources(locations, source_root, expected_worker_source)
    )

    def preflight():
        for path in (Path(declared["record_root"]), Path(declared["run_root"])):
            if path.exists() or path.is_symlink():
                raise ValueError("Fresh worker refuses an existing attempt or run directory")

    collective_phase(preflight)
    _consensus({"request": declared, "sources": sources})
    records = Path(declared["record_root"])
    rank = dist.get_rank()

    def start():
        if rank == 0:
            records.mkdir(parents=True, exist_ok=False)
            _write_new(
                records / STARTED,
                {
                    "scope": SCOPE,
                    "status": "started",
                    "started_at_utc": _utc(),
                    "request": declared,
                    "sources": sources,
                    "execution_authority_checked_by_this_component": False,
                    "scientific_admission": False,
                },
            )

    collective_phase(start)
    phase = "factory"
    try:
        trainer, creation = factory.prepare_trainer(
            locations,
            declared["calibrations"],
            state,
            operator,
            seed,
            source_root,
            output_root,
            project=project,
            entity=entity,
        )
        component = collective_phase(lambda: _require_created(trainer, creation, declared, sources))
        _consensus({"creation": creation, "component": component})
        collective_phase(
            lambda: (
                _write_new(
                    records / "factory.json",
                    {
                        "creation": creation,
                        "component_identity": component,
                    },
                )
                if rank == 0
                else None
            )
        )

        phase = "train"
        # This is the actual inherited fixed-horizon training call. No callback
        # stops it early and no checkpoint is implicitly selected for resumption.
        result = trainer.train(resume_from_checkpoint=None)
        phase = "full_horizon_check"
        local = collective_phase(lambda: _rank_result(rank, result, trainer))
        results = [None] * 4
        dist.all_gather_object(results, local)
        collective_phase(lambda: _require_rank_results(results))
        phase = "save_run_completion"
        binding = trainer.save_run_completion()
        phase = "deep_native_readback"

        def finish():
            if rank != 0:
                return None
            native = _read_native(locations, declared, binding)
            require_same(sources, require_sources(locations, source_root, expected_worker_source))
            _require_native(native, component["bound_factorial_run"])
            completed = {
                "scope": SCOPE,
                "status": "training_and_native_readback_complete",
                "completed_at_utc": _utc(),
                "request": declared,
                "started": file_identity(records / STARTED),
                "factory_record": file_identity(records / "factory.json"),
                "native_run_completion": _binding(binding),
                "native_readback": native,
                "rank_results": results,
                "training_executed": True,
                "process_exit_observed": False,
                "scientific_admission": False,
                "source_release_or_resource_gate_waived": False,
                "independent_fresh_process_readback": False,
            }
            _write_new(records / COMPLETE, completed)
            return file_identity(records / COMPLETE)

        completed_binding = collective_phase(finish)
        shared = [completed_binding]
        dist.broadcast_object_list(shared, src=0)
        return shared[0]
    except BaseException as exc:
        # Never serialize exception text, environment or credentials. A failed
        # record is not permission to retry or to release another rank's leases.
        failure = records / f"rank-{rank}-failed.json"
        if not failure.exists():
            try:
                _write_new(
                    failure,
                    {
                        "scope": SCOPE,
                        "status": "failed",
                        "rank": rank,
                        "phase": phase,
                        "observed_at_utc": _utc(),
                        "exception_type": type(exc).__name__,
                        "scientific_admission": False,
                    },
                )
            except OSError:
                pass
        raise


def read_execution(locations, declared, binding):
    """Fresh CPU-capable reader of worker provenance and the full native run.

    This recomputes original complete-run acceptance, not training. Authenticating
    an execution artifact does not prove OS exit, resource authority or release.
    No result from this reader can replace real GPU admission or full outcomes.
    """
    expected = request(
        locations,
        declared["calibrations"],
        declared["state"],
        declared["operator"],
        declared["seed"],
        declared["source_root"],
        declared["output_root"],
        declared["record_root"],
        project=declared["project"],
        entity=declared["entity"],
        expected_worker_source=declared["worker_source"],
    )
    require_same(declared, expected)
    sources = require_sources(locations, declared["source_root"], declared["worker_source"])
    root = Path(declared["record_root"])
    verify_file(root / COMPLETE, _binding(binding))
    complete = read_json(root / COMPLETE)
    if set(complete) != {
        "scope",
        "status",
        "completed_at_utc",
        "request",
        "started",
        "factory_record",
        "native_run_completion",
        "native_readback",
        "rank_results",
        "training_executed",
        "process_exit_observed",
        "scientific_admission",
        "source_release_or_resource_gate_waived",
        "independent_fresh_process_readback",
    }:
        raise ValueError("Incomplete or extended worker completion record")
    require_same(complete["request"], declared)
    if (
        complete["scope"] != SCOPE
        or complete["status"] != "training_and_native_readback_complete"
        or complete["training_executed"] is not True
        or any(
            complete[k] is not False
            for k in (
                "process_exit_observed",
                "scientific_admission",
                "source_release_or_resource_gate_waived",
                "independent_fresh_process_readback",
            )
        )
    ):
        raise ValueError("Not a complete bounded worker execution record")
    entries = list(root.iterdir())
    if {p.name for p in entries} != {STARTED, "factory.json", COMPLETE} or any(
        not p.is_file() or p.is_symlink() for p in entries
    ):
        raise ValueError("Worker evidence contains an incomplete, extra or failed attempt")
    verify_file(root / STARTED, complete["started"])
    verify_file(root / "factory.json", complete["factory_record"])
    started = read_json(root / STARTED)
    if set(started) != {
        "scope",
        "status",
        "started_at_utc",
        "request",
        "sources",
        "execution_authority_checked_by_this_component",
        "scientific_admission",
    }:
        raise ValueError("Incomplete or extended worker start record")
    require_same(started["request"], declared)
    require_same(started["sources"], sources)
    if (
        started["scope"] != SCOPE
        or started["status"] != "started"
        or started["execution_authority_checked_by_this_component"] is not False
        or started["scientific_admission"] is not False
    ):
        raise ValueError("Worker start record changed its scope or authority boundary")
    _require_rank_results(complete["rank_results"])
    start_time = datetime.fromisoformat(started["started_at_utc"])
    end_time = datetime.fromisoformat(complete["completed_at_utc"])
    if start_time.tzinfo is None or end_time.tzinfo is None or end_time < start_time:
        raise ValueError("Worker completion has invalid or reversed timestamps")
    native = _read_native(locations, declared, complete["native_run_completion"])
    require_same(native, complete["native_readback"])
    created = read_json(root / "factory.json")
    if set(created) != {"creation", "component_identity"}:
        raise ValueError("Incomplete or extended worker factory record")
    identity = native["run_identity"]
    _require_native(native, identity)
    expected_identity = _require_factory_record(
        created["creation"], created["component_identity"], declared, sources
    )
    require_same(identity, expected_identity)
    verify_file(root / COMPLETE, binding)
    return {
        "scope": SCOPE,
        "worker_and_native_artifacts_verified": True,
        "worker_completion": dict(binding),
        "native_readback": native,
        "scientific_admission": False,
        "execution_authorized": False,
        "process_exit_observed": False,
        "training_reexecuted": False,
    }
