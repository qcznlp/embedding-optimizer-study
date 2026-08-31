"""Read-only W&B provenance audit for the frozen 34-run Dense study.

The canonical discovery publisher remains responsible for the immutable 12-run
discovery view.  This module performs no W&B mutation: it verifies the original
Trainer-created source runs that back discovery, routing controls, confirmation,
and the short shared-start branch, then writes a secret-free local receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .config import RunConfig, load_matrix, source_wandb_run_id
from .scope import load_scope_amendment
from .wandb_sync import canonical_history, history_sha256

EXPECTED_COUNTS = {
    "discovery": 12,
    "hybrid": 4,
    "confirmatory": 9,
    "short-branch": 9,
}
EXPECTED_TOTAL_RUNS = sum(EXPECTED_COUNTS.values())
EXPECTED_QUEUE_COUNTS = {
    "confirmatory_runs": 9,
    "short_branch_runs": 9,
    "total_runs": 18,
}
EXPECTED_GIT_REMOTE = "https://github.com/qcznlp/embedding-optimizer-study.git"
REMOTE_HISTORY_KEYS = {
    "train/epoch": "train/epoch",
    "train/grad_norm": "train/grad_norm",
    "train/learning_rate": "train/learning_rate",
    "train/loss": "train/loss",
}
PHASE_ORDER = {phase: index for index, phase in enumerate(EXPECTED_COUNTS)}
SECRET_MARKERS = (
    "wandb" + "_v1_",
    '"api_key"',
    '"apikey"',
    '"authorization"',
    '"credential"',
    '"password"',
)


@dataclass(frozen=True)
class SourceRunSpec:
    phase: str
    config: RunConfig
    matrix_path: Path
    source_run_id: str

    @property
    def label(self) -> str:
        return (
            f"{self.phase}/{self.config.model_family}/{self.config.run_id}/seed{self.config.seed}"
        )


@dataclass(frozen=True)
class FrozenDenseStudy:
    repository: Path
    scope_path: Path
    training_plan_path: Path
    matrix_paths: tuple[Path, ...]
    entity: str
    project: str
    runs: tuple[SourceRunSpec, ...]


@dataclass(frozen=True)
class NormalizedRemoteHistory:
    history: tuple[dict[str, int | float], ...]
    raw_rows: int
    duplicate_rows: int
    overwritten_values: int


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _normalized_json(payload: Any) -> Any:
    return json.loads(_json_bytes(payload))


def _under_repository(repository: Path, path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve() if candidate.is_absolute() else (repository / candidate).resolve()
    )
    try:
        resolved.relative_to(repository)
    except ValueError:
        raise ValueError(f"Path is outside the audited repository: {resolved}") from None
    return resolved


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _binding_map(
    repository: Path,
    bindings: Any,
    *,
    label: str,
) -> dict[Path, str]:
    if not isinstance(bindings, list):
        raise ValueError(f"{label} source bindings must be a list")
    result: dict[Path, str] = {}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError(f"{label} contains a non-object source binding")
        path = _under_repository(repository, str(binding.get("path", "")))
        digest = binding.get("sha256")
        if path in result or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{label} contains a duplicate or invalid source binding: {path}")
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"{label} source binding differs from disk: {path}")
        result[path] = digest
    return result


def _dense_configs(matrix_path: Path, *, expected: int, phase: str) -> list[RunConfig]:
    configs = [config for config in load_matrix(matrix_path) if config.model_family == "dense"]
    if len(configs) != expected:
        raise ValueError(
            f"{phase} matrix defines {len(configs)} Dense runs, expected {expected}: {matrix_path}"
        )
    return configs


def _queue_run_specs(
    repository: Path,
    plan: dict[str, Any],
) -> tuple[list[SourceRunSpec], set[Path]]:
    if plan.get("schema_version") != 1 or plan.get("status") != (
        "frozen_before_dense_confirmatory_or_short_branch_training"
    ):
        raise ValueError("Dense training plan is not the frozen pre-training plan")
    if plan.get("family") != "dense" or plan.get("expected") != EXPECTED_QUEUE_COUNTS:
        raise ValueError("Dense training plan has an unexpected family or run-count contract")
    pools = plan.get("pools")
    if not isinstance(pools, dict) or set(pools) != {"a", "b"}:
        raise ValueError("Dense training plan must define exactly pools a and b")

    matrix_cache: dict[Path, list[RunConfig]] = {}
    specs: list[SourceRunSpec] = []
    used_matrices: set[Path] = set()
    for pool in ("a", "b"):
        jobs = pools[pool]
        if not isinstance(jobs, list):
            raise ValueError(f"Dense training pool {pool} is not a job list")
        for job in jobs:
            if not isinstance(job, dict) or set(job) != {"phase", "matrix", "run_id"}:
                raise ValueError(f"Dense training pool {pool} contains a malformed job")
            phase = job["phase"]
            if phase not in {"confirmatory", "short-branch"}:
                raise ValueError(f"Dense training pool {pool} contains unknown phase {phase!r}")
            matrix_path = _under_repository(repository, str(job["matrix"]))
            used_matrices.add(matrix_path)
            configs = matrix_cache.setdefault(matrix_path, load_matrix(matrix_path))
            selected = [
                config
                for config in configs
                if config.model_family == "dense" and config.run_id == job["run_id"]
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Frozen queue job does not select exactly one Dense config: {phase}/{job['run_id']}"
                )
            config = selected[0]
            specs.append(
                SourceRunSpec(
                    phase=phase,
                    config=config,
                    matrix_path=matrix_path,
                    source_run_id=source_wandb_run_id(config),
                )
            )
    return specs, used_matrices


def load_frozen_dense_study(
    repository: str | Path,
    *,
    scope_amendment: str | Path,
    experiment_matrix: str | Path,
    hybrid_matrix: str | Path,
    training_plan: str | Path,
) -> FrozenDenseStudy:
    """Resolve and validate the exact pre-declared 34-run Dense source matrix."""

    root = Path(repository).resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError(f"Not an embedding-optimizer-study repository: {root}")
    scope_path = _under_repository(root, scope_amendment)
    experiment_path = _under_repository(root, experiment_matrix)
    hybrid_path = _under_repository(root, hybrid_matrix)
    plan_path = _under_repository(root, training_plan)

    verified_scope_path, scope = load_scope_amendment(scope_path, families=("dense",))
    if verified_scope_path != scope_path:
        raise ValueError("Resolved Dense scope amendment differs from the requested path")
    active = scope.get("active_scope")
    if not isinstance(active, dict) or any(
        active.get(key) != value
        for key, value in {
            "discovery_runs": EXPECTED_COUNTS["discovery"],
            "hybrid_adamw_runs": EXPECTED_COUNTS["hybrid"],
            "confirmatory_runs": EXPECTED_COUNTS["confirmatory"],
            "short_branch_runs": EXPECTED_COUNTS["short-branch"],
        }.items()
    ):
        raise ValueError("Dense scope amendment does not declare the frozen 34-run design")
    scope_bindings = _binding_map(root, scope.get("source_bindings"), label="Dense scope")
    if experiment_path not in scope_bindings or hybrid_path not in scope_bindings:
        raise ValueError("Dense scope does not bind both discovery and hybrid matrices")

    plan = _load_json(plan_path, label="Dense training plan")
    declared_scope = plan.get("scope_amendment")
    if not isinstance(declared_scope, dict) or set(declared_scope) != {"path"}:
        raise ValueError("Dense training plan has a malformed scope-amendment binding")
    if _under_repository(root, str(declared_scope["path"])) != scope_path:
        raise ValueError("Dense training plan binds a different scope amendment")
    queue_specs, used_queue_matrices = _queue_run_specs(root, plan)
    plan_bindings = _binding_map(root, plan.get("source_bindings"), label="Dense training plan")
    if set(plan_bindings) != used_queue_matrices:
        raise ValueError("Dense training plan source bindings differ from its queued matrices")

    discovery_specs = [
        SourceRunSpec("discovery", config, experiment_path, source_wandb_run_id(config))
        for config in _dense_configs(
            experiment_path,
            expected=EXPECTED_COUNTS["discovery"],
            phase="discovery",
        )
    ]
    hybrid_specs = [
        SourceRunSpec("hybrid", config, hybrid_path, source_wandb_run_id(config))
        for config in _dense_configs(
            hybrid_path,
            expected=EXPECTED_COUNTS["hybrid"],
            phase="hybrid",
        )
    ]
    specs = [*discovery_specs, *hybrid_specs, *queue_specs]
    counts = Counter(spec.phase for spec in specs)
    if dict(counts) != EXPECTED_COUNTS or len(specs) != EXPECTED_TOTAL_RUNS:
        raise ValueError(f"Frozen Dense source matrix counts differ: {dict(counts)}")
    source_ids = [spec.source_run_id for spec in specs]
    if len(set(source_ids)) != EXPECTED_TOTAL_RUNS:
        raise ValueError("Frozen Dense source matrix has colliding deterministic W&B run IDs")
    output_identities = [
        (spec.config.output_root, spec.config.model_family, spec.config.run_id) for spec in specs
    ]
    if len(set(output_identities)) != EXPECTED_TOTAL_RUNS:
        raise ValueError("Frozen Dense source matrix has colliding local output identities")

    projects = {(spec.config.wandb_entity, spec.config.wandb_project) for spec in specs}
    if len(projects) != 1:
        raise ValueError("Frozen Dense source configs do not share one W&B project")
    entity, project = projects.pop()
    if not isinstance(entity, str) or not entity or not isinstance(project, str) or not project:
        raise ValueError("Frozen Dense source configs do not declare a W&B entity/project")

    ordered = tuple(
        sorted(
            specs,
            key=lambda spec: (
                PHASE_ORDER[spec.phase],
                spec.config.seed,
                spec.config.run_id,
                spec.source_run_id,
            ),
        )
    )
    matrices = tuple(sorted({experiment_path, hybrid_path, *used_queue_matrices}))
    return FrozenDenseStudy(
        repository=root,
        scope_path=scope_path,
        training_plan_path=plan_path,
        matrix_paths=matrices,
        entity=entity,
        project=project,
        runs=ordered,
    )


def local_source_history(
    repository: Path,
    config: RunConfig,
) -> tuple[dict[str, int | float], ...]:
    output_dir = _source_output_dir(repository, config)
    completed = _load_json(output_dir / "completed.json", label="training completion receipt")
    state = _load_json(output_dir / "trainer_state_final.json", label="final Trainer state")
    if int(completed.get("global_step", -1)) != int(state.get("global_step", -2)):
        raise RuntimeError(
            f"Local completion/history step mismatch for {config.model_family}/{config.run_id}"
        )
    return canonical_history(state)


def _source_output_dir(repository: Path, config: RunConfig) -> Path:
    return _under_repository(
        repository,
        Path(config.output_root) / config.model_family / config.run_id,
    )


def _merge_source_wandb_config(
    *,
    model_config: dict[str, Any],
    training_arguments: dict[str, Any],
    run_config: RunConfig,
) -> dict[str, Any]:
    """Reconstruct the exact config dictionary logged by Trainer to W&B.

    Transformers adds the model config and serialized ``TrainingArguments`` to
    the user RunConfig. Its W&B callback omits the two empty mapping fields
    below, while ``TrainingArguments.to_dict()`` retains them. Refuse any value
    other than the observed empty mapping so this is a narrow, explicit
    normalization rather than a general allow-list for remote extras.
    """

    trainer_payload = dict(training_arguments)
    for key in ("learning_rate_mapping", "router_mapping"):
        if trainer_payload.pop(key, None) != {}:
            raise RuntimeError(
                f"Local TrainingArguments {key} is absent or non-empty; "
                "cannot reconstruct the exact W&B config"
            )
    expected = {
        **model_config,
        **trainer_payload,
        **run_config.as_dict(),
    }
    # Loading AutoConfig from the checkpoint reports the checkpoint directory;
    # the original source run was initialized before saving and records the
    # frozen base-model identity instead.
    expected["_name_or_path"] = run_config.model_name
    normalized = _normalized_json(expected)
    if not isinstance(normalized, dict):
        raise AssertionError("Normalized source W&B config is not an object")
    return normalized


def local_source_wandb_config(
    repository: Path,
    config: RunConfig,
) -> dict[str, Any]:
    """Rebuild the complete Trainer-created W&B config from frozen artifacts."""

    output_dir = _source_output_dir(repository, config)
    schedule = _load_json(
        output_dir / "checkpoint_schedule.json",
        label="checkpoint schedule",
    ).get("steps")
    if (
        not isinstance(schedule, list)
        or not schedule
        or any(isinstance(step, bool) or not isinstance(step, int) or step < 1 for step in schedule)
        or schedule != sorted(set(schedule))
    ):
        raise RuntimeError(
            f"Local checkpoint schedule is invalid for {config.model_family}/{config.run_id}"
        )
    completed = _load_json(output_dir / "completed.json", label="training completion receipt")
    completed_step = completed.get("global_step")
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or completed_step != schedule[-1]
    ):
        raise RuntimeError(
            f"Local completion/checkpoint step differs for {config.model_family}/{config.run_id}"
        )
    checkpoint = output_dir / f"checkpoint-{schedule[-1]}"
    arguments_path = checkpoint / "training_args.bin"
    if not arguments_path.is_file() or not (checkpoint / "config.json").is_file():
        raise RuntimeError(
            f"Final local checkpoint lacks config provenance for "
            f"{config.model_family}/{config.run_id}"
        )

    import torch
    from transformers import AutoConfig

    training_arguments = torch.load(
        arguments_path,
        map_location="cpu",
        weights_only=False,
    )
    to_dict = getattr(training_arguments, "to_dict", None)
    if not callable(to_dict):
        raise RuntimeError(
            f"Final TrainingArguments payload is invalid for {config.model_family}/{config.run_id}"
        )
    trainer_payload = to_dict()
    if not isinstance(trainer_payload, dict):
        raise RuntimeError(
            f"Final TrainingArguments serialization is invalid for "
            f"{config.model_family}/{config.run_id}"
        )
    model_payload = AutoConfig.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to_dict()
    if not isinstance(model_payload, dict):
        raise RuntimeError(
            f"Final model config is invalid for {config.model_family}/{config.run_id}"
        )
    return _merge_source_wandb_config(
        model_config=model_payload,
        training_arguments=trainer_payload,
        run_config=config,
    )


def normalize_remote_source_history(rows: Iterable[dict[str, Any]]) -> NormalizedRemoteHistory:
    """Last-write normalize Trainer W&B rows by train/global_step.

    A few resumed discovery source runs contain an overlapping prefix.  The
    canonical local Trainer history also resolves each step by last write, so
    the same deterministic rule is used here and duplicate/conflict counts are
    retained in the receipt.
    """

    by_step: dict[int, dict[str, int | float]] = {}
    raw_rows = 0
    duplicate_rows = 0
    overwritten_values = 0
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise RuntimeError(f"Remote W&B history row {index} is not an object")
        raw_rows += 1
        step_value = raw.get("train/global_step")
        if step_value is None:
            if any(raw.get(key) is not None for key in REMOTE_HISTORY_KEYS):
                raise RuntimeError(f"Remote W&B history row {index} has metrics but no global step")
            continue
        if (
            isinstance(step_value, bool)
            or not isinstance(step_value, (int, float))
            or not math.isfinite(float(step_value))
            or not float(step_value).is_integer()
            or int(step_value) < 1
        ):
            raise RuntimeError(f"Remote W&B history row {index} has an invalid global step")
        step = int(step_value)
        if step in by_step:
            duplicate_rows += 1
        normalized = by_step.setdefault(step, {"global_step": step})
        for remote_key, local_key in REMOTE_HISTORY_KEYS.items():
            value = raw.get(remote_key)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(f"Remote W&B history row {index} has invalid {remote_key}")
            normalized_value = float(value)
            if local_key in normalized and normalized[local_key] != normalized_value:
                overwritten_values += 1
            normalized[local_key] = normalized_value
    history = tuple(by_step[step] for step in sorted(by_step))
    if not history:
        raise RuntimeError("Remote W&B source history has no Trainer rows")
    return NormalizedRemoteHistory(history, raw_rows, duplicate_rows, overwritten_values)


def _git_origin(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "remote", "get-url", "origin"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Cannot resolve the audited repository's origin remote")
    return result.stdout.strip()


def _git_commit_exists(repository: Path, commit: str) -> bool:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        return False
    result = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return False
    reachable = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "for-each-ref",
            f"--contains={commit}",
            "--format=%(refname)",
            "refs/heads",
            "refs/remotes",
            "refs/tags",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return reachable.returncode == 0 and bool(reachable.stdout.strip())


def audit_remote_source_run(
    spec: SourceRunSpec,
    run: Any,
    *,
    repository: Path,
    expected_git_remote: str,
    expected_remote_config: dict[str, Any],
    expected_history: tuple[dict[str, int | float], ...],
    commit_exists: Callable[[Path, str], bool] = _git_commit_exists,
) -> dict[str, Any]:
    label = spec.label
    if getattr(run, "id", None) != spec.source_run_id:
        raise RuntimeError(f"{label}: remote W&B run ID differs")
    if getattr(run, "state", None) != "finished":
        raise RuntimeError(f"{label}: remote W&B run is not finished")
    if getattr(run, "group", None) != "dense":
        raise RuntimeError(f"{label}: remote W&B group is not dense")
    required_tags = {
        "dense",
        spec.config.optimizer.name,
        f"seed-{spec.config.seed}",
    }
    tags = {str(tag) for tag in (getattr(run, "tags", None) or [])}
    if not required_tags.issubset(tags):
        raise RuntimeError(f"{label}: remote W&B run is missing required tags")

    remote_config = getattr(run, "config", None)
    if not isinstance(remote_config, dict):
        raise RuntimeError(f"{label}: remote W&B config is not an object")
    expected_config = _normalized_json(expected_remote_config)
    observed_config = _normalized_json(remote_config)
    expected_keys = set(expected_config)
    observed_keys = set(observed_config)
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    changed = sorted(
        key for key in expected_keys & observed_keys if observed_config[key] != expected_config[key]
    )
    if missing or extra or changed:
        differences = "; ".join(
            f"{kind}={','.join(keys)}"
            for kind, keys in (("missing", missing), ("extra", extra), ("changed", changed))
            if keys
        )
        raise RuntimeError(f"{label}: exact remote W&B config differs ({differences})")

    metadata = getattr(run, "metadata", None)
    git = metadata.get("git") if isinstance(metadata, dict) else None
    if not isinstance(git, dict) or git.get("remote") != expected_git_remote:
        raise RuntimeError(f"{label}: remote W&B Git origin differs")
    commit = git.get("commit")
    if not isinstance(commit, str) or not commit_exists(repository, commit):
        raise RuntimeError(f"{label}: remote W&B Git commit is not reachable from a local Git ref")

    normalized = normalize_remote_source_history(run.scan_history(page_size=1000))
    if normalized.history != expected_history:
        raise RuntimeError(
            f"{label}: normalized remote W&B history differs from local Trainer state"
        )
    local_digest = history_sha256(expected_history)
    remote_digest = history_sha256(normalized.history)
    if remote_digest != local_digest:
        raise AssertionError(f"{label}: equal histories produced different hashes")
    return {
        "phase": spec.phase,
        "model_family": spec.config.model_family,
        "run_id": spec.config.run_id,
        "seed": spec.config.seed,
        "optimizer": spec.config.optimizer.name,
        "source_wandb_run_id": spec.source_run_id,
        "group": "dense",
        "required_tags": sorted(required_tags),
        "config_sha256": _json_sha256(expected_config),
        "run_config_sha256": _json_sha256(_normalized_json(spec.config.as_dict())),
        "config_keys": len(expected_config),
        "history_sha256": local_digest,
        "normalized_history_rows": len(expected_history),
        "remote_raw_history_rows": normalized.raw_rows,
        "remote_duplicate_history_rows": normalized.duplicate_rows,
        "remote_overwritten_values": normalized.overwritten_values,
        "git_commit": commit,
        "git_commit_validation": "reachable-local-ref",
        "state": "finished",
    }


def audit_dense_source_provenance(
    study: FrozenDenseStudy,
    *,
    expected_git_remote: str,
    api: Any,
    history_loader: Callable[[Path, RunConfig], tuple[dict[str, int | float], ...]] = (
        local_source_history
    ),
    config_loader: Callable[[Path, RunConfig], dict[str, Any]] = local_source_wandb_config,
    commit_exists: Callable[[Path, str], bool] = _git_commit_exists,
    verify_local_origin: bool = True,
) -> dict[str, Any]:
    """Read and verify all remote source runs; never update remote state."""

    if verify_local_origin and _git_origin(study.repository) != expected_git_remote:
        raise RuntimeError("Audited repository origin differs from the frozen GitHub remote")
    project_runs = list(api.runs(f"{study.entity}/{study.project}", per_page=200))
    remote_by_id: dict[str, list[Any]] = {}
    for run in project_runs:
        remote_by_id.setdefault(str(getattr(run, "id", "")), []).append(run)

    records = []
    for spec in study.runs:
        matches = remote_by_id.get(spec.source_run_id, [])
        if len(matches) != 1:
            raise RuntimeError(f"{spec.label}: expected exactly one remote W&B source run")
        expected_history = history_loader(study.repository, spec.config)
        expected_remote_config = config_loader(study.repository, spec.config)
        records.append(
            audit_remote_source_run(
                spec,
                matches[0],
                repository=study.repository,
                expected_git_remote=expected_git_remote,
                expected_remote_config=expected_remote_config,
                expected_history=expected_history,
                commit_exists=commit_exists,
            )
        )
    counts = Counter(record["phase"] for record in records)
    if dict(counts) != EXPECTED_COUNTS or len(records) != EXPECTED_TOTAL_RUNS:
        raise AssertionError("Remote W&B audit did not preserve the frozen 34-run coverage")
    return {
        "schema_version": 1,
        "status": "passed",
        "audited_at_utc": _timestamp(),
        "remote_access": "read-only",
        "entity": study.entity,
        "project": study.project,
        "expected_git_remote": expected_git_remote,
        "expected_counts": EXPECTED_COUNTS,
        "verified_runs": len(records),
        "scope_amendment": {
            "path": study.scope_path.relative_to(study.repository).as_posix(),
            "sha256": _sha256(study.scope_path),
        },
        "training_plan": {
            "path": study.training_plan_path.relative_to(study.repository).as_posix(),
            "sha256": _sha256(study.training_plan_path),
        },
        "matrices": [
            {
                "path": path.relative_to(study.repository).as_posix(),
                "sha256": _sha256(path),
            }
            for path in study.matrix_paths
        ],
        "runs": records,
    }


def receipt_envelope(audit: dict[str, Any]) -> dict[str, Any]:
    envelope = {
        "schema_version": 1,
        "kind": "dense-wandb-source-provenance-audit",
        "audit": audit,
        "audit_sha256": _json_sha256(audit),
    }
    serialized = _json_bytes(envelope).decode("utf-8").lower()
    if any(marker in serialized for marker in SECRET_MARKERS):
        raise RuntimeError("Refusing to write a W&B audit receipt containing a secret marker")
    return envelope


def write_receipt(path: Path, envelope: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--scope-amendment",
        type=Path,
        default=Path("configs/dense_scope_amendment.json"),
    )
    parser.add_argument(
        "--experiment-matrix",
        type=Path,
        default=Path("configs/experiment.yaml"),
    )
    parser.add_argument(
        "--hybrid-matrix",
        type=Path,
        default=Path("configs/hybrid_adamw.yaml"),
    )
    parser.add_argument(
        "--training-plan",
        type=Path,
        default=Path("configs/dense_training_queue.json"),
    )
    parser.add_argument("--expected-git-remote", default=EXPECTED_GIT_REMOTE)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("reports/wandb/dense_source_provenance_audit.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repository = args.repository.resolve()
    study = load_frozen_dense_study(
        repository,
        scope_amendment=args.scope_amendment,
        experiment_matrix=args.experiment_matrix,
        hybrid_matrix=args.hybrid_matrix,
        training_plan=args.training_plan,
    )
    import wandb

    audit = audit_dense_source_provenance(
        study,
        expected_git_remote=args.expected_git_remote,
        api=wandb.Api(),
    )
    envelope = receipt_envelope(audit)
    receipt_path = _under_repository(repository, args.receipt)
    write_receipt(receipt_path, envelope)
    print(
        json.dumps(
            {
                "audit_sha256": envelope["audit_sha256"],
                "entity": study.entity,
                "project": study.project,
                "receipt": receipt_path.relative_to(repository).as_posix(),
                "verified_runs": audit["verified_runs"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
