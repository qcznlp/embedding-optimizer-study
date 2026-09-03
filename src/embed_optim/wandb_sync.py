"""Publish immutable, resume-safe training histories to Weights & Biases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RunConfig, load_matrix, source_wandb_run_id
from .scope import resolve_scope

SCALAR_HISTORY_KEYS = {
    "epoch": "train/epoch",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/learning_rate",
    "loss": "train/loss",
}
SYSTEM_HISTORY_KEYS = (
    "system/useful_training_wall_time_seconds",
    "system/useful_samples_per_second",
    "system/useful_steps_per_second",
)
REMOTE_HISTORY_KEYS = ("global_step", *SCALAR_HISTORY_KEYS.values(), *SYSTEM_HISTORY_KEYS)


@dataclass(frozen=True)
class CanonicalRun:
    config: RunConfig
    history: tuple[dict[str, int | float], ...]
    history_sha256: str
    wandb_run_id: str
    source_wandb_run_id: str


def _audited_system_metrics(config: RunConfig, completed: dict[str, Any]) -> dict[str, float]:
    """Reconstruct useful full-run timing without duplicated resume segments."""

    accepted_path = config.output_dir / "accepted_timing.json"
    if accepted_path.is_file():
        accepted = json.loads(accepted_path.read_text())
        segment_seconds = sum(
            float(segment["wall_time_seconds_max_rank"]) for segment in accepted.get("segments", [])
        )
    else:
        segment_seconds = float(
            completed.get("system_metrics", {}).get("wall_time_seconds_max_rank", 0)
        )
    adjustment_path = config.output_dir / "timing_adjustment.json"
    adjustment = json.loads(adjustment_path.read_text()) if adjustment_path.is_file() else {}
    prior_seconds = float(adjustment.get("prior_training_wall_time_seconds", 0))
    useful_seconds = segment_seconds + prior_seconds
    rows = int(completed.get("dataset_rows", 0))
    steps = int(completed.get("global_step", 0))
    if not math.isfinite(useful_seconds) or useful_seconds <= 0 or rows <= 0 or steps <= 0:
        raise ValueError(
            f"Invalid audited system metrics for {config.model_family}/{config.run_id}"
        )
    return {
        "system/useful_training_wall_time_seconds": useful_seconds,
        "system/useful_samples_per_second": rows / useful_seconds,
        "system/useful_steps_per_second": steps / useful_seconds,
    }


def canonical_history(state: dict[str, Any]) -> tuple[dict[str, int | float], ...]:
    """Normalize Trainer history to one deterministic record per optimizer step."""

    global_step = int(state["global_step"])
    by_step: dict[int, dict[str, int | float]] = {}
    for raw in state.get("log_history", []):
        step = raw.get("step")
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            continue
        step = int(step)
        if step < 1 or step > global_step:
            raise ValueError(f"History step {step} is outside [1, {global_step}]")
        normalized = by_step.setdefault(step, {"global_step": step})
        for source, destination in SCALAR_HISTORY_KEYS.items():
            value = raw.get(source)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            normalized[destination] = float(value)

    # Trainer may log the last loss on the preceding logging interval. Preserve
    # the true terminal step explicitly so every canonical curve reaches 100%.
    terminal = by_step.setdefault(global_step, {"global_step": global_step})
    epoch = state.get("epoch")
    if not isinstance(epoch, bool) and isinstance(epoch, (int, float)):
        terminal.setdefault("train/epoch", float(epoch))

    history = tuple(by_step[step] for step in sorted(by_step))
    if not history:
        raise ValueError("Trainer state contains no scalar step history")
    if history[-1]["global_step"] > global_step:
        raise AssertionError("Canonical history extends beyond Trainer global_step")
    return history


def history_sha256(history: tuple[dict[str, int | float], ...]) -> str:
    payload = json.dumps(history, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_canonical_run(config: RunConfig) -> CanonicalRun:
    output_dir = config.output_dir
    completed_path = output_dir / "completed.json"
    state_path = output_dir / "trainer_state_final.json"
    if not completed_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"Run is not complete: {output_dir}")

    completed = json.loads(completed_path.read_text())
    state = json.loads(state_path.read_text())
    if int(completed["global_step"]) != int(state["global_step"]):
        raise ValueError(
            f"Completion/state step mismatch for {config.model_family}/{config.run_id}: "
            f"{completed['global_step']} != {state['global_step']}"
        )
    history = list(canonical_history(state))
    history[-1] = {**history[-1], **_audited_system_metrics(config, completed)}
    history = tuple(history)
    digest = history_sha256(history)
    source_id = source_wandb_run_id(config)
    run_id = f"canonical-{config.model_family}-{config.run_id}-{digest[:12]}"
    return CanonicalRun(config, history, digest, run_id, source_id)


def _existing_run(path: str):
    import wandb

    try:
        return wandb.Api().run(path)
    except wandb.errors.CommError as error:
        if "Could not find run" in str(error) or "not found" in str(error).lower():
            return None
        raise


def _mark_canonical_current(run: Any) -> bool:
    """Persist the non-destructive marker used to select the authoritative matrix."""

    tags = [str(tag) for tag in (run.tags or [])]
    changed = False
    if "canonical-current" not in tags:
        tags.append("canonical-current")
        changed = True
    if "canonical-superseded" in tags:
        tags = [tag for tag in tags if tag != "canonical-superseded"]
        changed = True
    if run.summary.get("canonical_status") != "current":
        run.summary["canonical_status"] = "current"
        changed = True
    if changed:
        run.tags = tags
        run.update()
    return changed


def _remote_canonical_history(run: Any) -> tuple[dict[str, int | float], ...]:
    """Read and normalize every canonical history row from the W&B backend."""

    history: list[dict[str, int | float]] = []
    for index, raw in enumerate(run.scan_history(page_size=1000)):
        step = raw.get("global_step")
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(float(step))
            or not float(step).is_integer()
        ):
            raise RuntimeError(f"Remote canonical history row {index} has an invalid global_step")
        normalized: dict[str, int | float] = {"global_step": int(step)}
        for key in REMOTE_HISTORY_KEYS[1:]:
            value = raw.get(key)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(f"Remote canonical history row {index} has invalid {key}")
            normalized[key] = float(value)
        history.append(normalized)
    steps = [int(row["global_step"]) for row in history]
    if not history or steps != sorted(set(steps)):
        raise RuntimeError("Remote canonical history is empty, duplicated, or out of order")
    return tuple(history)


def verify_remote_canonical_history(run: Any, expected: CanonicalRun) -> int:
    """Re-read remote rows and require byte-equivalent normalized history content."""

    observed = _remote_canonical_history(run)
    observed_digest = history_sha256(observed)
    if observed_digest != expected.history_sha256 or observed != expected.history:
        raise RuntimeError(
            "Remote canonical history does not match local history: "
            f"rows={len(observed)}, sha256={observed_digest}, "
            f"expected_rows={len(expected.history)}, expected_sha256={expected.history_sha256}"
        )
    return len(observed)


def verify_remote_current_matrix(runs: list[Any], expected: list[CanonicalRun]) -> dict[str, int]:
    """Require exactly one current remote run for every selected matrix identity."""

    expected_by_identity = {
        (spec.config.model_family, spec.config.run_id): spec.wandb_run_id for spec in expected
    }
    if len(expected_by_identity) != len(expected):
        raise ValueError("Selected canonical matrix contains duplicate run identities")
    observed: dict[tuple[str, str], list[Any]] = {identity: [] for identity in expected_by_identity}
    project_current = 0
    for run in runs:
        tags = {str(tag) for tag in (run.tags or [])}
        if "canonical-current" not in tags:
            continue
        project_current += 1
        identity = (run.config.get("model_family"), run.config.get("run_id"))
        if identity in observed:
            observed[identity].append(run)
    problems = []
    for identity, expected_id in expected_by_identity.items():
        matches = observed[identity]
        if len(matches) != 1:
            problems.append(f"{identity[0]}/{identity[1]} has {len(matches)} current runs")
            continue
        run = matches[0]
        if run.id != expected_id or run.summary.get("canonical_status") != "current":
            problems.append(f"{identity[0]}/{identity[1]} current run identity/status differs")
    if problems:
        raise RuntimeError("Remote canonical-current matrix is invalid: " + "; ".join(problems))
    return {
        "selected_runs": len(expected),
        "selected_current_runs": sum(len(matches) for matches in observed.values()),
        "project_current_runs": project_current,
    }


def verify_remote_only_selected_current(
    runs: list[Any], expected: list[CanonicalRun]
) -> dict[str, int]:
    """Require the selected matrix to be the entire project's current canonical scope."""

    audit = verify_remote_current_matrix(runs, expected)
    if audit["project_current_runs"] != len(expected):
        raise RuntimeError(
            "Remote project still has canonical-current runs outside the selected matrix: "
            f"{audit['project_current_runs']} != {len(expected)}"
        )
    return audit


def verify_remote_retirement_precondition(
    runs: list[Any],
    selected: list[CanonicalRun],
    excluded: list[CanonicalRun],
    scope_amendment: dict[str, Any],
) -> dict[str, int]:
    """Reject any current canonical run outside the exact retirement scope."""

    audit = verify_remote_current_matrix(runs, selected)
    excluded_matches = _matched_excluded_current_runs(runs, excluded, scope_amendment)
    excluded_current = sum(
        "canonical-current" in {str(tag) for tag in (run.tags or [])} for run, _ in excluded_matches
    )
    expected_current = len(selected) + excluded_current
    if audit["project_current_runs"] != expected_current:
        raise RuntimeError(
            "Remote project has canonical-current runs outside the exact retirement scope: "
            f"{audit['project_current_runs']} != {expected_current}"
        )
    return {**audit, "excluded_current_runs": excluded_current}


def _historical_scope_summary(scope_amendment: dict[str, Any]) -> dict[str, Any]:
    """Return the portable, explicit reason an excluded canonical run became historical."""

    return {
        "status": "excluded_by_user_directed_dense_scope_amendment",
        "active_families": ["dense"],
        "scope_amendment": scope_amendment,
    }


def _flatten_summary(prefix: str, value: Any) -> dict[str, Any]:
    """Mirror W&B's dotted-key representation for nested summary dictionaries."""

    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_summary(child_prefix, child))
        return flattened
    return {prefix: value}


def _historical_scope_matches(summary: Any, expected_scope: dict[str, Any]) -> bool:
    """Accept the nested input form or W&B's exact dotted-key round trip."""

    if summary.get("historical_scope") == expected_scope:
        return True
    expected_flat = _flatten_summary("historical_scope", expected_scope)
    observed_keys = {str(key) for key in summary.keys() if str(key).startswith("historical_scope.")}
    return observed_keys == set(expected_flat) and all(
        summary.get(key) == value for key, value in expected_flat.items()
    )


def _matched_excluded_current_runs(
    runs: list[Any],
    excluded: list[CanonicalRun],
    scope_amendment: dict[str, Any],
) -> list[tuple[Any, CanonicalRun]]:
    """Fail closed unless every excluded current run exactly matches the frozen matrix."""

    expected = {(spec.config.model_family, spec.config.run_id): spec for spec in excluded}
    matched: dict[tuple[str, str], list[Any]] = {identity: [] for identity in expected}
    unexpected = []
    expected_historical_scope = _historical_scope_summary(scope_amendment)
    for run in runs:
        tags = {str(tag) for tag in (run.tags or [])}
        config = run.config or {}
        identity = (config.get("model_family"), config.get("run_id"))
        if identity[0] != "late" or "canonical" not in tags:
            continue
        if identity not in expected:
            if "canonical-current" in tags:
                unexpected.append(f"{run.id} ({identity[0]}/{identity[1]})")
            continue
        spec = expected[identity]
        if run.id != spec.wandb_run_id:
            if "canonical-current" in tags:
                unexpected.append(f"{run.id} ({identity[0]}/{identity[1]})")
            continue
        digest = config.get("canonical_history_sha256")
        is_current = (
            "canonical-current" in tags and run.summary.get("canonical_status") == "current"
        )
        is_historical = (
            "canonical-current" not in tags
            and "canonical-historical" in tags
            and run.summary.get("canonical_status") == "historical"
            and _historical_scope_matches(run.summary, expected_historical_scope)
        )
        if digest != spec.history_sha256 or not (is_current or is_historical):
            raise RuntimeError(
                "Excluded canonical run does not exactly match the frozen matrix and scope: "
                f"{run.id}"
            )
        matched[identity].append(run)
    problems = [
        f"{family}/{run_id} has {len(matches)} exact current runs"
        for (family, run_id), matches in matched.items()
        if len(matches) != 1
    ]
    if unexpected:
        problems.append("unexpected Late canonical-current runs: " + ", ".join(unexpected))
    if problems:
        raise RuntimeError("Excluded canonical-current matrix is invalid: " + "; ".join(problems))
    return [(matched[identity][0], spec) for identity, spec in expected.items()]


def retire_excluded_canonical_runs(
    runs: list[Any],
    excluded: list[CanonicalRun],
    scope_amendment: dict[str, Any],
    *,
    dry_run: bool = False,
) -> list[str]:
    """Retire exact excluded canonical runs without touching any other remote run."""

    if not scope_amendment or scope_amendment.get("status") != (
        "user_directed_post_hoc_scope_amendment"
    ):
        raise ValueError("Retirement requires a verified Dense scope amendment")
    matches = _matched_excluded_current_runs(runs, excluded, scope_amendment)
    historical_scope = _historical_scope_summary(scope_amendment)
    results = []
    for run, spec in matches:
        if "canonical-historical" in {str(tag) for tag in (run.tags or [])}:
            results.append(f"already-historical {run.id} sha256={spec.history_sha256}")
            continue
        if dry_run:
            results.append(f"would-retire {run.id} sha256={spec.history_sha256}")
            continue
        tags = [str(tag) for tag in (run.tags or []) if str(tag) != "canonical-current"]
        if "canonical-historical" not in tags:
            tags.append("canonical-historical")
        run.tags = tags
        run.summary["canonical_status"] = "historical"
        run.summary["historical_scope"] = historical_scope
        run.update()
        results.append(f"retired {run.id} sha256={spec.history_sha256}")
    return results


def verify_excluded_canonical_histories(
    runs: list[Any],
    excluded: list[CanonicalRun],
    scope_amendment: dict[str, Any],
) -> int:
    """Content-verify every exact excluded run before any remote mutation."""

    matches = _matched_excluded_current_runs(runs, excluded, scope_amendment)
    for run, spec in matches:
        verify_remote_canonical_history(run, spec)
    return len(matches)


def verify_remote_historical_matrix(
    runs: list[Any],
    excluded: list[CanonicalRun],
    scope_amendment: dict[str, Any],
) -> dict[str, int]:
    """Verify the exact excluded matrix is historical after a retirement operation."""

    expected_scope = _historical_scope_summary(scope_amendment)
    expected = {(spec.config.model_family, spec.config.run_id): spec for spec in excluded}
    observed: dict[tuple[str, str], list[Any]] = {identity: [] for identity in expected}
    for run in runs:
        config = run.config or {}
        identity = (config.get("model_family"), config.get("run_id"))
        if identity not in expected or run.id != expected[identity].wandb_run_id:
            continue
        tags = {str(tag) for tag in (run.tags or [])}
        if (
            "canonical" in tags
            and "canonical-historical" in tags
            and "canonical-current" not in tags
            and config.get("canonical_history_sha256") == expected[identity].history_sha256
            and run.summary.get("canonical_status") == "historical"
            and _historical_scope_matches(run.summary, expected_scope)
        ):
            observed[identity].append(run)
    problems = [
        f"{family}/{run_id} has {len(matches)} exact historical runs"
        for (family, run_id), matches in observed.items()
        if len(matches) != 1
    ]
    if problems:
        raise RuntimeError("Remote historical matrix is invalid: " + "; ".join(problems))
    return {"excluded_runs": len(excluded), "verified_historical_runs": len(excluded)}


def publish_canonical_run(
    spec: CanonicalRun,
    dry_run: bool = False,
    *,
    verify_remote_history: bool = False,
) -> str:
    """Publish one immutable history, or verify and skip an identical existing run."""

    config = spec.config
    if not config.wandb_entity:
        raise ValueError(f"wandb_entity is required for {config.model_family}/{config.run_id}")
    run_path = f"{config.wandb_entity}/{config.wandb_project}/{spec.wandb_run_id}"
    if dry_run:
        return f"would-publish {run_path} rows={len(spec.history)} sha256={spec.history_sha256}"

    existing = _existing_run(run_path)
    if existing is not None:
        remote_digest = existing.config.get("canonical_history_sha256")
        remote_step = existing.summary.get("final_global_step")
        remote_rows = existing.summary.get("history_rows")
        expected_step = int(spec.history[-1]["global_step"])
        if (
            remote_digest != spec.history_sha256
            or int(remote_step or -1) != expected_step
            or int(remote_rows or -1) != len(spec.history)
        ):
            raise RuntimeError(f"Existing canonical run does not match local history: {run_path}")
        verified_rows = (
            verify_remote_canonical_history(existing, spec) if verify_remote_history else None
        )
        marked = _mark_canonical_current(existing)
        detail = f" rows={verified_rows} sha256={spec.history_sha256}" if verified_rows else ""
        return f"verified{'-and-marked-current' if marked else ''} {run_path}{detail}"

    import wandb

    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        id=spec.wandb_run_id,
        name=f"{config.model_family}-{config.run_id}-canonical",
        group=config.model_family,
        job_type="canonical-history",
        tags=[
            config.model_family,
            config.optimizer.name,
            f"seed-{config.seed}",
            "canonical",
            "canonical-current",
        ],
        resume="never",
        config={
            **config.as_dict(),
            "canonical_history_sha256": spec.history_sha256,
            "source_wandb_run_id": spec.source_wandb_run_id,
        },
    )
    try:
        run.define_metric("global_step")
        run.define_metric("train/*", step_metric="global_step")
        run.define_metric("system/*", step_metric="global_step")
        for record in spec.history:
            run.log(record)
        run.summary.update(
            {
                "canonical_history_sha256": spec.history_sha256,
                "final_global_step": int(spec.history[-1]["global_step"]),
                "history_rows": len(spec.history),
                "source_wandb_run_id": spec.source_wandb_run_id,
                "canonical_status": "current",
                **{
                    key: value
                    for key, value in spec.history[-1].items()
                    if key.startswith("system/")
                },
            }
        )
    finally:
        run.finish()
    if verify_remote_history:
        remote = _existing_run(run_path)
        if remote is None:
            raise RuntimeError(f"Published canonical run is not readable: {run_path}")
        verified_rows = verify_remote_canonical_history(remote, spec)
        return (
            f"published-and-verified {run_path} rows={verified_rows} sha256={spec.history_sha256}"
        )
    return f"published {run_path}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument("--families", nargs="+", choices=["dense", "late"], default=["dense"])
    parser.add_argument("--scope-amendment", type=Path)
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retire-excluded-families",
        action="store_true",
        help=(
            "After exact remote verification, mark canonical-current runs excluded by the "
            "verified Dense scope amendment as canonical-historical"
        ),
    )
    parser.add_argument(
        "--skip-remote-history-verification",
        action="store_true",
        help=(
            "Trust a matching remote summary instead of reading and rehashing every history row; "
            "the formal completion pipeline never uses this shortcut"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    families, scope_amendment = resolve_scope(args.families, args.scope_amendment)
    if args.retire_excluded_families and families != ("dense",):
        raise ValueError("Excluded-family retirement is only valid for verified Dense-only scope")
    if args.retire_excluded_families and args.run_ids:
        raise ValueError("Excluded-family retirement requires the complete Dense matrix")
    if args.retire_excluded_families and args.skip_remote_history_verification:
        raise ValueError("Excluded-family retirement requires remote history verification")
    args.families = list(families)
    matrix_configs = load_matrix(args.matrix)
    configs = [
        config
        for config in matrix_configs
        if config.model_family in args.families
        and (not args.run_ids or config.run_id in args.run_ids)
    ]
    if not configs:
        raise SystemExit("No runs selected")
    specs = [build_canonical_run(config) for config in configs]
    if args.retire_excluded_families:
        excluded = [
            build_canonical_run(config)
            for config in matrix_configs
            if config.model_family not in families
        ]
        if not excluded or {spec.config.model_family for spec in excluded} != {"late"}:
            raise RuntimeError("Training matrix does not define the excluded Late canonical matrix")
        entity, project = specs[0].config.wandb_entity, specs[0].config.wandb_project
        if not entity or any(
            (spec.config.wandb_entity, spec.config.wandb_project) != (entity, project)
            for spec in [*specs, *excluded]
        ):
            raise RuntimeError("Dense and excluded canonical runs do not share one W&B project")
        import wandb

        api = wandb.Api()
        project_runs = list(api.runs(f"{entity}/{project}"))
        verify_remote_retirement_precondition(project_runs, specs, excluded, scope_amendment)
        for spec in specs:
            remote = next((run for run in project_runs if run.id == spec.wandb_run_id), None)
            if remote is None:
                raise RuntimeError(f"Current Dense canonical run is missing: {spec.wandb_run_id}")
            verify_remote_canonical_history(remote, spec)
        verify_excluded_canonical_histories(project_runs, excluded, scope_amendment)
        for result in retire_excluded_canonical_runs(
            project_runs, excluded, scope_amendment, dry_run=args.dry_run
        ):
            print(result, flush=True)
        if not args.dry_run:
            refreshed = list(api.runs(f"{entity}/{project}"))
            audit = verify_remote_historical_matrix(refreshed, excluded, scope_amendment)
            verify_remote_only_selected_current(refreshed, specs)
            print(json.dumps({"remote_historical_matrix": audit}, sort_keys=True), flush=True)
        return

    for spec in specs:
        print(
            publish_canonical_run(
                spec,
                dry_run=args.dry_run,
                verify_remote_history=not args.skip_remote_history_verification,
            ),
            flush=True,
        )

    if not args.dry_run and not args.skip_remote_history_verification:
        projects = {(spec.config.wandb_entity, spec.config.wandb_project) for spec in specs}
        if len(projects) != 1:
            raise RuntimeError("Selected canonical runs do not share one W&B project")
        entity, project = projects.pop()
        if not entity:
            raise RuntimeError("Selected canonical runs do not declare a W&B entity")
        import wandb

        audit = verify_remote_current_matrix(list(wandb.Api().runs(f"{entity}/{project}")), specs)
        print(json.dumps({"remote_current_matrix": audit}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
