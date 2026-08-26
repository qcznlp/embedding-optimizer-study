"""Publish immutable, resume-safe training histories to Weights & Biases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .config import RunConfig, load_matrix, source_wandb_run_id

SCALAR_HISTORY_KEYS = {
    "epoch": "train/epoch",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/learning_rate",
    "loss": "train/loss",
}


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


def publish_canonical_run(spec: CanonicalRun, dry_run: bool = False) -> str:
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
        expected_step = int(spec.history[-1]["global_step"])
        if remote_digest != spec.history_sha256 or int(remote_step or -1) != expected_step:
            raise RuntimeError(f"Existing canonical run does not match local history: {run_path}")
        marked = _mark_canonical_current(existing)
        return f"verified{'-and-marked-current' if marked else ''} {run_path}"

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
    return f"published {run_path}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/experiment.yaml")
    parser.add_argument(
        "--families", nargs="+", choices=["dense", "late"], default=["dense", "late"]
    )
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configs = [
        config
        for config in load_matrix(args.matrix)
        if config.model_family in args.families
        and (not args.run_ids or config.run_id in args.run_ids)
    ]
    if not configs:
        raise SystemExit("No runs selected")
    for config in configs:
        spec = build_canonical_run(config)
        print(publish_canonical_run(spec, dry_run=args.dry_run), flush=True)


if __name__ == "__main__":
    main()
