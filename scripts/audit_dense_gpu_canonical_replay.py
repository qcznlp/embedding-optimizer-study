"""Isolated fixed-layout reduction + deterministic-backward acceptance control.

Select the existing full-model global-gradient oracle or the actual checkpoint
replay campaign. Export exact baseline fingerprints for a subsequent independent
process restart, without changing any old receipt or production code.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel

from scripts import audit_dense_gpu_normalization as gpu
from scripts import audit_dense_gpu_resume as resume_gpu
from scripts import audit_dense_trainer_resume as replay
from scripts import dense_canonical_reduction_control as canonical
from scripts.audit_dense_gpu_deterministic_control import KERNEL_SOURCES, change_owned_model
from scripts.audit_dense_gpu_replay_update import identity


def fingerprint(value, cache=None):
    cache = {} if cache is None else cache
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in cache:
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Cannot seal non-finite state")
            cache[key] = {
                "kind": "tensor",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": replay.tensor_hash(value),
            }
        return cache[key]
    if isinstance(value, dict):
        items = [[fingerprint(k, cache), fingerprint(v, cache)] for k, v in value.items()]
        items.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
        return {
            "kind": "mapping",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "items": items,
        }
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "items": [fingerprint(x, cache) for x in value]}
    if value is None or type(value) in (str, bool, int, float):
        return {"kind": type(value).__name__, "value": value}
    raise ValueError(f"Unsupported state type: {type(value)}")


def replay_fingerprint(snapshot, start, cache=None):
    # Same fields as the existing strict replay test, plus the exact restart entry.
    cache = {} if cache is None else cache
    final = dict(snapshot)
    final["trace"] = [x for x in snapshot["trace"] if x["global_step"] >= start]
    final["gradients"] = [x for x in snapshot["gradients"] if x["global_step"] >= start]
    if not final["trace"] or not final["gradients"]:
        raise ValueError("A replay fingerprint requires at least one real update")
    for key in ("entry_states", "gradient_values"):
        final[key] = {k: v for k, v in snapshot[key].items() if k >= start}
    return {
        "entry": fingerprint(snapshot["entry_states"][start], cache),
        "replayed_state": fingerprint(final, cache),
    }


@contextmanager
def controls(observations):
    """Only owned models/DDP instances are changed; all source files stay intact."""
    for source, expected in KERNEL_SOURCES.items():
        if identity(Path(source))["sha256"] != expected:
            raise ValueError("Unreviewed attention dispatch source")
    original_load, original_prepare = gpu.load_model, Accelerator.prepare_model
    states = []

    def load(*a, **kw):
        model = original_load(*a, **kw)
        observations["attention_controls"].append(change_owned_model(model))
        return model

    def prepare(accelerator, *a, **kw):
        for state in states:
            state.drain()
        model = original_prepare(accelerator, *a, **kw)
        if isinstance(model, DistributedDataParallel):
            named = [(n, p) for n, p in model.module.named_parameters() if p.requires_grad]
            if len(named) != 134 or any(
                not p.is_cuda or p.dtype != torch.float32 for _, p in named
            ):
                raise ValueError("Require the actual full DenseOn FP32 CUDA model")
            states.append(canonical.attach(model))
        return model

    try:
        with (
            patch.object(gpu, "load_model", side_effect=load),
            patch.object(Accelerator, "prepare_model", prepare),
        ):
            yield states
    finally:
        observations["reducers"] = [state.snapshot() for state in states]


def checkpoint_files(checkpoint):
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Require an ordinary completed diagnostic checkpoint")
    files = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError("Do not seal symlinked checkpoint inputs")
        if path.is_file():
            files.append(identity(path))
    required = {"model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"}
    required.update(f"rng_state_{rank}.pth" for rank in range(4))
    if not required.issubset({Path(x["path"]).name for x in files}):
        raise ValueError("Missing required diagnostic continuation files")
    return files


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control-output", type=Path, required=True)
    parser.add_argument("--target", choices=("normalization", "resume"), required=True)
    args, remaining = parser.parse_known_args()
    output = args.control_output.absolute()
    rank = int(os.environ["RANK"])
    if (
        output.parent != Path("/tmp")
        or not output.name.startswith("dense-canonical-replay.")
        or not output.is_dir()
        or output.is_symlink()
    ):
        raise ValueError("Require a dedicated mktemp control namespace")
    if rank == 0 and any(output.iterdir()):
        raise ValueError("Control output must be empty")
    if (
        args.target == "normalization"
        and remaining[remaining.index("--trainer") + 1] != "candidate"
    ):
        raise ValueError("Only the isolated single-normalization candidate is in scope")
    report = {
        "scope": "engineering_canonical_reduction_deterministic_backward_control",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": False,
        "target": args.target,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "command_arguments": remaining,
        "source": identity(__file__),
        "observations": {"attention_controls": [], "reducers": []},
        "baseline_exports": [],
        "boundary": "After bucket-layout origin localization, explicitly change only owned DDP communication to one lexical-name-ordered FP32 predivided SUM and owned attention backward flags to deterministic. The pinned single-normalization Trainer candidate is unchanged. Verify actual full-model gradients against the prior manual oracle or strict full Trainer checkpoint replay. No default-path failure is relabeled, no tolerance is changed, and no primary/production run is executed. Baseline state fingerprints and diagnostic file inventories are exported for a later independent-process test; this campaign alone is same-process-group replay.",
    }
    original_segment = replay.train_segment

    def segment(config, fixture, segment_output, *a, **kw):
        snapshot, metadata = original_segment(config, fixture, segment_output, *a, **kw)
        if segment_output.name == "uninterrupted":
            cache = {}
            expected = {str(step): replay_fingerprint(snapshot, step, cache) for step in (1, 2)}
            path = output / f"{config.run_id}-baseline-rank-{rank}.json"
            with path.open("x") as handle:
                json.dump(
                    {
                        "run_id": config.run_id,
                        "rank": rank,
                        "expected": expected,
                        "metadata": metadata,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
            inventory = None
            if rank == 0:
                inventory = {
                    str(step): checkpoint_files(segment_output / f"checkpoint-{step}")
                    for step in (1, 2, 3)
                }
            rank_exports = [None] * 4
            dist.all_gather_object(rank_exports, identity(path))
            if rank == 0:
                report["baseline_exports"].append(
                    {
                        "run_id": config.run_id,
                        "ranks": rank_exports,
                        "checkpoint_root": str(segment_output),
                        "checkpoint_files": inventory,
                    }
                )
            dist.barrier()
        return snapshot, metadata

    if rank == 0:
        with (output / "started.json").open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    try:
        with (
            controls(report["observations"]),
            patch.object(replay, "train_segment", side_effect=segment),
            patch.object(sys, "argv", [sys.argv[0], *remaining]),
        ):
            (gpu if args.target == "normalization" else resume_gpu).main()
    finally:
        with (output / f"observations-rank-{rank}.json").open("x") as handle:
            json.dump(
                {"rank": rank, "observations": report["observations"]},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        if rank == 0:
            parent = Path(remaining[remaining.index("--workdir") + 1]) / (
                "audit.json" if args.target == "normalization" else "result.json"
            )
            parent_result = json.loads(parent.read_text()) if parent.is_file() else {}
            report.update(
                finished_at_utc=datetime.now(UTC).isoformat(),
                audit_execution_complete=parent_result.get("audit_execution_complete", False),
                parent_result=identity(parent) if parent.is_file() else None,
                acceptance_passed=parent_result.get(
                    "normalization_acceptance_passed"
                    if args.target == "normalization"
                    else "strict_bitwise_replay_passed",
                    False,
                ),
                source_bindings=[
                    identity(Path(inspect.getsourcefile(m)))
                    for m in (
                        canonical,
                        gpu,
                        resume_gpu,
                        replay,
                        Accelerator,
                        DistributedDataParallel,
                        change_owned_model,
                    )
                ],
            )
            with (output / "result.json").open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")


if __name__ == "__main__":
    main()
