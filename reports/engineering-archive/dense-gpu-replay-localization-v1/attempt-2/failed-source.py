"""Localize checkpoint-replay differences without altering the training operator.

Wrap the existing full four-GPU save/resume diagnostic. Capture the first replayed
gradients, then independently execute the unmodified production optimizer from
the exact same entry state. Trace arithmetic only after the actual update; every
trace is checked bitwise against the production helper. Engineering evidence only.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from accelerate import Accelerator

from embed_optim import optimizers, train
from scripts import audit_dense_gpu_normalization as gpu
from scripts import audit_dense_gpu_resume as campaign
from scripts import audit_dense_trainer_resume as replay


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Require an ordinary evidence file")
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha}


def difference(left, right):
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError("Tensor topology differs")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Non-finite trace")
    delta = left.double() - right.double()
    return {
        "dtype": str(left.dtype),
        "shape": list(left.shape),
        "elements": left.numel(),
        "unequal_elements": int(torch.count_nonzero(left != right)),
        "bitwise_equal": torch.equal(left, right),
        "max_absolute_error": float(delta.abs().max()),
        "relative_l2_error": float(delta.norm() / left.double().norm().clamp_min(1e-30)),
    }


def optimizer_layout(optimizer, named):
    by_id = {id(p): name for name, p in named}
    if len(by_id) != len(named):
        raise ValueError("Aliased parameter topology")
    return [[by_id[id(p)] for p in group["params"]] for group in optimizer.param_groups]


@torch.no_grad()
def direct_step(config, entry, names, layout, gradients):
    if len(set(names)) != len(names) or set(names) != set(entry["weights"]):
        raise ValueError("Replay requires exactly the observed trainable state topology")
    if len(names) != len(gradients):
        raise ValueError("Gradient count differs")
    named = [(name, torch.nn.Parameter(entry["weights"][name].clone())) for name in names]
    model = SimpleNamespace(named_parameters=lambda: iter(named))
    optimizer, _ = optimizers.build_optimizer(model, config)
    if optimizer_layout(optimizer, named) != layout:
        raise ValueError("Production parameter routing changed")
    optimizer.load_state_dict(copy.deepcopy(entry["optimizer"]))
    replay.assert_exact_tree(optimizer.state_dict(), entry["optimizer"], "direct_entry")
    for (_, parameter), gradient in zip(named, gradients, strict=True):
        if parameter.shape != gradient.shape or parameter.dtype != gradient.dtype:
            raise ValueError("Gradient topology differs")
        parameter.grad = gradient.clone()
    optimizer.step()
    weights = copy.deepcopy(entry["weights"])
    for name, parameter in named:
        weights[name] = parameter.detach().clone()
    return {
        "weights": weights,
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }


@torch.no_grad()
def operator_trace(gradient, state, group):
    """Literal stage observation, not an alternative optimizer implementation."""
    algorithm = group["algorithm"]
    if algorithm not in ("muon", "normuon"):
        raise ValueError("Trace is limited to Muon-family hidden matrices")
    momentum = state["momentum_buffer"].clone()
    momentum.lerp_(gradient, 1 - group["momentum"])
    nesterov = gradient.lerp(momentum, group["momentum"])
    x = nesterov.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    denominator = x.norm().clamp_min(1e-7)
    stages = {
        "clipped_gradient": gradient,
        "momentum": momentum,
        "nesterov_fp32": nesterov,
        "cast_bf16": x,
        "denominator_bf16": denominator,
    }
    x = x / denominator
    stages["normalized_bf16"] = x
    for step in range(group["ns_steps"]):
        gram = x @ x.T
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * gram @ gram) @ x
        stages[f"newton_schulz_{step + 1}"] = x
    update = x.T if transposed else x
    state_after = {"momentum_buffer": momentum}
    if algorithm == "normuon":
        update = update.to(gradient.dtype)
        original_norm = update.norm()
        row_second_moment = torch.mean(update.square(), dim=-1, keepdim=True)
        second = state["second_moment"].clone()
        second.lerp_(row_second_moment, 1 - group["beta2"])
        stages["row_second_moment"] = row_second_moment
        stages["second_moment"] = second
        update = update * second.sqrt().add_(1e-10).reciprocal()
        stages["row_rescaled"] = update
        update = update * (original_norm / update.norm().add_(1e-10))
        update = update * math.sqrt(max(1, gradient.size(0) / gradient.size(1)))
        state_after["second_moment"] = second
    stages["operator_update"] = update
    actual_state = copy.deepcopy(state)
    if algorithm == "muon":
        actual = optimizers._muon_update(
            gradient, actual_state["momentum_buffer"], group["momentum"], group["ns_steps"]
        )
    else:
        actual = optimizers._normuon_update(
            gradient,
            actual_state["momentum_buffer"],
            actual_state["second_moment"],
            group["momentum"],
            group["beta2"],
            group["ns_steps"],
        )
    replay.assert_exact_tree(actual, update, "trace_operator_output")
    replay.assert_exact_tree(actual_state, state_after, "trace_operator_state")
    return stages


def exact_update(actual, expected):
    for key in ("weights", "optimizer"):
        replay.assert_exact_tree(actual[key], expected[key], f"direct_post_update/{key}")


def trace_pair(entry, names, layout, left, right):
    by_name = {name: i for i, name in enumerate(names)}
    records = []
    for group, group_names in zip(entry["optimizer"]["param_groups"], layout, strict=True):
        if group["algorithm"] == "adamw":
            continue
        for state_id, name in zip(group["params"], group_names, strict=True):
            state = entry["optimizer"]["state"][state_id]
            a = operator_trace(left[by_name[name]], state, group)
            b = operator_trace(right[by_name[name]], state, group)
            stages = {stage: difference(a[stage], b[stage]) for stage in a}
            records.append({"name": name, "stages": stages, "trace_matches_production": True})
    return records


def analyze(config, reference, restored, names, layout, output, rank):
    replay.assert_exact_tree(reference["entry_states"][1], restored["entry_states"][1])
    entry = reference["entry_states"][1]
    left = reference["gradient_values"][1]["clipped"]
    right = restored["gradient_values"][1]["clipped"]
    left_update = direct_step(config.optimizer, entry, names, layout, left)
    exact_update(left_update, reference["entry_states"][2])
    right_update = direct_step(config.optimizer, entry, names, layout, right)
    exact_update(right_update, restored["entry_states"][2])
    repeat = direct_step(config.optimizer, entry, names, layout, left)
    exact_update(repeat, left_update)
    del repeat
    result = {
        "rank": rank,
        "same_entry_exact": True,
        "baseline_direct_update_matches_actual_exactly": True,
        "resumed_direct_update_matches_actual_exactly": True,
        "same_gradient_repeat_exact": True,
        "names": names,
        "layout": layout,
        "clipped_gradient_pairs": {
            n: difference(a, b) for n, a, b in zip(names, left, right, strict=True)
        },
        "post_update_weight_pairs": {
            n: difference(left_update["weights"][n], right_update["weights"][n]) for n in names
        },
        "baseline_clipped_hashes": [replay.tensor_hash(g) for g in left],
        "resumed_clipped_hashes": [replay.tensor_hash(g) for g in right],
    }
    if rank == 0:
        # Only newly generated diagnostic gradients are written. No source payload is altered.
        payload = output / f"{config.run_id}-gradients.pt"
        with payload.open("xb") as handle:
            torch.save(
                {
                    "names": names,
                    "layout": layout,
                    "baseline": {
                        k: [t.cpu() for t in v] for k, v in reference["gradient_values"][1].items()
                    },
                    "resumed": {
                        k: [t.cpu() for t in v] for k, v in restored["gradient_values"][1].items()
                    },
                },
                handle,
            )
        result["captured_gradient_payload"] = identity(payload)
        result["operator_trace_pairs"] = trace_pair(entry, names, layout, left, right)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output = args.probe_output.absolute()
    rank = int(os.environ["RANK"])
    if output.parent != Path("/tmp") or not output.name.startswith("dense-replay-update-probe."):
        raise ValueError("Require a dedicated mktemp probe namespace")
    if not output.is_dir() or output.is_symlink():
        raise ValueError("Require an existing ordinary output directory")
    if rank == 0 and any(output.iterdir()):
        raise ValueError("Probe output directory must be empty")
    report = {
        "scope": "engineering_gpu_replay_update_localization",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source": identity(__file__),
        "command_arguments": remaining,
        "records": [],
        "boundary": "Same pinned short-input full-model four-rank GPU campaign, fresh candidate optimizers, same process group, three optimizers and resume points 1/2. Capture resume-1 gradients on every rank; replay actual optimizer from identical entry weights/state, check exact next-step reproduction and same-gradient repeatability. Stage traces are measured on rank zero and checked against production helpers. This does not identify the original gradient-difference source, certify historical trajectories, relax any failed replay requirement, or establish optimizer-quality/scientific findings.",
    }
    original_segment = replay.train_segment
    original_build = train.build_optimizer
    original_clip = Accelerator.clip_grad_norm_
    observed, references = {}, {}

    def build(model, config):
        optimizer, summary = original_build(model, config)
        named = list(model.named_parameters())
        observed.update(
            names=[n for n, _ in named],
            parameter_ids=[id(p) for _, p in named],
            layout=optimizer_layout(optimizer, named),
            clip_calls=0,
        )
        return optimizer, summary

    def clip(accelerator, parameters, *a, **kw):
        parameters = list(parameters)
        if [id(p) for p in parameters] != observed["parameter_ids"]:
            raise ValueError("Captured gradient order is not the observed named-parameter order")
        observed["clip_calls"] += 1
        return original_clip(accelerator, parameters, *a, **kw)

    def segment(config, fixture, segment_output, *a, **kw):
        snapshot, metadata = original_segment(config, fixture, segment_output, *a, **kw)
        if observed["clip_calls"] != len(snapshot["gradient_values"]):
            raise ValueError("Not every clipping call was observed")
        topology = {k: copy.deepcopy(observed[k]) for k in ("names", "layout")}
        if segment_output.name == "uninterrupted":
            references[config.run_id] = (snapshot, topology)
        elif segment_output.name == "resume-1":
            reference, previous_topology = references.pop(config.run_id)
            replay.assert_exact_tree(topology, previous_topology, "restored_topology")
            measured = analyze(config, reference, snapshot, **topology, output=output, rank=rank)
            gathered = [None] * 4
            dist.all_gather_object(gathered, measured)
            for key in ("baseline_clipped_hashes", "resumed_clipped_hashes"):
                if any(x[key] != gathered[0][key] for x in gathered):
                    raise ValueError("First-update gradients differ across ranks")
            if rank == 0:
                case = {"run_id": config.run_id, "ranks": gathered}
                path = output / f"{config.run_id}.json"
                with path.open("x") as handle:
                    json.dump(case, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                report["records"].append(identity(path))
                print(
                    json.dumps(
                        {
                            "run": config.run_id,
                            "direct_update_localization": "exact",
                            "all_ranks": 4,
                        }
                    ),
                    flush=True,
                )
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        return snapshot, metadata

    if rank == 0:
        with (output / "started.json").open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    try:
        with (
            patch.object(train, "build_optimizer", side_effect=build),
            patch.object(Accelerator, "clip_grad_norm_", clip),
            patch.object(replay, "train_segment", side_effect=segment),
            patch.object(sys, "argv", [sys.argv[0], *remaining]),
        ):
            campaign.main()
    finally:
        if rank == 0:
            parent = Path(remaining[remaining.index("--workdir") + 1]) / "result.json"
            parent_result = json.loads(parent.read_text()) if parent.is_file() else {}
            report.update(
                finished_at_utc=datetime.now(UTC).isoformat(),
                parent_result=identity(parent) if parent.is_file() else None,
                audit_execution_complete=(
                    len(report["records"]) == 3
                    and parent_result.get("audit_execution_complete", False)
                ),
                upstream_strict_replay_passed=parent_result.get(
                    "strict_bitwise_replay_passed", False
                ),
                source_bindings=[
                    identity(Path(inspect.getsourcefile(m)))
                    for m in (campaign, replay, gpu, optimizers, train)
                ],
            )
            with (output / "result.json").open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")


if __name__ == "__main__":
    main()
