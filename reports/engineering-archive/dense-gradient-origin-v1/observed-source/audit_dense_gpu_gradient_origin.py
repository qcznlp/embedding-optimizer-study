"""Observe first-replay local gradients and replay actual DDP bucket layouts.

No communication hook, gradient replacement, production/library edit or tolerance
change. A paired CLI control changes only owned attention determinism flags.
The existing candidate GPU campaign is narrowed explicitly to Muon 3e-4 to isolate
the shared first-update input, not to establish a new optimizer-quality finding.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel

from scripts import audit_dense_gpu_normalization as gpu
from scripts import audit_dense_gpu_resume as campaign
from scripts import audit_dense_trainer_resume as replay
from scripts.audit_dense_gpu_deterministic_control import KERNEL_SOURCES, change_owned_model
from scripts.audit_dense_gpu_replay_update import difference, identity

RUN_ID = "padded-muon-3e-4"
PRIOR = Path("reports/engineering-archive/dense-gpu-replay-localization-v1/validation.json")
PRIOR_SHA = "829f29f1cc132d15fcbcfebf70c2b6c1fcf23bdb01dd6dda87d1ca95117574ed"
LOG_KEYS = (
    "bucket_sizes",
    "rebuilt_bucket_sizes",
    "rebuilt_per_bucket_param_indices",
    "has_rebuilt_buckets",
    "prev_iteration_grad_ready_order_indices",
    "iteration",
    "gradient_as_bucket_view",
    "find_unused_parameters",
    "comm_hook",
)


def validate_layout(layout, names):
    flattened = [name for bucket in layout for name in bucket["names"]]
    if (
        len(flattened) != len(names)
        or len(set(flattened)) != len(names)
        or set(flattened) != set(names)
    ):
        raise ValueError("Bucket layout must cover each observed parameter exactly once")
    if [b["index"] for b in layout] != list(range(len(layout))):
        raise ValueError("Unexpected bucket order")


def get_layout(ddp, names, parameter_ids):
    if ddp._comm_hooks:
        raise ValueError("Do not replace actual DDP communication")
    by_id = dict(zip(parameter_ids, names, strict=True))
    buckets = ddp.reducer._get_zeros_like_grad_buckets()
    layout = [
        {
            "index": b.index(),
            "names": [by_id[id(p)] for p in b.parameters()],
            "numel": b.buffer().numel(),
            "dtype": str(b.buffer().dtype),
        }
        for b in buckets
    ]
    validate_layout(layout, names)
    return layout


@torch.no_grad()
def local_sum(values, names):
    result = {}
    for name in names:
        parts = values[name]
        if len(parts) != 4:
            raise ValueError("Require all four first-replay microbatch gradients")
        total = parts[0].clone()
        for part in parts[1:]:
            total.add_(part)
        result[name] = total
    return result


@torch.no_grad()
def project_buckets(local, layout):
    """Replay observed partition/order, FP32 predivision by four, NCCL SUM."""
    validate_layout(layout, list(local))
    result = {}
    for bucket in layout:
        values = [local[name] for name in bucket["names"]]
        if any(x.dtype != torch.float32 or not x.is_cuda for x in values):
            raise ValueError("Require actual FP32 CUDA gradient tensors")
        buffer = torch.cat([x.reshape(-1) for x in values]).mul_(0.25)
        if buffer.numel() != bucket["numel"] or bucket["dtype"] != str(buffer.dtype):
            raise ValueError("Observed bucket size/dtype differs")
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        offset = 0
        for name, value in zip(bucket["names"], values, strict=True):
            result[name] = buffer[offset : offset + value.numel()].view_as(value).clone()
            offset += value.numel()
    return result


def compare_named(left, right, names):
    if set(left) != set(names) or set(right) != set(names):
        raise ValueError("Named comparison topology changed")
    return {name: difference(left[name], right[name]) for name in names}


def all_exact(comparison):
    return bool(comparison) and all(x["bitwise_equal"] for x in comparison.values())


def analyze(reference, restored, left_capture, right_capture, output, rank, prior_binding):
    replay.assert_exact_tree(reference["entry_states"][1], restored["entry_states"][1])
    names = left_capture["names"]
    if names != right_capture["names"]:
        raise ValueError("Parameter name/order changed")
    local_pairs = {
        name: [
            difference(a, b)
            for a, b in zip(
                left_capture["values"][name], right_capture["values"][name], strict=True
            )
        ]
        for name in names
    }
    left = local_sum(left_capture["values"], names)
    right = local_sum(right_capture["values"], names)
    layouts = [left_capture["layout"], right_capture["layout"]]
    rank_layouts = [None] * 4
    dist.all_gather_object(rank_layouts, layouts)
    if any(x != layouts for x in rank_layouts):
        raise ValueError("Ranks disagree on observed bucket layouts")
    actual_left = dict(zip(names, reference["gradient_values"][1]["raw"], strict=True))
    actual_right = dict(zip(names, restored["gradient_values"][1]["raw"], strict=True))
    reconstructed_left = project_buckets(left, layouts[0])
    reconstructed_right = project_buckets(right, layouts[1])
    crossed = project_buckets(left, layouts[1])
    repeated = project_buckets(left, layouts[0])
    comparisons = {
        "local_accumulated": compare_named(left, right, names),
        "actual_ddp_raw": compare_named(actual_left, actual_right, names),
        "baseline_bucket_replay_vs_actual": compare_named(reconstructed_left, actual_left, names),
        "resumed_bucket_replay_vs_actual": compare_named(reconstructed_right, actual_right, names),
        "same_input_resumed_layout_vs_actual_resumed": compare_named(crossed, actual_right, names),
        "same_input_same_layout_repeat": compare_named(repeated, reconstructed_left, names),
        "same_input_crossed_layout": compare_named(reconstructed_left, crossed, names),
    }
    result = {
        "rank": rank,
        "exact_entry_state": True,
        "names": names,
        "baseline_layout": layouts[0],
        "resumed_layout": layouts[1],
        "baseline_ddp_logging": left_capture["ddp_logging"],
        "resumed_ddp_logging": right_capture["ddp_logging"],
        "baseline_hook_counts": left_capture["counts"],
        "resumed_hook_counts": right_capture["counts"],
        "local_microbatch_pairs": local_pairs,
        "comparisons": comparisons,
        "all_local_microbatch_tensors_exact": all(
            x["bitwise_equal"] for rows in local_pairs.values() for x in rows
        ),
        "all_local_accumulated_tensors_exact": all_exact(comparisons["local_accumulated"]),
        "both_actual_bucket_replays_exact": all_exact(
            comparisons["baseline_bucket_replay_vs_actual"]
        )
        and all_exact(comparisons["resumed_bucket_replay_vs_actual"]),
        "same_gradient_same_layout_repeat_exact": all_exact(
            comparisons["same_input_same_layout_repeat"]
        ),
        "fixed_local_gradient_cross_layout_reproduces_resumed_exactly": all_exact(
            comparisons["same_input_resumed_layout_vs_actual_resumed"]
        ),
        "actual_raw_gradients_differ": not all_exact(comparisons["actual_ddp_raw"]),
    }
    path = output / f"local-sums-rank-{rank}.pt"
    with path.open("xb") as handle:
        torch.save(
            {
                "names": names,
                "baseline": {n: x.cpu() for n, x in left.items()},
                "resumed": {n: x.cpu() for n, x in right.items()},
            },
            handle,
        )
    result["captured_local_sum_payload"] = identity(path)
    if rank == 0:
        prior_path = Path(prior_binding["path"])
        if identity(prior_path) != prior_binding:
            raise ValueError("Prior gradient payload changed before deserialization")
        prior = torch.load(prior_path, map_location="cpu", weights_only=True)
        if prior["names"] != names:
            raise ValueError("Prior gradient topology differs")
        result["prior_uninstrumented_gradient_comparisons"] = {
            label: {
                name: difference(value.cpu(), old)
                for name, value, old in zip(
                    names, snapshot["gradient_values"][1]["raw"], prior[label]["raw"], strict=True
                )
            }
            for label, snapshot in (("baseline", reference), ("resumed", restored))
        }
        result["prior_gradient_payload"] = prior_binding
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--origin-output", type=Path, required=True)
    parser.add_argument("--deterministic-backward", action="store_true")
    args, remaining = parser.parse_known_args()
    output = args.origin_output.absolute()
    rank = int(os.environ["RANK"])
    root = Path(remaining[remaining.index("--repository") + 1]).resolve()
    if (
        output.parent != Path("/tmp")
        or not output.name.startswith("dense-gradient-origin.")
        or not output.is_dir()
        or output.is_symlink()
    ):
        raise ValueError("Require a dedicated mktemp origin namespace")
    if rank == 0 and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    if identity(root / PRIOR)["sha256"] != PRIOR_SHA:
        raise ValueError("Prior capture authorization/digest receipt changed")
    prior = json.loads((root / PRIOR).read_text())
    prior_binding = next(
        x
        for x in prior["captured_external_gradient_payloads"]
        if Path(x["path"]).name == f"{RUN_ID}-gradients.pt"
    )
    for source, sha in KERNEL_SOURCES.items():
        if identity(Path(source))["sha256"] != sha:
            raise ValueError("Unreviewed attention source")
    report = {
        "scope": "engineering_first_gradient_origin_bucket_layout_control",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": False,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "deterministic_backward_control": args.deterministic_backward,
        "run_ids": [RUN_ID],
        "command_arguments": remaining,
        "source": identity(__file__),
        "prior_validation": identity(root / PRIOR),
        "records": [],
        "model_controls": [],
        "boundary": "Actual candidate Trainer/NCCL checkpoint replay with observational leaf tensor hooks returning None and no communication hook. Capture all four first-replay microbatch gradients on all 134 parameters, query actual DDP zero-bucket metadata after backward, and independently replay each observed bucket partition on captured local sums. Explicit Muon-only matched control for the shared first-update fixture, not independent optimizer trials, performance or retrieval evidence. Deterministic mode changes only 22 owned attention flags. Hook allocations may affect timing; compare actual raw gradients with the authenticated prior capture separately.",
    }
    original_load, original_segment = gpu.load_model, replay.train_segment
    original_prepare, original_clip = Accelerator.prepare_model, Accelerator.clip_grad_norm_
    current, references = {}, {}

    def load(*a, **kw):
        model = original_load(*a, **kw)
        if args.deterministic_backward:
            report["model_controls"].append(change_owned_model(model))
        named = list(model.named_parameters())
        current.update(
            names=[n for n, _ in named],
            parameter_ids=[id(p) for _, p in named],
            counts={n: 0 for n, _ in named},
            values={n: [] for n, _ in named},
            handles=[],
        )
        start = 4 if current["phase"] == "uninterrupted" else 0
        capture = current
        for name, parameter in named:

            def observe(gradient, name=name):
                index = capture["counts"][name]
                capture["counts"][name] += 1
                if capture["phase"] != "resume-2" and start <= index < start + 4:
                    capture["values"][name].append(gradient.detach().clone())
                return None

            capture["handles"].append(parameter.register_hook(observe))
        return model

    def prepare(accelerator, *a, **kw):
        model = original_prepare(accelerator, *a, **kw)
        if isinstance(model, DistributedDataParallel):
            if "ddp" in current:
                raise ValueError("Unexpected second DDP preparation")
            current["ddp"] = model
        return model

    def clip(accelerator, parameters, *a, **kw):
        parameters = list(parameters)
        if [id(p) for p in parameters] != current["parameter_ids"]:
            raise ValueError("Snapshot gradients do not follow observed named-parameter order")
        index = current["clip_calls"]
        current["clip_calls"] += 1
        target = 1 if current["phase"] == "uninterrupted" else 0
        if current["phase"] != "resume-2" and index == target:
            ddp = current["ddp"]
            current["layout"] = get_layout(ddp, current["names"], current["parameter_ids"])
            data = ddp._get_ddp_logging_data()
            current["ddp_logging"] = {key: data[key] for key in LOG_KEYS if key in data}
        return original_clip(accelerator, parameters, *a, **kw)

    def segment(config, fixture, segment_output, *a, **kw):
        nonlocal current
        current = {"phase": segment_output.name, "clip_calls": 0}
        snapshot, metadata = original_segment(config, fixture, segment_output, *a, **kw)
        for handle in current["handles"]:
            handle.remove()
        expected = {"uninterrupted": 9, "resume-1": 5, "resume-2": 1}[current["phase"]]
        if set(current["counts"].values()) != {expected}:
            raise ValueError("Tensor hooks did not observe each expected backward exactly once")
        capture = {
            key: current[key]
            for key in ("names", "counts", "values", "layout", "ddp_logging")
            if key in current
        }
        current.clear()
        if segment_output.name == "uninterrupted":
            references[config.run_id] = (snapshot, capture)
        elif segment_output.name == "resume-1":
            reference, left_capture = references.pop(config.run_id)
            measured = analyze(
                reference, snapshot, left_capture, capture, output, rank, prior_binding
            )
            gathered = [None] * 4
            dist.all_gather_object(gathered, measured)
            if rank == 0:
                with (output / "first-gradient.json").open("x") as handle:
                    json.dump(
                        {"run_id": config.run_id, "ranks": gathered},
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
                    handle.write("\n")
                report["records"].append(identity(output / "first-gradient.json"))
                print(
                    json.dumps(
                        {
                            "gradient_origin_capture_complete": True,
                            "local_gradients_exact_all_ranks": all(
                                x["all_local_microbatch_tensors_exact"] for x in gathered
                            ),
                            "bucket_replays_exact_all_ranks": all(
                                x["both_actual_bucket_replays_exact"] for x in gathered
                            ),
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
            patch.object(gpu, "load_model", side_effect=load),
            patch.object(replay, "train_segment", side_effect=segment),
            patch.object(replay.previous, "RUN_IDS", (RUN_ID,)),
            patch.object(Accelerator, "prepare_model", prepare),
            patch.object(Accelerator, "clip_grad_norm_", clip),
            patch.object(sys, "argv", [sys.argv[0], *remaining]),
        ):
            campaign.main()
    finally:
        if rank == 0:
            parent = Path(remaining[remaining.index("--workdir") + 1]) / "result.json"
            parent_result = json.loads(parent.read_text()) if parent.is_file() else {}
            report.update(
                finished_at_utc=datetime.now(UTC).isoformat(),
                audit_execution_complete=(
                    len(report["records"]) == 1
                    and parent_result.get("audit_execution_complete", False)
                ),
                parent_result=identity(parent) if parent.is_file() else None,
                upstream_strict_replay_passed=parent_result.get(
                    "strict_bitwise_replay_passed", False
                ),
                source_bindings=[
                    identity(Path(inspect.getsourcefile(m)))
                    for m in (
                        campaign,
                        replay,
                        gpu,
                        Accelerator,
                        DistributedDataParallel,
                        change_owned_model,
                        difference,
                    )
                ],
            )
            with (output / "result.json").open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")


if __name__ == "__main__":
    main()
