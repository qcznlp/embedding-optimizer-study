"""Value-preserving device placement for restored fused AdamW step counters.

This adapter is explicit: it does not replace loading, validate a checkpoint's
provenance, change group settings or choose a different numerical kernel.
Call after the original named optimizer/scheduler restore, before any update.
"""

from __future__ import annotations

import torch

SCOPE = "named-adam-step-device-placement-v1"


def place_adam_step_counters(optimizer, *, expected_step: int) -> dict:
    """Place scalar counters with their parameters, preserving exact FP32 bits.

    The caller supplies the independently authenticated checkpoint step. Validate
    the entire named state before moving anything; leave moments and group fields
    untouched. An error is a failed restore, never permission to initialize state.
    """
    if type(expected_step) is not int or expected_step <= 0:
        raise ValueError("Require a positive externally authenticated resume step")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("Require the actual loaded optimizer")
    plan, names, parameters = [], set(), set()
    for group in optimizer.param_groups:
        if group.get("algorithm") not in ("adamw", "muon", "normuon"):
            raise ValueError("Unknown optimizer group algorithm")
        labels, members = group.get("param_names"), group["params"]
        if not isinstance(labels, list) or len(labels) != len(members) or not members:
            raise ValueError("Require the complete named parameter group")
        for name, parameter in zip(labels, members, strict=True):
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or id(parameter) in parameters
            ):
                raise ValueError("Duplicate or invalid named parameter")
            names.add(name)
            parameters.add(id(parameter))
            if group["algorithm"] != "adamw":
                continue
            if parameter.dtype != torch.float32 or parameter.device.type not in ("cpu", "cuda"):
                raise ValueError("Require original FP32 CPU/CUDA parameters")
            values = optimizer.state.get(parameter)
            if not isinstance(values, dict) or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(
                    "Require complete original Adam state; never initialize missing moments"
                )
            step = values["step"]
            if (
                not isinstance(step, torch.Tensor)
                or step.dtype != torch.float32
                or step.layout != torch.strided
                or step.shape != torch.Size([])
                or step.device.type not in ("cpu", "cuda")
                or step.item() != expected_step
            ):
                raise ValueError("Step tensor differs from the authenticated scalar counter")
            for key in ("exp_avg", "exp_avg_sq"):
                moment = values[key]
                if (
                    not isinstance(moment, torch.Tensor)
                    or moment.layout != torch.strided
                    or moment.dtype != parameter.dtype
                    or moment.shape != parameter.shape
                    or moment.device != parameter.device
                ):
                    raise ValueError(
                        "Original loader did not place the named Adam moments correctly"
                    )
            plan.append((name, parameter, values, step))
    if not plan:
        raise ValueError("No authenticated Adam counters to restore")

    rows = []
    for name, parameter, values, before in plan:
        original_bits = bytes(before.detach().reshape(1).view(torch.uint8).cpu().tolist())
        placed = before.to(device=parameter.device)
        restored_bits = bytes(placed.detach().reshape(1).view(torch.uint8).cpu().tolist())
        if restored_bits != original_bits or placed.dtype != before.dtype:
            raise ValueError("Device placement altered the original counter bits")
        values["step"] = placed
        rows.append(
            {
                "name": name,
                "from_device": str(before.device),
                "to_device": str(placed.device),
                "moved": before.device != placed.device,
                "counter_fp32_bytes_hex": original_bits.hex(),
            }
        )
    return {
        "scope": SCOPE,
        "authenticated_step": expected_step,
        "named_adam_counters": len(rows),
        "counters_moved": sum(row["moved"] for row in rows),
        "counter_values_bitwise_preserved": True,
        "moments_and_group_fields_not_modified": True,
        "numerical_kernel_not_modified": True,
        "counters": rows,
    }
