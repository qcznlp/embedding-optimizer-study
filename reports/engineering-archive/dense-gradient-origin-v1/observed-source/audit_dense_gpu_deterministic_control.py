"""Paired diagnostic: only owned ModernBERT attention backward flags change.

The default-kernel 8192-token audit remains a failed tolerance check. This control
tests a possible numerical reproducibility cause; it is not a production repair
or permission to replace the original acceptance result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from scripts import audit_dense_gpu_max_context as stress
from scripts import audit_dense_gpu_normalization as gpu

KERNEL_SOURCES = {
    "/usr/local/lib/python3.12/dist-packages/transformers/models/modernbert/modeling_modernbert.py": "d657e768ab419cb4101d3564d346cfe95b69149dd45e1d2f3e0b2fb138c6700b",
    "/usr/local/lib/python3.12/dist-packages/transformers/modeling_flash_attention_utils.py": "2c5b839ceae298a7d995f2b8ae88cc5f8151ee00a8ddccceed1be3bd1ef9bb72",
    "/usr/local/lib/python3.12/dist-packages/flash_attn/flash_attn_interface.py": "d8eaf989a5401fedd23d11a336a7f38e67596d0a4d5a279df14cbe2d7de2ed75",
}


def change_owned_model(model):
    config = model[0].auto_model.config
    attention = [
        m for m in model[0].auto_model.modules() if type(m).__name__ == "ModernBertAttention"
    ]
    if len(attention) != 22 or config.deterministic_flash_attn is not False:
        raise ValueError("Unreviewed attention topology/default")
    if any(m.deterministic_flash_attn is not False for m in attention):
        raise ValueError("Default kernel control was already altered")
    config.deterministic_flash_attn = True
    for module in attention:
        module.deterministic_flash_attn = True
    return {"attention_modules": len(attention), "before": False, "after": True}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control-receipt", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    rank = int(os.environ["RANK"])
    if (
        not args.control_receipt.is_absolute()
        or not args.control_receipt.parent.is_dir()
        or args.control_receipt.exists()
    ):
        raise ValueError("Use a new absolute control receipt with an existing parent")
    for name, expected in KERNEL_SOURCES.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Unreviewed kernel dispatch source")
    original_load = gpu.load_model
    observations = []
    report = {
        "scope": "engineering_paired_deterministic_flash_backward_control",
        "scientific_completion": False,
        "production_deployed": False,
        "source_hashes": KERNEL_SOURCES,
        "source": gpu.previous.identity(Path(__file__).resolve()),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "command_arguments": remaining,
        "boundary": "After a failed default-kernel long-input gradient comparison, change only the 22 owned attention instances and config deterministic_flash_attn=False to True. Both actual and manual-reference model copies use this control. No file/library/operator formula/precision/loss/data/tolerance is changed; this does not turn the default-path failure into a pass.",
    }

    def load(*a, **kw):
        model = original_load(*a, **kw)
        observations.append(change_owned_model(model))
        return model

    if rank == 0:
        with args.control_receipt.with_suffix(".started.json").open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    try:
        with (
            patch.object(gpu, "load_model", side_effect=load),
            patch.object(sys, "argv", [sys.argv[0], *remaining]),
        ):
            stress.main()
    finally:
        if rank == 0:
            result_path = Path(remaining[remaining.index("--workdir") + 1]) / "result.json"
            report.update(
                finished_at_utc=datetime.now(UTC).isoformat(),
                model_observations=observations,
                upstream_result=gpu.previous.identity(result_path)
                if result_path.is_file()
                else None,
                stress_source=gpu.previous.identity(Path(inspect.getsourcefile(stress))),
            )
            with args.control_receipt.open("x") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
