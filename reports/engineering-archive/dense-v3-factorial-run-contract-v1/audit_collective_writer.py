"""Four-rank CPU structural writer/IO controls, not Trainer/GPU numerics.

The caller is an explicit fake Trainer and all checkpoint tensors are invalid
synthetic placeholders. The default deep reader must reject them. Only the real
completion writer and collective error propagation are exercised here.
"""

import argparse
import json
import os
import runpy
import shutil
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import TrainerState

from embed_optim import factorial_v3_bound_trainer as bound
from embed_optim import factorial_v3_checkpoint as checkpoints
from embed_optim import factorial_v3_run_contract as contract
from embed_optim.primary_contract import file_identity, read_json


class SyntheticWriterCaller:
    def __init__(self, root, identity, observed, *, corrupt_export=False):
        self.args = SimpleNamespace(output_dir=str(root))
        self._bound_run = identity
        self.observed = observed
        self.state = TrainerState(
            global_step=391, max_steps=391, num_train_epochs=1, train_batch_size=8, epoch=1.0
        )
        self.corrupt_export = corrupt_export

    def _require_identity(self):
        return self.observed

    def _raw_optimizer(self):
        return SimpleNamespace(completed_steps=391)

    def is_world_process_zero(self):
        return dist.get_rank() == 0

    def save_model(self, final):
        if not self.is_world_process_zero():
            return
        final = Path(final)
        final.mkdir()
        for name in (*contract.INFERENCE_FILES, "README.md", "training_args.bin"):
            target = final / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(self.args.output_dir) / "checkpoint-391" / name, target)
        if self.corrupt_export:
            (final / "model.safetensors").write_bytes(b"intentional synthetic export mismatch")


def expect_failure(action, text):
    try:
        action()
    except ValueError as exc:
        if text not in str(exc):
            raise
        return str(exc)
    raise AssertionError("The structural failure was unexpectedly accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This structural test requires CUDA hidden")
    os.nice(10)
    torch.set_num_threads(1)
    if Path(contract.__file__).resolve().parents[2] != args.source.resolve():
        raise ValueError("Imported another run-component assembly")
    fixture_binding = file_identity(args.fixtures)
    fixture = runpy.run_path(str(args.fixtures))
    identity = fixture["metadata_identity"]()
    observed = fixture["component"](identity)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise ValueError("Require four real CPU ranks")
    if rank == 0:
        args.workdir.mkdir()
        fixture["checkpoint_fixture"](args.workdir / "success", identity)
        fixture["checkpoint_fixture"](args.workdir / "corrupt-export", identity)
    dist.barrier()
    success_root = args.workdir / "success"
    caller = SyntheticWriterCaller(success_root, identity, observed)
    completed_binding = bound.RunBoundFactorialTrainer.save_run_completion(caller)
    receipts = [None] * 4
    dist.all_gather_object(receipts, completed_binding)
    if any(r != receipts[0] for r in receipts):
        raise ValueError("Four ranks received different completion bindings")
    before = checkpoints.inventory(success_root)
    overwrite = expect_failure(
        lambda: bound.RunBoundFactorialTrainer.save_run_completion(caller), "overwrite"
    )
    if checkpoints.inventory(success_root) != before:
        raise ValueError("Rejected overwrite changed the existing fixture")
    caller.state.global_step = 390
    partial = expect_failure(
        lambda: bound.RunBoundFactorialTrainer.save_run_completion(caller), "phase failed"
    )

    def rank_two_failure():
        if rank == 2:
            raise OSError("synthetic rank-two IO failure")

    propagated = expect_failure(
        lambda: bound.collective_phase(rank_two_failure), "synthetic rank-two IO failure"
    )
    bad_root = args.workdir / "corrupt-export"
    bad = SyntheticWriterCaller(bad_root, identity, observed, corrupt_export=True)
    export_error = expect_failure(
        lambda: bound.RunBoundFactorialTrainer.save_run_completion(bad),
        "File content identity differs",
    )
    if (bad_root / contract.RUN_COMPLETE).exists() or not (
        bad_root / "final/model.safetensors"
    ).exists():
        raise ValueError("Failure must preserve partial outputs without a completion receipt")
    deep_refusal = expect_failure(
        lambda: contract.inspect_complete_run(success_root, completed_binding, identity), ""
    )
    if file_identity(args.fixtures) != fixture_binding or torch.cuda.is_initialized():
        raise ValueError("Changed fixture source or unexpected CUDA initialization")
    result = {
        "rank": rank,
        "backend": dist.get_backend(),
        "world_size": 4,
        "complete_writer_binding": completed_binding,
        "all_rank_bindings_identical": True,
        "overwrite_refusal": overwrite,
        "partial_refusal": partial,
        "rank_two_failure_propagated": propagated,
        "rank_zero_export_failure_propagated": export_error,
        "partial_outputs_preserved": True,
        "default_deep_reader_rejects_fixture": deep_refusal,
        "synthetic_caller_and_payloads": True,
        "actual_trainer_executed": False,
        "gpu_execution": False,
        "scientific_admission": False,
    }
    fixture["write_new_json"](args.workdir / f"rank-{rank}.json", result)
    dist.barrier()
    if rank == 0:
        ranks = [read_json(args.workdir / f"rank-{r}.json") for r in range(4)]
        complete = {
            "scope": "four-cpu-rank-synthetic-completion-writer-controls",
            "audit_source": {"path": str(Path(__file__).resolve()), **file_identity(__file__)},
            "fixture_source": {"path": str(args.fixtures), **fixture_binding},
            "run_sources": identity["sources"],
            "ranks": ranks,
            "files": checkpoints.inventory(args.workdir),
            "numerical_trainer_or_formal_run_admitted": False,
        }
        fixture["write_new_json"](args.workdir / "complete.json", complete)
        print(
            json.dumps(
                {
                    "output": str(args.workdir / "complete.json"),
                    **file_identity(args.workdir / "complete.json"),
                    "four_rank_controls_passed": True,
                    "actual_trainer_executed": False,
                }
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
