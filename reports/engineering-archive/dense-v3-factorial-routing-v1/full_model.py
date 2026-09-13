"""CPU-only actual pretrained loading and reset routing under corrected sources.

Invoke from a separately assembled source-only test directory, never the live
training environment. No forward pass, gradient, optimizer update, GPU lease,
formal experiment admission, remote call, or controller action is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path


def identity(path):
    path = Path(path)
    with path.open("rb") as handle:
        return {
            "bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
        }


def verify(path, expected):
    actual = identity(path)
    if actual != {k: expected[k] for k in ("bytes", "sha256")}:
        raise ValueError(f"Pinned source/model payload differs: {Path(path).name}")
    return actual


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Require explicit CPU-only, offline diagnostic environment")
    import torch
    from safetensors import safe_open
    from sentence_transformers import SentenceTransformer

    from embed_optim import config, factorial_v3_optimizer, optimizers

    files = {}
    package = args.source_root.resolve() / "src/embed_optim"
    for module in (config, factorial_v3_optimizer, optimizers):
        path = Path(inspect.getsourcefile(module)).resolve()
        if path.parent != package:
            raise ValueError("Actual imported numerical module is outside the test assembly")
        files[module.__name__] = {"path": str(path), **identity(path)}
    for name, sha in (
        ("config.py", "25d85021a0d78918374cb995016e80563c5ede9b8e87499809e336323105436b"),
        ("optimizers.py", "7d7d4f200c410cb1b24e1773ae72811582f43a0c794a5d1b2af269c123e69b1e"),
    ):
        if identity(package / name)["sha256"] != sha:
            raise ValueError("This diagnostic requires the exact corrected candidate factory")
    inputs_path = args.repository.resolve() / "reports/dense-primary-v2/input-bindings.json"
    if (
        identity(inputs_path)["sha256"]
        != "02561536a7ac691290f127dc07ddef41cede4b15e9bfbabb97eee0b569a0fa59"
    ):
        raise ValueError("The trusted immutable base-input receipt changed")
    inputs = json.loads(inputs_path.read_text())
    # The caller uses the unchanged trusted receipt; this check binds the actual
    # model payload to its accepted independent-HF-digest parent.
    base = Path(inputs["initial_model"]["cached_root"])
    model_files = {}
    for row in inputs["common_identity"]["model"]["files"]:
        model_files[row["path"]] = verify(base / row["path"], row)
    if base.name != "0edbd55684eb782bce55ee74c95b25c97cbe7f43":
        raise ValueError("Unexpected pretrained checkpoint revision")
    model = SentenceTransformer(
        str(base),
        device="cpu",
        local_files_only=True,
        model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
    )
    parameters = dict(model.named_parameters())
    if {n: list(p.shape) for n, p in parameters.items()} != factorial_v3_optimizer.denseon_shapes():
        raise ValueError("Actual loaded full-model topology differs")
    with safe_open(base / "model.safetensors", framework="pt", device="cpu") as store:
        if {"0.model." + n for n in store.keys()} != set(parameters):
            raise ValueError("Actual model namespace differs from every saved tensor")
        for key in store.keys():
            if not torch.equal(
                parameters["0.model." + key].detach(), store.get_tensor(key).float()
            ):
                raise ValueError("A loaded parameter does not equal the pinned saved weight")
    layouts, resets = {}, {}
    for name in ("hybrid_adamw", "muon"):
        opt = factorial_v3_optimizer.FactorialOptimizer(
            model,
            config.OptimizerConfig(name=name, lr=0.0003),
        )
        scheduler = factorial_v3_optimizer.create_scheduler(opt)
        layouts[name] = opt.named_parameter_layout
        if [len(row["members"]) for row in layouts[name]] != [88, 1, 45]:
            raise ValueError("Full model did not route the same 88 hidden matrices")
        resets[name] = {
            "steps": opt.completed_steps,
            "state_tensors": len(opt.state_dict()["state"]),
            "partition": opt.partition_summary,
            "scheduler": factorial_v3_optimizer.inspect_scheduler(
                scheduler.state_dict(),
                opt.state_dict(),
                config.OptimizerConfig(name=name, lr=0.0003),
                step=0,
            ),
        }
    if layouts["hybrid_adamw"] != layouts["muon"]:
        raise ValueError("AdamW and Muon loaded-model routes differ")
    with safe_open(base / "model.safetensors", framework="pt", device="cpu") as store:
        for key in store.keys():
            if not torch.equal(
                parameters["0.model." + key].detach(), store.get_tensor(key).float()
            ):
                raise ValueError("Optimizer construction changed a model weight")
    for name, expected in model_files.items():
        verify(base / name, expected)
    for row in files.values():
        verify(row["path"], row)
    result = {
        "scope": "actual_pretrained_denseon_corrected_factory_reset_routing_only",
        "complete": True,
        "source_files": files,
        "diagnostic_source": {"path": str(Path(__file__).resolve()), **identity(__file__)},
        "input_receipt": {"path": str(inputs_path), **identity(inputs_path)},
        "model_files": model_files,
        "model_parameters": sum(p.numel() for p in parameters.values()),
        "model_tensors": len(parameters),
        "loaded_model_weights_equal_before_and_after": True,
        "routing": layouts["muon"],
        "resets": resets,
        "optimizer_updates_executed": False,
        "formal_training_executed": False,
        "scientific_completion": False,
        "training_candidate_whole_suite_admitted": False,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), **identity(args.output), "complete": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
