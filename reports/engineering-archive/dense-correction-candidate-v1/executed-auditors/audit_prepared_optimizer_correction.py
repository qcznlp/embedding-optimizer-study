"""Authenticate and test the preparation-only NorMuon correction, CPU or GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch

from embed_optim import optimizers
from embed_optim.config import NORMUON_NS_IMPLEMENTATION, OptimizerConfig
from scripts import audit_normuon_reference_edges as upstream
from scripts.audit_dense_gpu_normalization import require_handoff
from scripts.audit_dense_gpu_replay_update import identity

LEGACY = Path("/root/embedding-optimizer-story-refactor/src/embed_optim/optimizers.py")
LEGACY_SHA = "12158eee448b4a7cc4ae5d4dd9e2b3492f52e452b0cb7ed4b2a96e996e71d3a6"


def legacy_functions():
    if identity(LEGACY)["sha256"] != LEGACY_SHA:
        raise ValueError("Original optimizer evidence source changed")
    spec = importlib.util.spec_from_file_location("embed_optim._audited_legacy_optimizer", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(shape, scale, device, reference, legacy):
    generator = torch.Generator(device=device).manual_seed(20260906)
    parameter = torch.nn.Parameter(torch.randn(shape, generator=generator, device=device))
    model = torch.nn.Module()
    model.layers = torch.nn.ParameterList([parameter])
    config = OptimizerConfig(name="normuon", lr=3e-4, ns_implementation=NORMUON_NS_IMPLEMENTATION)
    optimizer, partition = optimizers.build_optimizer(model, config)
    if partition["hidden"]["tensors"] != 1 or len(optimizer.param_groups) != 1:
        raise ValueError("Wrong actual optimizer factory dispatch")
    expected_weight = parameter.detach().clone()
    ref_m, ref_v = torch.zeros_like(parameter), torch.zeros_like(parameter[..., :1])
    legacy_m, legacy_v = ref_m.clone(), ref_v.clone()
    muon_m, old_muon_m = ref_m.clone(), ref_m.clone()
    records = []
    for step in range(4):
        gradient = torch.randn(shape, generator=generator, device=device) * scale
        parameter.grad = gradient.clone()
        official = reference(gradient.clone(), ref_m, ref_v, beta=0.95, beta2=0.95, ns_steps=5)
        old = legacy._normuon_update(gradient.clone(), legacy_m, legacy_v, 0.95, 0.95, 5)
        optimizer.step()
        expected_weight.mul_(1 - config.lr * config.weight_decay).add_(official, alpha=-config.lr)
        state = optimizer.state[parameter]
        checked = (
            (parameter, expected_weight),
            (state["momentum_buffer"], ref_m),
            (state["second_moment"], ref_v),
        )
        for actual, expected in checked:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            if not torch.isfinite(actual).all():
                raise ValueError("Nonfinite optimizer state")
        current_muon = optimizers._muon_update(gradient.clone(), muon_m, 0.95, 5)
        previous_muon = legacy._muon_update(gradient.clone(), old_muon_m, 0.95, 5)
        torch.testing.assert_close(current_muon, previous_muon, atol=0, rtol=0)
        torch.testing.assert_close(muon_m, old_muon_m, atol=0, rtol=0)
        if not torch.equal(parameter.grad, gradient):
            raise ValueError("Candidate unexpectedly mutates incoming gradient")
        records.append(
            {
                "step": step + 1,
                "candidate_weight_and_state_bitwise_equal_to_official": True,
                "muon_update_and_state_bitwise_unchanged": True,
                "incoming_gradient_unchanged": True,
                "legacy_normuon_conforms_to_reference_tolerance": torch.allclose(
                    old, official, **upstream.TOLERANCES
                ),
                "legacy_normuon_max_absolute_update_error": float((old - official).abs().max()),
            }
        )
    return {"shape": list(shape), "scale": scale, "steps": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    root, output = args.candidate_root.resolve(), args.output.absolute()
    if Path(inspect.getsourcefile(optimizers)).resolve() != root / "src/embed_optim/optimizers.py":
        raise ValueError("Incorrect candidate import")
    if output.exists() or not output.parent.is_dir() or output.parent.parent != Path("/tmp"):
        raise ValueError("Require a new temporary diagnostic output")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ("" if args.device == "cpu" else "0"):
        raise ValueError("Require the explicit CPU or single owned diagnostic GPU visibility")
    handoff = (
        require_handoff(Path("/tmp/dense-gpu-handoff.4MuYjC/receipt/result.json"))
        if args.device == "cuda"
        else None
    )
    torch.set_num_threads(2)
    source_before = [
        identity(root / f"src/embed_optim/{name}.py")
        for name in ("config", "optimizers", "train", "dense_numerical_contract")
    ]
    raw = urllib.request.urlopen(upstream.URL, timeout=30).read()
    reference = upstream.reference_functions(raw)
    legacy = legacy_functions()
    records = [
        run_case(shape, scale, torch.device(args.device), reference, legacy)
        for scale in upstream.SCALES
        for shape in upstream.SHAPES
    ]
    if [identity(Path(x["path"])) for x in source_before] != source_before or identity(LEGACY)[
        "sha256"
    ] != LEGACY_SHA:
        raise ValueError("Numerical source changed during audit")
    if args.device == "cuda":
        require_handoff(Path(handoff["path"]))
    report = {
        "scope": "engineering_prepared_optimizer_reference_correction",
        "scientific_completion": False,
        "production_deployed": False,
        "audit_execution_complete": True,
        "candidate_reference_conformance_passed": True,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "tested_updates": 96,
        "candidate_exact_weight_state_checks": 96,
        "muon_unchanged_update_state_checks": 96,
        "legacy_nonconforming_updates": sum(
            not x["legacy_normuon_conforms_to_reference_tolerance"]
            for c in records
            for x in c["steps"]
        ),
        "source_bindings": source_before + [identity(Path(__file__).resolve()), identity(LEGACY)],
        "upstream": {
            "url": upstream.URL,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        },
        "handoff": handoff,
        "records": records,
        "boundary": "Four fixed small matrix shapes, six scales and four consecutive updates. Actual candidate factory/operator branch and weight decay versus the authenticated official functions; unchanged legacy Muon branch compared bitwise. CPU and GPU are distinct arithmetic checks, not model-quality or full-model gradient claims. No primary checkpoint or source modified.",
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "device",
                    "tested_updates",
                    "candidate_reference_conformance_passed",
                    "legacy_nonconforming_updates",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
