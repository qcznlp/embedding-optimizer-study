"""CPU-only pinned NorMuon reference comparison, including small gradients."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import torch

from embed_optim import optimizers

URL = "https://raw.githubusercontent.com/zichongli5/NorMuon/c6989a8354730695d9f5a9faa6c55eeb24865209/normuon.py"
SHA256 = "706c1a35fb35342f6ff207f8310b814a1c8f55c6ffba0e843537db434a696141"
FUNCTIONS = ("zeropower_via_newtonschulz5", "normuon_update")
SCALES = (1.0, 1e-3, 1e-5, 1e-7, 1e-9, 0.0)
SHAPES = ((8, 4), (4, 8), (8, 8), (64, 32))
TOLERANCES = {"atol": 1e-5, "rtol": 1.3e-6}


def reference_functions(raw, clamp_control=False):
    """Execute only two inspected functions after exact source authentication."""
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Pinned upstream source digest mismatch")
    nodes = [
        n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS
    ]
    if tuple(n.name for n in nodes) != FUNCTIONS:
        raise ValueError("Pinned upstream function topology mismatch")
    if clamp_control:
        expected = ast.parse("X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)").body[0]
        replacement = ast.parse("X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)").body[
            0
        ]
        matches = [i for i, n in enumerate(nodes[0].body) if ast.dump(n) == ast.dump(expected)]
        if len(matches) != 1:
            raise ValueError("Expected the exact single upstream normalization statement")
        nodes[0].body[matches[0]] = replacement
    namespace = {"torch": torch}
    selected = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(selected, URL, "exec"), namespace)
    return namespace["normuon_update"]


def run_case(reference, control, shape, scale):
    generator = torch.Generator().manual_seed(20260906)
    local_m, local_v = torch.zeros(shape), torch.zeros(shape[0], 1)
    ref_m, ref_v = local_m.clone(), local_v.clone()
    ctrl_m, ctrl_v = local_m.clone(), local_v.clone()
    records = []
    for step in range(4):
        gradient = torch.randn(shape, generator=generator) * scale
        local = optimizers._normuon_update(gradient.clone(), local_m, local_v, 0.95, 0.95, 5)
        expected = reference(gradient.clone(), ref_m, ref_v, beta=0.95, beta2=0.95, ns_steps=5)
        paired = control(gradient.clone(), ctrl_m, ctrl_v, beta=0.95, beta2=0.95, ns_steps=5)
        if any(not torch.isfinite(v).all() for v in (local, expected, paired)):
            raise ValueError("Non-finite diagnostic update")
        for left, right in (
            (local_m, ref_m),
            (local_m, ctrl_m),
            (local_v, ctrl_v),
            (local, paired),
        ):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        error = local - expected
        records.append(
            {
                "step": step + 1,
                "matches_pinned_reference": torch.allclose(local, expected, **TOLERANCES),
                "matches_clamp_only_control_exactly": True,
                "max_absolute_update_error": float(error.abs().max()),
                "relative_l2_update_error": float(error.norm() / expected.norm().clamp_min(1e-30)),
                "reference_update_norm": float(expected.norm()),
                "local_update_norm": float(local.norm()),
                "momentum_matches_reference_exactly": torch.equal(local_m, ref_m),
                "max_row_second_moment_error": float((local_v - ref_v).abs().max()),
            }
        )
    return {"shape": shape, "element_gradient_scale": scale, "steps": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require explicitly hidden GPUs")
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Require a new output file in an existing diagnostic directory")
    torch.set_num_threads(1)
    raw = urllib.request.urlopen(URL, timeout=30).read()
    reference, control = reference_functions(raw), reference_functions(raw, clamp_control=True)
    cases = [run_case(reference, control, shape, scale) for scale in SCALES for shape in SHAPES]
    failed = sum(not r["matches_pinned_reference"] for c in cases for r in c["steps"])
    source = Path(inspect.getsourcefile(optimizers)).resolve()
    report = {
        "scope": "engineering_normuon_pinned_reference_small_gradient_fidelity",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "audit_executed_completely": True,
        "conforms_to_pinned_reference_over_tested_envelope": failed == 0,
        "scientific_completion": False,
        "runtime_deployed": False,
        "tested_updates": 96,
        "nonconforming_updates": failed,
        "tolerances": TOLERANCES,
        "upstream": {"url": URL, "sha256": SHA256, "bytes": len(raw)},
        "source_bindings": [
            {
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in (Path(__file__).resolve(), source)
        ],
        "difference": "Local Newton-Schulz uses norm.clamp_min(1e-7); pinned official NorMuon uses norm + 1e-7. Changing just that statement in an isolated upstream namespace yields bit-identical local updates and states for all cases.",
        "boundary": "Four small synthetic matrix shapes, six gradient scales, four successive updates from zero optimizer state; CPU FP32 state/BF16 polynomial. Not actual primary gradients, GPU equivalence, performance impact, a model-quality result, or evidence that formal runs encountered this envelope. No production source changed. Audit execution success is not reference-conformance success.",
        "cases": cases,
    }
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "tested_updates": 96,
                "nonconforming_updates": failed,
                "conforms": failed == 0,
            }
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
