"""Exercise the actual factory on two genuine models, without GPU/calibration/training.

The run recipe metadata uses explicitly synthetic hidden rates. It is not the
public prepare_run calibration chain and cannot admit any formal branch.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Require explicitly hidden CUDA and offline loading")
    if args.output.exists():
        raise ValueError("Preserve earlier audit attempts")
    import torch

    from embed_optim import factorial_v3_factory as factory
    from embed_optim import factorial_v3_run_contract as contract
    from embed_optim.primary_contract import digest, file_identity

    if Path(factory.__file__).resolve().parent != args.source.resolve() / "src/embed_optim":
        raise ValueError("The actual factory import differs from the recorded assembly")
    sys.path.insert(0, str(args.repository / "tests"))
    from test_factorial_v3_run_contract import metadata_identity

    if torch.cuda.is_initialized():
        raise ValueError("No CUDA context may exist in this CPU loading diagnostic")
    records, plans = [], []
    for state in ("adamw_state", "muon_state"):
        identity = metadata_identity(state)
        sources = contract.source_identity(args.repository, args.source)
        source = Path(identity["source"]["checkpoint"])
        before = file_identity(source / "model.safetensors")
        for operator in ("adamw", "muon"):
            for seed in (314159, 271828, 161803):
                synthetic = metadata_identity(state, operator, seed)
                config, arguments = factory.recipe(
                    synthetic,
                    Path("/root/embedding-optimizer-study/data") / factory.BRANCH,
                    args.output.parent / "not-created-training-output",
                    project="factory-loading-diagnostic",
                    entity="local-only-no-logging",
                )
                plans.append(
                    {
                        "state": state,
                        "operator": operator,
                        "seed": seed,
                        "synthetic_rate_not_calibration": True,
                        "config": config.as_dict(),
                        "arguments": arguments,
                    }
                )
        config, _ = factory.recipe(
            identity,
            Path("/root/embedding-optimizer-study/data") / factory.BRANCH,
            args.output.parent / "not-created-training-output",
            project="factory-loading-diagnostic",
            entity="local-only-no-logging",
        )
        model, loss, observed = factory.load_model(identity, config, diagnostic_cpu=True)
        controls = []

        def reject_attribute(target, name, wrong):
            old = getattr(target, name)
            try:
                setattr(target, name, wrong)
                try:
                    factory.model_configuration(model, source, diagnostic_cpu=True)
                except ValueError:
                    controls.append({"attribute": name, "rejected": True})
                else:
                    raise AssertionError(f"Configuration mutation was accepted: {name}")
            finally:
                setattr(target, name, old)

        reject_attribute(model[0].model.config, "local_attention", 64)
        reject_attribute(model[0].model.config, "attention_dropout", 0.1)
        reject_attribute(model[0].model.config, "layer_norm_eps", 0.001)
        reject_attribute(model[1], "pooling_mode", "mean")
        reject_attribute(model, "prompts", {"query": "different: ", "document": "document: "})
        reject_attribute(model, "truncate_dim", 128)
        reject_attribute(model, "training", False)
        restored = factory.model_configuration(model, source, diagnostic_cpu=True)
        if observed != restored:
            raise ValueError("Controlled changes did not exactly restore loaded configuration")
        if before != file_identity(source / "model.safetensors"):
            raise ValueError("Original source weight file changed")
        if loss.model is not model or loss.temperature != 0.02:
            raise ValueError("Loaded actual loss differs")
        records.append(
            {
                "state": state,
                "source_checkpoint": str(source),
                "source_weights": before,
                "loaded_tensor_count": len(list(model.parameters())),
                "loaded_parameter_count": sum(p.numel() for p in model.parameters()),
                "every_saved_tensor_bitwise_equal": True,
                "configuration": observed,
                "configuration_controls": controls,
                "original_source_weights_unchanged": True,
                "synthetic_rate_and_data_metadata_not_run_admission": True,
                "source_files": sources,
            }
        )
        del model, loss
        gc.collect()
    if torch.cuda.is_initialized():
        raise ValueError("The factory initialized CUDA in a CPU-only diagnostic")
    receipt = {
        "scope": "dense-v3-factorial-actual-factory-loading-diagnostic-v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_admission": False,
        "execution_authorized": False,
        "gpu_execution": False,
        "forward_backward_or_training": False,
        "calibration_outputs_consumed": False,
        "formal_trainer_constructed": False,
        "actual_cpu_source_loads": records,
        "synthetic_rate_recipe_grid": plans,
        "factory_source": file_identity(factory.__file__),
        "worker_source": file_identity(__file__),
        "test_metadata_source": file_identity(
            args.repository / "tests/test_factorial_v3_run_contract.py"
        ),
        "run_component_sources_sha256": digest(records[0]["source_files"]),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "actual_cpu_model_loads": 2,
                "synthetic_recipe_cells": len(plans),
                "configuration_counterexamples": sum(
                    len(r["configuration_controls"]) for r in records
                ),
                "gpu_execution": False,
                "receipt": file_identity(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
