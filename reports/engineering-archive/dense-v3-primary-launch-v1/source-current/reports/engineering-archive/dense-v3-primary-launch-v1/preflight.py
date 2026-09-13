"""Read-only assembled-source/runtime/real-input primary preflight, no GPU work."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def run(root, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or output.exists():
        raise ValueError("Use CPU-only inspection and a new receipt path")
    from embed_optim import dense_numerical_contract as numerical
    from embed_optim import dense_run_contract as run_contract
    from embed_optim import train
    from embed_optim.config import RunConfig
    from embed_optim.primary_contract import (
        digest,
        file_identity,
        read_json,
        require_same,
        verify_file,
    )
    from embed_optim.primary_v3_contract import PrimaryV3Contract
    from embed_optim.runtime import verify_runtime_spec

    assembly = read_json(root / "source-assembly.json")
    for name, binding in assembly["files"].items():
        verify_file(root / name, binding["identity"])
    contract = PrimaryV3Contract.load(root / "configs/dense_primary_v3_protocol.json", root, root)
    runtime = verify_runtime_spec(root / "configs/formal_runtime.json")
    numerical.verify_stack(train.OptimizerTrainer)
    data_root = Path(contract.inputs["data_linkage_audit"]["dataset_path"])
    data_observation = contract.dataset(data_root, "training")
    amendment = read_json(root / contract.payload["data_amendment"]["path"])
    validation = contract.dataset(Path(amendment["datasets"]["validation"]["root"]), "validation")
    # Pure argument construction reads WORLD_SIZE. No TrainingArguments instance,
    # process group, model, optimizer or W&B client is created by this diagnostic.
    os.environ["WORLD_SIZE"] = "4"
    records = []
    common = None
    for row in contract.inputs["runs"]:
        expected = contract.expected_identity(row["run_id"])
        config = RunConfig.from_dict(
            {
                **expected["recipe"],
                "dataset_path": str(data_root),
                "output_root": str(output.parent / "uncreated-preflight-output"),
                "wandb_project": contract.payload["wandb"]["project"],
                "wandb_entity": contract.payload["wandb"]["entity"],
            }
        )
        arguments = train._training_argument_values(config)
        if common is None:
            observed, dataset = run_contract.prepare(config, arguments, 4)
            common = {
                k: observed[k] for k in ("schema_version", "scope", "source", "data", "model")
            }
            if (
                dataset._fingerprint
                != contract.inputs["data_linkage_audit"]["training_view_fingerprint"]
            ):
                raise ValueError("Actual Trainer-selected dataset fingerprint differs")
        else:
            # Data, base and source are shared and actually read above; do not
            # rehash the same multi-GB inputs twelve times. Each recipe and its
            # numerical/execution identities are independently reconstructed.
            observed = {
                **common,
                "recipe": run_contract.recipe_identity(config),
                "execution": run_contract.execution_identity(arguments, 4),
                "numerical_contract": numerical.receipt(
                    config.optimizer, arguments["gradient_accumulation_steps"]
                ),
            }
        require_same(observed, expected)
        if arguments["max_steps"] != -1 or config.epochs != 1.0:
            raise ValueError("Primary horizon override is not permitted")
        records.append(
            {
                "run_id": row["run_id"],
                "identity_sha256": digest(observed),
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "full_horizon": True,
                "batch_size": config.global_batch_size,
            }
        )
    imports = []
    for name, module in sorted(sys.modules.items()):
        if name == "embed_optim" or name.startswith("embed_optim."):
            source = Path(module.__file__).resolve()
            relative = source.relative_to(root).as_posix()
            if relative not in assembly["files"]:
                raise ValueError(f"Undeclared or foreign assembled package import: {name}")
            verify_file(source, assembly["files"][relative]["identity"])
            imports.append({"module": name, "path": relative, **file_identity(source)})
    for name, binding in assembly["files"].items():
        verify_file(root / name, binding["identity"])
    result = {
        "scope": "primary-v3-assembled-read-only-preflight",
        "read_only_preflight_passed": True,
        "formal_training_started": False,
        "scientific_completion": False,
        "source_assembly": file_identity(root / "source-assembly.json"),
        "protocol": file_identity(contract.path),
        "runtime": runtime,
        "actual_package_imports": imports,
        "training_data": data_observation,
        "validation_data": validation,
        "trainer_selected_fingerprint": dataset._fingerprint,
        "initial_model": common["model"],
        "runs": records,
        "shared_input_reading": "One actual prepared Trainer input read; twelve independently reconstructed recipe/execution/numerical identities.",
        "remaining": [
            "owner-approved local training-code commit and reviewed training-only execution release",
            "exact assembled source freeze, new data/output namespace and exclusive GPU leases",
            "formal full-horizon jobs and durable checkpoint verification",
        ],
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "read_only_preflight_passed": True,
                "runs": len(records),
                "imports": len(imports),
                "receipt": str(output),
                **file_identity(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())
