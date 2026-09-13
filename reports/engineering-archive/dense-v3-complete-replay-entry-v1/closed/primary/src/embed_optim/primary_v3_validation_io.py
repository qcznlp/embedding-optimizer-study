"""Full-v3 validation producer, saved-score reader and complete-grid recipe selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .primary_contract import canonical, digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_validation import (
    METRICS,
    SCOPE,
    SETTINGS,
    ValidationContract,
    score_rows,
    select_recipes,
    summarize_records,
)


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def job_plan(contract, run_root, run_id, data_root):
    dataset, data = contract.data(data_root)
    checked = contract.primary.complete_run(run_root, run_id)
    expected = contract.primary.expected_identity(run_id)
    if expected["recipe"]["temperature"] != SETTINGS["temperature"]:
        raise ValueError("Validation temperature differs from the actual primary recipe")
    plan = {
        "scope": SCOPE,
        "validation_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "run_id": run_id,
        "run_identity_sha256": checked["run_identity_sha256"],
        "checkpoint": checked["checkpoints"][-1],
        "dataset_content_sha256": data["content"]["content_sha256"],
        "row_identities_sha256": data["row_identities_sha256"],
        "settings": SETTINGS,
        "scientific_completion": False,
    }
    return dataset, data["row_identities"], plan


def read_record_file(path):
    # Retain the shared decoder's duplicate-key and finite-number policy for each line.
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Require an ordinary saved validation score file")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate validation record key")
            value[key] = item
        return value

    result = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("A saved validation record cannot be blank")
            row = json.loads(line, object_pairs_hook=unique)
            canonical(row)
            result.append(row)
    return result


def inspect_saved(output, plan, identities):
    """Content reader also usable with explicitly diagnostic expected plans."""
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Require an ordinary validation output directory")
    manifest = read_json(output / "manifest.json")
    if set(manifest) != {"status", "plan", "outputs"}:
        raise ValueError("Unexpected saved validation manifest fields")
    if manifest.get("status") != "complete" or set(manifest.get("outputs", {})) != {
        "sample_scores",
        "summary",
    }:
        raise ValueError("Incomplete saved validation result")
    require_same(manifest["plan"], plan)
    require_same(read_json(output / "admission.json"), plan)
    names = {"sample_scores": "sample_scores.jsonl", "summary": "summary.json"}
    for key, name in names.items():
        if manifest["outputs"][key].get("path") != name:
            raise ValueError("Validation output path differs")
        verify_file(output / name, manifest["outputs"][key])
    records = read_record_file(output / "sample_scores.jsonl")
    computed = summarize_records(records, identities)
    require_same(read_json(output / "summary.json"), computed)
    for key, name in names.items():
        verify_file(output / name, manifest["outputs"][key])
    if sorted(p.name for p in output.iterdir()) != sorted(
        ["manifest.json", "admission.json", *names.values()]
    ):
        raise ValueError("Unexpected or partial validation output inventory")
    return {
        "plan": plan,
        "summary": computed,
        "manifest": {
            "path": str(output / "manifest.json"),
            **file_identity(output / "manifest.json"),
        },
        "outputs": manifest["outputs"],
        "scientific_completion": False,
    }


def save_scoring(output, plan, records, identities):
    """Write a new complete bundle. Never overwrite a prior or partial calculation."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Use a new validation output directory")
    summary = summarize_records(records, identities)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "admission.json", plan)
    with (output / "sample_scores.jsonl").open("x") as stream:
        for row in records:
            stream.write(canonical(row).decode() + "\n")
    write_new(output / "summary.json", summary)
    outputs = {
        k: {"path": name, **file_identity(output / name)}
        for k, name in (("sample_scores", "sample_scores.jsonl"), ("summary", "summary.json"))
    }
    write_new(output / "manifest.json", {"status": "complete", "plan": plan, "outputs": outputs})
    return inspect_saved(output, plan, identities)


def inspect_validation(contract, run_root, run_id, data_root, output_root):
    _, identities, plan = job_plan(contract, run_root, run_id, data_root)
    return inspect_saved(Path(output_root) / digest(plan), plan, identities)


def execute(contract, run_root, run_id, data_root, output_root):
    contract = contract.require_execution()
    from .gpu_lease import acquire_gpu_lease, parse_gpu_tokens
    from .runtime import verify_runtime_spec

    tokens = parse_gpu_tokens(os.environ.get("CUDA_VISIBLE_DEVICES", ""), expected_count=1)
    if not set(tokens).issubset(
        {g for pool in contract.primary.payload["gpu_pools"] for g in pool}
    ):
        raise ValueError("Validation requested an undeclared GPU")
    verify_runtime_spec(contract.primary.repository / "configs/formal_runtime.json")
    dataset, identities, plan = job_plan(contract, run_root, run_id, data_root)
    output = Path(output_root) / digest(plan)
    if output.exists():
        return inspect_saved(output, plan, identities)
    with acquire_gpu_lease(
        tokens,
        lock_dir=contract.primary.payload["gpu_lease_root"],
        timeout_seconds=60,
        purpose=f"primary-v3-validation:{run_id}",
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        from .corrected_input_execution import require_independently_padded_dense

        checkpoint = Path(run_root) / f"checkpoint-{SETTINGS['checkpoint']}"
        model = SentenceTransformer(
            str(checkpoint),
            trust_remote_code=True,
            model_kwargs={"dtype": torch.float32, "attn_implementation": "flash_attention_2"},
        )
        model.max_seq_length = SETTINGS["max_length"]
        require_independently_padded_dense(model)
        model.to("cuda:0").eval()
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Validation model parameters are not float32")
        try:
            records = score_rows(
                model,
                dataset,
                device="cuda:0",
                batch_size=SETTINGS["batch_size"],
                forward_dtype=SETTINGS["forward_dtype"],
            )
        finally:
            del model
            torch.cuda.empty_cache()
    # Recheck actual complete-run/data content after forward computation, before publication.
    contract.require_execution()
    _, after_identities, after_plan = job_plan(contract, run_root, run_id, data_root)
    require_same(after_plan, plan)
    require_same(after_identities, identities)
    return save_scoring(output, plan, records, identities)


def collect_selection(contract, experiment_root, data_root, output_root):
    # All twelve full runs are required before any validation metric can select a recipe.
    primary = contract.primary
    roots = {
        r["run_id"]: Path(experiment_root) / primary.payload["output_root"] / "dense" / r["run_id"]
        for r in primary.inputs["runs"]
    }
    for run_id, root in roots.items():
        primary.complete_run(root, run_id)
    rows, receipts = [], []
    for run_id, root in roots.items():
        result = inspect_validation(contract, root, run_id, data_root, output_root)
        overall = [g for g in result["summary"]["groups"] if g["group"] == "__all__"]
        if len(overall) != 1 or overall[0]["samples"] != SETTINGS["rows"]:
            raise ValueError("Missing full-row validation aggregate")
        optimizer = primary.expected_identity(run_id)["recipe"]["optimizer"]
        rows.append(
            {
                "run_id": run_id,
                "optimizer": optimizer["name"],
                "learning_rate": optimizer["lr"],
                **{k: overall[0][k] for k in METRICS},
            }
        )
        receipts.append(result)
    return {
        "scope": "dense_primary_v3_validation_selection",
        "primary_protocol_sha256": primary.sha256,
        "validation_protocol_sha256": contract.sha256,
        "selected": select_recipes(rows, primary.inputs["runs"]),
        "run_metrics": rows,
        "source_receipts": receipts,
        "scientific_completion": False,
        "boundary": "Held-out selection only. No BEIR input; not retrieval inference or paper release.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "execute", "select"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary-protocol", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    primary = PrimaryV3Contract.load(args.primary_protocol, args.repository, args.training_root)
    contract = ValidationContract.load(
        args.protocol, primary, require_released=args.mode == "execute"
    )
    if args.mode == "select":
        if args.experiment_root is None:
            parser.error("select requires --experiment-root")
        value = collect_selection(contract, args.experiment_root, args.data_root, args.output_root)
    else:
        if args.run_root is None or args.run_id is None:
            parser.error("inspect/execute require --run-root and --run-id")
        fn = execute if args.mode == "execute" else inspect_validation
        value = fn(contract, args.run_root, args.run_id, args.data_root, args.output_root)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
