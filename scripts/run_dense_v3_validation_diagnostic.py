"""One-GPU diagnostic score comparison, never primary validation or recipe selection."""

import argparse
import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import (
    digest,
    file_identity,
    inspect_sealed_checkpoint,
    read_json,
    require_same,
    verify_file,
)
from embed_optim.primary_v3_validation import SETTINGS, row_identity, score_rows
from embed_optim.primary_v3_validation_io import save_scoring, write_new


def weights_identity(model):
    result = {}
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy()
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if len(result) != 134:
        raise ValueError("Diagnostic model parameter count differs from DenseOn")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise ValueError("Require exactly the parent-leased diagnostic device")
    if file_identity(args.plan)["sha256"] != args.plan_sha256:
        raise ValueError("The declared diagnostic plan differs")
    plan = read_json(args.plan)
    root, work = Path(plan["repository"]), args.plan.parent
    for row in plan["source_bindings"] + plan["input_bindings"]:
        verify_file(row["path"], row)
    require_same(plan["settings"], SETTINGS)
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer

    from embed_optim.collators import DenseGroupCollator
    from embed_optim.corrected_input_execution import require_independently_padded_dense
    from embed_optim.functional_intervention import group_scores, score_metrics
    from embed_optim.gradient_probe import _collect_features
    from embed_optim.runtime import verify_runtime_spec

    torch.set_num_threads(2)
    if torch.cuda.device_count() != 1:
        raise ValueError("A diagnostic worker must see one leased GPU")
    verify_runtime_spec(root / "configs/formal_runtime.json")
    dataset = Dataset.load_from_disk(str(work / "dataset"))
    identities = [row_identity(dataset[i], i) for i in range(len(dataset))]
    require_same(identities, plan["selection"]["row_identities"])
    records = []
    for item in plan["models"]:
        checkpoint = Path(item["checkpoint"])
        expected = read_json(checkpoint / "dense_run_contract.json")
        checked = inspect_sealed_checkpoint(checkpoint, expected, 3)
        require_same(checked, item["checked"])
        torch.cuda.reset_peak_memory_stats()
        model = SentenceTransformer(
            str(checkpoint),
            trust_remote_code=True,
            model_kwargs={"dtype": torch.float32, "attn_implementation": "flash_attention_2"},
        )
        model.max_seq_length = 8192
        require_independently_padded_dense(model)
        model.to("cuda:0").eval()
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Diagnostic model parameters must retain float32")
        before = weights_identity(model)
        scored = score_rows(
            model, dataset, device="cuda:0", batch_size=16, forward_dtype="bfloat16"
        )
        collator = DenseGroupCollator(model.preprocess)
        old_records = []
        with torch.inference_mode():
            for start in range(0, len(dataset), 16):
                rows = [dataset[i] for i in range(start, min(start + 16, len(dataset)))]
                features = _collect_features(collator(rows), "cuda:0")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    old_scores = group_scores(model, features, "dense")
                old_metrics = score_metrics(old_scores.float(), 0.02)
                for offset, row in enumerate(rows):
                    old_records.append(
                        {
                            "row": row_identity(row, start + offset),
                            "scores": old_scores.float()[offset].cpu().tolist(),
                            "metrics": {k: float(v[offset]) for k, v in old_metrics.items()},
                        }
                    )
        require_same(scored, old_records)
        after = weights_identity(model)
        require_same(before, after)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        output_plan = {
            "scope": "engineering_dense_v3_validation_diagnostic",
            "algorithm": item["algorithm"],
            "actual_diagnostic_run_identity_sha256": digest(expected),
            "actual_diagnostic_step": 3,
            "diagnostic_plan_sha256": args.plan_sha256,
            "selection_sha256": digest(plan["selection"]),
            "scientific_completion": False,
            "primary_validation": False,
        }
        output = work / "scores" / item["algorithm"]
        saved = save_scoring(output, output_plan, scored, identities)
        require_same(inspect_sealed_checkpoint(checkpoint, expected, 3), checked)
        records.append(
            {
                "algorithm": item["algorithm"],
                "rows": len(scored),
                "all_scores_and_metrics_equal_unchanged_scorer": True,
                "all_134_parameter_values_unchanged": True,
                "parameter_identity_sha256": digest(before),
                "saved": saved,
                "output": str(output),
                "plan": output_plan,
                "gpu": gpu,
            }
        )
        del collator, model
        torch.cuda.empty_cache()
        print(
            {
                "completed_diagnostic": item["algorithm"],
                "rows": len(scored),
                "unchanged_scorer_match": True,
            },
            flush=True,
        )
    for row in plan["source_bindings"] + plan["input_bindings"]:
        verify_file(row["path"], row)
    value = {
        "scope": "engineering_dense_v3_validation_gpu_comparison",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "records": records,
        "scoring_comparison_passed": True,
        "model_updates": 0,
        "primary_validation_executed": False,
        "scientific_completion": False,
        "formal_replication_ready": False,
    }
    write_new(work / "worker-result.json", value)
    print({"diagnostic_models": len(records), "scoring_comparison_passed": True}, flush=True)


if __name__ == "__main__":
    main()
