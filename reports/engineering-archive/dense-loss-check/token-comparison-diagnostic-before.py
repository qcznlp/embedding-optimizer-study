"""CPU engineering check of real Dense loss gradients and MTEB encoding semantics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from embed_optim.collators import TEXT_COLUMNS, DenseGroupCollator
from embed_optim.config import load_matrix
from embed_optim.incremental_checkpoint_backup import (
    local_checkpoint_inventory,
    stat_signature,
    validate_sealed_checkpoint,
)
from embed_optim.losses import ExplicitDenseInfoNCELoss

# Declared before real checkpoint execution. These cover CPU FP32 rounding,
# not BF16/FlashAttention equivalence or distributed gradient accumulation.
TOLERANCES = {
    "loss_atol": 2e-5,
    "loss_rtol": 1e-5,
    "gradient_atol": 3e-6,
    "gradient_rtol": 3e-4,
    "encoding_atol": 2e-5,
    "encoding_rtol": 2e-4,
}


def reference_scores(embeddings, temperature):
    """Independent row-wise float64 cosine and log-sum-exp reference."""
    if len(embeddings) != 9 or temperature <= 0:
        raise ValueError("Require exactly nine columns and positive temperature")
    normalized = [
        x.double() / x.double().square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        for x in embeddings
    ]
    scores = (
        torch.stack(
            [
                torch.stack([(normalized[0][i] * doc[i]).sum() for doc in normalized[1:]])
                for i in range(normalized[0].shape[0])
            ]
        )
        / temperature
    )
    losses = torch.logsumexp(scores, dim=1) - scores[:, 0]
    return scores, losses


def check_loss_graph(model, features, temperature, loss_type=ExplicitDenseInfoNCELoss):
    captured = []

    def observe(_module, _inputs, output):
        captured.append(output["sentence_embedding"])

    with model.register_forward_hook(observe):
        actual = loss_type(model, temperature=temperature)(copy.deepcopy(features))
    scores, losses = reference_scores(captured, temperature)
    reference = losses.mean()
    torch.testing.assert_close(
        actual.double(), reference, atol=TOLERANCES["loss_atol"], rtol=TOLERANCES["loss_rtol"]
    )
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("The forward/backward audit requires trainable parameters")
    tensors = tuple(p for _, p in parameters)
    actual_grad = torch.autograd.grad(actual, tensors, retain_graph=True)
    reference_grad = torch.autograd.grad(reference, tensors)
    gradient_errors = []
    for (name, _), left, right in zip(parameters, actual_grad, reference_grad, strict=True):
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError(f"Non-finite gradient: {name}")
        torch.testing.assert_close(
            left,
            right,
            atol=TOLERANCES["gradient_atol"],
            rtol=TOLERANCES["gradient_rtol"],
            msg=lambda m: f"{name}: {m}",
        )
        gradient_errors.append(
            {
                "name": name,
                "elements": left.numel(),
                "max_absolute_error": float((left - right).abs().max()),
            }
        )
    return {
        "complete_for_loss_graph": True,
        "query_count": scores.shape[0],
        "candidate_count_per_query": scores.shape[1],
        "loss_absolute_error": float((actual.double() - reference).abs().detach()),
        "parameter_gradients_compared": len(parameters),
        "gradient_elements_compared": sum(x["elements"] for x in gradient_errors),
        "gradient_errors": gradient_errors,
    }, [x.detach() for x in captured]


def synthetic_rows():
    queries = ["How do plants convert sunlight into energy?", "When does water freeze?"]
    positives = [
        "Plants convert sunlight through photosynthesis.",
        "Pure water freezes at zero degrees Celsius at ordinary pressure.",
    ]
    negatives = [
        "The museum opens on Tuesday.",
        "Saturn has rings made of ice.",
        "A violin has four strings.",
        "The path crosses a wooden bridge.",
        "Copper conducts electricity.",
        "An atlas contains maps of many countries.",
        "The train arrived after the rain ended.",
    ]
    return [
        {
            "query": q,
            "positive": pos,
            **{
                f"negative_{j}": f"{text} {'Details follow. ' * (i + j % 3)}".strip()
                for j, text in enumerate(negatives)
            },
        }
        for i, (q, pos) in enumerate(zip(queries, positives, strict=True))
    ]


def check_encoding(model, rows, captured, setup_budget_encode):
    from datasets import Dataset
    from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
    from mteb.types import PromptType
    from torch.utils.data import DataLoader

    wrapper = SentenceTransformerEncoderWrapper(model)
    # Force multiple length-ordered chunks in this tiny engineering fixture.
    # This is not a change to the formal evaluator's character budget.
    setup_budget_encode(model, char_budget=140)
    metadata = SimpleNamespace(name="EngineeringDenseContract", type="Retrieval")
    values = {}
    for kind, texts, expected in [
        (PromptType.query, [row["query"] for row in rows], captured[0]),
        (
            PromptType.document,
            [row[col] for col in TEXT_COLUMNS[1:] for row in rows],
            torch.cat(captured[1:]),
        ),
    ]:
        loader = DataLoader(Dataset.from_dict({"text": texts}), batch_size=3, shuffle=False)
        encoded = torch.as_tensor(
            wrapper.encode(
                loader,
                task_metadata=metadata,
                hf_split="engineering",
                hf_subset="synthetic",
                prompt_type=kind,
                show_progress_bar=False,
                convert_to_tensor=True,
            )
        )
        normalized = torch.nn.functional.normalize(encoded, dim=-1)
        target = torch.nn.functional.normalize(expected, dim=-1)
        torch.testing.assert_close(
            normalized, target, atol=TOLERANCES["encoding_atol"], rtol=TOLERANCES["encoding_rtol"]
        )
        values[kind.value] = {
            "embeddings": encoded,
            "target": target,
            "maximum_error": float((normalized - target).abs().max()),
        }
    actual = torch.as_tensor(
        wrapper.similarity(values["query"]["embeddings"], values["document"]["embeddings"])
    )
    expected = values["query"]["target"] @ values["document"]["target"].T
    torch.testing.assert_close(
        actual, expected, atol=TOLERANCES["encoding_atol"], rtol=TOLERANCES["encoding_rtol"]
    )
    return {
        "complete_for_cpu_mteb_encoding": True,
        "query_vectors": len(rows),
        "document_vectors": 8 * len(rows),
        "engineering_char_budget": 140,
        "normalized_query_max_error": values["query"]["maximum_error"],
        "normalized_document_max_error": values["document"]["maximum_error"],
        "cosine_score_max_error": float((actual - expected).abs().max()),
    }


def identity(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=2345)
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=["padded-adamw-3e-5", "padded-muon-3e-4", "padded-normuon-3e-4"],
    )
    args = parser.parse_args()
    root, output = args.repository.resolve(), args.output.resolve()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise ValueError("Require hidden GPUs and offline model loading")
    if output.exists() or output.is_relative_to(root / "outputs"):
        raise ValueError("Use a new engineering report outside checkpoint outputs")
    for function, relative in [
        (ExplicitDenseInfoNCELoss, "src/embed_optim/losses.py"),
        (DenseGroupCollator, "src/embed_optim/collators.py"),
    ]:
        if Path(inspect.getsourcefile(function)).resolve() != root / relative:
            raise ValueError("Set PYTHONPATH to the audited checkout's absolute src")
    os.chdir(root)
    torch.set_num_threads(2)
    sys.path.insert(0, str(root / "scripts/eval"))
    from dense_sequential import setup_st_forward_compat

    setup_st_forward_compat()
    from dense_no_packing_parallel import _load_padded_model
    from dense_parallel import setup_budget_encode
    from mteb.models import sentence_transformer_wrapper
    from sentence_transformers import SentenceTransformerTrainer

    configs = {c.run_id: c for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")}
    if len(set(args.run_ids)) != len(args.run_ids) or set(args.run_ids) - set(configs):
        raise ValueError("Unknown or duplicate primary run")
    rows = synthetic_rows()
    records = []
    for run_id in args.run_ids:
        config = configs[run_id]
        checkpoint = validate_sealed_checkpoint(config, args.step)
        before = stat_signature(checkpoint)
        inventory = local_checkpoint_inventory(checkpoint)
        model = _load_padded_model(str(checkpoint), bf16=False, fa2=False, local=True)
        model.eval()
        if (
            model.max_seq_length != config.max_length
            or model.prompts != {"query": "query: ", "document": "document: "}
            or model.similarity_fn_name != "cosine"
            or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())
        ):
            raise ValueError("Unexpected CPU dtype, context, prompts or similarity")
        batch = DenseGroupCollator(model.preprocess)(rows)
        features, labels = SentenceTransformerTrainer.collect_features(None, batch)
        if labels is not None or len(features) != 9:
            raise ValueError("Unexpected trainer feature extraction")
        for column, feature in zip(TEXT_COLUMNS, features, strict=True):
            prefix = "query: " if column == "query" else "document: "
            manual = model.preprocess([prefix + row[column] for row in rows])
            if feature["input_ids"] != manual["input_ids"]:
                raise ValueError(f"Column order, tokenization or prompt differs: {column}")
        checks, captured = check_loss_graph(model, features, config.resolved_temperature)
        encoding = check_encoding(model, rows, captured, setup_budget_encode)
        if before != stat_signature(checkpoint) or inventory != local_checkpoint_inventory(
            checkpoint
        ):
            raise ValueError("Source checkpoint changed during diagnostic")
        records.append(
            {
                "run_id": run_id,
                "checkpoint": str(checkpoint),
                "checkpoint_inventory": inventory,
                "temperature": config.resolved_temperature,
                "max_length": model.max_seq_length,
                "tokenized_columns": list(TEXT_COLUMNS),
                "loss_and_gradients": checks,
                "mteb_encoding": encoding,
            }
        )
        print(f"Verified CPU loss, gradients and MTEB encoding: {run_id}", flush=True)
        del model, captured
    source_paths = [
        Path(__file__).resolve(),
        root / "src/embed_optim/losses.py",
        root / "src/embed_optim/collators.py",
        root / "scripts/eval/dense_sequential.py",
        root / "scripts/eval/dense_parallel.py",
        root / "scripts/eval/dense_no_packing_parallel.py",
        root / "src/embed_optim/corrected_input_execution.py",
        Path(inspect.getsourcefile(SentenceTransformerTrainer)),
        Path(sentence_transformer_wrapper.__file__),
    ]
    report = {
        "schema_version": 1,
        "scope": "engineering_cpu_dense_loss_and_encoding_contract",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scientific_completion": False,
        "complete_for_cpu_diagnostic": True,
        "runtime_deployed": False,
        "source_checkpoints_unchanged": True,
        "tolerances": TOLERANCES,
        "synthetic_rows": rows,
        "source_bindings": [identity(p) for p in source_paths],
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "sentence-transformers", "transformers", "mteb")
        },
        "records": records,
        "boundary": "Two short synthetic queries per real checkpoint, CPU float32, eval mode with autograd. Checks production collator/feature order, exact prompts, nine encoder calls/eight explicit candidates, float64-reference loss and all trainable gradients, actual MTEB wrapper and length-budget ordering/cosine scoring. No optimizer step, real evaluation corpus, training-mode dropout, distributed accumulation, BF16/FlashAttention equivalence or 8192-token forward pass is tested. These checks supply no retrieval or optimizer-quality result.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
