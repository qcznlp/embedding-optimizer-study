"""CPU-only full-DenseOn extension of the complete/tail repair-candidate audit.

Load only an independently downloaded, digest-authenticated checkpoint. Start fresh
diagnostic optimizers on synthetic data; this is not resumed primary training.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import inspect
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from sentence_transformers import SentenceTransformer

from embed_optim import train
from embed_optim.config import load_matrix
from embed_optim.corrected_input_execution import require_independently_padded_dense
from embed_optim.optimizers import parameter_partition
from scripts import audit_dense_trainer_normalization as small
from scripts.audit_checkpoint_download import select_download, verify_download
from scripts.dense_trainer_normalization_candidate import SingleNormalizationTrainerCandidate


def validate_model(model):
    parameters = list(model.named_parameters())
    if (
        len(parameters) != 134
        or model.get_sentence_embedding_dimension() != 768
        or model.max_seq_length != 8192
        or model.prompts != {"query": "query: ", "document": "document: "}
        or model[0].can_flatten_inputs is not False
        or model[0].auto_model.config._attn_implementation != "sdpa"
        or not model[0].auto_model.is_gradient_checkpointing
    ):
        raise ValueError("Unexpected full DenseOn configuration")
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for _, p in parameters):
        raise ValueError("Require CPU FP32 parameters throughout")
    if any(m.p != 0 for m in model.modules() if isinstance(m, torch.nn.Dropout)):
        raise ValueError("The deterministic full-batch reference requires zero dropout")
    partition = parameter_partition(model)
    if len(partition["hidden"]) != 88 or len(partition["aux_decay"]) != 1:
        raise ValueError("Unexpected full-model optimizer routing")
    if any(not torch.isfinite(p).all() for _, p in parameters):
        raise ValueError("Non-finite source-model parameter")
    return {
        "parameter_tensors": len(parameters),
        "parameter_elements": sum(p.numel() for _, p in parameters),
        "hidden_matrices": 88,
        "embedding_dimensions": 768,
        "configured_context_length": 8192,
        "maximum_context_execution_tested": False,
        "gradient_checkpointing": True,
        "dropout": 0.0,
        "parameter_dtype": "float32",
        "attention": "sdpa",
        "device": "cpu",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--source-run", default="padded-normuon-3e-4")
    parser.add_argument("--source-step", type=int, default=3126)
    args = parser.parse_args()
    required_env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WORLD_SIZE": "4",
    }
    if any(os.environ.get(k) != v for k, v in required_env.items()):
        raise ValueError("Require four CPU-only torchrun ranks and offline model loading")
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-full-normalization.")
        or not work.is_dir()
    ):
        raise ValueError("Use a fresh mktemp -d /tmp/dense-full-normalization.XXXXXX directory")
    download = args.download_root.absolute()
    if not download.is_relative_to(Path("/tmp")) or work.is_relative_to(download):
        raise ValueError("Use an independent temporary download and separate output")
    if Path(inspect.getsourcefile(train)).resolve() != root / "src/embed_optim/train.py":
        raise ValueError("PYTHONPATH does not select the declared audited source")
    torch.set_num_threads(2)
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo", timeout=timedelta(seconds=900))
    report = {
        "scope": "engineering_four_rank_cpu_full_denseon_normalization_candidate",
        "scientific_completion": False,
        "runtime_deployed": False,
        "complete_for_full_model_cpu_fixture": False,
        "records": [],
        "tolerances": small.previous.TOLERANCES,
        "boundary": "Real downloaded DenseOn weights, four CPU/Gloo ranks, FP32/SDPA, zero dropout, non-reentrant gradient checkpointing and fresh diagnostic AdamW/Muon/NorMuon optimizers. Each sees 288 distinct synthetic queries in global batches 128, 128, 32. Checks raw/clipped gradients and LR cadence through the prospective candidate. Does not resume a primary optimizer, validate GPU/BF16/FlashAttention/8192-token inputs/rank-RNG or produce retrieval/scientific results. Only the audit fixture loader is substituted; no production/library function is patched.",
    }
    before = None
    try:
        if rank == 0 and any(work.iterdir()):
            raise ValueError("Diagnostic work directory is not empty")
        selection = select_download(
            args.audit, args.audit_sha256, args.source_run, args.source_step
        )
        before = verify_download(selection, download)
        checkpoint = Path(before["checkpoint_root"])
        report["download_verification"] = before
        report["download_selection"] = {k: v for k, v in selection.items() if k != "files"}
        dist.barrier()

        def load_full(path):
            if Path(path).resolve() != checkpoint.resolve():
                raise ValueError("Refuse model loading outside the authenticated checkpoint")
            model = SentenceTransformer(
                str(checkpoint),
                device="cpu",
                local_files_only=True,
                model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
            )
            require_independently_padded_dense(model)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            report["model_configuration"] = validate_model(model)
            return model

        configs = {c.run_id: c for c in load_matrix(root / "configs/dense_no_packing_retrain.yaml")}
        # This replaces only the old diagnostic's tiny-fixture loader for the
        # duration of this call. Training_step, loss, backward and optimizers are
        # not monkeypatched; the repair remains the explicit candidate subclass.
        with patch.object(small.previous, "load_fixture", side_effect=load_full):
            for run_id in small.previous.RUN_IDS:
                if rank == 0:
                    print(f"Starting full-model CPU normalization: {run_id}", flush=True)
                record = small.run_case(
                    configs[run_id],
                    checkpoint,
                    work / run_id,
                    small.previous.synthetic_rows(small.ROW_COUNT),
                    rank,
                )
                if rank == 0:
                    report["records"].append(record)
                    with (work / f"{run_id}-verified.json").open("x") as handle:
                        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
                    print(f"Verified full-model CPU normalization: {run_id}", flush=True)
                gc.collect()
                dist.barrier()
        if verify_download(selection, download) != before:
            raise ValueError("Downloaded source payload changed during the audit")
        report["source_checkpoint_unchanged"] = True
        report["complete_for_full_model_cpu_fixture"] = True
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if rank == 0:
            objects = (
                small,
                small.previous,
                small.reference_scores,
                select_download,
                SingleNormalizationTrainerCandidate,
                require_independently_padded_dense,
                SentenceTransformer,
            )
            sources = [
                Path(__file__).resolve(),
                *[Path(inspect.getsourcefile(o)) for o in objects],
                *[
                    root / f"src/embed_optim/{n}.py"
                    for n in ("train", "losses", "collators", "optimizers", "config")
                ],
                root / "configs/dense_no_packing_retrain.yaml",
            ]
            report.update(
                observed_at_utc=datetime.now(UTC).isoformat(),
                source_bindings=[small.previous.identity(p) for p in dict.fromkeys(sources)],
                versions={
                    p: importlib.metadata.version(p)
                    for p in (
                        "torch",
                        "transformers",
                        "accelerate",
                        "sentence-transformers",
                        "datasets",
                    )
                },
            )
            with (work / "audit.json").open("x") as handle:
                handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
