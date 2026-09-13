"""Observe the pinned ST/Accelerate sampler's tail, without any training/device.

The row identities are synthetic cardinality labels. Actual upstream sampler
objects are used. This is a counterexample to copying the primary loader into
the 50K continuation, not an executed factorial run or a repaired data loader.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def identity(path):
    path = Path(path)
    with path.open("rb") as handle:
        return {
            "bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
        }


def observe(rows, *, drop_last, seed):
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer
    from sentence_transformers.base.sampler import BatchSamplers

    data = Dataset.from_dict({"sample_id": range(rows)})
    builder = SimpleNamespace(args=SimpleNamespace(batch_sampler=BatchSamplers.BATCH_SAMPLER))
    ranks = []
    all_ids = []
    for rank in range(4):
        # Each DDP process constructs the same independently seeded sampler.
        batch_sampler = SentenceTransformerTrainer.get_batch_sampler(
            builder,
            data,
            batch_size=8,
            drop_last=drop_last,
            valid_label_columns=[],
            generator=torch.Generator().manual_seed(seed),
        )
        shard = BatchSamplerShard(
            batch_sampler,
            num_processes=4,
            process_index=rank,
            split_batches=False,
            even_batches=False,
        )
        batches = list(shard)
        ids = [index for batch in batches for index in batch]
        all_ids.extend(ids)
        updates = [sum(map(len, batches[i : i + 4])) for i in range(0, len(batches), 4)]
        ranks.append(
            {
                "rank": rank,
                "micro_batches": len(batches),
                "reported_length": len(shard),
                "query_groups": len(ids),
                "optimizer_steps": len(updates),
                "last_update_queries": updates[-1],
                "last_update_micro_sizes": [
                    len(b) for b in batches[-((len(batches) - 1) % 4 + 1) :]
                ],
                "sample_order_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    count = Counter(all_ids)
    return {
        "rows": rows,
        "drop_last": drop_last,
        "even_batches": False,
        "split_batches": False,
        "seed": seed,
        "sampler_class": type(batch_sampler).__module__ + "." + type(batch_sampler).__name__,
        "ranks": ranks,
        "draws": len(all_ids),
        "unique_groups": len(count),
        "missing_groups": rows - len(count),
        "duplicate_draws": sum(v - 1 for v in count.values()),
        "same_micro_batch_count_on_all_ranks": len({r["micro_batches"] for r in ranks}) == 1,
        "last_global_update_queries": sum(r["last_update_queries"] for r in ranks),
        "declared_full_epoch_steps": math.ceil(rows / 128),
    }


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require explicit CPU-only execution")
    import accelerate.data_loader
    import sentence_transformers.base.trainer
    import sentence_transformers.base.training_args
    import sentence_transformers.base.sampler
    import transformers.trainer

    versions = {
        p: importlib.metadata.version(p)
        for p in ("torch", "accelerate", "sentence-transformers", "transformers")
    }
    if versions != {
        "torch": "2.9.1+cu129",
        "accelerate": "1.13.0",
        "sentence-transformers": "5.7.0",
        "transformers": "5.3.0",
    }:
        raise ValueError("Unexpected pinned sampler runtime")
    sources = [
        {
            "path": str(Path(inspect.getsourcefile(m)).resolve()),
            **identity(inspect.getsourcefile(m)),
        }
        for m in (
            accelerate.data_loader,
            sentence_transformers.base.trainer,
            sentence_transformers.base.training_args,
            sentence_transformers.base.sampler,
            transformers.trainer,
        )
    ]
    primary = observe(500000, drop_last=True, seed=42)
    branches = [observe(50000, drop_last=True, seed=s) for s in (314159, 271828, 161803)]
    naive = observe(50000, drop_last=False, seed=314159)
    if (
        primary["draws"] != 500000
        or primary["missing_groups"]
        or primary["last_global_update_queries"] != 32
    ):
        raise ValueError("Primary cardinality control differs from expectation")
    if any(
        row["draws"] != 49984
        or row["missing_groups"] != 16
        or row["last_global_update_queries"] != 64
        for row in branches
    ):
        raise ValueError("Observed branch truncation differs from the source-based prediction")
    if naive["draws"] != 50000 or naive["same_micro_batch_count_on_all_ranks"]:
        raise ValueError("Naive drop-last toggle did not reproduce the predicted rank imbalance")
    for row in sources:
        if identity(row["path"]) != {k: row[k] for k in ("bytes", "sha256")}:
            raise ValueError("Sampler source changed during observation")
    result = {
        "scope": "upstream_sampler_cardinality_counterexample_only",
        "observation_complete": True,
        "versions": versions,
        "sources": sources,
        "diagnostic_source": {"path": str(Path(__file__).resolve()), **identity(__file__)},
        "primary_control": primary,
        "direct_factorial_reuse": branches,
        "naive_drop_last_false_counterfactual": naive,
        "factorial_loader_integrated": False,
        "training_executed": False,
        "scientific_completion": False,
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {"output": str(args.output), **identity(args.output), "observation_complete": True}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
