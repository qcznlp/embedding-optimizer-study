"""Dense-only v3 validation with full row identity and independently replayable scores.

Raw GPU/FP32 metric values remain the selection inputs. The independent float64
replay is an arithmetic guard, not a replacement metric or a tie tolerance.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .primary_contract import (
    DRAFT,
    RELEASED,
    committed_file,
    digest,
    file_identity,
    read_json,
    relative_path,
    require_same,
    verify_file,
)
from .primary_v3_contract import PrimaryV3Contract

SCOPE = "dense_primary_v3_validation"
PRIMARY_SHA = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
CORE_ACCEPTANCE_SHA = "efa06bd154e6e67035654e214b1a278eede3a74e6c24cb8f5ff48183feb4bdf8"
METRICS = (
    "contrastive_loss",
    "positive_score",
    "hardest_negative_score",
    "positive_margin",
    "reciprocal_rank",
    "top1_accuracy",
)
TEXT_COLUMNS = ("query", "positive", *(f"negative_{i}" for i in range(7)))
SETTINGS = {
    "rows": 4096,
    "model_dtype": "float32",
    "forward_dtype": "bfloat16",
    "score_dtype": "float32",
    "model_mode": "eval",
    "flash_attention": True,
    "batch_size": 16,
    "max_length": 8192,
    "temperature": 0.02,
    "candidate_count": 8,
    "positive_column": 0,
    "in_batch_negatives": False,
    "checkpoint": 3907,
    "aggregation": "mean over every validation row, preserving row order; not mean of source means",
    "selection": "minimum mean contrastive_loss within optimizer; exact ties choose lower learning_rate",
    "positive_margin_is_a_tie_breaker": False,
    "beir_is_a_selection_input": False,
    "reference_replay": {
        "dtype": "float64",
        "atol": 2e-5,
        "rtol": 2e-6,
        "changes_recorded_metrics": False,
        "is_selection_tolerance": False,
    },
}
SOURCES = (
    "src/embed_optim/primary_v3_validation.py",
    "src/embed_optim/primary_v3_validation_io.py",
    "src/embed_optim/collators.py",
    "src/embed_optim/corrected_input_execution.py",
    "src/embed_optim/gpu_lease.py",
    "src/embed_optim/runtime.py",
    "configs/formal_runtime.json",
    "configs/validation_probe.json",
    "scripts/prepare_dense_v3_validation.py",
)


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Validation numbers must be finite and non-boolean")
    return float(value)


def row_identity(row, position):
    if type(position) is not int or position < 0:
        raise ValueError("Require an explicit nonnegative row position")
    id_names = ("sample_id", "query_id", "positive_id", *(f"negative_{i}_id" for i in range(7)))
    if any(type(row.get(k)) is not int or row[k] < 0 for k in id_names):
        raise ValueError("Require actual integral query/document/sample identities")
    if type(row.get("length")) is not int or row["length"] <= 0:
        raise ValueError("Require the original positive integral length metadata")
    if not isinstance(row.get("source"), str) or not row["source"]:
        raise ValueError("Validation row lacks its source")
    if any(not isinstance(row.get(k), str) or not row[k].strip() for k in TEXT_COLUMNS):
        raise ValueError("Validation row contains invalid original text")
    return {
        "position": position,
        "sample_id": row["sample_id"],
        "source": row["source"],
        "query_id": row["query_id"],
        "positive_id": row["positive_id"],
        "negative_ids": [row[f"negative_{i}_id"] for i in range(7)],
        "row_sha256": digest(row),
    }


def reference_metrics(scores, temperature):
    """Independent scalar stable-logsumexp reference; no Torch or production metric call."""
    if not isinstance(scores, list) or len(scores) != 8:
        raise ValueError("Require exactly eight positive-first candidate scores")
    values = [finite(x) for x in scores]
    temperature = finite(temperature)
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = [s / temperature for s in values]
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("Scaled validation logits must be finite")
    maximum = max(logits)
    loss = maximum - logits[0] + math.log(sum(math.exp(x - maximum) for x in logits))
    hardest = max(values[1:])
    rank = 1 + sum(n >= values[0] for n in values[1:])
    return {
        "contrastive_loss": loss,
        "positive_score": values[0],
        "hardest_negative_score": hardest,
        "positive_margin": values[0] - hardest,
        "reciprocal_rank": 1 / rank,
        "top1_accuracy": float(rank == 1),
    }


def summarize_records(records, identities, *, settings=SETTINGS):
    if not identities or len(records) != len(identities):
        raise ValueError("Validation records do not cover the complete declared view")
    if len({(r["source"], r["query_id"]) for r in identities}) != len(identities):
        raise ValueError("Duplicate source-qualified validation query identity")
    if [r["position"] for r in identities] != list(range(len(identities))) or len(
        {r["sample_id"] for r in identities}
    ) != len(identities):
        raise ValueError("Validation positions or sample identities are duplicated or incomplete")
    tolerance = settings["reference_replay"]
    maximum_errors = {metric: 0.0 for metric in METRICS}
    for record, expected in zip(records, identities, strict=True):
        if set(record) != {"row", "scores", "metrics"} or set(record["metrics"]) != set(METRICS):
            raise ValueError("A validation record has missing or extra fields")
        require_same(record["row"], expected)
        replay = reference_metrics(record["scores"], settings["temperature"])
        for metric in METRICS:
            actual = finite(record["metrics"][metric])
            error = abs(actual - replay[metric])
            maximum_errors[metric] = max(maximum_errors[metric], error)
            if not math.isclose(
                actual, replay[metric], rel_tol=tolerance["rtol"], abs_tol=tolerance["atol"]
            ):
                raise ValueError(
                    f"Stored validation metric disagrees with independent replay: {metric}"
                )
    groups = {"__all__": records}
    for source in sorted({row["source"] for row in identities}):
        groups[source] = [r for r in records if r["row"]["source"] == source]
    summaries = [
        {
            "group": name,
            "samples": len(rows),
            **{
                metric: sum(float(r["metrics"][metric]) for r in rows) / len(rows)
                for metric in METRICS
            },
        }
        for name, rows in sorted(groups.items())
    ]
    return {
        "groups": summaries,
        "maximum_scalar_replay_absolute_error": maximum_errors,
        "raw_metrics_preserved": True,
        "records": len(records),
    }


def score_rows(model, dataset, *, device, batch_size=16, forward_dtype="bfloat16"):
    """Exact dense eight-way scorer; usable in explicitly separated diagnostics too."""
    import torch
    import torch.nn.functional as F

    from .collators import DenseGroupCollator

    if model.training or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Validation requires eval mode and a positive batch size")
    if forward_dtype not in {"float32", "bfloat16"} or not len(dataset):
        raise ValueError("Invalid validation dtype or empty dataset")
    collator, records = DenseGroupCollator(model.preprocess), []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            batch = collator(rows)
            features = []
            for column in TEXT_COLUMNS:
                prefix = f"{column}_"
                feature = {
                    k[len(prefix) :]: v.to(device)
                    for k, v in batch.items()
                    if k.startswith(prefix) and isinstance(v, torch.Tensor)
                }
                if "input_ids" not in feature:
                    raise ValueError("Collator omitted a declared candidate column")
                features.append(feature)
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=forward_dtype == "bfloat16",
            ):
                query = F.normalize(model(features[0])["sentence_embedding"], p=2, dim=-1)
                documents = torch.stack(
                    [
                        F.normalize(model(f)["sentence_embedding"], p=2, dim=-1)
                        for f in features[1:]
                    ],
                    dim=1,
                )
                scores = torch.einsum("bh,bnh->bn", query, documents).float()
            if scores.shape != (len(rows), 8) or not torch.isfinite(scores).all():
                raise ValueError("Invalid eight-way model scores")
            positive, hardest = scores[:, 0], scores[:, 1:].max(dim=1).values
            rank = 1 + (scores[:, 1:] >= positive[:, None]).sum(dim=1)
            metrics = {
                "contrastive_loss": F.cross_entropy(
                    scores / SETTINGS["temperature"],
                    torch.zeros(len(rows), device=scores.device, dtype=torch.long),
                    reduction="none",
                ),
                "positive_score": positive,
                "hardest_negative_score": hardest,
                "positive_margin": positive - hardest,
                "reciprocal_rank": rank.float().reciprocal(),
                "top1_accuracy": rank.eq(1).float(),
            }
            cpu_scores = scores.cpu().tolist()
            cpu_metrics = {k: v.cpu().tolist() for k, v in metrics.items()}
            for offset, row in enumerate(rows):
                records.append(
                    {
                        "row": row_identity(row, start + offset),
                        "scores": cpu_scores[offset],
                        "metrics": {k: v[offset] for k, v in cpu_metrics.items()},
                    }
                )
    identities = [row_identity(dataset[i], i) for i in range(len(dataset))]
    summarize_records(records, identities)
    return records


def select_recipes(rows, expected_runs):
    expected = {r["run_id"]: r["identity_fields"]["recipe"]["optimizer"] for r in expected_runs}
    if len(expected_runs) != 12 or len(expected) != 12 or len(rows) != 12:
        raise ValueError("Recipe selection requires every one of twelve completed validations")
    if Counter(r["run_id"] for r in rows) != Counter(expected.keys()):
        raise ValueError("Missing, duplicated or historical validation recipe")
    groups = {}
    for row in rows:
        configuration = expected[row["run_id"]]
        if row["optimizer"] != configuration["name"] or row["learning_rate"] != configuration["lr"]:
            raise ValueError("Validation metric recipe disagrees with the primary identity")
        if finite(row["contrastive_loss"]) < 0:
            raise ValueError("Mean contrastive loss cannot be negative")
        groups.setdefault(row["optimizer"], []).append(row)
    if set(groups) != {"adamw", "muon", "normuon"} or any(len(v) != 4 for v in groups.values()):
        raise ValueError("Require the complete three-optimizer/four-rate validation grid")
    return {
        name: min(values, key=lambda r: (r["contrastive_loss"], r["learning_rate"]))["run_id"]
        for name, values in groups.items()
    }


@dataclass(frozen=True)
class ValidationContract:
    path: Path
    primary: PrimaryV3Contract
    payload: dict
    sha256: str

    @classmethod
    def load(cls, path, primary, *, require_released=False):
        path = Path(path).resolve()
        value = read_json(path)
        if (
            value.get("scope") != SCOPE
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("status") not in {DRAFT, RELEASED}
            or value.get("scientific_completion") is not False
        ):
            raise ValueError("Not the v3 validation contract")
        if require_released and value["status"] != RELEASED:
            raise ValueError("Prepared v3 validation is not execution authorized")
        if primary.sha256 != PRIMARY_SHA or value["primary"]["sha256"] != PRIMARY_SHA:
            raise ValueError("Validation binds another primary revision")
        if value["primary"]["path"] != primary.path.relative_to(primary.repository).as_posix():
            raise ValueError("Validation primary path label differs from its actual parent")
        verify_file(primary.path, value["primary"])
        require_same(value["settings"], SETTINGS)
        require_same(value["dataset"], primary.payload["datasets"]["validation"])
        if value["core_acceptance"]["sha256"] != CORE_ACCEPTANCE_SHA:
            raise ValueError("The v3 core-chain acceptance differs")
        verify_file(
            primary.repository / relative_path(value["core_acceptance"]["path"]),
            value["core_acceptance"],
        )
        if set(value["sources"]) != set(SOURCES):
            raise ValueError("Incomplete v3 validation source closure")
        for name, binding in value["sources"].items():
            verify_file(primary.repository / relative_path(name), binding)
            if require_released:
                committed_file(primary.repository, name, binding)
        for name in ("primary_v3_validation.py", "primary_v3_validation_io.py"):
            verify_file(
                Path(__file__).resolve().parent / name, value["sources"][f"src/embed_optim/{name}"]
            )
        if require_released:
            primary.require_execution()
            committed_file(
                primary.repository,
                path.relative_to(primary.repository).as_posix(),
                file_identity(path),
            )
        return cls(path, primary, value, file_identity(path)["sha256"])

    def require_execution(self):
        current = type(self).load(self.path, self.primary, require_released=True)
        if current.sha256 != self.sha256:
            raise ValueError("Validation contract changed after admission")
        return current

    def data(self, root):
        from datasets import Dataset

        checked = self.primary.dataset(root, "validation")
        dataset = Dataset.load_from_disk(str(Path(root) / "dataset"))
        if len(dataset) != SETTINGS["rows"] or list(dataset["sample_id"]) != list(
            range(SETTINGS["rows"])
        ):
            raise ValueError("Validation view lacks its complete ordered sample identity")
        identities = [row_identity(dataset[i], i) for i in range(len(dataset))]
        return dataset, {
            "content": checked,
            "row_identities": identities,
            "row_identities_sha256": digest(identities),
        }
