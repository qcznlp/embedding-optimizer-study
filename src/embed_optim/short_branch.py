from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import Dataset

from .config import OptimizerConfig, load_matrix, resolve_matrix_path
from .confirmatory_data import (
    _canonical,
    _dataset_file_identities,
    _identity,
    _iter_jsonl,
    _receipt_path,
)
from .data import SPLITS, _seed_for, allocate_quotas
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

ALGORITHMS = ("adamw", "muon", "normuon")
FAMILIES = ("dense", "late")


def load_short_branch_protocol(
    path: str | Path = "configs/short_branch_protocol.json",
) -> tuple[Path, dict[str, Any]]:
    protocol_path = resolve_matrix_path(path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = protocol.get("source", {})
    subset = protocol.get("subset", {})
    shared = protocol.get("shared_start", {})
    calibration = protocol.get("scale_calibration", {})
    training = protocol.get("training", {})
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported short-branch protocol schema")
    if protocol.get("status") != "prospective_completion_lock":
        raise ValueError("Short-branch protocol is not prospectively locked")
    if source.get("queries") != 500_000 or subset.get("count") != 50_000:
        raise ValueError("Short branch requires the frozen 500K source and 50K subset")
    if subset.get("preserve_query_positive_and_negative_groups") is not True:
        raise ValueError("Short branch must preserve complete source contrastive groups")
    if (
        shared.get("fraction") != 0.6
        or shared.get("run_id") != "adamw-lr1e-5"
        or shared.get("checkpoint_step") != 2345
        or set(shared.get("checkpoints", {})) != set(FAMILIES)
    ):
        raise ValueError("Short branch must start from the fixed 60% AdamW anchor")
    target = calibration.get("target_global_hidden_update_to_weight")
    if not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
        raise ValueError("Short-branch scale target must be finite and positive")
    if calibration.get("weight_decay_included") is not False:
        raise ValueError("Short-branch calibration must isolate the data-update operator")
    seeds = training.get("order_seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Short branch requires exactly three distinct order seeds")
    if training.get("optimizer_operators") != list(ALGORITHMS):
        raise ValueError("Short branch must compare AdamW, Muon, and NorMuon")
    if training.get("adamw_operator_uses_hybrid_routing") is not True:
        raise ValueError("Short-branch AdamW must use the Muon hidden/auxiliary routing")
    if training.get("expected_runs") != len(seeds) * len(FAMILIES) * len(ALGORITHMS):
        raise ValueError("Short-branch run count disagrees with the frozen matrix")
    return protocol_path, protocol


def _load_source(
    protocol: dict[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    root = Path(protocol["source"]["training_data"]).resolve()
    manifest_path = root / "manifest.json"
    ledger_path = root / "rows.jsonl"
    if (
        _sha256(manifest_path) != protocol["source"]["manifest_sha256"]
        or _sha256(ledger_path) != protocol["source"]["row_ledger_sha256"]
    ):
        raise ValueError("Short-branch source differs from its frozen identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(_iter_jsonl(ledger_path))
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    for sample_id, row in enumerate(rows):
        digest.update(_canonical(row))
        if row.get("sample_id") != sample_id or row.get("source") not in SPLITS:
            raise ValueError(f"Invalid source row at sample {sample_id}")
        counts[row["source"]] += 1
    if (
        len(rows) != protocol["source"]["queries"]
        or digest.hexdigest() != manifest.get("row_manifest_sha256")
        or dict(counts) != manifest.get("quotas")
    ):
        raise ValueError("Short-branch source ledger coverage differs")
    return root, manifest, rows


def _selected_source_ids(
    rows: list[dict[str, Any]], manifest: dict[str, Any], protocol: dict[str, Any]
) -> tuple[list[int], dict[str, int]]:
    quota = allocate_quotas(manifest["quotas"], int(protocol["subset"]["count"]))
    by_split: dict[str, list[int]] = {split: [] for split in SPLITS}
    for row in rows:
        by_split[row["source"]].append(int(row["sample_id"]))
    selected = []
    for split in SPLITS:
        candidates = np.asarray(by_split[split], dtype=np.int64)
        order = np.random.default_rng(
            _seed_for(protocol["subset"]["selection_seed"], split, "short-branch")
        ).permutation(len(candidates))[: quota[split]]
        selected.extend(int(candidates[index]) for index in order)
    selected.sort()
    if len(selected) != protocol["subset"]["count"] or len(set(selected)) != len(selected):
        raise ValueError("Short-branch subset selection is not exact and unique")
    return selected, quota


def prepare_short_branch_subset(
    protocol_path: str | Path = "configs/short_branch_protocol.json",
) -> Path:
    resolved_protocol, protocol = load_short_branch_protocol(protocol_path)
    output = Path(protocol["subset"]["output"]).resolve()
    if output.exists():
        audit_short_branch_subset(resolved_protocol)
        return output
    source_root, source_manifest, source_rows = _load_source(protocol)
    selected_ids, quota = _selected_source_ids(source_rows, source_manifest, protocol)
    source_dataset = Dataset.load_from_disk(str(source_root / "dataset"))
    if len(source_dataset) != len(source_rows):
        raise ValueError("Short-branch source Dataset and ledger row counts differ")
    selected_dataset = source_dataset.select(selected_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".short-branch-subset-", dir=output.parent) as temp:
        artifact = Path(temp) / "artifact"
        artifact.mkdir()
        dataset_root = artifact / "dataset"
        selected_dataset.save_to_disk(str(dataset_root), num_proc=min(14, len(SPLITS) * 2))
        serialized = Dataset.load_from_disk(str(dataset_root))
        rows_path = artifact / "rows.jsonl"
        row_digest = hashlib.sha256()
        selected_id_digest = hashlib.sha256()
        with rows_path.open("w", encoding="utf-8") as handle:
            for sample_id in selected_ids:
                row = source_rows[sample_id]
                row_digest.update(_canonical(row))
                selected_id_digest.update(f"{sample_id}\n".encode())
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "protocol": _identity(resolved_protocol),
            "source_training_manifest_sha256": protocol["source"]["manifest_sha256"],
            "source_training_row_ledger_sha256": protocol["source"]["row_ledger_sha256"],
            "source_training_row_manifest_sha256": source_manifest["row_manifest_sha256"],
            "selection_seed": protocol["subset"]["selection_seed"],
            "rows": len(selected_ids),
            "quotas": quota,
            "selected_sample_ids_sha256": selected_id_digest.hexdigest(),
            "row_manifest_sha256": row_digest.hexdigest(),
            "row_ledger_sha256": _sha256(rows_path),
            "materialized_dataset_fingerprint": selected_dataset._fingerprint,
            "dataset_fingerprint": serialized._fingerprint,
            "dataset_files": _dataset_file_identities(dataset_root),
        }
        _atomic_json(artifact / "manifest.json", manifest)
        os.replace(artifact, output)
    audit_short_branch_subset(resolved_protocol)
    return output


def audit_short_branch_subset(
    protocol_path: str | Path = "configs/short_branch_protocol.json",
) -> dict[str, Any]:
    resolved_protocol, protocol = load_short_branch_protocol(protocol_path)
    source_root, source_manifest, source_rows = _load_source(protocol)
    expected_ids, expected_quota = _selected_source_ids(source_rows, source_manifest, protocol)
    output = Path(protocol["subset"]["output"]).resolve()
    manifest_path = output / "manifest.json"
    ledger_path = output / "rows.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("protocol", {}).get("sha256") != _sha256(resolved_protocol)
        or manifest.get("source_training_manifest_sha256") != protocol["source"]["manifest_sha256"]
        or manifest.get("source_training_row_ledger_sha256")
        != protocol["source"]["row_ledger_sha256"]
        or manifest.get("source_training_row_manifest_sha256")
        != source_manifest["row_manifest_sha256"]
        or manifest.get("rows") != protocol["subset"]["count"]
        or manifest.get("quotas") != expected_quota
        or manifest.get("row_ledger_sha256") != _sha256(ledger_path)
    ):
        raise ValueError("Short-branch subset manifest is inconsistent")
    observed_files = _dataset_file_identities(output / "dataset")
    if observed_files != manifest.get("dataset_files"):
        raise ValueError("Short-branch subset Dataset files differ")
    source_dataset = Dataset.load_from_disk(str(source_root / "dataset"))
    subset = Dataset.load_from_disk(str(output / "dataset"))
    if len(subset) != len(expected_ids) or subset._fingerprint != manifest["dataset_fingerprint"]:
        raise ValueError("Short-branch subset count or fingerprint differs")

    digest = hashlib.sha256()
    selected_id_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    ledger = _iter_jsonl(ledger_path)
    expected_columns = [
        "sample_id",
        "source",
        "query_id",
        "positive_id",
        "query",
        "positive",
        *(f"negative_{index}_id" for index in range(7)),
        *(f"negative_{index}" for index in range(7)),
    ]
    source_selected = source_dataset.select(expected_ids).select_columns(expected_columns)
    observed_selected = subset.select_columns(expected_columns)
    row_index = 0
    for expected_batch, observed_batch in zip(
        source_selected.iter(batch_size=2_048),
        observed_selected.iter(batch_size=2_048),
        strict=True,
    ):
        for index in range(len(observed_batch["sample_id"])):
            try:
                ledger_row = next(ledger)
            except StopIteration as error:
                raise ValueError("Short-branch ledger ended before its Dataset") from error
            source_id = expected_ids[row_index]
            if (
                any(
                    observed_batch[column][index] != expected_batch[column][index]
                    for column in expected_columns
                )
                or ledger_row != source_rows[source_id]
            ):
                raise ValueError(f"Short-branch source drift at subset row {row_index}")
            digest.update(_canonical(ledger_row))
            selected_id_digest.update(f"{source_id}\n".encode())
            counts[ledger_row["source"]] += 1
            row_index += 1
    try:
        next(ledger)
        raise ValueError("Short-branch ledger has extra rows")
    except StopIteration:
        pass
    if (
        row_index != protocol["subset"]["count"]
        or digest.hexdigest() != manifest["row_manifest_sha256"]
        or selected_id_digest.hexdigest() != manifest["selected_sample_ids_sha256"]
        or dict(counts) != expected_quota
    ):
        raise ValueError("Short-branch subset identities differ")
    return {
        "status": "complete",
        "path": _receipt_path(output, resolved_protocol.parent.parent),
        "manifest_sha256": _sha256(manifest_path),
        "rows": row_index,
        "quotas": dict(counts),
        "selected_sample_ids_sha256": selected_id_digest.hexdigest(),
        "dataset_fingerprint": subset._fingerprint,
    }


def _load_update_metrics(
    family: str,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    metrics_path = Path(protocol["scale_calibration"]["update_metrics"][family]).resolve()
    update_root = metrics_path.parent
    manifest_path = update_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    common_spec = Path(protocol["scale_calibration"]["common_state_spec"]).resolve()
    checkpoint = Path(protocol["shared_start"]["checkpoints"][family]).resolve()
    metrics_output = manifest.get("outputs", {}).get("metrics", {})
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or Path(manifest.get("checkpoint", {}).get("path", "")).resolve() != checkpoint
        or manifest.get("common_state_spec", {}).get("sha256") != _sha256(common_spec)
        or manifest.get("gradient_steps") != protocol["scale_calibration"]["gradient_history_steps"]
        or metrics_output.get("path") != metrics_path.name
        or metrics_output.get("bytes") != metrics_path.stat().st_size
        or metrics_output.get("sha256") != _sha256(metrics_path)
        or manifest.get("analysis_config", {}).get("weight_decay_included") is not False
    ):
        raise ValueError(f"{family}: common-state calibration manifest is inconsistent")
    rows = list(_iter_jsonl(metrics_path))
    if len(rows) != 88 or len({row.get("tensor") for row in rows}) != 88:
        raise ValueError(f"{family}: scale calibration must cover 88 unique hidden matrices")
    weight_sq = 0.0
    direction_sq = {algorithm: 0.0 for algorithm in ALGORITHMS}
    parameters = 0
    for row in rows:
        weight_norm = float(row.get("weight_frobenius_norm", math.nan))
        if not math.isfinite(weight_norm) or weight_norm <= 0:
            raise ValueError(f"{family}: invalid calibration weight norm")
        weight_sq += weight_norm**2
        parameters += int(row["parameters"])
        algorithms = row.get("algorithms", {})
        for algorithm in ALGORITHMS:
            direction_norm = float(algorithms.get(algorithm, {}).get("frobenius_norm", math.nan))
            if not math.isfinite(direction_norm) or direction_norm <= 0:
                raise ValueError(f"{family}: invalid {algorithm} direction norm")
            direction_sq[algorithm] += direction_norm**2
    if parameters != 110_297_088:
        raise ValueError(f"{family}: calibration parameter partition differs")
    ratios = {algorithm: math.sqrt(value / weight_sq) for algorithm, value in direction_sq.items()}
    if not all(math.isfinite(value) and value > 0 for value in ratios.values()):
        raise ValueError(f"{family}: invalid aggregate calibration ratios")
    return {
        "manifest": _identity(manifest_path),
        "metrics": _identity(metrics_path),
        "checkpoint": manifest["checkpoint"],
        "tensors": len(rows),
        "parameters": parameters,
    }, ratios


def _optimizer_payload(algorithm: str, learning_rate: float, auxiliary_lr: float) -> dict[str, Any]:
    name = "hybrid_adamw" if algorithm == "adamw" else algorithm
    config = OptimizerConfig(name=name, lr=learning_rate, aux_lr=auxiliary_lr)
    payload = dataclasses.asdict(config)
    if algorithm in {"muon", "normuon"}:
        payload["ns_implementation"] = "unfused-bfloat16-v1"
    return payload


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def generate_short_branch_matrices(
    protocol_path: str | Path = "configs/short_branch_protocol.json",
    *,
    experiment_matrix: str | Path = "configs/experiment.yaml",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_protocol, protocol = load_short_branch_protocol(protocol_path)
    subset = audit_short_branch_subset(resolved_protocol)
    source_matrix = resolve_matrix_path(experiment_matrix).resolve()
    source_runs = load_matrix(source_matrix)
    source_configs = {
        family: next(
            run
            for run in source_runs
            if run.model_family == family and run.run_id == protocol["shared_start"]["run_id"]
        )
        for family in FAMILIES
    }
    calibration = {}
    ratios = {}
    target = float(protocol["scale_calibration"]["target_global_hidden_update_to_weight"])
    learning_rates = {}
    for family in FAMILIES:
        calibration[family], ratios[family] = _load_update_metrics(family, protocol)
        learning_rates[family] = {
            algorithm: target / ratios[family][algorithm] for algorithm in ALGORITHMS
        }
    output = Path(output_dir or protocol["training"]["matrix_output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    formal_runtime = source_matrix.parent / "formal_runtime.json"
    matrices = []
    for seed in protocol["training"]["order_seeds"]:
        common = source_configs["dense"]
        payload = {
            "formal_runtime": os.path.relpath(formal_runtime, output),
            "common": {
                "dataset_path": protocol["subset"]["output"],
                "output_root": f"{protocol['training']['output_root']}/seed{seed}",
                "seed": seed,
                "epochs": protocol["training"]["epochs"],
                "global_batch_size": common.global_batch_size,
                "micro_batch_size": common.micro_batch_size,
                "max_length": common.max_length,
                "warmup_ratio": common.warmup_ratio,
                "max_grad_norm": common.max_grad_norm,
                "dataloader_workers": common.dataloader_workers,
                "gradient_checkpointing": common.gradient_checkpointing,
                "flash_attention": common.flash_attention,
                "wandb_project": common.wandb_project,
                "wandb_entity": common.wandb_entity,
                "checkpoint_fractions": protocol["training"]["checkpoint_fractions"],
            },
            "models": {
                family: {
                    "model_name": protocol["shared_start"]["checkpoints"][family],
                    "temperature": source_configs[family].resolved_temperature,
                }
                for family in FAMILIES
            },
            "runs": [
                {
                    "id": f"{algorithm}-scale-matched",
                    "model_family": family,
                    "optimizer": _optimizer_payload(
                        algorithm,
                        learning_rates[family][algorithm],
                        float(protocol["training"]["auxiliary_adamw_lr"]),
                    ),
                }
                for family in FAMILIES
                for algorithm in ALGORITHMS
            ],
            "provenance": {
                "short_branch_protocol_sha256": _sha256(resolved_protocol),
                "subset_manifest_sha256": subset["manifest_sha256"],
                "calibration_metrics_sha256": {
                    family: calibration[family]["metrics"]["sha256"] for family in FAMILIES
                },
                "target_global_hidden_update_to_weight": target,
                "derived_hidden_learning_rates": learning_rates,
            },
        }
        path = output / f"seed{seed}.yaml"
        _atomic_yaml(path, payload)
        matrices.append(_identity(path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": _identity(resolved_protocol),
        "source_matrix": _identity(source_matrix),
        "subset": subset,
        "calibration": calibration,
        "global_per_unit_lr_update_to_weight": ratios,
        "target_global_hidden_update_to_weight": target,
        "derived_hidden_learning_rates": learning_rates,
        "order_seeds": protocol["training"]["order_seeds"],
        "expected_runs": protocol["training"]["expected_runs"],
        "matrices": matrices,
    }
    _atomic_json(output / "manifest.json", manifest)
    audit_short_branch_matrices(
        resolved_protocol, experiment_matrix=source_matrix, output_dir=output
    )
    return manifest


def audit_short_branch_matrices(
    protocol_path: str | Path = "configs/short_branch_protocol.json",
    *,
    experiment_matrix: str | Path = "configs/experiment.yaml",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_protocol, protocol = load_short_branch_protocol(protocol_path)
    subset = audit_short_branch_subset(resolved_protocol)
    source_matrix = resolve_matrix_path(experiment_matrix).resolve()
    output = Path(output_dir or protocol["training"]["matrix_output_dir"]).resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = [output / f"seed{seed}.yaml" for seed in protocol["training"]["order_seeds"]]
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("protocol", {}).get("sha256") != _sha256(resolved_protocol)
        or manifest.get("source_matrix", {}).get("sha256") != _sha256(source_matrix)
        or manifest.get("subset", {}).get("manifest_sha256") != subset["manifest_sha256"]
        or manifest.get("expected_runs") != protocol["training"]["expected_runs"]
        or manifest.get("matrices") != [_identity(path) for path in expected_paths]
    ):
        raise ValueError("Short-branch matrix manifest is inconsistent")
    calibration = {}
    ratios = {}
    target = float(protocol["scale_calibration"]["target_global_hidden_update_to_weight"])
    expected_lrs = {}
    for family in FAMILIES:
        calibration[family], ratios[family] = _load_update_metrics(family, protocol)
        expected_lrs[family] = {
            algorithm: target / ratios[family][algorithm] for algorithm in ALGORITHMS
        }
    if manifest.get("derived_hidden_learning_rates") != expected_lrs:
        raise ValueError("Short-branch derived learning rates differ from common-state metrics")

    observed_runs = 0
    for seed, path in zip(protocol["training"]["order_seeds"], expected_paths, strict=True):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        runs = load_matrix(path)
        if (
            len(runs) != protocol["training"]["runs_per_seed"]
            or {run.seed for run in runs} != {seed}
            or {run.dataset_path for run in runs} != {protocol["subset"]["output"]}
            or {run.output_root for run in runs}
            != {f"{protocol['training']['output_root']}/seed{seed}"}
            or raw.get("provenance", {}).get("short_branch_protocol_sha256")
            != _sha256(resolved_protocol)
            or raw.get("provenance", {}).get("subset_manifest_sha256") != subset["manifest_sha256"]
        ):
            raise ValueError(f"Seed {seed}: short-branch matrix invariants differ")
        indexed = {
            (
                run.model_family,
                "adamw" if run.optimizer.name == "hybrid_adamw" else run.optimizer.name,
            ): run
            for run in runs
        }
        expected_keys = set(itertools.product(FAMILIES, ALGORITHMS))
        if set(indexed) != expected_keys:
            raise ValueError(f"Seed {seed}: short-branch operator coverage differs")
        for (family, algorithm), run in indexed.items():
            if (
                not math.isclose(
                    run.optimizer.lr,
                    expected_lrs[family][algorithm],
                    rel_tol=1e-15,
                    abs_tol=0.0,
                )
                or run.optimizer.aux_lr != protocol["training"]["auxiliary_adamw_lr"]
                or run.model_name != protocol["shared_start"]["checkpoints"][family]
                or run.model_revision is not None
            ):
                raise ValueError(
                    f"Seed {seed}: short-branch recipe differs for {(family, algorithm)}"
                )
        observed_runs += len(runs)
    if observed_runs != protocol["training"]["expected_runs"]:
        raise ValueError("Short-branch run coverage differs")
    return {
        "status": "complete",
        "manifest_sha256": _sha256(manifest_path),
        "matrices": len(expected_paths),
        "runs": observed_runs,
        "derived_hidden_learning_rates": expected_lrs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or audit the 50K shared-checkpoint scale-matched short branch"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/short_branch_protocol.json"))
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--subset-only", action="store_true")
    parser.add_argument(
        "--subset-receipt",
        type=Path,
        default=Path("reports/short-branch/subset-receipt.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.subset_only:
        result = (
            audit_short_branch_subset(args.protocol)
            if args.audit_only
            else {"path": str(prepare_short_branch_subset(args.protocol))}
        )
        if args.audit_only:
            _atomic_json(args.subset_receipt, result)
    elif args.audit_only:
        result = audit_short_branch_matrices(
            args.protocol,
            experiment_matrix=args.experiment_matrix,
            output_dir=args.output_dir,
        )
    else:
        prepare_short_branch_subset(args.protocol)
        result = generate_short_branch_matrices(
            args.protocol,
            experiment_matrix=args.experiment_matrix,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
