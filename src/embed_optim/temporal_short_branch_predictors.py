from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from .geometry import SCHEMA_VERSION, TensorStore, _atomic_json, _resolve_source, _sha256
from .optimizers import parameter_partition_name
from .probe_matrix import _declared_checkpoint_steps
from .scope import resolve_scope
from .short_branch_evaluation import _load_branch_configs
from .temporal_short_branch import _load_spec

FIELDS = [
    "family",
    "seed",
    "operator",
    "stage",
    "update_stable_rank_fraction",
    "update_entropy_rank_fraction",
    "update_head_energy_fraction",
    "update_middle_energy_fraction",
    "update_tail_energy_fraction",
    "update_row_norm_cv",
    "update_frobenius_norm",
    "weight_frobenius_norm",
]


def _operator(name: str) -> str:
    return "adamw" if name == "hybrid_adamw" else name


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    files, _ = _resolve_source(path)
    return {"path": str(path.resolve()), "files": [_file_identity(item) for item in files]}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 45 or any(list(row) != FIELDS for row in rows):
        raise ValueError("Temporal predictor output must contain 45 canonical rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {**_file_identity(path), "rows": len(rows), "fields": FIELDS}


def _matrix_metrics(delta: torch.Tensor) -> dict[str, float]:
    singular = torch.linalg.svdvals(delta)
    energy = singular.square()
    total = energy.sum()
    rank = min(delta.shape)
    if total <= 0 or rank <= 0:
        raise ValueError("A hidden update matrix has zero Frobenius energy")
    probabilities = singular / singular.sum()
    stable = total / energy.max()
    entropy = torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
    head_end = max(1, round(rank * 0.25))
    middle_end = max(head_end, round(rank * 0.75))
    rows = delta.square().sum(dim=1).sqrt()
    row_mean = rows.mean()
    return {
        "stable": float((stable / rank).item()),
        "entropy": float((entropy / rank).item()),
        "head": float((energy[:head_end].sum() / total).item()),
        "middle": float((energy[head_end:middle_end].sum() / total).item()),
        "tail": float((energy[middle_end:].sum() / total).item()),
        "row_cv": float((rows.std(unbiased=False) / row_mean).item()) if row_mean > 0 else 0.0,
        "energy": float(total.item()),
    }


def checkpoint_predictors(
    reference: Path, current: Path, device: str
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    weighted = {name: 0.0 for name in ("stable", "entropy", "head", "middle", "tail", "row_cv")}
    update_energy = 0.0
    weight_energy = 0.0
    parameters = 0
    shapes = []
    with TensorStore(reference) as left, TensorStore(current) as right:
        if left.keys() != right.keys():
            raise ValueError("Adjacent checkpoints have different tensor names")
        for name in left.keys():
            shape = left.shape(name)
            if shape != right.shape(name):
                raise ValueError(f"Adjacent checkpoint shape differs for {name}")
            if parameter_partition_name(name, len(shape)) != "hidden":
                continue
            if len(shape) != 2:
                raise AssertionError("Hidden selector returned a non-matrix")
            count = math.prod(shape)
            before = left.tensor(name).to(device=device, dtype=torch.float32)
            after = right.tensor(name).to(device=device, dtype=torch.float32)
            delta = after - before
            metrics = _matrix_metrics(delta)
            for key in weighted:
                weighted[key] += metrics[key] * count
            update_energy += metrics["energy"]
            weight_energy += float(after.square().sum().item())
            parameters += count
            shapes.append({"name": name, "shape": list(shape), "parameters": count})
            del before, after, delta
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not shapes or parameters != 110_297_088:
        raise ValueError(
            f"Dense hidden partition differs: {len(shapes)} tensors / {parameters} parameters"
        )
    result = {name: value / parameters for name, value in weighted.items()}
    result.update(
        update_frobenius_norm=math.sqrt(update_energy),
        weight_frobenius_norm=math.sqrt(weight_energy),
    )
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite temporal predictor")
    return result, shapes


def _jobs(
    protocol: Path, experiment_matrix: Path, matrix_dir: Path | None
) -> tuple[list[dict[str, Any]], Path]:
    resolved, _, configs, generated = _load_branch_configs(
        protocol,
        experiment_matrix=experiment_matrix,
        matrix_dir=matrix_dir,
        audit_matrices=False,
        families=("dense",),
    )
    jobs = []
    for seed, runs in sorted(configs.items()):
        for config in sorted(runs, key=lambda item: item.run_id):
            steps = _declared_checkpoint_steps(config)
            if len(steps) != 5 or steps != sorted(set(map(int, steps))):
                raise ValueError(f"Seed {seed}: invalid short-branch schedule")
            reference = Path(config.model_name).resolve()
            for stage, step in enumerate(map(int, steps), start=1):
                current = (config.output_dir / f"checkpoint-{step}").resolve()
                jobs.append(
                    {
                        "seed": seed,
                        "operator": _operator(config.optimizer.name),
                        "stage": stage,
                        "reference": reference,
                        "current": current,
                    }
                )
                reference = current
    if len(jobs) != 45:
        raise ValueError(f"Expected 45 Dense temporal predictor jobs, found {len(jobs)}")
    return jobs, generated / "manifest.json"


def build_predictors(
    *,
    protocol: Path,
    analysis_protocol: Path,
    families: tuple[str, ...],
    scope_amendment: Path,
    experiment_matrix: Path,
    matrix_dir: Path | None,
    output_csv: Path,
    manifest_path: Path,
    cache_dir: Path,
    device: str,
) -> dict[str, Any]:
    _load_spec(analysis_protocol)
    resolved_families, scope = resolve_scope(families, scope_amendment)
    if resolved_families != ("dense",) or scope is None:
        raise ValueError("Temporal predictors require the frozen Dense-only scope amendment")
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError("--device must be cpu or an explicit cuda:N device")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA predictor extraction requested but CUDA is unavailable")
    jobs, matrix_manifest = _jobs(protocol, experiment_matrix, matrix_dir)
    missing = sorted(
        {
            str(path)
            for job in jobs
            for path in (job["reference"], job["current"])
            if not path.is_dir()
        }
    )
    base = {
        "schema_version": SCHEMA_VERSION,
        "family": "dense",
        "selector": {
            "call": "parameter_partition_name(name, ndim) == 'hidden'",
            "expected_tensors": 88,
            "expected_parameters": 110_297_088,
        },
        "reference_rule": "stage1-minus-common-start; stage2-5-minus-previous-checkpoint",
        "decomposition": "torch.linalg.svdvals float32 exact, one hidden matrix at a time",
        "device": device,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "analysis_protocol": _file_identity(analysis_protocol),
        "scope_amendment": _file_identity(scope_amendment),
        "cache_dir": str(cache_dir.resolve()),
    }
    if missing:
        payload = {
            **base,
            "complete": False,
            "claimable": False,
            "status": "pending-not-claimable",
            "missing_checkpoints": missing,
        }
        _atomic_json(manifest_path, payload)
        return payload
    rows = []
    sources = []
    identity_cache: dict[Path, dict[str, Any]] = {}
    canonical_shapes = None
    cache_dir.resolve().mkdir(parents=True, exist_ok=True)
    for job in jobs:

        def cached_identity(path: Path) -> dict[str, Any]:
            resolved = path.resolve()
            if resolved not in identity_cache:
                identity_cache[resolved] = _checkpoint_identity(resolved)
            return identity_cache[resolved]

        reference_identity = cached_identity(job["reference"])
        current_identity = cached_identity(job["current"])
        cache_path = cache_dir / (f"seed{job['seed']}__{job['operator']}__stage{job['stage']}.json")
        expected_cache_inputs = {
            "reference": reference_identity,
            "current": current_identity,
            "analysis_protocol": base["analysis_protocol"],
            "protocol": _file_identity(protocol),
            "scope_amendment": base["scope_amendment"],
            "device": device,
            "torch": base["runtime"],
        }
        cached = None
        if cache_path.is_file():
            candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            candidate_metrics = candidate.get("metrics")
            expected_metric_names = {
                "stable",
                "entropy",
                "head",
                "middle",
                "tail",
                "row_cv",
                "update_frobenius_norm",
                "weight_frobenius_norm",
            }
            if (
                candidate.get("schema_version") == SCHEMA_VERSION
                and candidate.get("complete") is True
                and candidate.get("job") == {key: job[key] for key in ("seed", "operator", "stage")}
                and candidate.get("inputs") == expected_cache_inputs
                and isinstance(candidate_metrics, dict)
                and set(candidate_metrics) == expected_metric_names
                and all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in candidate_metrics.values()
                )
                and isinstance(candidate.get("hidden_shapes"), list)
            ):
                cached = candidate
        if cached is None:
            values, shapes = checkpoint_predictors(job["reference"], job["current"], device)
            cached = {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "job": {key: job[key] for key in ("seed", "operator", "stage")},
                "inputs": expected_cache_inputs,
                "metrics": values,
                "hidden_shapes": shapes,
            }
            _atomic_json(cache_path, cached)
        values = cached["metrics"]
        shapes = cached["hidden_shapes"]
        if canonical_shapes is None:
            canonical_shapes = shapes
        elif shapes != canonical_shapes:
            raise ValueError("Dense hidden tensor names/shapes changed across checkpoints")
        rows.append(
            {
                "family": "dense",
                "seed": job["seed"],
                "operator": job["operator"],
                "stage": job["stage"],
                "update_stable_rank_fraction": values["stable"],
                "update_entropy_rank_fraction": values["entropy"],
                "update_head_energy_fraction": values["head"],
                "update_middle_energy_fraction": values["middle"],
                "update_tail_energy_fraction": values["tail"],
                "update_row_norm_cv": values["row_cv"],
                "update_frobenius_norm": values["update_frobenius_norm"],
                "weight_frobenius_norm": values["weight_frobenius_norm"],
            }
        )

        sources.append(
            {
                "seed": job["seed"],
                "operator": job["operator"],
                "stage": job["stage"],
                "reference": reference_identity,
                "current": current_identity,
                "cache_receipt": _file_identity(cache_path),
            }
        )
    output = _write_csv(output_csv, rows)
    payload = {
        **base,
        "complete": True,
        "claimable": True,
        "status": "complete",
        "protocol": _file_identity(protocol),
        "experiment_matrix": _file_identity(experiment_matrix),
        "matrix_manifest": _file_identity(matrix_manifest),
        "hidden_shapes": canonical_shapes,
        "sources": sources,
        "output": output,
    }
    _atomic_json(manifest_path, payload)
    return payload


def audit_predictors(
    manifest_path: Path,
    *,
    protocol: Path,
    analysis_protocol: Path,
    families: tuple[str, ...],
    scope_amendment: Path,
    experiment_matrix: Path,
    matrix_dir: Path | None,
    output_csv: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True or payload.get("claimable") is not True:
        raise RuntimeError("Temporal predictor receipt is pending/not claimable")
    resolved_families, scope = resolve_scope(families, scope_amendment)
    if resolved_families != ("dense",) or scope is None:
        raise RuntimeError("Temporal predictor audit requires the frozen Dense-only scope")
    _load_spec(analysis_protocol)
    expected_bindings = {
        "protocol": _file_identity(protocol),
        "analysis_protocol": _file_identity(analysis_protocol),
        "scope_amendment": _file_identity(scope_amendment),
        "experiment_matrix": _file_identity(experiment_matrix),
    }
    if payload.get("family") != "dense" or any(
        payload.get(label) != identity for label, identity in expected_bindings.items()
    ):
        raise RuntimeError("Temporal predictor receipt differs from CLI protocol/scope bindings")
    jobs, expected_matrix_manifest = _jobs(protocol, experiment_matrix, matrix_dir)
    if payload.get("matrix_manifest") != _file_identity(expected_matrix_manifest):
        raise RuntimeError("Temporal predictor matrix manifest differs from canonical jobs")
    expected_jobs = {
        (str(job["seed"]), job["operator"], str(job["stage"])): (
            job["reference"].resolve(),
            job["current"].resolve(),
        )
        for job in jobs
    }
    if len(expected_jobs) != 45:
        raise RuntimeError("Temporal predictor canonical job grid differs")
    output = payload.get("output")
    path = Path(output.get("path", "")) if isinstance(output, dict) else Path()
    if (
        not path.is_file()
        or path.resolve() != output_csv.resolve()
        or path.stat().st_size != output.get("bytes")
        or _sha256(path) != output.get("sha256")
        or output.get("rows") != 45
        or output.get("fields") != FIELDS
        or len(_read_rows(path)) != 45
    ):
        raise RuntimeError("Temporal predictor output differs from its manifest")
    rows = _read_rows(path)
    if list(rows[0]) != FIELDS:
        raise RuntimeError("Temporal predictor CSV schema differs")
    expected_grid = {
        (str(seed), operator, str(stage))
        for seed in (314159, 271828, 161803)
        for operator in ("adamw", "muon", "normuon")
        for stage in range(1, 6)
    }
    observed_grid = {(row["seed"], row["operator"], row["stage"]) for row in rows}
    if (
        len(rows) != 45
        or len(observed_grid) != 45
        or observed_grid != expected_grid
        or {row["family"] for row in rows} != {"dense"}
        or any(not math.isfinite(float(row[field])) for row in rows for field in FIELDS[4:])
    ):
        raise RuntimeError("Temporal predictor CSV grid or numeric values differ")
    sources = payload.get("sources")
    source_grid = (
        {(str(row["seed"]), row["operator"], str(row["stage"])) for row in sources}
        if isinstance(sources, list)
        else set()
    )
    if not isinstance(sources, list) or len(sources) != 45 or source_grid != expected_grid:
        raise RuntimeError("Temporal predictor source identities do not match the CSV grid")
    for source in sources:
        identity = (str(source["seed"]), source["operator"], str(source["stage"]))
        expected_reference, expected_current = expected_jobs[identity]
        observed_reference = Path(source.get("reference", {}).get("path", "")).resolve()
        observed_current = Path(source.get("current", {}).get("path", "")).resolve()
        expected_cache = (
            cache_dir / f"seed{source['seed']}__{source['operator']}__stage{source['stage']}.json"
        ).resolve()
        observed_cache = Path(source.get("cache_receipt", {}).get("path", "")).resolve()
        if (
            observed_reference != expected_reference
            or observed_current != expected_current
            or observed_cache != expected_cache
        ):
            raise RuntimeError("Temporal predictor source differs from canonical job paths")
    if payload.get("cache_dir") != str(cache_dir.resolve()):
        raise RuntimeError("Temporal predictor cache directory differs from CLI binding")
    shapes = payload.get("hidden_shapes")
    if (
        not isinstance(shapes, list)
        or len(shapes) != 88
        or len({item.get("name") for item in shapes}) != 88
        or sum(int(item.get("parameters", 0)) for item in shapes) != 110_297_088
        or any(math.prod(item.get("shape", [])) != item.get("parameters") for item in shapes)
    ):
        raise RuntimeError("Temporal predictor hidden shape contract differs")
    if (
        payload.get("selector")
        != {
            "call": "parameter_partition_name(name, ndim) == 'hidden'",
            "expected_tensors": 88,
            "expected_parameters": 110_297_088,
        }
        or payload.get("reference_rule")
        != "stage1-minus-common-start; stage2-5-minus-previous-checkpoint"
        or payload.get("decomposition")
        != "torch.linalg.svdvals float32 exact, one hidden matrix at a time"
        or not isinstance(payload.get("device"), str)
        or not isinstance(payload.get("runtime"), dict)
    ):
        raise RuntimeError("Temporal predictor extraction contract differs")
    row_index = {(row["seed"], row["operator"], row["stage"]): row for row in rows}
    for source in sources:
        for side in ("reference", "current"):
            for item in source[side]["files"]:
                file = Path(item["path"])
                if (
                    not file.is_file()
                    or file.stat().st_size != item["bytes"]
                    or _sha256(file) != item["sha256"]
                ):
                    raise RuntimeError(f"Temporal predictor source differs: {file}")
        receipt = source.get("cache_receipt")
        cache_path = Path(receipt.get("path", "")) if isinstance(receipt, dict) else Path()
        if (
            not cache_path.is_file()
            or cache_path.stat().st_size != receipt.get("bytes")
            or _sha256(cache_path) != receipt.get("sha256")
        ):
            raise RuntimeError("Temporal predictor cache receipt differs")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        identity = (str(source["seed"]), source["operator"], str(source["stage"]))
        expected_metrics = {
            "stable": float(row_index[identity]["update_stable_rank_fraction"]),
            "entropy": float(row_index[identity]["update_entropy_rank_fraction"]),
            "head": float(row_index[identity]["update_head_energy_fraction"]),
            "middle": float(row_index[identity]["update_middle_energy_fraction"]),
            "tail": float(row_index[identity]["update_tail_energy_fraction"]),
            "row_cv": float(row_index[identity]["update_row_norm_cv"]),
            "update_frobenius_norm": float(row_index[identity]["update_frobenius_norm"]),
            "weight_frobenius_norm": float(row_index[identity]["weight_frobenius_norm"]),
        }
        inputs = cache.get("inputs", {})
        if (
            cache.get("complete") is not True
            or cache.get("job")
            != {"seed": source["seed"], "operator": source["operator"], "stage": source["stage"]}
            or inputs.get("reference") != source["reference"]
            or inputs.get("current") != source["current"]
            or inputs.get("analysis_protocol") != payload.get("analysis_protocol")
            or inputs.get("protocol") != payload.get("protocol")
            or inputs.get("scope_amendment") != payload.get("scope_amendment")
            or inputs.get("device") != payload.get("device")
            or inputs.get("torch") != payload.get("runtime")
            or cache.get("metrics") != expected_metrics
            or cache.get("hidden_shapes") != payload.get("hidden_shapes")
        ):
            raise RuntimeError("Temporal predictor cache content differs from final aggregation")
    for label in (
        "protocol",
        "analysis_protocol",
        "experiment_matrix",
        "matrix_manifest",
        "scope_amendment",
    ):
        item = payload.get(label)
        source = Path(item.get("path", "")) if isinstance(item, dict) else Path()
        if (
            not source.is_file()
            or source.stat().st_size != item.get("bytes")
            or _sha256(source) != item.get("sha256")
        ):
            raise RuntimeError(f"Temporal predictor {label} provenance differs")
    return payload


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract exact adjacent-update spectra for Dense short branches"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/short_branch_protocol.json"))
    parser.add_argument(
        "--analysis-protocol",
        type=Path,
        default=Path("configs/causal_chain_analysis.json"),
    )
    parser.add_argument("--families", nargs="+", choices=("dense", "late"), default=["dense"])
    parser.add_argument(
        "--scope-amendment", type=Path, default=Path("configs/dense_scope_amendment.json")
    )
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/short-branch/temporal_mechanism_predictors.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/short-branch/temporal_mechanism_predictors.manifest.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("reports/short-branch/temporal-predictor-cache"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = (
        audit_predictors(
            args.manifest.resolve(),
            protocol=args.protocol.resolve(),
            analysis_protocol=args.analysis_protocol.resolve(),
            families=tuple(args.families),
            scope_amendment=args.scope_amendment.resolve(),
            experiment_matrix=args.experiment_matrix.resolve(),
            matrix_dir=None if args.matrix_dir is None else args.matrix_dir.resolve(),
            output_csv=args.output_csv.resolve(),
            cache_dir=args.cache_dir.resolve(),
        )
        if args.audit
        else build_predictors(
            protocol=args.protocol.resolve(),
            analysis_protocol=args.analysis_protocol.resolve(),
            families=tuple(args.families),
            scope_amendment=args.scope_amendment.resolve(),
            experiment_matrix=args.experiment_matrix.resolve(),
            matrix_dir=None if args.matrix_dir is None else args.matrix_dir.resolve(),
            output_csv=args.output_csv.resolve(),
            manifest_path=args.manifest.resolve(),
            cache_dir=args.cache_dir.resolve(),
            device=args.device,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
