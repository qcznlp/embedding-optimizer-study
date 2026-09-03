from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from .config import RunConfig, load_matrix
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

PAIR_CONTRAST_FIELDS = [
    "model_family",
    "learning_rate",
    "stage",
    "step",
    "muon_run_id",
    "normuon_run_id",
    "muon_reference_displacement_frobenius_norm",
    "normuon_reference_displacement_frobenius_norm",
    "normuon_to_muon_displacement_ratio",
    "muon_reference_delta_row_cv_parameter_weighted",
    "normuon_reference_delta_row_cv_parameter_weighted",
    "normuon_to_muon_row_cv_ratio",
    "muon_reference_delta_top_1pct_row_energy_parameter_weighted",
    "normuon_reference_delta_top_1pct_row_energy_parameter_weighted",
    "normuon_to_muon_top_1pct_row_energy_ratio",
]


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _validate_input_files(manifest: dict[str, Any]) -> None:
    for source in manifest["inputs"]:
        for category in ("files", "metadata_files"):
            for item in source.get(category, []):
                path = Path(item["path"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
                    raise ValueError(f"Geometry input differs from its manifest: {path}")


def _read_records(
    analysis_dir: Path,
    metadata: dict[str, Any],
    *,
    expected_step: int,
    expected_tensors: int,
) -> list[dict[str, Any]]:
    path = analysis_dir / metadata["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    if len(raw) != metadata["bytes"] or _sha256(path) != metadata["sha256"]:
        raise ValueError(f"Geometry record differs from its manifest: {path}")
    records = [json.loads(line) for line in raw.splitlines() if line]
    if len(records) != metadata["tensors"] or len(records) != expected_tensors:
        raise ValueError(f"Expected {expected_tensors} records in {path}, found {len(records)}")
    names = {record.get("tensor") for record in records}
    if len(names) != len(records):
        raise ValueError(f"Duplicate tensor names in {path}")
    for record in records:
        if record.get("step") != expected_step or record.get("partition") != "hidden":
            raise ValueError(f"Invalid step or partition in {path}: {record.get('tensor')}")
        if not _all_finite(record):
            raise ValueError(f"Non-finite geometry value in {path}: {record.get('tensor')}")
    return records


def _root_sum_square(records: list[dict[str, Any]], section: str) -> float:
    return math.sqrt(sum(record[section]["frobenius_norm"] ** 2 for record in records))


def _weighted_mean(records: list[dict[str, Any]], section: str, *path: str) -> float:
    parameters = sum(record["parameters"] for record in records)
    total = 0.0
    for record in records:
        value: Any = record[section]
        for key in path:
            value = value[key]
        total += record["parameters"] * value
    return total / parameters


def _optimizer_pair_contrasts(
    checkpoint_rows: list[dict[str, Any]], *, allow_partial: bool
) -> list[dict[str, Any]]:
    indexed_rows: dict[tuple[str, str, float, int, int], dict[str, Any]] = {}
    for row in checkpoint_rows:
        key = (
            row["model_family"],
            row["optimizer"],
            row["learning_rate"],
            row["stage"],
            row["step"],
        )
        if key in indexed_rows:
            raise ValueError(f"Duplicate checkpoint contrast row: {key}")
        indexed_rows[key] = row

    muon_keys = {
        (family, learning_rate, stage, step)
        for family, optimizer, learning_rate, stage, step in indexed_rows
        if optimizer == "muon"
    }
    normuon_keys = {
        (family, learning_rate, stage, step)
        for family, optimizer, learning_rate, stage, step in indexed_rows
        if optimizer == "normuon"
    }
    if not allow_partial and muon_keys and normuon_keys and muon_keys != normuon_keys:
        raise ValueError(
            "Muon and NorMuon rows do not have the same family/learning-rate/checkpoint coverage"
        )

    contrasts: list[dict[str, Any]] = []
    for family, learning_rate, stage, step in sorted(muon_keys & normuon_keys):
        muon = indexed_rows[(family, "muon", learning_rate, stage, step)]
        normuon = indexed_rows[(family, "normuon", learning_rate, stage, step)]
        displacement_muon = muon["reference_displacement_frobenius_norm"]
        row_cv_muon = muon["reference_delta_row_cv_parameter_weighted"]
        top_energy_muon = muon["reference_delta_top_1pct_row_energy_parameter_weighted"]
        if min(displacement_muon, row_cv_muon, top_energy_muon) <= 0:
            raise ValueError(
                f"Muon contrast denominators must be positive for {family} at lr={learning_rate}"
            )
        contrasts.append(
            {
                "model_family": family,
                "learning_rate": learning_rate,
                "stage": stage,
                "step": step,
                "muon_run_id": muon["run_id"],
                "normuon_run_id": normuon["run_id"],
                "muon_reference_displacement_frobenius_norm": displacement_muon,
                "normuon_reference_displacement_frobenius_norm": normuon[
                    "reference_displacement_frobenius_norm"
                ],
                "normuon_to_muon_displacement_ratio": normuon[
                    "reference_displacement_frobenius_norm"
                ]
                / displacement_muon,
                "muon_reference_delta_row_cv_parameter_weighted": row_cv_muon,
                "normuon_reference_delta_row_cv_parameter_weighted": normuon[
                    "reference_delta_row_cv_parameter_weighted"
                ],
                "normuon_to_muon_row_cv_ratio": normuon["reference_delta_row_cv_parameter_weighted"]
                / row_cv_muon,
                "muon_reference_delta_top_1pct_row_energy_parameter_weighted": top_energy_muon,
                "normuon_reference_delta_top_1pct_row_energy_parameter_weighted": normuon[
                    "reference_delta_top_1pct_row_energy_parameter_weighted"
                ],
                "normuon_to_muon_top_1pct_row_energy_ratio": normuon[
                    "reference_delta_top_1pct_row_energy_parameter_weighted"
                ]
                / top_energy_muon,
            }
        )
    return contrasts


def _validate_run_manifest(
    manifest: dict[str, Any],
    config: RunConfig,
    *,
    verify_inputs: bool,
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported geometry schema for {config.model_family}/{config.run_id}")
    run = manifest.get("run", {})
    optimizer = run.get("optimizer", {})
    if (
        run.get("run_id") != config.run_id
        or run.get("model_family") != config.model_family
        or optimizer.get("name") != config.optimizer.name
        or optimizer.get("lr") != config.optimizer.lr
    ):
        raise ValueError(f"Geometry run identity does not match matrix: {config.output_dir}")
    analysis = manifest.get("analysis_config", {})
    if analysis.get("partitions") != ["hidden"]:
        raise ValueError(f"Geometry analysis is not hidden-only: {config.output_dir}")
    partition = manifest.get("partition_summary", {}).get("hidden", {})
    if not partition.get("tensors") or not partition.get("parameters"):
        raise ValueError(f"Missing hidden partition summary: {config.output_dir}")
    if manifest.get("reference") is None:
        raise ValueError(f"Geometry analysis has no pretrained reference: {config.output_dir}")
    checkpoint_inputs = [
        item for item in manifest.get("inputs", []) if item.get("kind") == "checkpoint"
    ]
    if len(checkpoint_inputs) != len(config.checkpoint_fractions):
        raise ValueError(
            f"Expected {len(config.checkpoint_fractions)} checkpoint inputs for {config.run_id}, "
            f"found {len(checkpoint_inputs)}"
        )
    input_steps = {int(item["step"]) for item in checkpoint_inputs}
    record_steps = {int(step) for step in manifest.get("records", {})}
    if input_steps != record_steps:
        raise ValueError(f"Geometry checkpoint inputs and records differ for {config.run_id}")
    if verify_inputs:
        _validate_input_files(manifest)


def summarize_geometry(
    geometry_root: Path,
    output_dir: Path,
    *,
    matrix_path: Path = Path("configs/experiment.yaml"),
    allow_partial: bool = False,
    verify_inputs: bool = False,
) -> dict[str, Any]:
    geometry_root = geometry_root.resolve()
    output_dir = output_dir.resolve()
    configs = load_matrix(matrix_path)
    checkpoint_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    dataset_fingerprints: set[str] = set()

    for config in sorted(
        configs,
        key=lambda item: (
            item.model_family,
            item.optimizer.name,
            item.optimizer.lr,
            item.run_id,
        ),
    ):
        analysis_dir = geometry_root / config.model_family / f"{config.run_id}-exact"
        manifest_path = analysis_dir / "manifest.json"
        if not manifest_path.is_file():
            if allow_partial:
                continue
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_run_manifest(manifest, config, verify_inputs=verify_inputs)
        fingerprint = manifest["run"].get("dataset_fingerprint")
        if not fingerprint:
            raise ValueError(f"Missing dataset fingerprint in {manifest_path}")
        dataset_fingerprints.add(fingerprint)
        source_manifests.append(
            {
                "family": config.model_family,
                "run_id": config.run_id,
                "path": str(manifest_path.relative_to(geometry_root)),
                "sha256": _sha256(manifest_path),
            }
        )

        hidden_tensors = manifest["partition_summary"]["hidden"]["tensors"]
        path_length = 0.0
        final_displacement = 0.0
        records_by_step = manifest["records"]
        steps = sorted(int(step) for step in records_by_step)
        for stage, step in enumerate(steps, start=1):
            records = _read_records(
                analysis_dir,
                records_by_step[str(step)],
                expected_step=step,
                expected_tensors=hidden_tensors,
            )
            weight_norm = _root_sum_square(records, "weight")
            reference_displacement = _root_sum_square(records, "delta_from_reference")
            if stage == 1:
                previous_displacement = None
                segment = reference_displacement
            else:
                previous_displacement = _root_sum_square(records, "delta_from_previous")
                segment = previous_displacement
            path_length += segment
            final_displacement = reference_displacement
            checkpoint_rows.append(
                {
                    "model_family": config.model_family,
                    "optimizer": config.optimizer.name,
                    "learning_rate": config.optimizer.lr,
                    "run_id": config.run_id,
                    "stage": stage,
                    "step": step,
                    "hidden_tensors": hidden_tensors,
                    "hidden_parameters": manifest["partition_summary"]["hidden"]["parameters"],
                    "weight_frobenius_norm": weight_norm,
                    "reference_displacement_frobenius_norm": reference_displacement,
                    "previous_checkpoint_displacement_frobenius_norm": previous_displacement,
                    "reference_displacement_to_weight_ratio": reference_displacement / weight_norm,
                    "weight_row_cv_parameter_weighted": _weighted_mean(
                        records, "weight", "row_norms", "cv"
                    ),
                    "reference_delta_row_cv_parameter_weighted": _weighted_mean(
                        records, "delta_from_reference", "row_norms", "cv"
                    ),
                    "weight_top_1pct_row_energy_parameter_weighted": _weighted_mean(
                        records, "weight", "top_1pct_row_energy"
                    ),
                    "reference_delta_top_1pct_row_energy_parameter_weighted": _weighted_mean(
                        records, "delta_from_reference", "top_1pct_row_energy"
                    ),
                }
            )
        run_rows.append(
            {
                "model_family": config.model_family,
                "optimizer": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "run_id": config.run_id,
                "checkpoints": len(steps),
                "final_step": steps[-1],
                "final_reference_displacement_frobenius_norm": final_displacement,
                "coarse_checkpoint_path_length": path_length,
                "coarse_checkpoint_path_efficiency": final_displacement / path_length,
            }
        )

    if not run_rows:
        raise ValueError(f"No geometry analyses found under {geometry_root}")
    if len(dataset_fingerprints) != 1:
        raise ValueError(
            f"Geometry runs use different dataset fingerprints: {dataset_fingerprints}"
        )

    checkpoint_path = output_dir / "checkpoint_trajectory.csv"
    run_path = output_dir / "run_trajectory_summary.csv"
    contrast_path = output_dir / "optimizer_pair_contrasts.csv"
    contrast_trajectory_path = output_dir / "optimizer_pair_contrast_trajectory.csv"
    contrast_trajectory_rows = _optimizer_pair_contrasts(
        checkpoint_rows, allow_partial=allow_partial
    )
    final_stage_by_pair: dict[tuple[str, float], int] = {}
    for row in contrast_trajectory_rows:
        key = (row["model_family"], row["learning_rate"])
        final_stage_by_pair[key] = max(final_stage_by_pair.get(key, 0), row["stage"])
    contrast_rows = [
        row
        for row in contrast_trajectory_rows
        if row["stage"] == final_stage_by_pair[(row["model_family"], row["learning_rate"])]
    ]
    _atomic_csv(checkpoint_path, checkpoint_rows, list(checkpoint_rows[0]))
    _atomic_csv(run_path, run_rows, list(run_rows[0]))
    _atomic_csv(contrast_path, contrast_rows, PAIR_CONTRAST_FIELDS)
    _atomic_csv(contrast_trajectory_path, contrast_trajectory_rows, PAIR_CONTRAST_FIELDS)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "complete": len(run_rows) == len(configs),
        "expected_runs": len(configs),
        "observed_runs": len(run_rows),
        "checkpoint_rows": len(checkpoint_rows),
        "dataset_fingerprint": next(iter(dataset_fingerprints)),
        "verify_inputs": verify_inputs,
        "sources": source_manifests,
        "outputs": {
            checkpoint_path.name: {
                "rows": len(checkpoint_rows),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            },
            run_path.name: {
                "rows": len(run_rows),
                "bytes": run_path.stat().st_size,
                "sha256": _sha256(run_path),
            },
            contrast_path.name: {
                "rows": len(contrast_rows),
                "bytes": contrast_path.stat().st_size,
                "sha256": _sha256(contrast_path),
            },
            contrast_trajectory_path.name: {
                "rows": len(contrast_trajectory_rows),
                "bytes": contrast_trajectory_path.stat().st_size,
                "sha256": _sha256(contrast_trajectory_path),
            },
        },
        "interpretation": (
            "coarse_checkpoint_path_length joins only saved checkpoints and is not the optimizer's "
            "per-step path length"
        ),
        "contrast_interpretation": (
            "optimizer_pair_contrasts are matched-learning-rate, one-seed integrated trajectories; "
            "they are not individual optimizer updates or causal effects"
        ),
    }
    if not allow_partial and not summary["complete"]:
        raise ValueError(f"Expected {len(configs)} runs, found {len(run_rows)}")
    _atomic_json(output_dir / "summary_manifest.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly aggregate weight-space trajectory records"
    )
    parser.add_argument("--geometry-root", type=Path, default=Path("results/weight-space"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/weight-space"))
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--verify-inputs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    summary = summarize_geometry(
        args.geometry_root,
        args.output_dir,
        matrix_path=args.matrix,
        allow_partial=args.allow_partial,
        verify_inputs=args.verify_inputs,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
