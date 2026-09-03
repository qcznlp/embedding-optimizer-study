"""Run source-bound weight-space analysis for corrected Dense checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .aggregate import audit_dataset_artifacts, audit_training_artifacts
from .config import RunConfig, load_matrix, resolve_matrix_path
from .corrected_geometry_summary import summarize_corrected_geometry
from .corrected_input_execution import require_corrected_training_receipt
from .geometry import _sha256, analyze_run
from .matrix import _run_is_complete


def _load_protocol(path: Path, repository: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "prospective_corrected_analysis_lock":
        raise ValueError(f"Unexpected corrected analysis protocol status: {path}")
    for identity in protocol.get("source_bindings", {}).values():
        source = repository / identity["path"]
        if (
            not source.is_file()
            or source.stat().st_size != int(identity["bytes"])
            or _sha256(source) != identity["sha256"]
        ):
            raise ValueError(f"Corrected analysis source binding mismatch: {source}")
    for identity in protocol.get("parent_bindings", {}).values():
        source = repository / identity["path"]
        if not source.is_file() or _sha256(source) != identity["sha256"]:
            raise ValueError(f"Corrected analysis parent binding mismatch: {source}")
    return protocol


def _validate_matrix(configs: list[RunConfig]) -> None:
    if (
        len(configs) != 12
        or any(config.model_family != "dense" for config in configs)
        or any(config.dense_can_flatten_inputs for config in configs)
        or {config.optimizer.name for config in configs} != {"adamw", "muon", "normuon"}
        or any(len(config.checkpoint_fractions) != 5 for config in configs)
    ):
        raise ValueError("Corrected geometry requires the frozen 12-run padded Dense matrix")


def _selected_complete_configs(
    configs: list[RunConfig], run_ids: list[str], *, allow_partial: bool
) -> list[RunConfig]:
    requested = set(run_ids)
    unknown = requested - {config.run_id for config in configs}
    if unknown:
        raise ValueError(f"Unknown corrected run IDs: {sorted(unknown)}")
    candidates = [config for config in configs if not requested or config.run_id in requested]
    complete = []
    incomplete = []
    for config in candidates:
        if _run_is_complete(config):
            receipt = json.loads((config.output_dir / "completed.json").read_text(encoding="utf-8"))
            require_corrected_training_receipt(receipt)
            complete.append(config)
        else:
            incomplete.append(config.run_id)
    if incomplete and not allow_partial:
        raise RuntimeError(f"Corrected training is incomplete: {incomplete}")
    if not complete:
        raise RuntimeError("No deeply complete corrected runs are available for geometry")
    return complete


def _audit_training(configs: list[RunConfig]) -> None:
    dataset = audit_dataset_artifacts(configs)
    if not dataset["complete"]:
        raise RuntimeError("Corrected geometry training-data audit failed")
    training = audit_training_artifacts(
        configs,
        deep=True,
        expected_dataset_fingerprint=dataset.get("training_view_fingerprint"),
    )
    if not training["complete"]:
        raise RuntimeError(
            "Corrected geometry checkpoint audit failed: " + "; ".join(training["errors"][:5])
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path, repository)
    matrix_path = resolve_matrix_path(args.matrix).resolve()
    matrix_identity = protocol["parent_bindings"]["matrix"]
    if _sha256(matrix_path) != matrix_identity["sha256"]:
        raise ValueError("Corrected geometry matrix differs from the frozen analysis protocol")
    configs = load_matrix(matrix_path)
    _validate_matrix(configs)
    selected = _selected_complete_configs(configs, args.run_ids, allow_partial=args.allow_partial)
    _audit_training(selected)
    reference = Path(
        snapshot_download(
            repo_id=configs[0].model_name,
            revision=configs[0].model_revision,
            local_files_only=args.local_files_only,
        )
    ).resolve()
    settings = protocol["weight_space"]
    geometry_root = args.geometry_root.resolve()
    manifests = []
    for config in selected:
        output = geometry_root / "dense" / f"{config.run_id}-rank64"
        manifest = analyze_run(
            config.output_dir,
            output,
            reference=reference,
            partitions=("hidden",),
            sketch_rank=int(settings["sketch_rank"]),
            oversample=int(settings["oversample"]),
            power_iterations=int(settings["power_iterations"]),
            seed=int(settings["seed"]),
        )
        manifests.append(
            {
                "run_id": config.run_id,
                "path": str((output / "manifest.json").resolve()),
                "checkpoints": len(manifest["records"]),
            }
        )
    summary = None
    if len(selected) == len(configs) and not args.skip_summary:
        summary = summarize_corrected_geometry(
            geometry_root,
            args.output_dir.resolve(),
            configs,
            reference,
            protocol_path=protocol_path,
            sketch_rank=int(settings["sketch_rank"]),
            subspace_rank=int(settings["subspace_rank"]),
            oversample=int(settings["oversample"]),
            power_iterations=int(settings["power_iterations"]),
            seed=int(settings["seed"]),
        )
    return {
        "status": "complete" if len(selected) == len(configs) else "partial",
        "analyzed_runs": len(selected),
        "expected_runs": len(configs),
        "manifests": manifests,
        "summary": summary,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", type=Path, default=Path("configs/dense_no_packing_retrain.yaml")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/dense_no_packing_analysis_protocol.json"),
    )
    parser.add_argument(
        "--geometry-root", type=Path, default=Path("results/dense-no-packing-weight-space")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/dense-no-packing-weight-space")
    )
    parser.add_argument("--run-ids", nargs="*", default=[])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
