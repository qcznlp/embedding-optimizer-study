"""Query-disjoint validation with verified Dense input flattening disabled."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import validation_evaluation as _base
from .corrected_input_execution import (
    PADDED_DENSE_RECEIPT,
    require_independently_padded_dense,
)
from .geometry import _atomic_json


def _source_receipt() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "src/embed_optim/corrected_input_execution.py",
        root / "src/embed_optim/validation_evaluation.py",
    )
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    }


def run_corrected_validation_evaluation(
    checkpoint: str | Path,
    probe_root: str | Path,
    output_dir: str | Path,
    *,
    validation_spec: str | Path = "configs/validation_probe.json",
    device: str = "cuda",
    audit_probe: bool = True,
) -> dict:
    """Run the historical metric contract through the corrected padded encoder."""

    original_load_model = _base._load_model

    def load_padded_model(family, *args, **kwargs):
        if family != "dense":
            raise RuntimeError("Corrected validation accepts Dense checkpoints only")
        model = original_load_model(family, *args, **kwargs)
        require_independently_padded_dense(model)
        return model

    _base._load_model = load_padded_model
    try:
        manifest = _base.run_validation_evaluation(
            checkpoint,
            probe_root,
            output_dir,
            family="dense",
            validation_spec=validation_spec,
            device=device,
            audit_probe=audit_probe,
        )
    finally:
        _base._load_model = original_load_model

    source = _source_receipt()
    if "input_execution" in manifest:
        if manifest.get("input_execution") != PADDED_DENSE_RECEIPT:
            raise RuntimeError("Existing corrected validation has a different input mode")
        if manifest.get("corrected_source_files") != source:
            raise RuntimeError(
                "Existing corrected validation was produced by different source files"
            )
        return manifest
    manifest = {
        **manifest,
        "input_execution": dict(PADDED_DENSE_RECEIPT),
        "corrected_source_files": source,
    }
    _atomic_json(Path(output_dir).resolve() / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("data/validation-4096-seed20260826"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-probe-audit", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_corrected_validation_evaluation(
        args.checkpoint,
        args.probe,
        args.output_dir,
        validation_spec=args.validation_spec,
        device=args.device,
        audit_probe=not args.skip_probe_audit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
