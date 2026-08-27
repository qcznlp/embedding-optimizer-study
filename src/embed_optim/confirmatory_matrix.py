from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .config import load_matrix, resolve_matrix_path
from .confirmatory_data import audit_confirmatory_data, load_confirmatory_protocol
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

OPTIMIZERS = ("adamw", "muon", "normuon")
FAMILIES = ("dense", "late")


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _common_payload(config, *, seed: int, dataset_path: str, output_root: str) -> dict[str, Any]:
    return {
        "dataset_path": dataset_path,
        "output_root": f"{output_root}/seed{seed}",
        "seed": seed,
        "epochs": config.epochs,
        "global_batch_size": config.global_batch_size,
        "micro_batch_size": config.micro_batch_size,
        "max_length": config.max_length,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "dataloader_workers": config.dataloader_workers,
        "gradient_checkpointing": config.gradient_checkpointing,
        "flash_attention": config.flash_attention,
        "wandb_project": config.wandb_project,
        "wandb_entity": config.wandb_entity,
        "checkpoint_fractions": list(config.checkpoint_fractions),
    }


def _load_selection(
    path: Path,
    *,
    validation_spec_path: Path,
    expected_recipes: int,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "complete"
        or not isinstance(selected, list)
        or len(selected) != expected_recipes
        or payload.get("validation_spec", {}).get("sha256") != _sha256(validation_spec_path)
    ):
        raise ValueError(
            "Recipe selection is incomplete or not bound to the frozen validation spec"
        )
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected:
        key = (row.get("family"), row.get("optimizer"))
        optimizer_config = row.get("optimizer_config")
        if (
            key[0] not in FAMILIES
            or key[1] not in OPTIMIZERS
            or key in indexed
            or not isinstance(optimizer_config, dict)
            or optimizer_config.get("name") != key[1]
            or float(optimizer_config.get("lr", -1)) != float(row.get("learning_rate", -2))
        ):
            raise ValueError(f"Invalid selected recipe {row!r}")
        indexed[key] = row
    expected = {(family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS}
    if set(indexed) != expected:
        raise ValueError("Recipe selection does not cover the exact family/optimizer grid")
    return payload, indexed


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def generate_confirmatory_matrices(
    protocol_path: str | Path = "configs/confirmatory_protocol.json",
    *,
    experiment_matrix: str | Path = "configs/experiment.yaml",
    validation_spec: str | Path = "configs/validation_probe.json",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_protocol, protocol = load_confirmatory_protocol(protocol_path)
    source_matrix = resolve_matrix_path(experiment_matrix).resolve()
    validation_spec_path = resolve_matrix_path(validation_spec).resolve()
    selection_path = Path(protocol["recipe_selection"]["source"]).resolve()
    selection, selected = _load_selection(
        selection_path,
        validation_spec_path=validation_spec_path,
        expected_recipes=int(protocol["recipe_selection"]["expected_recipes"]),
    )
    data_receipt = audit_confirmatory_data(resolved_protocol)
    source_runs = load_matrix(source_matrix)
    representatives = {
        family: next(run for run in source_runs if run.model_family == family)
        for family in FAMILIES
    }
    if len(source_runs) != 24:
        raise ValueError(
            "Confirmatory matrices must derive from the complete 24-run discovery grid"
        )
    if any(
        tuple(run.checkpoint_fractions) != tuple(protocol["training"]["checkpoint_fractions"])
        for run in representatives.values()
    ):
        raise ValueError(
            "Confirmatory checkpoint fractions differ from the training implementation"
        )
    for (family, _optimizer), recipe in selected.items():
        reference = representatives[family]
        if (
            recipe.get("model_name") != reference.model_name
            or recipe.get("model_revision") != reference.model_revision
        ):
            raise ValueError(f"Selected {family} recipe refers to a different base model")

    output = Path(output_dir or protocol["training"]["matrix_output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    formal_runtime = source_matrix.parent / "formal_runtime.json"
    view_by_seed = {int(view["seed"]): view for view in data_receipt["views"]}
    matrices = []
    for seed in protocol["confirmatory_data"]["seeds"]:
        seed = int(seed)
        view = view_by_seed.get(seed)
        if view is None:
            raise ValueError(f"Confirmatory data receipt is missing seed {seed}")
        reference = representatives["dense"]
        payload = {
            "formal_runtime": os.path.relpath(formal_runtime, output),
            "common": _common_payload(
                reference,
                seed=seed,
                dataset_path=protocol["confirmatory_data"]["outputs"][str(seed)],
                output_root=protocol["training"]["output_root"],
            ),
            "models": {
                family: {
                    "model_name": representatives[family].model_name,
                    "model_revision": representatives[family].model_revision,
                    "temperature": representatives[family].resolved_temperature,
                }
                for family in FAMILIES
            },
            "runs": [
                {
                    "id": f"{optimizer}-selected",
                    "model_family": family,
                    "optimizer": selected[(family, optimizer)]["optimizer_config"],
                }
                for family in FAMILIES
                for optimizer in OPTIMIZERS
            ],
            "provenance": {
                "confirmatory_protocol_sha256": _sha256(resolved_protocol),
                "recipe_selection_sha256": _sha256(selection_path),
                "data_manifest_sha256": view["manifest_sha256"],
            },
        }
        matrix_path = output / f"seed{seed}.yaml"
        _atomic_yaml(matrix_path, payload)
        matrices.append(_identity(matrix_path))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol": _identity(resolved_protocol),
        "source_matrix": _identity(source_matrix),
        "validation_spec": _identity(validation_spec_path),
        "recipe_selection": _identity(selection_path),
        "recipe_selection_status": selection["status"],
        "confirmatory_data": data_receipt,
        "seeds": protocol["confirmatory_data"]["seeds"],
        "expected_runs": protocol["training"]["expected_runs"],
        "runs_per_seed": protocol["training"]["runs_per_seed"],
        "matrices": matrices,
    }
    _atomic_json(output / "manifest.json", manifest)
    audit_confirmatory_matrices(
        resolved_protocol,
        experiment_matrix=source_matrix,
        validation_spec=validation_spec_path,
        output_dir=output,
    )
    return manifest


def audit_confirmatory_matrices(
    protocol_path: str | Path = "configs/confirmatory_protocol.json",
    *,
    experiment_matrix: str | Path = "configs/experiment.yaml",
    validation_spec: str | Path = "configs/validation_probe.json",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_protocol, protocol = load_confirmatory_protocol(protocol_path)
    source_matrix = resolve_matrix_path(experiment_matrix).resolve()
    validation_spec_path = resolve_matrix_path(validation_spec).resolve()
    output = Path(output_dir or protocol["training"]["matrix_output_dir"]).resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_path = Path(protocol["recipe_selection"]["source"]).resolve()
    _, selected = _load_selection(
        selection_path,
        validation_spec_path=validation_spec_path,
        expected_recipes=int(protocol["recipe_selection"]["expected_recipes"]),
    )
    data_receipt = audit_confirmatory_data(resolved_protocol)
    views = {int(view["seed"]): view for view in data_receipt["views"]}
    expected_matrix_paths = [
        output / f"seed{int(seed)}.yaml" for seed in protocol["confirmatory_data"]["seeds"]
    ]
    expected_identities = [_identity(path) for path in expected_matrix_paths]
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("protocol", {}).get("sha256") != _sha256(resolved_protocol)
        or manifest.get("source_matrix", {}).get("sha256") != _sha256(source_matrix)
        or manifest.get("validation_spec", {}).get("sha256") != _sha256(validation_spec_path)
        or manifest.get("recipe_selection", {}).get("sha256") != _sha256(selection_path)
        or manifest.get("expected_runs") != protocol["training"]["expected_runs"]
        or manifest.get("runs_per_seed") != protocol["training"]["runs_per_seed"]
        or manifest.get("matrices") != expected_identities
    ):
        raise ValueError("Confirmatory matrix manifest is inconsistent")

    observed_runs = 0
    for seed, matrix_path in zip(
        protocol["confirmatory_data"]["seeds"], expected_matrix_paths, strict=True
    ):
        seed = int(seed)
        raw = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
        runs = load_matrix(matrix_path)
        if (
            len(runs) != protocol["training"]["runs_per_seed"]
            or {run.seed for run in runs} != {seed}
            or {run.dataset_path for run in runs}
            != {protocol["confirmatory_data"]["outputs"][str(seed)]}
            or {run.output_root for run in runs}
            != {f"{protocol['training']['output_root']}/seed{seed}"}
            or raw.get("provenance", {}).get("confirmatory_protocol_sha256")
            != _sha256(resolved_protocol)
            or raw.get("provenance", {}).get("recipe_selection_sha256") != _sha256(selection_path)
            or raw.get("provenance", {}).get("data_manifest_sha256")
            != views[seed]["manifest_sha256"]
        ):
            raise ValueError(f"Seed {seed}: generated matrix invariants differ")
        indexed = {(run.model_family, run.optimizer.name): run for run in runs}
        expected_keys = {(family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS}
        if set(indexed) != expected_keys:
            raise ValueError(f"Seed {seed}: generated recipe coverage differs")
        for key, run in indexed.items():
            if run.as_dict()["optimizer"] != selected[key]["optimizer_config"]:
                raise ValueError(f"Seed {seed}: optimizer recipe differs for {key}")
        observed_runs += len(runs)
    if observed_runs != protocol["training"]["expected_runs"]:
        raise ValueError(f"Expected 18 confirmatory runs, found {observed_runs}")
    return {
        "status": "complete",
        "manifest_sha256": _sha256(manifest_path),
        "matrices": len(expected_matrix_paths),
        "runs": observed_runs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or audit validation-selected confirmatory training matrices"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/confirmatory_protocol.json"))
    parser.add_argument("--experiment-matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument(
        "--validation-spec", type=Path, default=Path("configs/validation_probe.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    function = audit_confirmatory_matrices if args.audit_only else generate_confirmatory_matrices
    try:
        result = function(
            args.protocol,
            experiment_matrix=args.experiment_matrix,
            validation_spec=args.validation_spec,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        status = "pending" if isinstance(error, FileNotFoundError) else "invalid"
        print(
            json.dumps(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "mode": "audit" if args.audit_only else "generate",
                    "status": status,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
