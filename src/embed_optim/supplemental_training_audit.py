from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .aggregate import audit_training_artifacts
from .config import RunConfig
from .geometry import _sha256


def _load_json(path: Path, *, context: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{context}: missing/invalid JSON ({error})")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{context}: expected a JSON object")
        return None
    return payload


def audit_derived_training_artifacts(
    configs: list[RunConfig],
    dataset_receipt: dict[str, Any],
    *,
    deep: bool = True,
) -> dict[str, Any]:
    """Bind deep training validation to a specialized derived-dataset receipt.

    Confirmatory views and the 50K shared-start branch intentionally use a
    provenance-rich manifest schema with ``rows`` instead of the discovery
    manifest's ``total_queries``. The shared checkpoint auditor predates that
    schema and therefore emits one predictable row-count error per otherwise
    valid run. This adapter removes only that schema-only error after independently
    verifying the derived manifest hash, copied manifest, exact row count, and
    training-view fingerprint.
    """

    errors: list[str] = []
    rows = dataset_receipt.get("rows")
    fingerprint = dataset_receipt.get("training_view_fingerprint")
    manifest_sha256 = dataset_receipt.get("manifest_sha256")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        errors.append("derived dataset receipt has an invalid row count")
    if not isinstance(fingerprint, str) or not fingerprint:
        errors.append("derived dataset receipt has an invalid training-view fingerprint")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        errors.append("derived dataset receipt has an invalid manifest SHA-256")
    if errors:
        return {
            "complete": False,
            "verified_runs": 0,
            "expected_runs": len(configs),
            "verified_checkpoints": 0,
            "expected_checkpoints": len(configs) * 5,
            "deep_validation": deep,
            "errors": errors,
        }

    generic = audit_training_artifacts(
        configs,
        deep=deep,
        expected_dataset_fingerprint=fingerprint,
    )
    remaining = list(generic["errors"])
    for config in configs:
        label = f"{config.model_family}/{config.run_id}"
        source_manifest_path = Path(config.dataset_path) / "manifest.json"
        copied_manifest_path = config.output_dir / "dataset_manifest.json"
        completion_path = config.output_dir / "completed.json"
        local_errors: list[str] = []
        source = _load_json(
            source_manifest_path,
            context=f"{label}: derived source manifest",
            errors=local_errors,
        )
        copied = _load_json(
            copied_manifest_path,
            context=f"{label}: copied derived manifest",
            errors=local_errors,
        )
        completed = _load_json(
            completion_path,
            context=f"{label}: completion receipt",
            errors=local_errors,
        )
        if source is not None:
            if _sha256(source_manifest_path) != manifest_sha256:
                local_errors.append(f"{label}: derived source manifest differs from its receipt")
            if source.get("rows") != rows or "total_queries" in source:
                local_errors.append(f"{label}: derived source manifest row schema differs")
        if source is not None and copied is not None and copied != source:
            local_errors.append(f"{label}: copied derived manifest differs from source")
        if completed is not None:
            if completed.get("dataset_rows") != rows:
                local_errors.append(f"{label}: completion row count differs from derived dataset")
            if completed.get("dataset_fingerprint") != fingerprint:
                local_errors.append(
                    f"{label}: completion training-view fingerprint differs from derived dataset"
                )

        schema_error = f"{label}: completion dataset row count does not match manifest"
        if not local_errors and schema_error in remaining:
            remaining.remove(schema_error)
        errors.extend(local_errors)

    errors = [*remaining, *errors]
    errored_runs = {
        f"{config.model_family}/{config.run_id}"
        for config in configs
        if any(
            error.startswith(f"{config.model_family}/{config.run_id}:")
            or error.startswith(f"{config.model_family}/{config.run_id}/")
            for error in errors
        )
    }
    result = dict(generic)
    result["errors"] = errors
    result["complete"] = not errors
    result["verified_runs"] = max(0, len(configs) - len(errored_runs))
    result["derived_dataset_rows"] = rows
    result["derived_training_view_fingerprint"] = fingerprint
    return result
