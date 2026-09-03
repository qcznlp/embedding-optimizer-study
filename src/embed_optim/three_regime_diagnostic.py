from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

from .geometry import SCHEMA_VERSION

OPTIMIZERS = ("adamw", "muon", "normuon")
RUN_FIELDS = (
    "model_family",
    "optimizer",
    "run_id",
    "learning_rate",
    "trailing_training_loss",
    "validation_contrastive_loss",
    "validation_positive_margin",
    "full_corpus_mean_ndcg_at_10",
)
CONTRAST_FIELDS = (
    "optimizer",
    "validation_selected_run_id",
    "validation_selected_lr",
    "retrieval_oracle_run_id",
    "retrieval_oracle_lr",
    "delta_trailing_training_loss",
    "delta_validation_contrastive_loss",
    "delta_validation_positive_margin",
    "delta_full_corpus_mean_ndcg_at_10",
    "training_fit_worse",
    "validation_shortlist_better",
    "full_corpus_worse",
    "three_regime_reversal",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {label}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value for {label}")
    return result


def _repository_root(protocol_path: Path) -> Path:
    return protocol_path.parent.parent if protocol_path.parent.name == "configs" else Path.cwd()


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    protocol = json.loads(resolved.read_text(encoding="utf-8"))
    timing = protocol.get("timing", {})
    scope = protocol.get("scope", {})
    comparisons = protocol.get("comparisons")
    expected_timing = {
        "discovery_training_losses_visible": True,
        "query_disjoint_validation_metrics_visible": True,
        "discovery_full_corpus_beir_visible": True,
        "confirmatory_final_beir_visible": True,
        "confirmatory_dynamics_results_visible": False,
        "candidate_breadth_data_or_scores_visible": False,
    }
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("status") != "post_hoc_three_regime_diagnostic"
        or timing != expected_timing
        or scope.get("family") != "dense"
        or not isinstance(protocol.get("frozen_at_utc"), str)
        or not isinstance(protocol.get("claim_boundary"), str)
        or not isinstance(comparisons, list)
        or {row.get("optimizer") for row in comparisons} != {"muon", "normuon"}
    ):
        raise ValueError(f"Invalid three-regime diagnostic protocol: {resolved}")

    root = _repository_root(resolved)
    scope_source = scope.get("scope_amendment", {})
    scope_path = _resolve(root, str(scope_source.get("path", "")))
    if not scope_path.is_file() or _sha256(scope_path) != scope_source.get("sha256"):
        raise ValueError("Three-regime scope amendment differs from the frozen source")
    for label, source in protocol.get("sources", {}).items():
        manifest = _resolve(root, str(source.get("manifest", "")))
        if not manifest.is_file() or _sha256(manifest) != source.get("manifest_sha256"):
            raise ValueError(f"Three-regime {label} manifest differs from the frozen source")
    return resolved, protocol


def _declared_output(root: Path, source: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest_path = _resolve(root, str(source["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("outputs", {}).get(source["output_key"])
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        raise ValueError(f"Manifest does not declare {source['output_key']}: {manifest_path}")
    producer_output = Path(declared["path"])
    colocated_output = manifest_path.parent / producer_output.name
    output = colocated_output if colocated_output.is_file() else producer_output
    output = output.resolve()
    if (
        not output.is_file()
        or output.stat().st_size != declared.get("bytes")
        or _sha256(output) != declared.get("sha256")
    ):
        raise ValueError(f"Declared source output differs: {output}")
    return output, {
        "manifest": {
            "path": _portable_path(manifest_path, root),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "output": {
            "path": _portable_path(output, root),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty three-regime source table: {path}")
    return rows


def _joined_run_rows(bridge_path: Path, validation_path: Path) -> list[dict[str, Any]]:
    bridge: dict[str, dict[str, Any]] = {}
    for row in _read_csv(bridge_path):
        if row.get("model_family") != "dense" or int(row.get("stage", 0)) != 5:
            continue
        run_id = row.get("run_id", "")
        if run_id in bridge:
            raise ValueError(f"Duplicate final bridge run: {run_id}")
        optimizer = row.get("optimizer")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unexpected optimizer in final bridge: {optimizer}")
        bridge[run_id] = {
            "model_family": "dense",
            "optimizer": optimizer,
            "run_id": run_id,
            "learning_rate": _finite(row.get("learning_rate"), f"bridge lr/{run_id}"),
            "trailing_training_loss": _finite(
                row.get("mean_training_loss"), f"training loss/{run_id}"
            ),
            "full_corpus_mean_ndcg_at_10": _finite(
                row.get("mean_beir_ndcg_at_10"), f"BEIR/{run_id}"
            ),
        }
    validation: dict[str, dict[str, Any]] = {}
    for row in _read_csv(validation_path):
        if row.get("family") != "dense":
            continue
        run_id = row.get("run_id", "")
        if run_id in validation:
            raise ValueError(f"Duplicate validation run: {run_id}")
        validation[run_id] = {
            "optimizer": row.get("optimizer"),
            "learning_rate": _finite(row.get("learning_rate"), f"validation lr/{run_id}"),
            "validation_contrastive_loss": _finite(
                row.get("contrastive_loss"), f"validation loss/{run_id}"
            ),
            "validation_positive_margin": _finite(
                row.get("positive_margin"), f"validation margin/{run_id}"
            ),
        }
    if set(bridge) != set(validation) or len(bridge) != 12:
        raise ValueError("Three-regime sources do not cover the same 12 DenseOn runs")

    rows: list[dict[str, Any]] = []
    for run_id in sorted(bridge):
        left, right = bridge[run_id], validation[run_id]
        if left["optimizer"] != right["optimizer"] or not math.isclose(
            left["learning_rate"], right["learning_rate"], rel_tol=0, abs_tol=1e-15
        ):
            raise ValueError(f"Three-regime source identity differs for {run_id}")
        rows.append(
            {
                **left,
                "validation_contrastive_loss": right["validation_contrastive_loss"],
                "validation_positive_margin": right["validation_positive_margin"],
            }
        )
    return [{field: row[field] for field in RUN_FIELDS} for row in rows]


def _contrast_rows(
    rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    indexed = {str(row["run_id"]): row for row in rows}
    output = []
    for comparison in comparisons:
        optimizer = str(comparison["optimizer"])
        selected = indexed.get(str(comparison["validation_selected_run_id"]))
        oracle = indexed.get(str(comparison["retrieval_oracle_run_id"]))
        if (
            selected is None
            or oracle is None
            or selected["optimizer"] != optimizer
            or oracle["optimizer"] != optimizer
        ):
            raise ValueError(f"Invalid three-regime comparison for {optimizer}")
        train_delta = selected["trailing_training_loss"] - oracle["trailing_training_loss"]
        validation_loss_delta = (
            selected["validation_contrastive_loss"] - oracle["validation_contrastive_loss"]
        )
        validation_margin_delta = (
            selected["validation_positive_margin"] - oracle["validation_positive_margin"]
        )
        retrieval_delta = (
            selected["full_corpus_mean_ndcg_at_10"] - oracle["full_corpus_mean_ndcg_at_10"]
        )
        training_worse = train_delta > 0
        validation_better = validation_loss_delta < 0 and validation_margin_delta > 0
        corpus_worse = retrieval_delta < 0
        record = {
            "optimizer": optimizer,
            "validation_selected_run_id": selected["run_id"],
            "validation_selected_lr": selected["learning_rate"],
            "retrieval_oracle_run_id": oracle["run_id"],
            "retrieval_oracle_lr": oracle["learning_rate"],
            "delta_trailing_training_loss": train_delta,
            "delta_validation_contrastive_loss": validation_loss_delta,
            "delta_validation_positive_margin": validation_margin_delta,
            "delta_full_corpus_mean_ndcg_at_10": retrieval_delta,
            "training_fit_worse": training_worse,
            "validation_shortlist_better": validation_better,
            "full_corpus_worse": corpus_worse,
            "three_regime_reversal": training_worse and validation_better and corpus_worse,
        }
        output.append({field: record[field] for field in CONTRAST_FIELDS})
    return sorted(output, key=lambda row: str(row["optimizer"]))


def _csv_bytes(fields: tuple[str, ...], rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _markdown(contrasts: list[dict[str, Any]], claim_boundary: str) -> str:
    lines = [
        "# Training, shortlist validation, and full-corpus retrieval disagree",
        "",
        "This is a **post-hoc descriptive diagnostic**. The contrast is the query-disjoint "
        "validation-selected `3e-3` run minus the within-optimizer discovery-BEIR oracle at "
        "`3e-4`.",
        "",
        "| Optimizer | Δ trailing train loss | Δ validation loss | Δ validation margin | Δ BEIR nDCG@10 | Three-regime reversal |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in contrasts:
        lines.append(
            "| {optimizer} | {train:+.4f} | {validation:+.4f} | {margin:+.4f} | "
            "{beir:+.4f} | {decision} |".format(
                optimizer=row["optimizer"],
                train=row["delta_trailing_training_loss"],
                validation=row["delta_validation_contrastive_loss"],
                margin=row["delta_validation_positive_margin"],
                beir=row["delta_full_corpus_mean_ndcg_at_10"],
                decision="yes" if row["three_regime_reversal"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "For both Muon and NorMuon, the larger dose fits the sampled training tuples less "
            "well, generalizes better to the held-out eight-way shortlist, and retrieves worse "
            "against the complete corpus. This rules out simple training-set memorization as a "
            "sufficient explanation, but it does not identify missing-candidate coverage causally.",
            "",
            f"> Claim boundary: {claim_boundary}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_or_audit(
    path: Path, payload: bytes, *, audit_only: bool, repository_root: Path
) -> dict[str, Any]:
    if audit_only:
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Three-regime output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {
        "path": _portable_path(path, repository_root),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_diagnostic(
    protocol_path: str | Path = "configs/three_regime_diagnostic.json",
    *,
    output_dir: str | Path = "reports/three-regime-diagnostic",
    audit_only: bool = False,
) -> dict[str, Any]:
    resolved, protocol = load_protocol(protocol_path)
    root = _repository_root(resolved)
    bridge_path, bridge_source = _declared_output(root, protocol["sources"]["mechanism_bridge"])
    validation_path, validation_source = _declared_output(
        root, protocol["sources"]["query_disjoint_validation"]
    )
    rows = _joined_run_rows(bridge_path, validation_path)
    contrasts = _contrast_rows(rows, protocol["comparisons"])
    if len(contrasts) != 2 or not all(row["three_regime_reversal"] for row in contrasts):
        decision = "not_supported"
    else:
        decision = "observed_for_both_muon_family_optimizers"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "analysis_status": protocol["status"],
        "decision": decision,
        "runs": len(rows),
        "comparisons": len(contrasts),
        "timing": protocol["timing"],
        "claim_boundary": protocol["claim_boundary"],
        "sources": {
            "mechanism_bridge": bridge_source,
            "query_disjoint_validation": validation_source,
        },
    }
    output = Path(output_dir).resolve()
    artifacts = {
        "run_metrics": _write_or_audit(
            output / "run_metrics.csv",
            _csv_bytes(RUN_FIELDS, rows),
            audit_only=audit_only,
            repository_root=root,
        ),
        "high_dose_contrasts": _write_or_audit(
            output / "high_dose_contrasts.csv",
            _csv_bytes(CONTRAST_FIELDS, contrasts),
            audit_only=audit_only,
            repository_root=root,
        ),
        "summary": _write_or_audit(
            output / "summary.json",
            _json_bytes(summary),
            audit_only=audit_only,
            repository_root=root,
        ),
        "readme": _write_or_audit(
            output / "README.md",
            _markdown(contrasts, protocol["claim_boundary"]).encode("utf-8"),
            audit_only=audit_only,
            repository_root=root,
        ),
    }
    manifest = {
        **summary,
        "protocol": {
            "path": _portable_path(resolved, root),
            "bytes": resolved.stat().st_size,
            "sha256": _sha256(resolved),
        },
        "outputs": artifacts,
    }
    manifest_path = output / "summary_manifest.json"
    _write_or_audit(
        manifest_path,
        _json_bytes(manifest),
        audit_only=audit_only,
        repository_root=root,
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or audit the post-hoc training/validation/corpus diagnostic"
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/three_regime_diagnostic.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/three-regime-diagnostic"))
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_diagnostic(args.protocol, output_dir=args.output_dir, audit_only=args.audit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
