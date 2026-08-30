from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

ALGORITHMS = ("adamw", "muon", "normuon")
CHALLENGERS = ("muon", "normuon")
FAMILIES = ("dense", "late")

REVERSAL_FIELDS = [
    "family",
    "challenger",
    "relative_scale",
    "local_margin_delta_adamw",
    "local_margin_delta_challenger",
    "local_margin_contrast",
    "final_lr_points",
    "final_median_unseen_margin_adamw",
    "final_median_unseen_margin_challenger",
    "final_unseen_margin_contrast",
    "final_median_beir_adamw",
    "final_median_beir_challenger",
    "final_beir_contrast",
    "local_global_reversal",
]

FRONTIER_FIELDS = [
    "family",
    "optimizer",
    "validation_selected_run_id",
    "validation_selected_lr",
    "validation_contrastive_loss",
    "validation_positive_margin",
    "best_discovery_beir_run_id",
    "best_discovery_beir_lr",
    "selected_discovery_beir",
    "best_discovery_beir",
    "discovery_beir_regret",
    "selected_unseen_margin",
    "best_beir_unseen_margin",
    "selected_score_drift_rms",
    "best_beir_score_drift_rms",
    "score_drift_excess",
    "selected_top1_agreement",
    "best_beir_top1_agreement",
    "selection_matches_beir_oracle",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows or any(list(row) != fields for row in rows):
        raise ValueError(f"Cannot write inconsistent or empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {label}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite numeric value for {label}")
    return parsed


def _declared_file(
    manifest_path: Path,
    output_key: str,
    *,
    required: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Source manifest field differs: {manifest_path}/{key}")
    declared = manifest.get("outputs", {}).get(output_key)
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        raise ValueError(f"Source manifest has no declared {output_key}: {manifest_path}")
    path = Path(declared["path"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    else:
        try:
            declared_path_available = path.is_file()
        except OSError:
            declared_path_available = False
        if not declared_path_available:
            # Checked-in report manifests preserve the producer's absolute path for provenance.
            # A clean checkout may live elsewhere, so content-address the colocated copy instead.
            path = manifest_path.parent / path.name
    path = path.resolve()
    if (
        not path.is_file()
        or path.stat().st_size != declared.get("bytes")
        or _sha256(path) != declared.get("sha256")
    ):
        raise ValueError(f"Declared source differs from its manifest: {path}")
    return path, {
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256(manifest_path),
        },
        "table": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        },
    }


def _load_protocol(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    local = protocol.get("local_effect", {})
    long_horizon = protocol.get("long_horizon_effect", {})
    reversal = protocol.get("reversal_definition", {})
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("analysis_status") != "post_hoc_exploratory"
        or not isinstance(protocol.get("frozen_at_utc"), str)
        or local.get("direction") != "descent"
        or _finite(local.get("relative_scale"), "protocol local scale") != 0.001
        or local.get("metric") != "mean_anchor_delta_positive_margin"
        or long_horizon.get("stage") != 5
        or long_horizon.get("aggregation")
        != "median over the four frozen learning-rate points within family and optimizer"
        or long_horizon.get("margin_metric") != "unseen_margin_mean"
        or long_horizon.get("retrieval_metric") != "mean_beir_ndcg_at_10"
        or reversal.get("reference") != "adamw"
        or reversal.get("challengers") != list(CHALLENGERS)
        or not isinstance(protocol.get("claim_boundary"), str)
    ):
        raise ValueError(f"Local-to-global protocol is invalid: {path}")
    return path, protocol


def _load_local_effects(path: Path, protocol: dict[str, Any]) -> dict[tuple[str, str], float]:
    rows = _read_csv(path)
    required_fields = {
        "family",
        "algorithm",
        "direction",
        "relative_scale",
        "anchors",
        "mean_anchor_delta_positive_margin",
    }
    if not rows or not required_fields.issubset(rows[0]):
        raise ValueError("Functional-intervention family summary schema differs")
    direction = protocol["local_effect"]["direction"]
    scale = float(protocol["local_effect"]["relative_scale"])
    indexed: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["direction"] != direction or not math.isclose(
            _finite(row["relative_scale"], "local relative scale"),
            scale,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            continue
        identity = (row["family"], row["algorithm"])
        if identity in indexed or identity[0] not in FAMILIES or identity[1] not in ALGORITHMS:
            raise ValueError(f"Duplicate or unexpected local effect: {identity}")
        if int(row["anchors"]) != 10:
            raise ValueError(f"Local effect does not contain ten anchors: {identity}")
        indexed[identity] = _finite(
            row["mean_anchor_delta_positive_margin"], f"local margin/{identity}"
        )
    expected = {(family, algorithm) for family in FAMILIES for algorithm in ALGORITHMS}
    if set(indexed) != expected:
        raise ValueError("Local effect table does not cover both families and all optimizers")
    return indexed


def _load_final_rows(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows = _read_csv(path)
    required_fields = {
        "model_family",
        "optimizer",
        "learning_rate",
        "run_id",
        "stage",
        "unseen_margin_mean",
        "unseen_reference_top1_agreement",
        "unseen_reference_score_drift_rms",
        "mean_beir_ndcg_at_10",
    }
    if not rows or not required_fields.issubset(rows[0]):
        raise ValueError("Mechanism bridge schema differs")
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_runs: set[tuple[str, str]] = set()
    for row in rows:
        if int(row["stage"]) != 5:
            continue
        family, algorithm = row["model_family"], row["optimizer"]
        if family not in FAMILIES or algorithm not in ALGORITHMS:
            raise ValueError("Final bridge row has an unexpected family or optimizer")
        run_identity = (family, row["run_id"])
        if run_identity in seen_runs:
            raise ValueError(f"Duplicate final bridge run: {run_identity}")
        seen_runs.add(run_identity)
        parsed = {
            "family": family,
            "optimizer": algorithm,
            "run_id": row["run_id"],
            "learning_rate": _finite(row["learning_rate"], f"lr/{run_identity}"),
            "unseen_margin": _finite(row["unseen_margin_mean"], f"margin/{run_identity}"),
            "top1_agreement": _finite(
                row["unseen_reference_top1_agreement"], f"agreement/{run_identity}"
            ),
            "score_drift": _finite(
                row["unseen_reference_score_drift_rms"], f"drift/{run_identity}"
            ),
            "beir": _finite(row["mean_beir_ndcg_at_10"], f"BEIR/{run_identity}"),
        }
        indexed.setdefault((family, algorithm), []).append(parsed)
    expected = {(family, algorithm) for family in FAMILIES for algorithm in ALGORITHMS}
    if set(indexed) != expected or any(len(group) != 4 for group in indexed.values()):
        raise ValueError("Final bridge rows do not contain four LR points for every comparison")
    for group in indexed.values():
        group.sort(key=lambda row: row["learning_rate"])
    return indexed


def _load_selection(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("selected")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "complete"
        or not isinstance(rows, list)
        or len(rows) != 6
    ):
        raise ValueError("Recipe selection is incomplete")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        identity = (row.get("family"), row.get("optimizer"))
        if identity in indexed or identity[0] not in FAMILIES or identity[1] not in ALGORITHMS:
            raise ValueError(f"Duplicate or unexpected recipe selection: {identity}")
        indexed[identity] = {
            "run_id": row.get("run_id"),
            "learning_rate": _finite(row.get("learning_rate"), f"selected lr/{identity}"),
            "validation_loss": _finite(
                row.get("validation_contrastive_loss"), f"selected loss/{identity}"
            ),
            "validation_margin": _finite(
                row.get("validation_positive_margin"), f"selected margin/{identity}"
            ),
        }
    expected = {(family, algorithm) for family in FAMILIES for algorithm in ALGORITHMS}
    if set(indexed) != expected:
        raise ValueError("Recipe selection does not cover every family and optimizer")
    return indexed


def _reversal_rows(
    local: dict[tuple[str, str], float],
    final: dict[tuple[str, str], list[dict[str, Any]]],
    relative_scale: float,
) -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        adam = final[(family, "adamw")]
        adam_margin = statistics.median(row["unseen_margin"] for row in adam)
        adam_beir = statistics.median(row["beir"] for row in adam)
        for challenger in CHALLENGERS:
            candidate = final[(family, challenger)]
            candidate_margin = statistics.median(row["unseen_margin"] for row in candidate)
            candidate_beir = statistics.median(row["beir"] for row in candidate)
            local_contrast = local[(family, challenger)] - local[(family, "adamw")]
            margin_contrast = candidate_margin - adam_margin
            beir_contrast = candidate_beir - adam_beir
            rows.append(
                {
                    "family": family,
                    "challenger": challenger,
                    "relative_scale": relative_scale,
                    "local_margin_delta_adamw": local[(family, "adamw")],
                    "local_margin_delta_challenger": local[(family, challenger)],
                    "local_margin_contrast": local_contrast,
                    "final_lr_points": len(candidate),
                    "final_median_unseen_margin_adamw": adam_margin,
                    "final_median_unseen_margin_challenger": candidate_margin,
                    "final_unseen_margin_contrast": margin_contrast,
                    "final_median_beir_adamw": adam_beir,
                    "final_median_beir_challenger": candidate_beir,
                    "final_beir_contrast": beir_contrast,
                    "local_global_reversal": local_contrast < 0
                    and margin_contrast > 0
                    and beir_contrast > 0,
                }
            )
    return rows


def _frontier_rows(
    final: dict[tuple[str, str], list[dict[str, Any]]],
    selection: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        for optimizer in ALGORITHMS:
            candidates = final[(family, optimizer)]
            selected = selection[(family, optimizer)]
            selected_matches = [row for row in candidates if row["run_id"] == selected["run_id"]]
            if len(selected_matches) != 1 or not math.isclose(
                selected_matches[0]["learning_rate"],
                selected["learning_rate"],
                rel_tol=0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"Selected recipe is absent from final bridge: {family}/{optimizer}"
                )
            selected_result = selected_matches[0]
            best = max(candidates, key=lambda row: (row["beir"], -row["learning_rate"]))
            rows.append(
                {
                    "family": family,
                    "optimizer": optimizer,
                    "validation_selected_run_id": selected["run_id"],
                    "validation_selected_lr": selected["learning_rate"],
                    "validation_contrastive_loss": selected["validation_loss"],
                    "validation_positive_margin": selected["validation_margin"],
                    "best_discovery_beir_run_id": best["run_id"],
                    "best_discovery_beir_lr": best["learning_rate"],
                    "selected_discovery_beir": selected_result["beir"],
                    "best_discovery_beir": best["beir"],
                    "discovery_beir_regret": best["beir"] - selected_result["beir"],
                    "selected_unseen_margin": selected_result["unseen_margin"],
                    "best_beir_unseen_margin": best["unseen_margin"],
                    "selected_score_drift_rms": selected_result["score_drift"],
                    "best_beir_score_drift_rms": best["score_drift"],
                    "score_drift_excess": selected_result["score_drift"] - best["score_drift"],
                    "selected_top1_agreement": selected_result["top1_agreement"],
                    "best_beir_top1_agreement": best["top1_agreement"],
                    "selection_matches_beir_oracle": selected_result["run_id"] == best["run_id"],
                }
            )
    return rows


def _render_markdown(
    reversal: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    claim_boundary: str,
) -> str:
    lines = [
        "# Local steps lose, trajectories win",
        "",
        "This is a **post-hoc exploratory analysis** declared after all 1,680 discovery BEIR units, "
        "the common-state intervention, and the mechanism bridge were complete, but before any "
        "confirmatory or shared-start-branch result existed.",
        "",
        "## Local-to-global reversal",
        "",
        "The local column compares per-tensor Frobenius-matched virtual steps at relative scale "
        "`1e-3`. Long-horizon columns compare final-stage medians over all four frozen learning "
        "rates. Positive values favor the challenger over AdamW.",
        "",
        "| Family | Challenger | Local margin Δ vs AdamW | Final unseen-margin Δ | Final BEIR Δ | Reversal |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in reversal:
        lines.append(
            "| {family} | {challenger} | {local:+.3e} | {margin:+.4f} | {beir:+.4f} | {reversal} |".format(
                family=row["family"],
                challenger=row["challenger"],
                local=row["local_margin_contrast"],
                margin=row["final_unseen_margin_contrast"],
                beir=row["final_beir_contrast"],
                reversal="yes" if row["local_global_reversal"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "All four contrasts reverse sign. The completed native trajectories therefore cannot "
            "be explained by Muon-family directions producing a larger immediate margin increase "
            "under a matched parameter-space step budget.",
            "",
            "## Acquisition–preservation mismatch",
            "",
            "The validation-selected recipe is the prospectively valid choice for confirmation. "
            "The best discovery-BEIR point is shown only as a descriptive oracle and never replaces "
            "that choice.",
            "",
            "| Family | Optimizer | Validation-selected LR | BEIR-oracle LR | Selected BEIR | Oracle BEIR | Regret | Drift excess |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in frontier:
        lines.append(
            "| {family} | {optimizer} | {selected_lr:.0e} | {best_lr:.0e} | {selected:.4f} | "
            "{best:.4f} | {regret:+.4f} | {drift:+.4f} |".format(
                family=row["family"],
                optimizer=row["optimizer"],
                selected_lr=row["validation_selected_lr"],
                best_lr=row["best_discovery_beir_lr"],
                selected=row["selected_discovery_beir"],
                best=row["best_discovery_beir"],
                regret=row["discovery_beir_regret"],
                drift=row["score_drift_excess"],
            )
        )
    lines.extend(
        [
            "",
            "Dense Muon and NorMuon provide the sharpest mismatch: independent validation selects "
            "`3e-3`, yet those recipes lose roughly 0.03 mean BEIR nDCG@10 and approximately double "
            "the unseen score drift relative to the within-optimizer discovery oracle at `3e-4`. "
            "The LateOn mismatch is much smaller at its validation-selected `1e-3` recipe.",
            "",
            "The defensible mechanism hypothesis is consequently trajectory-level: spectral "
            "reweighting changes future gradients and the acquisition–preservation frontier. At "
            "moderate strength this can accumulate useful retrieval margins; at excessive strength "
            "it can optimize the training-domain objective while eroding zero-shot rankings. The "
            "shared-start branches and spectral interventions must decide whether that hypothesis is "
            "causal.",
            "",
            f"> Claim boundary: {claim_boundary}",
            "",
        ]
    )
    return "\n".join(lines)


def build_local_global_reversal(
    protocol_path: Path,
    functional_manifest: Path,
    bridge_manifest: Path,
    validation_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol_path, protocol = _load_protocol(protocol_path)
    functional_path, functional_source = _declared_file(
        functional_manifest,
        "family_summary",
        required={
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "anchors": 20,
            "family_summary_records": 24,
        },
    )
    bridge_path, bridge_source = _declared_file(
        bridge_manifest,
        "checkpoint_bridge",
        required={"schema_version": SCHEMA_VERSION, "complete": True, "checkpoints": 120},
    )
    selection_path, validation_source = _declared_file(
        validation_manifest,
        "recipe_selection",
        required={
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "jobs": 24,
            "selected_recipes": 6,
        },
    )
    local = _load_local_effects(functional_path, protocol)
    final = _load_final_rows(bridge_path)
    selection = _load_selection(selection_path)
    reversal = _reversal_rows(local, final, float(protocol["local_effect"]["relative_scale"]))
    frontier = _frontier_rows(final, selection)
    if len(reversal) != 4 or not all(row["local_global_reversal"] for row in reversal):
        raise ValueError(
            "The declared local-to-global reversal is not present in all four contrasts"
        )
    if len(frontier) != 6:
        raise AssertionError("Acquisition-preservation table cardinality changed")

    output_dir = output_dir.resolve()
    reversal_path = output_dir / "local_to_global_reversal.csv"
    frontier_path = output_dir / "acquisition_preservation_frontier.csv"
    readme_path = output_dir / "README.md"
    _write_csv(reversal_path, REVERSAL_FIELDS, reversal)
    _write_csv(frontier_path, FRONTIER_FIELDS, frontier)
    _write_text(readme_path, _render_markdown(reversal, frontier, protocol["claim_boundary"]))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "analysis_status": protocol["analysis_status"],
        "protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": _sha256(protocol_path),
        },
        "sources": {
            "functional_intervention": functional_source,
            "mechanism_bridge": bridge_source,
            "recipe_validation": validation_source,
        },
        "outputs": {
            "local_to_global_reversal": {
                "path": str(reversal_path),
                "bytes": reversal_path.stat().st_size,
                "sha256": _sha256(reversal_path),
                "rows": len(reversal),
            },
            "acquisition_preservation_frontier": {
                "path": str(frontier_path),
                "bytes": frontier_path.stat().st_size,
                "sha256": _sha256(frontier_path),
                "rows": len(frontier),
            },
            "readme": {
                "path": str(readme_path),
                "bytes": readme_path.stat().st_size,
                "sha256": _sha256(readme_path),
            },
        },
        "reversal_contrasts": len(reversal),
        "reversal_contrasts_observed": sum(row["local_global_reversal"] for row in reversal),
        "frontier_rows": len(frontier),
        "selection_oracle_matches": sum(row["selection_matches_beir_oracle"] for row in frontier),
        "claim_boundary": protocol["claim_boundary"],
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the audited post-hoc local-to-global reversal analysis"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/local_global_reversal.json"))
    parser.add_argument(
        "--functional-manifest",
        type=Path,
        default=Path("reports/functional-intervention/manifest.json"),
    )
    parser.add_argument(
        "--bridge-manifest",
        type=Path,
        default=Path("reports/mechanism-bridge/summary_manifest.json"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("reports/recipe-validation/manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/local-global-reversal"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_local_global_reversal(
        args.protocol,
        args.functional_manifest,
        args.bridge_manifest,
        args.validation_manifest,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
