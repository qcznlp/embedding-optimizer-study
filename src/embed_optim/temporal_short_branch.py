from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import SCHEMA_VERSION, _atomic_json, _sha256
from .tail_stability import SHORT_BRANCH_FIELDS

IDENTITY = ("family", "seed", "operator", "stage")


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Refusing to write empty temporal table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("Temporal table rows have inconsistent schemas")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return _identity(path)


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {label}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite value for {label}")
    return result


def _load_spec(path: Path) -> dict[str, Any]:
    root = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(root)
        != {
            "schema_version",
            "status",
            "frozen_at_utc",
            "family",
            "source_bindings",
            "freeze_context",
            "amendments",
            "temporal_short_branch",
            "dose_band",
        }
        or root.get("schema_version") != 1
        or root.get("family") != "dense"
    ):
        raise ValueError("Causal-chain protocol is malformed or non-Dense")
    amendments = root.get("amendments")
    if (
        not isinstance(amendments, list)
        or len(amendments) != 3
        or amendments[0].get("previous_protocol_sha256")
        != "74d92cf270bf1a1006cad7f2ed17705631adf44899fd31c4d23ee1f6632da9fa"
        or amendments[0].get("short_branch_results_available") is not False
        or amendments[0].get("spectral_transplant_results_available") is not False
        or amendments[0].get("confirmatory_beir_results_available") is not False
        or amendments[1].get("short_branch_results_available") is not False
        or amendments[1].get("previous_protocol_sha256")
        != "20758007b4a0754e416f11ac74d55c01432d95ad97dbfdd7fdbb89038eef1037"
        or amendments[1].get("spectral_transplant_results_available") is not False
        or amendments[1].get("confirmatory_beir_results_available") is not False
        or amendments[2].get("previous_protocol_sha256")
        != "cd881dd66804b49a6897ec56a3a31216f4c15ad4449607d47486b25214d337b6"
        or amendments[2].get("short_branch_results_available") is not False
        or amendments[2].get("spectral_transplant_results_available") is not False
        or amendments[2].get("confirmatory_beir_results_available") is not False
    ):
        raise ValueError("Causal-chain amendment disclosure differs")
    repository = path.resolve().parent.parent
    bindings = root.get("source_bindings")
    if not isinstance(bindings, list) or len(bindings) != 8:
        raise ValueError("Causal-chain source bindings differ from the frozen protocol")
    for item in bindings:
        source = (repository / item.get("path", "")).resolve()
        if not source.is_file() or _sha256(source) != item.get("sha256"):
            raise ValueError(f"Causal-chain source binding differs: {source}")
    temporal = root.get("temporal_short_branch")
    if not isinstance(temporal, dict):
        raise ValueError("Causal-chain protocol lacks temporal_short_branch")
    units = temporal.get("randomized_units", {})
    extraction = temporal.get("predictor_extraction", {})
    analysis = temporal.get("analysis", {})
    if units.get("seeds") != [314159, 271828, 161803] or units.get("operators") != [
        "adamw",
        "muon",
        "normuon",
    ]:
        raise ValueError("Temporal short-branch randomized units changed")
    if analysis.get("early_predictor_stages") != [1, 2] or analysis.get("final_outcome_stage") != 5:
        raise ValueError("Temporal predictor/outcome stages changed")
    predictors = [analysis.get("primary_predictor"), *analysis.get("secondary_predictors", [])]
    controls = analysis.get("negative_controls")
    outcomes = analysis.get("outcomes")
    if (
        predictors
        != [
            "update_tail_energy_fraction",
            "update_stable_rank_fraction",
            "update_entropy_rank_fraction",
            "update_head_energy_fraction",
            "update_middle_energy_fraction",
            "update_row_norm_cv",
        ]
        or set(predictors) != set(extraction.get("per_tensor_metrics", {}))
        or controls != ["update_frobenius_norm", "weight_frobenius_norm"]
        or outcomes
        != {"validation_loss_p95": "lower_is_better", "unseen_margin_p05": "higher_is_better"}
    ):
        raise ValueError("Temporal predictor, control, or outcome contract changed")
    return {
        "seeds": units["seeds"],
        "operators": units["operators"],
        "predictors": predictors,
        "negative_controls": controls,
        "outcomes": list(outcomes),
        "beneficial_direction": {
            key: "negative" if value == "lower_is_better" else "positive"
            for key, value in outcomes.items()
        },
        "analysis": {
            **analysis,
            "claim_rule": analysis["primary_support_rule"]["decision"],
        },
        "claim_boundary": temporal["claim_boundary"],
    }


def _verify_declared_csv(
    manifest_path: Path, csv_path: Path, *, expected_rows: int
) -> list[dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("output")
    declared_path = Path(declared.get("path", "")) if isinstance(declared, dict) else Path()
    if not declared_path.is_absolute():
        declared_path = manifest_path.resolve().parent / declared_path
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or not isinstance(declared, dict)
        or declared_path.resolve() != csv_path.resolve()
        or declared.get("bytes") != csv_path.stat().st_size
        or declared.get("sha256") != _sha256(csv_path)
        or declared.get("rows") != expected_rows
    ):
        raise ValueError(f"Input manifest does not bind the expected table: {manifest_path}")
    rows = _read_csv(csv_path)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {csv_path}, found {len(rows)}")
    return rows


def _load_inputs(
    spec: dict[str, Any],
    predictor_csv: Path,
    predictor_manifest: Path,
    outcome_csv: Path,
    outcome_manifest: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictors = _verify_declared_csv(predictor_manifest, predictor_csv, expected_rows=45)
    outcome_manifest_payload = json.loads(outcome_manifest.read_text(encoding="utf-8"))
    declared = (outcome_manifest_payload.get("outputs") or {}).get("short_branch_checkpoint_tail")
    declared_path = Path(declared.get("path", "")) if isinstance(declared, dict) else Path()
    if not declared_path.is_absolute():
        declared_path = outcome_manifest.resolve().parent / declared_path
    if (
        outcome_manifest_payload.get("schema_version") != SCHEMA_VERSION
        or outcome_manifest_payload.get("complete") is not True
        or not isinstance(declared, dict)
        or declared_path.resolve() != outcome_csv.resolve()
        or declared.get("sha256") != _sha256(outcome_csv)
        or declared.get("bytes") != outcome_csv.stat().st_size
        or declared.get("rows") != 45
    ):
        raise ValueError("Tail-stability manifest does not bind 45 Dense checkpoint outcomes")
    outcomes = _read_csv(outcome_csv)
    if len(outcomes) != 45 or list(outcomes[0]) != SHORT_BRANCH_FIELDS:
        raise ValueError("Temporal outcome table schema or cardinality differs")
    expected = {
        ("dense", str(seed), operator, str(stage))
        for seed in spec["seeds"]
        for operator in spec["operators"]
        for stage in range(1, 6)
    }
    predictor_fields = [*IDENTITY, *spec["predictors"], *spec["negative_controls"]]
    if not predictors or list(predictors[0]) != predictor_fields:
        raise ValueError("Temporal predictor table schema differs from the frozen protocol")
    for label, rows in (("predictor", predictors), ("outcome", outcomes)):
        observed = {(row["family"], row["seed"], row["operator"], row["stage"]) for row in rows}
        if observed != expected or len(observed) != 45:
            raise ValueError(f"Temporal {label} identities do not cover the frozen 45 checkpoints")
    sources = [
        _identity(path)
        for path in (predictor_manifest, predictor_csv, outcome_manifest, outcome_csv)
    ]
    return predictors, outcomes, sources


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def analyze_rows(
    spec: dict[str, Any], predictors: list[dict[str, str]], outcomes: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pidx = {(int(r["seed"]), r["operator"], int(r["stage"])): r for r in predictors}
    oidx = {(int(r["seed"]), r["operator"], int(r["stage"])): r for r in outcomes}
    mediators = [*spec["predictors"], *spec["negative_controls"]]
    units = []
    for seed in spec["seeds"]:
        for operator in spec["operators"]:
            row = {"seed": seed, "operator": operator}
            for name in mediators:
                row[name] = statistics.fmean(
                    _finite(pidx[(seed, operator, stage)][name], name) for stage in (1, 2)
                )
            for outcome in spec["outcomes"]:
                row[outcome] = _finite(oidx[(seed, operator, 5)][outcome], outcome)
            units.append(row)
    paired = []
    by_unit = {(r["seed"], r["operator"]): r for r in units}
    for seed in spec["seeds"]:
        adam = by_unit[(seed, "adamw")]
        for challenger in ("muon", "normuon"):
            current = by_unit[(seed, challenger)]
            paired.append(
                {
                    "seed": seed,
                    "challenger": challenger,
                    "reference": "adamw",
                    **{
                        f"delta_{name}": current[name] - adam[name]
                        for name in [*mediators, *spec["outcomes"]]
                    },
                }
            )

    predictions = []
    estimates = []
    for outcome in spec["outcomes"]:
        for mediator in mediators:
            for held_seed in spec["seeds"]:
                train = [r for r in paired if r["seed"] != held_seed]
                test = [r for r in paired if r["seed"] == held_seed]
                mean = statistics.fmean(r[f"delta_{mediator}"] for r in train)
                scale = statistics.pstdev(r[f"delta_{mediator}"] for r in train) or 1.0
                x = np.asarray(
                    [
                        [1.0, r["challenger"] == "normuon", (r[f"delta_{mediator}"] - mean) / scale]
                        for r in train
                    ],
                    dtype=float,
                )
                y = np.asarray([r[f"delta_{outcome}"] for r in train], dtype=float)
                beta = _ols(x, y)
                baseline = {
                    c: statistics.fmean(
                        r[f"delta_{outcome}"] for r in train if r["challenger"] == c
                    )
                    for c in ("muon", "normuon")
                }
                for r in test:
                    prediction = float(
                        np.dot(
                            [
                                1.0,
                                r["challenger"] == "normuon",
                                (r[f"delta_{mediator}"] - mean) / scale,
                            ],
                            beta,
                        )
                    )
                    actual = r[f"delta_{outcome}"]
                    predictions.append(
                        {
                            "outcome": outcome,
                            "predictor": mediator,
                            "held_out_seed": held_seed,
                            "challenger": r["challenger"],
                            "actual": actual,
                            "label_only_prediction": baseline[r["challenger"]],
                            "mediator_prediction": prediction,
                            "label_only_squared_error": (actual - baseline[r["challenger"]]) ** 2,
                            "mediator_squared_error": (actual - prediction) ** 2,
                        }
                    )
            members = [
                r for r in predictions if r["outcome"] == outcome and r["predictor"] == mediator
            ]
            base_rmse = math.sqrt(statistics.fmean(r["label_only_squared_error"] for r in members))
            med_rmse = math.sqrt(statistics.fmean(r["mediator_squared_error"] for r in members))
            raw_x = np.asarray(
                [[1.0, r["operator"] == "muon", r["operator"] == "normuon"] for r in units], float
            )
            raw_y = np.asarray([r[outcome] for r in units], float)
            base_beta = _ols(raw_x, raw_y)
            values = np.asarray([r[mediator] for r in units], float)
            z = (values - values.mean()) / (values.std() or 1.0)
            full_beta = _ols(np.column_stack([raw_x, z]), raw_y)
            estimates.append(
                {
                    "outcome": outcome,
                    "predictor": mediator,
                    "predictor_kind": "mechanism"
                    if mediator in spec["predictors"]
                    else "negative_control",
                    "label_only_rmse": base_rmse,
                    "mediator_rmse": med_rmse,
                    "relative_rmse_improvement": (base_rmse - med_rmse) / base_rmse
                    if base_rmse
                    else 0.0,
                    "muon_coefficient_label_only": base_beta[1],
                    "muon_coefficient_with_predictor": full_beta[1],
                    "muon_absolute_coefficient_shrinkage": 1 - abs(full_beta[1]) / abs(base_beta[1])
                    if base_beta[1]
                    else 0.0,
                    "normuon_coefficient_label_only": base_beta[2],
                    "normuon_coefficient_with_predictor": full_beta[2],
                    "normuon_absolute_coefficient_shrinkage": 1
                    - abs(full_beta[2]) / abs(base_beta[2])
                    if base_beta[2]
                    else 0.0,
                }
            )
    return paired, predictions, estimates


def support_decision(
    spec: dict[str, Any], paired: list[dict[str, Any]], estimates: list[dict[str, Any]]
) -> dict[str, Any]:
    primary = spec["analysis"]["primary_predictor"]
    indexed = {(row["outcome"], row["predictor"]): row for row in estimates}
    treatment_shift = all(
        sum(row["challenger"] == challenger and row[f"delta_{primary}"] > 0 for row in paired) >= 2
        for challenger in ("muon", "normuon")
    )
    outcome_shift = all(
        sum(
            row["challenger"] == challenger
            and all(
                row[f"delta_{outcome}"] < 0
                if spec["beneficial_direction"][outcome] == "negative"
                else row[f"delta_{outcome}"] > 0
                for outcome in spec["outcomes"]
            )
            for row in paired
        )
        >= 2
        for challenger in ("muon", "normuon")
    )
    held_out = all(
        indexed[(outcome, primary)]["relative_rmse_improvement"] > 0 for outcome in spec["outcomes"]
    )
    negative_control = all(
        indexed[(outcome, control)]["relative_rmse_improvement"]
        < indexed[(outcome, primary)]["relative_rmse_improvement"]
        for outcome in spec["outcomes"]
        for control in spec["negative_controls"]
    )
    coefficient_behavior = all(
        abs(indexed[(outcome, primary)][f"{challenger}_coefficient_with_predictor"])
        <= abs(indexed[(outcome, primary)][f"{challenger}_coefficient_label_only"])
        for outcome in spec["outcomes"]
        for challenger in ("muon", "normuon")
    )
    criteria = {
        "treatment_shift": treatment_shift,
        "outcome_shift": outcome_shift,
        "held_out_prediction": held_out,
        "negative_control": negative_control,
        "coefficient_behavior": coefficient_behavior,
    }
    return {"criteria": criteria, "spectral_temporal_bridge_supported": all(criteria.values())}


def _markdown(
    spec: dict[str, Any],
    estimates: list[dict[str, Any]],
    status: str,
    reason: str | None = None,
    decision: dict[str, Any] | None = None,
) -> str:
    lines = ["# Dense short-branch temporal mechanism", "", f"Status: **{status}**.", ""]
    if status != "complete":
        lines += [f"No scientific claim is permitted: {reason}.", ""]
    else:
        if decision is None:
            raise ValueError("A complete temporal report requires its frozen decision")
        supported = decision["spectral_temporal_bridge_supported"]
        lines += [
            "## Frozen decision",
            "",
            "Overall spectral temporal bridge: "
            + ("**supported**." if supported else "**not supported (claimable negative result)**."),
            "",
            "| Criterion | Passed |",
            "| --- | --- |",
        ]
        for criterion, passed in decision["criteria"].items():
            lines.append(f"| {criterion} | {str(bool(passed)).lower()} |")
        lines += ["", "## Predictor diagnostics", ""]
        lines += [
            "| Outcome | Predictor | Kind | LOSO RMSE improvement | Muon shrinkage | NorMuon shrinkage |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for r in estimates:
            lines.append(
                f"| {r['outcome']} | {r['predictor']} | {r['predictor_kind']} | {r['relative_rmse_improvement']:+.3f} | {r['muon_absolute_coefficient_shrinkage']:+.3f} | {r['normuon_absolute_coefficient_shrinkage']:+.3f} |"
            )
        lines.append("")
    lines += [f"> {spec['claim_boundary']}", ""]
    return "\n".join(lines)


def build_report(
    *,
    protocol: Path,
    predictor_csv: Path,
    predictor_manifest: Path,
    outcome_csv: Path,
    outcome_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol = protocol.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = _load_spec(protocol)
    missing = [
        str(p.resolve())
        for p in (predictor_csv, predictor_manifest, outcome_csv, outcome_manifest)
        if not p.is_file()
    ]
    if missing:
        reason = "missing required upstream artifacts: " + ", ".join(missing)
        readme = output_dir / "README.md"
        readme.write_text(_markdown(spec, [], "pending-not-claimable", reason), encoding="utf-8")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "status": "pending-not-claimable",
            "claimable": False,
            "protocol": _identity(protocol),
            "missing": missing,
            "reason": reason,
            "outputs": {"README.md": _identity(readme)},
        }
        _atomic_json(output_dir / "summary_manifest.json", receipt)
        return receipt
    predictor_receipt = json.loads(predictor_manifest.read_text(encoding="utf-8"))
    if predictor_receipt.get("analysis_protocol") != _identity(protocol):
        raise ValueError("Temporal predictor receipt is bound to a different causal-chain protocol")
    predictors, outcomes, sources = _load_inputs(
        spec,
        predictor_csv.resolve(),
        predictor_manifest.resolve(),
        outcome_csv.resolve(),
        outcome_manifest.resolve(),
    )
    paired, predictions, estimates = analyze_rows(spec, predictors, outcomes)
    decision = support_decision(spec, paired, estimates)
    outputs = {
        "paired_contrasts.csv": _write_csv(output_dir / "paired_contrasts.csv", paired),
        "loso_predictions.csv": _write_csv(output_dir / "loso_predictions.csv", predictions),
        "estimates.csv": _write_csv(output_dir / "estimates.csv", estimates),
    }
    readme = output_dir / "README.md"
    readme.write_text(_markdown(spec, estimates, "complete", decision=decision), encoding="utf-8")
    outputs["README.md"] = _identity(readme)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "status": "complete",
        "claimable": True,
        "family": "dense",
        "protocol": _identity(protocol),
        "sources": sources,
        "coverage": {
            "seeds": 3,
            "operators": 3,
            "checkpoints": 45,
            "paired_contrasts": len(paired),
            "loso_predictions": len(predictions),
        },
        "outputs": outputs,
        "decision": decision,
        "claim_rule": spec["analysis"]["claim_rule"],
        "claim_boundary": spec["claim_boundary"],
    }
    _atomic_json(output_dir / "summary_manifest.json", manifest)
    return manifest


def audit_report(
    output_dir: Path,
    *,
    protocol: Path,
    scope_amendment: Path,
    predictor_csv: Path,
    predictor_manifest: Path,
    outcome_csv: Path,
    outcome_manifest: Path,
) -> dict[str, Any]:
    path = output_dir.resolve() / "summary_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("complete") is not True
        or payload.get("claimable") is not True
        or payload.get("status") != "complete"
    ):
        raise RuntimeError("Temporal short-branch report is pending/not claimable")
    expected_keys = {
        "schema_version",
        "complete",
        "status",
        "claimable",
        "family",
        "protocol",
        "sources",
        "coverage",
        "outputs",
        "decision",
        "claim_rule",
        "claim_boundary",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("Temporal short-branch manifest schema differs")
    expected_protocol = _identity(protocol)
    expected_source_paths = tuple(
        source.resolve()
        for source in (predictor_manifest, predictor_csv, outcome_manifest, outcome_csv)
    )
    declared_sources = payload.get("sources")
    declared_source_paths = (
        tuple(Path(item.get("path", "")).resolve() for item in declared_sources)
        if isinstance(declared_sources, list)
        and all(isinstance(item, dict) for item in declared_sources)
        else ()
    )
    if (
        payload.get("protocol") != expected_protocol
        or declared_source_paths != expected_source_paths
    ):
        raise RuntimeError("Temporal short-branch receipt differs from CLI protocol/input bindings")
    for item in payload.get("outputs", {}).values():
        target = Path(item["path"])
        if (
            not target.is_file()
            or target.stat().st_size != item["bytes"]
            or _sha256(target) != item["sha256"]
        ):
            raise RuntimeError(f"Temporal short-branch output differs: {target}")
    for item in [payload.get("protocol"), *payload.get("sources", [])]:
        source = Path(item.get("path", "")) if isinstance(item, dict) else Path()
        if (
            not source.is_file()
            or source.stat().st_size != item.get("bytes")
            or _sha256(source) != item.get("sha256")
        ):
            raise RuntimeError(f"Temporal short-branch source differs: {source}")
    protocol = protocol.resolve()
    spec = _load_spec(protocol)
    sources = payload["sources"]
    if len(sources) != 4:
        raise RuntimeError("Temporal short-branch input source cardinality differs")
    predictor_manifest, predictor_csv, outcome_manifest, outcome_csv = map(
        lambda item: Path(item["path"]), sources
    )
    predictor_receipt = json.loads(predictor_manifest.read_text(encoding="utf-8"))
    if predictor_receipt.get("analysis_protocol") != expected_protocol or predictor_receipt.get(
        "scope_amendment"
    ) != _identity(scope_amendment):
        raise RuntimeError("Temporal predictor receipt uses a different protocol/scope")
    predictors, outcomes, fresh_sources = _load_inputs(
        spec, predictor_csv, predictor_manifest, outcome_csv, outcome_manifest
    )
    paired, predictions, estimates = analyze_rows(spec, predictors, outcomes)
    decision = support_decision(spec, paired, estimates)
    tables = {
        "paired_contrasts.csv": paired,
        "loso_predictions.csv": predictions,
        "estimates.csv": estimates,
    }
    for name, rows in tables.items():
        if Path(payload["outputs"][name]["path"]).read_bytes() != _csv_bytes(rows):
            raise RuntimeError(f"Temporal short-branch recomputed table differs: {name}")
    expected_readme = _markdown(spec, estimates, "complete", decision=decision).encode()
    if Path(payload["outputs"]["README.md"]["path"]).read_bytes() != expected_readme:
        raise RuntimeError("Temporal short-branch recomputed Markdown differs")
    expected_coverage = {
        "seeds": 3,
        "operators": 3,
        "checkpoints": 45,
        "paired_contrasts": len(paired),
        "loso_predictions": len(predictions),
    }
    if (
        payload["protocol"] != _identity(protocol)
        or payload["sources"] != fresh_sources
        or payload["coverage"] != expected_coverage
        or payload["decision"] != decision
        or payload["claim_rule"] != spec["analysis"]["claim_rule"]
        or payload["claim_boundary"] != spec["claim_boundary"]
    ):
        raise RuntimeError("Temporal short-branch recomputed manifest fields differ")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense short-branch five-stage temporal mechanism analysis"
    )
    parser.add_argument("--protocol", type=Path, default=Path("configs/causal_chain_analysis.json"))
    parser.add_argument(
        "--predictor-csv",
        type=Path,
        default=Path("reports/short-branch/temporal_mechanism_predictors.csv"),
    )
    parser.add_argument(
        "--predictor-manifest",
        type=Path,
        default=Path("reports/short-branch/temporal_mechanism_predictors.manifest.json"),
    )
    parser.add_argument(
        "--outcome-csv",
        type=Path,
        default=Path("reports/tail-stability/short_branch_checkpoint_tail.csv"),
    )
    parser.add_argument(
        "--outcome-manifest",
        type=Path,
        default=Path("reports/tail-stability/summary_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/temporal-short-branch"))
    parser.add_argument(
        "--scope-amendment", type=Path, default=Path("configs/dense_scope_amendment.json")
    )
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = (
        audit_report(
            args.output_dir,
            protocol=args.protocol,
            scope_amendment=args.scope_amendment,
            predictor_csv=args.predictor_csv,
            predictor_manifest=args.predictor_manifest,
            outcome_csv=args.outcome_csv,
            outcome_manifest=args.outcome_manifest,
        )
        if args.audit
        else build_report(
            protocol=args.protocol,
            predictor_csv=args.predictor_csv,
            predictor_manifest=args.predictor_manifest,
            outcome_csv=args.outcome_csv,
            outcome_manifest=args.outcome_manifest,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
