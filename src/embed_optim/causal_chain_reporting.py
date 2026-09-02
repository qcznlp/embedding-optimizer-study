"""Strict, standalone reader for the frozen causal-chain reporting artifacts.

This module intentionally does not call either analysis implementation.  It
validates their persisted contracts and independently recomputes every decision
that can be recovered from the reporting tables.  The returned rows are plain
JSON-compatible dictionaries suitable for Markdown or LaTeX renderers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CAUSAL_PROTOCOL_SHA256 = "813538910dbd32d6fcfb65ee8736816cd573ae122ec907d25d7208229720cd7f"
CAUSAL_SOURCE_PATHS = (
    "configs/short_branch_protocol.json",
    "configs/spectral_transplant_intervention.json",
    "configs/dense_scope_amendment.json",
    "configs/validation_probe.json",
    "configs/beir_representation_probe.json",
    "configs/retrieval_dynamics_protocol.json",
    "configs/experiment.yaml",
    "configs/common_state_probe.json",
)
TEMPORAL_SOURCE_PATHS = (
    "reports/short-branch/temporal_mechanism_predictors.manifest.json",
    "reports/short-branch/temporal_mechanism_predictors.csv",
    "reports/tail-stability/summary_manifest.json",
    "reports/tail-stability/short_branch_checkpoint_tail.csv",
)
DOSE_CANONICAL_SOURCE_PATHS = (
    "reports/spectral-transplant/summary_manifest.json",
    "reports/spectral-transplant/anchor_query_tail_effects.csv",
    "reports/dense-discovery/evaluation_long.csv",
    "reports/dense-discovery/coverage.json",
)

TEMPORAL_DIR = Path("reports/temporal-short-branch")
DOSE_DIR = Path("reports/dose-band")
PROTOCOL_PATH = Path("configs/causal_chain_analysis.json")

SEEDS = (314159, 271828, 161803)
OPERATORS = ("adamw", "muon", "normuon")
CHALLENGERS = ("muon", "normuon")
PREDICTORS = (
    "update_tail_energy_fraction",
    "update_stable_rank_fraction",
    "update_entropy_rank_fraction",
    "update_head_energy_fraction",
    "update_middle_energy_fraction",
    "update_row_norm_cv",
)
CONTROLS = ("update_frobenius_norm", "weight_frobenius_norm")
MEDIATORS = (*PREDICTORS, *CONTROLS)
OUTCOMES = ("validation_loss_p95", "unseen_margin_p05")
TEMPORAL_CRITERIA = (
    "treatment_shift",
    "outcome_shift",
    "held_out_prediction",
    "negative_control",
    "coefficient_behavior",
)

SOURCE_RUNS = ("adamw-lr1e-5", "muon-lr1e-3", "normuon-lr1e-3")
SOURCE_LEARNING_RATES = {
    "adamw-lr1e-5": 1e-5,
    "muon-lr1e-3": 1e-3,
    "normuon-lr1e-3": 1e-3,
}
TASKS = (
    "ArguAna",
    "ClimateFEVER",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "TRECCOVID",
    "Touche2020",
)
TRANSITION_STEPS = {"stage1-to-2": 782, "stage3-to-4": 2345}
DOSE_PREDICTORS = ("spectrum_loss", "spectrum_margin", "basis_loss", "basis_margin")
MATCHED_CONTROLS = {"spectrum_loss": "basis_loss", "spectrum_margin": "basis_margin"}
SPECTRAL_CONDITIONS = (
    "muon-native",
    "adam-basis__spectrum-lambda-0.25",
    "adam-basis__spectrum-lambda-0.50",
    "adam-basis__spectrum-lambda-0.75",
    "adam-basis__muon-spectrum",
    "muon-basis__adam-spectrum",
    "adam-basis__muon-head-spectrum",
    "adam-basis__muon-middle-spectrum",
    "adam-basis__muon-tail-spectrum",
)

PAIRED_FIELDS = (
    "seed",
    "challenger",
    "reference",
    *(f"delta_{name}" for name in (*MEDIATORS, *OUTCOMES)),
)
LOSO_FIELDS = (
    "outcome",
    "predictor",
    "held_out_seed",
    "challenger",
    "actual",
    "label_only_prediction",
    "mediator_prediction",
    "label_only_squared_error",
    "mediator_squared_error",
)
ESTIMATE_FIELDS = (
    "outcome",
    "predictor",
    "predictor_kind",
    "label_only_rmse",
    "mediator_rmse",
    "relative_rmse_improvement",
    "muon_coefficient_label_only",
    "muon_coefficient_with_predictor",
    "muon_absolute_coefficient_shrinkage",
    "normuon_coefficient_label_only",
    "normuon_coefficient_with_predictor",
    "normuon_absolute_coefficient_shrinkage",
)
ANCHOR_FIELDS = (
    "family",
    "anchor",
    "loss_dose_monotone",
    "margin_dose_monotone",
    "tail_band_best_both_metrics",
    "basis_swap_negative_control",
    *(f"loss_lambda_{dose:.2f}" for dose in (0.0, 0.25, 0.5, 0.75, 1.0)),
    *(f"margin_lambda_{dose:.2f}" for dose in (0.0, 0.25, 0.5, 0.75, 1.0)),
    *(f"loss_band_{band}" for band in ("head", "middle", "tail")),
    *(f"margin_band_{band}" for band in ("head", "middle", "tail")),
    "anchor_passed",
)
HELDOUT_FIELDS = (
    "family",
    "held_out_run",
    "held_out_learning_rate",
    "task",
    "transition",
    "anchor",
    "observed_increment",
    "baseline_prediction",
    "fold",
    "spectrum_loss_prediction",
    "spectrum_margin_prediction",
    "basis_loss_prediction",
    "basis_margin_prediction",
)


class CausalChainReportingError(ValueError):
    """Raised when causal-chain reporting evidence violates its contract."""


def _fail(message: str) -> CausalChainReportingError:
    return CausalChainReportingError(message)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicates
        )
    except CausalChainReportingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail(f"Cannot read valid JSON from {path}") from error
    if not isinstance(payload, dict):
        raise _fail(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = set(value) if isinstance(value, dict) else type(value).__name__
        raise _fail(f"{label} keys differ: expected {sorted(expected)}, found {observed}")
    return value


def _json_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise _fail(f"{label} must be a JSON boolean")
    return value


def _json_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise _fail(f"{label} must be an integer")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise _fail(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise _fail(f"{label} is non-finite")
    return result


def _integer(value: str, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise _fail(f"{label} is not an integer") from error
    if str(result) != value.strip():
        raise _fail(f"{label} is not a canonical integer")
    return result


def _csv_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise _fail(f"{label} must be True or False")


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not _close(observed, expected):
        raise _fail(f"{label} differs: expected {expected!r}, found {observed!r}")


def _candidate_paths(raw: str, root: Path, manifest_dir: Path) -> tuple[Path, ...]:
    root = root.resolve()
    path = Path(raw)
    if path.is_absolute():
        # Completed receipts produced before the portable-path contract embedded
        # their checkout root.  Retain that literal target and every possible
        # repository-root relocation of its suffix.  Identity validation below
        # requires a unique match, so a stale checkout cannot silently win over
        # the current clone.
        candidates = [path.resolve()]
        for index in range(1, len(path.parts) - 1):
            candidate = (root / Path(*path.parts[index:])).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            candidates.append(candidate)
        return tuple(dict.fromkeys(candidates))
    candidates = []
    for candidate in ((root / path).resolve(), (manifest_dir / path).resolve()):
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        candidates.append(candidate)
    return tuple(dict.fromkeys(candidates))


def _validate_record(
    record: Any,
    *,
    root: Path,
    manifest_dir: Path,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    record = _require_keys(record, {"path", "bytes", "sha256"}, label)
    raw_path = record["path"]
    size = record["bytes"]
    digest = record["sha256"]
    if not isinstance(raw_path, str) or not raw_path:
        raise _fail(f"{label}.path must be a non-empty string")
    if type(size) is not int or size < 0:
        raise _fail(f"{label}.bytes must be a nonnegative integer")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise _fail(f"{label}.sha256 must be a lowercase SHA-256 digest")
    candidates = _candidate_paths(raw_path, root, manifest_dir)
    if expected_path is not None:
        expected_path = expected_path.resolve()
        if expected_path not in candidates:
            raise _fail(f"{label}.path does not resolve to {expected_path}")
        target = expected_path
    else:
        matches = [
            candidate
            for candidate in candidates
            if candidate.is_file()
            and candidate.stat().st_size == size
            and _sha256(candidate) == digest
        ]
        if not matches:
            raise _fail(f"{label} does not bind an existing file with the declared identity")
        if len(matches) != 1:
            raise _fail(f"{label}.path is ambiguously relocatable within the current repository")
        target = matches[0]
    if not target.is_file():
        raise _fail(f"{label} target is missing: {target}")
    if target.stat().st_size != size or _sha256(target) != digest:
        raise _fail(f"{label} content identity differs: {target}")
    return _identity(target)


def _read_csv(path: Path, fields: tuple[str, ...], rows: int, label: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != list(fields):
                raise _fail(f"{label} columns differ from the exact contract")
            raw_rows = list(reader)
    except CausalChainReportingError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise _fail(f"Cannot read {label}: {path}") from error
    if len(raw_rows) != rows:
        raise _fail(f"{label} row count differs: expected {rows}, found {len(raw_rows)}")
    if any(len(row) != len(fields) for row in raw_rows):
        raise _fail(f"{label} contains a row with the wrong column count")
    return [dict(zip(fields, row, strict=True)) for row in raw_rows]


def _validate_protocol(root: Path) -> dict[str, Any]:
    path = (root / PROTOCOL_PATH).resolve()
    if not path.is_file():
        raise _fail(f"Frozen causal-chain protocol is missing: {path}")
    payload = _read_json(path)
    if (
        _sha256(path) != CAUSAL_PROTOCOL_SHA256
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("family") != "dense"
    ):
        raise _fail("Frozen causal-chain protocol is not schema-1 Dense")
    source_bindings = payload.get("source_bindings")
    if (
        not isinstance(source_bindings, list)
        or len(source_bindings) != len(CAUSAL_SOURCE_PATHS)
        or tuple(item.get("path") for item in source_bindings if isinstance(item, dict))
        != CAUSAL_SOURCE_PATHS
    ):
        raise _fail("Frozen causal-chain protocol source-binding ledger differs")
    binding_records = []
    for index, (relative, binding) in enumerate(
        zip(CAUSAL_SOURCE_PATHS, source_bindings, strict=True)
    ):
        binding = _require_keys(binding, {"path", "sha256"}, f"protocol.source_bindings[{index}]")
        target = (root / relative).resolve()
        if (
            not target.is_file()
            or not isinstance(binding["sha256"], str)
            or _sha256(target) != binding["sha256"]
        ):
            raise _fail(f"Frozen causal-chain source binding differs: {target}")
        binding_records.append(_identity(target))
    temporal = payload.get("temporal_short_branch")
    dose = payload.get("dose_band")
    if not isinstance(temporal, dict) or not isinstance(dose, dict):
        raise _fail("Frozen causal-chain protocol lacks temporal or dose branch")
    units = temporal.get("randomized_units", {})
    analysis = temporal.get("analysis", {})
    predictors = (analysis.get("primary_predictor"), *analysis.get("secondary_predictors", []))
    if (
        tuple(units.get("seeds", ())) != SEEDS
        or tuple(units.get("operators", ())) != OPERATORS
        or units.get("reference") != "adamw"
        or tuple(units.get("challengers", ())) != CHALLENGERS
        or units.get("expected_runs") != 9
        or units.get("expected_checkpoints") != 45
        or tuple(predictors) != PREDICTORS
        or tuple(analysis.get("negative_controls", ())) != CONTROLS
        or analysis.get("outcomes")
        != {"validation_loss_p95": "lower_is_better", "unseen_margin_p05": "higher_is_better"}
        or tuple(analysis.get("primary_support_rule", {})) != TEMPORAL_CRITERIA + ("decision",)
    ):
        raise _fail("Temporal protocol differs from the exact reporting contract")
    anchor_scope = dose.get("anchor_scope", {})
    dose_response = dose.get("dose_response", {})
    bridge = dose.get("forward_retrieval_bridge", {})
    if (
        anchor_scope.get("expected_anchors") != 10
        or tuple(anchor_scope.get("source_runs", ())) != SOURCE_RUNS
        or tuple(anchor_scope.get("anchor_fractions", ())) != (0.0, 0.2, 0.6, 1.0)
        or tuple(dose_response.get("lambdas", ())) != (0.0, 0.25, 0.5, 0.75, 1.0)
        or dose_response.get("support_threshold")
        != "at least 8 of 10 anchors are monotone for each primary metric"
        or dose.get("band_localization", {}).get("prespecified_band") != "tail"
        or bridge.get("task_groups") != 14
        or bridge.get("expected_anchor_run_transitions") != 6
        or bridge.get("expected_rows") != 84
    ):
        raise _fail("Dose/band protocol differs from the exact reporting contract")
    temporal_boundary = temporal.get("claim_boundary")
    dose_boundary = dose.get("claim_boundary")
    if not isinstance(temporal_boundary, str) or not temporal_boundary:
        raise _fail("Temporal claim boundary is missing")
    if not isinstance(dose_boundary, str) or not dose_boundary:
        raise _fail("Dose claim boundary is missing")
    return {
        "path": path,
        "identity": _identity(path),
        "temporal_claim_boundary": temporal_boundary,
        "dose_claim_boundary": dose_boundary,
        "temporal_claim_rule": analysis["primary_support_rule"]["decision"],
        "source_binding_records": binding_records,
    }


def _validate_output_records(
    manifest: dict[str, Any],
    *,
    root: Path,
    directory: Path,
    names: tuple[str, ...],
    label: str,
) -> dict[str, dict[str, Any]]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or len(outputs) != len(names) or set(outputs) != set(names):
        raise _fail(f"{label} outputs must contain exactly {list(names)}")
    return {
        name: _validate_record(
            outputs[name],
            root=root,
            manifest_dir=directory,
            label=f"{label}.outputs[{name}]",
            expected_path=directory / name,
        )
        for name in names
    }


def _validate_source_records(
    records: Any, *, root: Path, manifest_dir: Path, label: str, expected_count: int | None = None
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or (
        expected_count is not None and len(records) != expected_count
    ):
        raise _fail(f"{label} has the wrong source-record cardinality")
    validated = [
        _validate_record(record, root=root, manifest_dir=manifest_dir, label=f"{label}[{index}]")
        for index, record in enumerate(records)
    ]
    paths = [record["path"] for record in validated]
    if len(paths) != len(set(paths)):
        raise _fail(f"{label} contains duplicate source paths")
    return validated


def _pending_temporal(
    manifest: dict[str, Any], root: Path, directory: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    _require_keys(
        manifest,
        {
            "schema_version",
            "complete",
            "status",
            "claimable",
            "protocol",
            "missing",
            "reason",
            "outputs",
        },
        "pending temporal manifest",
    )
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["complete"] is not False
        or manifest["status"] != "pending-not-claimable"
        or manifest["claimable"] is not False
        or not isinstance(manifest["missing"], list)
        or not manifest["missing"]
        or len(manifest["missing"]) != len(set(map(str, manifest["missing"])))
        or not isinstance(manifest["reason"], str)
        or not manifest["reason"]
    ):
        raise _fail("Pending temporal manifest has contradictory status/provenance")
    _validate_record(
        manifest["protocol"],
        root=root,
        manifest_dir=directory,
        label="pending temporal protocol",
        expected_path=protocol["path"],
    )
    outputs = _validate_output_records(
        manifest, root=root, directory=directory, names=("README.md",), label="temporal"
    )
    return {
        "status": "pending",
        "complete": False,
        "claimable": False,
        "supported": None,
        "pending_reason": manifest["reason"],
        "missing_inputs": list(map(str, manifest["missing"])),
        "claim_boundary": protocol["temporal_claim_boundary"],
        "manifest": _identity(directory / "summary_manifest.json"),
        "outputs": outputs,
        "source_tables": [],
        "paired_rows": [],
        "loso_rows": [],
        "estimate_rows": [],
        "criteria_rows": [],
        "rmse_rows": [],
    }


def _parse_paired(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: list[tuple[int, str]] = []
    for index, raw in enumerate(raw_rows):
        seed = _integer(raw["seed"], f"paired[{index}].seed")
        challenger = raw["challenger"]
        if seed not in SEEDS or challenger not in CHALLENGERS or raw["reference"] != "adamw":
            raise _fail(f"paired[{index}] identity differs from the 3x2 paired grid")
        row: dict[str, Any] = {"seed": seed, "challenger": challenger, "reference": "adamw"}
        for field in PAIRED_FIELDS[3:]:
            row[field] = _finite(raw[field], f"paired[{index}].{field}")
        identities.append((seed, challenger))
        rows.append(row)
    expected = {(seed, challenger) for seed in SEEDS for challenger in CHALLENGERS}
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise _fail("Temporal paired contrasts contain duplicates or incomplete identities")
    seed_order = {value: index for index, value in enumerate(SEEDS)}
    challenger_order = {value: index for index, value in enumerate(CHALLENGERS)}
    return sorted(
        rows, key=lambda row: (seed_order[row["seed"]], challenger_order[row["challenger"]])
    )


def _parse_loso(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: list[tuple[str, str, int, str]] = []
    numeric = LOSO_FIELDS[4:]
    for index, raw in enumerate(raw_rows):
        outcome = raw["outcome"]
        predictor = raw["predictor"]
        seed = _integer(raw["held_out_seed"], f"loso[{index}].held_out_seed")
        challenger = raw["challenger"]
        if (
            outcome not in OUTCOMES
            or predictor not in MEDIATORS
            or seed not in SEEDS
            or challenger not in CHALLENGERS
        ):
            raise _fail(f"loso[{index}] identity differs from the exact 2x8x3x2 grid")
        row: dict[str, Any] = {
            "outcome": outcome,
            "predictor": predictor,
            "held_out_seed": seed,
            "challenger": challenger,
        }
        for field in numeric:
            row[field] = _finite(raw[field], f"loso[{index}].{field}")
        if row["label_only_squared_error"] < 0 or row["mediator_squared_error"] < 0:
            raise _fail(f"loso[{index}] has a negative squared error")
        _assert_close(
            row["label_only_squared_error"],
            (row["actual"] - row["label_only_prediction"]) ** 2,
            f"loso[{index}].label_only_squared_error",
        )
        _assert_close(
            row["mediator_squared_error"],
            (row["actual"] - row["mediator_prediction"]) ** 2,
            f"loso[{index}].mediator_squared_error",
        )
        identities.append((outcome, predictor, seed, challenger))
        rows.append(row)
    expected = {
        (outcome, predictor, seed, challenger)
        for outcome in OUTCOMES
        for predictor in MEDIATORS
        for seed in SEEDS
        for challenger in CHALLENGERS
    }
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise _fail("Temporal LOSO rows contain duplicates or incomplete identities")
    outcome_order = {value: index for index, value in enumerate(OUTCOMES)}
    predictor_order = {value: index for index, value in enumerate(MEDIATORS)}
    seed_order = {value: index for index, value in enumerate(SEEDS)}
    challenger_order = {value: index for index, value in enumerate(CHALLENGERS)}
    return sorted(
        rows,
        key=lambda row: (
            outcome_order[row["outcome"]],
            predictor_order[row["predictor"]],
            seed_order[row["held_out_seed"]],
            challenger_order[row["challenger"]],
        ),
    )


def _parse_estimates(
    raw_rows: list[dict[str, str]], loso_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: list[tuple[str, str]] = []
    numeric = ESTIMATE_FIELDS[3:]
    for index, raw in enumerate(raw_rows):
        outcome = raw["outcome"]
        predictor = raw["predictor"]
        expected_kind = "mechanism" if predictor in PREDICTORS else "negative_control"
        if (
            outcome not in OUTCOMES
            or predictor not in MEDIATORS
            or raw["predictor_kind"] != expected_kind
        ):
            raise _fail(f"estimate[{index}] identity/kind differs from the exact 2x8 grid")
        row: dict[str, Any] = {
            "outcome": outcome,
            "predictor": predictor,
            "predictor_kind": expected_kind,
        }
        for field in numeric:
            row[field] = _finite(raw[field], f"estimate[{index}].{field}")
        if row["label_only_rmse"] < 0 or row["mediator_rmse"] < 0:
            raise _fail(f"estimate[{index}] has a negative RMSE")
        members = [
            member
            for member in loso_rows
            if member["outcome"] == outcome and member["predictor"] == predictor
        ]
        base = math.sqrt(sum(member["label_only_squared_error"] for member in members) / 6)
        mediator = math.sqrt(sum(member["mediator_squared_error"] for member in members) / 6)
        relative = (base - mediator) / base if base else 0.0
        _assert_close(row["label_only_rmse"], base, f"estimate[{index}].label_only_rmse")
        _assert_close(row["mediator_rmse"], mediator, f"estimate[{index}].mediator_rmse")
        _assert_close(
            row["relative_rmse_improvement"],
            relative,
            f"estimate[{index}].relative_rmse_improvement",
        )
        for challenger in CHALLENGERS:
            before = row[f"{challenger}_coefficient_label_only"]
            after = row[f"{challenger}_coefficient_with_predictor"]
            expected_shrinkage = 1 - abs(after) / abs(before) if before else 0.0
            _assert_close(
                row[f"{challenger}_absolute_coefficient_shrinkage"],
                expected_shrinkage,
                f"estimate[{index}].{challenger}_absolute_coefficient_shrinkage",
            )
        identities.append((outcome, predictor))
        rows.append(row)
    expected = {(outcome, predictor) for outcome in OUTCOMES for predictor in MEDIATORS}
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise _fail("Temporal estimates contain duplicates or incomplete identities")
    outcome_order = {value: index for index, value in enumerate(OUTCOMES)}
    predictor_order = {value: index for index, value in enumerate(MEDIATORS)}
    return sorted(
        rows,
        key=lambda row: (outcome_order[row["outcome"]], predictor_order[row["predictor"]]),
    )


def _temporal_decision(
    paired: list[dict[str, Any]], estimates: list[dict[str, Any]]
) -> dict[str, bool]:
    primary = PREDICTORS[0]
    indexed = {(row["outcome"], row["predictor"]): row for row in estimates}
    treatment = all(
        sum(row["challenger"] == challenger and row[f"delta_{primary}"] > 0 for row in paired) >= 2
        for challenger in CHALLENGERS
    )
    outcomes = all(
        sum(
            row["challenger"] == challenger
            and row["delta_validation_loss_p95"] < 0
            and row["delta_unseen_margin_p05"] > 0
            for row in paired
        )
        >= 2
        for challenger in CHALLENGERS
    )
    held_out = all(
        indexed[(outcome, primary)]["relative_rmse_improvement"] > 0 for outcome in OUTCOMES
    )
    controls = all(
        indexed[(outcome, control)]["relative_rmse_improvement"]
        < indexed[(outcome, primary)]["relative_rmse_improvement"]
        for outcome in OUTCOMES
        for control in CONTROLS
    )
    coefficients = all(
        abs(indexed[(outcome, primary)][f"{challenger}_coefficient_with_predictor"])
        <= abs(indexed[(outcome, primary)][f"{challenger}_coefficient_label_only"])
        for outcome in OUTCOMES
        for challenger in CHALLENGERS
    )
    return dict(
        zip(TEMPORAL_CRITERIA, (treatment, outcomes, held_out, controls, coefficients), strict=True)
    )


def _load_temporal(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    directory = (root / TEMPORAL_DIR).resolve()
    manifest_path = directory / "summary_manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "pending",
            "complete": False,
            "claimable": False,
            "supported": None,
            "pending_reason": f"missing manifest: {manifest_path}",
            "missing_inputs": [str(manifest_path)],
            "claim_boundary": protocol["temporal_claim_boundary"],
            "manifest": None,
            "outputs": {},
            "source_tables": [],
            "paired_rows": [],
            "loso_rows": [],
            "estimate_rows": [],
            "criteria_rows": [],
            "rmse_rows": [],
        }
    manifest = _read_json(manifest_path)
    if manifest.get("complete") is not True:
        return _pending_temporal(manifest, root, directory, protocol)
    _require_keys(
        manifest,
        {
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
        },
        "complete temporal manifest",
    )
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["status"] != "complete"
        or manifest["claimable"] is not True
        or manifest["family"] != "dense"
        or manifest["claim_rule"] != protocol["temporal_claim_rule"]
        or manifest["claim_boundary"] != protocol["temporal_claim_boundary"]
    ):
        raise _fail("Complete temporal manifest has contradictory status/protocol fields")
    _validate_record(
        manifest["protocol"],
        root=root,
        manifest_dir=directory,
        label="temporal protocol",
        expected_path=protocol["path"],
    )
    sources = _validate_source_records(
        manifest["sources"],
        root=root,
        manifest_dir=directory,
        label="temporal.sources",
        expected_count=4,
    )
    expected_source_paths = tuple(str((root / path).resolve()) for path in TEMPORAL_SOURCE_PATHS)
    if tuple(record["path"] for record in sources) != expected_source_paths:
        raise _fail("Temporal sources differ from the exact canonical paths/order")
    coverage = _require_keys(
        manifest["coverage"],
        {"seeds", "operators", "checkpoints", "paired_contrasts", "loso_predictions"},
        "temporal.coverage",
    )
    if coverage != {
        "seeds": 3,
        "operators": 3,
        "checkpoints": 45,
        "paired_contrasts": 6,
        "loso_predictions": 96,
    }:
        raise _fail("Temporal manifest coverage is not exactly 3/3/45/6/96")
    outputs = _validate_output_records(
        manifest,
        root=root,
        directory=directory,
        names=("paired_contrasts.csv", "loso_predictions.csv", "estimates.csv", "README.md"),
        label="temporal",
    )
    paired = _parse_paired(
        _read_csv(directory / "paired_contrasts.csv", PAIRED_FIELDS, 6, "temporal paired contrasts")
    )
    loso = _parse_loso(
        _read_csv(directory / "loso_predictions.csv", LOSO_FIELDS, 96, "temporal LOSO predictions")
    )
    estimates = _parse_estimates(
        _read_csv(directory / "estimates.csv", ESTIMATE_FIELDS, 16, "temporal estimates"), loso
    )
    criteria = _temporal_decision(paired, estimates)
    decision = _require_keys(
        manifest["decision"],
        {"criteria", "spectral_temporal_bridge_supported"},
        "temporal.decision",
    )
    declared_criteria = _require_keys(
        decision["criteria"], set(TEMPORAL_CRITERIA), "temporal.decision.criteria"
    )
    for criterion in TEMPORAL_CRITERIA:
        if (
            _json_bool(declared_criteria[criterion], f"temporal criterion {criterion}")
            != criteria[criterion]
        ):
            raise _fail(f"Temporal criterion contradicts reporting tables: {criterion}")
    supported = all(criteria.values())
    if (
        _json_bool(
            decision["spectral_temporal_bridge_supported"], "spectral_temporal_bridge_supported"
        )
        != supported
    ):
        raise _fail("Temporal overall decision contradicts the five criteria")
    criteria_rows = [{"criterion": name, "passed": criteria[name]} for name in TEMPORAL_CRITERIA]
    rmse_rows = [
        {
            "outcome": row["outcome"],
            "predictor": row["predictor"],
            "predictor_kind": row["predictor_kind"],
            "label_only_rmse": row["label_only_rmse"],
            "predictor_rmse": row["mediator_rmse"],
            "relative_rmse_improvement": row["relative_rmse_improvement"],
        }
        for row in estimates
    ]
    table_names = ("paired_contrasts.csv", "loso_predictions.csv", "estimates.csv")
    return {
        "status": "supported" if supported else "negative",
        "complete": True,
        "claimable": True,
        "supported": supported,
        "pending_reason": None,
        "missing_inputs": [],
        "claim_boundary": protocol["temporal_claim_boundary"],
        "manifest": _identity(manifest_path),
        "outputs": outputs,
        "sources": sources,
        "source_tables": [outputs[name] for name in table_names],
        "paired_rows": paired,
        "loso_rows": loso,
        "estimate_rows": estimates,
        "criteria_rows": criteria_rows,
        "rmse_rows": rmse_rows,
    }


def _pending_dose(
    manifest: dict[str, Any], root: Path, directory: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    _require_keys(
        manifest,
        {
            "schema_version",
            "status",
            "complete",
            "claimability",
            "falsification",
            "missing_inputs",
            "available_sources",
            "protocol",
            "claim_boundary",
            "outputs",
        },
        "pending dose manifest",
    )
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["status"] != "pending"
        or manifest["complete"] is not False
        or manifest["claimability"] != "pending"
        or manifest["falsification"] != "not_tested"
        or not isinstance(manifest["missing_inputs"], list)
        or not manifest["missing_inputs"]
        or len(manifest["missing_inputs"]) != len(set(map(str, manifest["missing_inputs"])))
        or manifest["claim_boundary"] != protocol["dose_claim_boundary"]
    ):
        raise _fail("Pending dose manifest has contradictory status/provenance")
    protocol_record = _require_keys(
        manifest["protocol"], {"protocol", "source_bindings"}, "dose.protocol"
    )
    _validate_record(
        protocol_record["protocol"],
        root=root,
        manifest_dir=directory,
        label="pending dose protocol",
        expected_path=protocol["path"],
    )
    protocol_sources = _validate_source_records(
        protocol_record["source_bindings"],
        root=root,
        manifest_dir=directory,
        label="pending dose protocol source bindings",
    )
    if protocol_sources != protocol["source_binding_records"]:
        raise _fail("Pending dose protocol bindings differ from the frozen source ledger")
    available = _validate_source_records(
        manifest["available_sources"],
        root=root,
        manifest_dir=directory,
        label="dose.available_sources",
    )
    outputs = _validate_output_records(
        manifest,
        root=root,
        directory=directory,
        names=("anchor_tests.csv", "heldout_predictions.csv", "report.md"),
        label="dose",
    )
    return {
        "status": "pending",
        "complete": False,
        "claimable": False,
        "supported": None,
        "pending_reason": "missing required upstream inputs",
        "missing_inputs": list(map(str, manifest["missing_inputs"])),
        "claim_boundary": protocol["dose_claim_boundary"],
        "manifest": _identity(directory / "summary_manifest.json"),
        "outputs": outputs,
        "sources": available,
        "protocol_sources": protocol_sources,
        "evaluation_records": [],
        "source_tables": [],
        "anchor_rows": [],
        "anchor_criteria_rows": [],
        "criteria_rows": [],
        "heldout_rows": [],
        "rmse_rows": [],
        "bridge_rows": [],
    }


def _expected_anchors() -> set[str]:
    return set(_ordered_anchors())


def _ordered_anchors() -> tuple[str, ...]:
    return (
        "dense/pretrained",
        *(f"dense/{run}/checkpoint-{step}" for run in SOURCE_RUNS for step in (782, 2345, 3907)),
    )


def _parse_anchors(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: list[tuple[str, str]] = []
    boolean_fields = (
        "loss_dose_monotone",
        "margin_dose_monotone",
        "tail_band_best_both_metrics",
        "basis_swap_negative_control",
        "anchor_passed",
    )
    numeric_fields = tuple(
        field
        for field in ANCHOR_FIELDS
        if field.startswith(("loss_", "margin_")) and field not in boolean_fields
    )
    for index, raw in enumerate(raw_rows):
        if raw["family"] != "dense" or raw["anchor"] not in _expected_anchors():
            raise _fail(f"anchor[{index}] is outside the exact ten-anchor Dense grid")
        row: dict[str, Any] = {"family": "dense", "anchor": raw["anchor"]}
        for field in boolean_fields:
            row[field] = _csv_bool(raw[field], f"anchor[{index}].{field}")
        for field in numeric_fields:
            row[field] = _finite(raw[field], f"anchor[{index}].{field}")
        losses = [row[f"loss_lambda_{dose:.2f}"] for dose in (0.0, 0.25, 0.5, 0.75, 1.0)]
        margins = [row[f"margin_lambda_{dose:.2f}"] for dose in (0.0, 0.25, 0.5, 0.75, 1.0)]
        if losses[0] != 0.0 or margins[0] != 0.0:
            raise _fail(f"anchor[{index}] lambda-zero identity contrast must be exactly zero")
        computed = {
            "loss_dose_monotone": all(right <= left for left, right in zip(losses, losses[1:])),
            "margin_dose_monotone": all(right >= left for left, right in zip(margins, margins[1:])),
            "tail_band_best_both_metrics": all(
                row["loss_band_tail"] < row[f"loss_band_{band}"]
                and row["margin_band_tail"] > row[f"margin_band_{band}"]
                for band in ("head", "middle")
            ),
        }
        for field, value in computed.items():
            if row[field] != value:
                raise _fail(f"anchor[{index}].{field} contradicts its numeric contrasts")
        passed = all(
            row[field]
            for field in (
                "loss_dose_monotone",
                "margin_dose_monotone",
                "tail_band_best_both_metrics",
                "basis_swap_negative_control",
            )
        )
        if row["anchor_passed"] != passed:
            raise _fail(f"anchor[{index}].anchor_passed contradicts its four criteria")
        identities.append(("dense", row["anchor"]))
        rows.append(row)
    expected = {("dense", anchor) for anchor in _expected_anchors()}
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise _fail("Dose anchor rows contain duplicates or incomplete identities")
    anchor_order = {value: index for index, value in enumerate(_ordered_anchors())}
    return sorted(rows, key=lambda row: anchor_order[row["anchor"]])


def _parse_heldout(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: list[tuple[str, str, str, str]] = []
    numeric_fields = (
        "held_out_learning_rate",
        "observed_increment",
        "baseline_prediction",
        *(f"{name}_prediction" for name in DOSE_PREDICTORS),
    )
    for index, raw in enumerate(raw_rows):
        run = raw["held_out_run"]
        task = raw["task"]
        transition = raw["transition"]
        if (
            raw["family"] != "dense"
            or run not in SOURCE_RUNS
            or task not in TASKS
            or transition not in TRANSITION_STEPS
            or raw["fold"] != "leave-one-run-and-learning-rate-out"
            or raw["anchor"] != f"dense/{run}/checkpoint-{TRANSITION_STEPS.get(transition, -1)}"
        ):
            raise _fail(f"heldout[{index}] differs from the exact 3x14x2 grid")
        row: dict[str, Any] = {
            "family": "dense",
            "held_out_run": run,
            "task": task,
            "transition": transition,
            "anchor": raw["anchor"],
            "fold": raw["fold"],
        }
        for field in numeric_fields:
            row[field] = _finite(raw[field], f"heldout[{index}].{field}")
        if row["held_out_learning_rate"] != SOURCE_LEARNING_RATES[run]:
            raise _fail(f"heldout[{index}].held_out_learning_rate differs from its frozen run")
        identities.append(("dense", run, task, transition))
        rows.append(row)
    expected = {
        ("dense", run, task, transition)
        for run in SOURCE_RUNS
        for task in TASKS
        for transition in TRANSITION_STEPS
    }
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise _fail("Dose held-out rows contain duplicates or incomplete identities")
    for run in SOURCE_RUNS:
        learning_rates = {
            row["held_out_learning_rate"] for row in rows if row["held_out_run"] == run
        }
        if len(learning_rates) != 1:
            raise _fail(f"Held-out learning rate is inconsistent within {run}")
    run_order = {value: index for index, value in enumerate(SOURCE_RUNS)}
    task_order = {value: index for index, value in enumerate(TASKS)}
    transition_order = {value: index for index, value in enumerate(TRANSITION_STEPS)}
    return sorted(
        rows,
        key=lambda row: (
            run_order[row["held_out_run"]],
            task_order[row["task"]],
            transition_order[row["transition"]],
        ),
    )


def _dose_rmse(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    rmse = {
        "baseline": math.sqrt(
            sum((row["baseline_prediction"] - row["observed_increment"]) ** 2 for row in rows) / 84
        )
    }
    for predictor in DOSE_PREDICTORS:
        rmse[predictor] = math.sqrt(
            sum((row[f"{predictor}_prediction"] - row["observed_increment"]) ** 2 for row in rows)
            / 84
        )
    improvements = {predictor: rmse["baseline"] - rmse[predictor] for predictor in DOSE_PREDICTORS}
    return rmse, improvements


def _validate_anchor_source(
    root: Path, sources: list[dict[str, Any]], anchors: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Cross-check all persisted anchor criteria against the canonical spectral table."""

    manifest_path = (root / DOSE_CANONICAL_SOURCE_PATHS[0]).resolve()
    table_path = (root / DOSE_CANONICAL_SOURCE_PATHS[1]).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("complete") is not True:
        raise _fail("Canonical spectral-transplant summary is not complete")
    outputs = manifest.get("outputs")
    record = outputs.get("anchor_query_tail_effects") if isinstance(outputs, dict) else None
    record = _require_keys(
        record,
        {"path", "bytes", "sha256", "rows"},
        "spectral outputs.anchor_query_tail_effects",
    )
    if _json_int(record["rows"], "spectral anchor_query_tail_effects rows") != 90:
        raise _fail("Canonical spectral anchor table must contain exactly 90 rows")
    bound_table = _validate_record(
        {key: record[key] for key in ("path", "bytes", "sha256")},
        root=root,
        manifest_dir=manifest_path.parent,
        label="spectral outputs.anchor_query_tail_effects",
        expected_path=table_path,
    )
    if bound_table != sources[1]:
        raise _fail("Dose source record differs from the spectral manifest's anchor table")

    upstream_sources = manifest.get("sources")
    if not isinstance(upstream_sources, list) or len(upstream_sources) != 10:
        raise _fail("Canonical spectral manifest must bind ten raw anchor sources")
    raw_records = []
    labels = []
    for index, item in enumerate(upstream_sources):
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise _fail(f"Malformed spectral source record at index {index}")
        labels.append(item["label"])
        raw_records.append(
            _validate_record(
                item.get("sample_metrics"),
                root=root,
                manifest_dir=manifest_path.parent,
                label=f"spectral.sources[{index}].sample_metrics",
            )
        )
    if (
        len(labels) != len(set(labels))
        or set(labels) != _expected_anchors()
        or raw_records != sources[4:]
    ):
        raise _fail("Dose raw sources differ from the spectral manifest's ten anchors")

    required = {
        "family",
        "anchor",
        "condition",
        "p95_pairwise_loss_contrast",
        "p05_pairwise_margin_contrast",
    }
    try:
        with table_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or not required.issubset(fields):
                raise _fail("Canonical spectral anchor table lacks required unique columns")
            rows = list(reader)
    except CausalChainReportingError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise _fail(f"Cannot read canonical spectral anchor table: {table_path}") from error
    if len(rows) != 90 or any(None in row for row in rows):
        raise _fail("Canonical spectral anchor table does not contain 90 rectangular rows")
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for index, row in enumerate(rows):
        if row["family"] != "dense" or row["anchor"] not in _expected_anchors():
            raise _fail(f"Canonical spectral row {index} is outside the Dense anchor grid")
        key = (row["anchor"], row["condition"])
        if row["condition"] not in SPECTRAL_CONDITIONS or key in indexed:
            raise _fail(f"Canonical spectral row {index} has a duplicate/unexpected condition")
        _finite(row["p95_pairwise_loss_contrast"], f"spectral row {index} loss")
        _finite(row["p05_pairwise_margin_contrast"], f"spectral row {index} margin")
        indexed[key] = row
    expected = {
        (anchor, condition) for anchor in _ordered_anchors() for condition in SPECTRAL_CONDITIONS
    }
    if set(indexed) != expected:
        raise _fail("Canonical spectral table does not cover the exact 10x9 anchor-condition grid")

    dose_conditions = {
        0.25: "adam-basis__spectrum-lambda-0.25",
        0.50: "adam-basis__spectrum-lambda-0.50",
        0.75: "adam-basis__spectrum-lambda-0.75",
        1.00: "adam-basis__muon-spectrum",
    }
    band_conditions = {
        "head": "adam-basis__muon-head-spectrum",
        "middle": "adam-basis__muon-middle-spectrum",
        "tail": "adam-basis__muon-tail-spectrum",
    }
    basis_margins: dict[str, dict[str, float]] = {}
    for anchor_row in anchors:
        anchor = anchor_row["anchor"]
        for dose, condition in dose_conditions.items():
            source = indexed[(anchor, condition)]
            _assert_close(
                anchor_row[f"loss_lambda_{dose:.2f}"],
                _finite(source["p95_pairwise_loss_contrast"], f"{anchor}/{condition} loss"),
                f"{anchor} loss lambda {dose:.2f}",
            )
            _assert_close(
                anchor_row[f"margin_lambda_{dose:.2f}"],
                _finite(source["p05_pairwise_margin_contrast"], f"{anchor}/{condition} margin"),
                f"{anchor} margin lambda {dose:.2f}",
            )
        for band, condition in band_conditions.items():
            source = indexed[(anchor, condition)]
            _assert_close(
                anchor_row[f"loss_band_{band}"],
                _finite(source["p95_pairwise_loss_contrast"], f"{anchor}/{condition} loss"),
                f"{anchor} loss band {band}",
            )
            _assert_close(
                anchor_row[f"margin_band_{band}"],
                _finite(source["p05_pairwise_margin_contrast"], f"{anchor}/{condition} margin"),
                f"{anchor} margin band {band}",
            )
        spectrum = indexed[(anchor, "adam-basis__muon-spectrum")]
        basis = indexed[(anchor, "muon-basis__adam-spectrum")]
        spectrum_loss = _finite(spectrum["p95_pairwise_loss_contrast"], f"{anchor} spectrum loss")
        basis_loss = _finite(basis["p95_pairwise_loss_contrast"], f"{anchor} basis loss")
        spectrum_margin = _finite(
            spectrum["p05_pairwise_margin_contrast"], f"{anchor} spectrum margin"
        )
        basis_margin = _finite(basis["p05_pairwise_margin_contrast"], f"{anchor} basis margin")
        loss_gap = basis_loss - spectrum_loss
        margin_gap = spectrum_margin - basis_margin
        basis_passed = loss_gap > 0 and margin_gap > 0
        if anchor_row["basis_swap_negative_control"] != basis_passed:
            raise _fail(f"{anchor} basis negative-control decision contradicts canonical contrasts")
        basis_margins[anchor] = {
            "basis_loss_decision_gap": loss_gap,
            "basis_margin_decision_gap": margin_gap,
        }
    return basis_margins


def _bridge_supported(improvements: dict[str, float]) -> bool:
    return any(
        improvements[spectrum] > 0 and improvements[spectrum] > improvements[control]
        for spectrum, control in MATCHED_CONTROLS.items()
    )


def _load_dose(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    directory = (root / DOSE_DIR).resolve()
    manifest_path = directory / "summary_manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "pending",
            "complete": False,
            "claimable": False,
            "supported": None,
            "pending_reason": f"missing manifest: {manifest_path}",
            "missing_inputs": [str(manifest_path)],
            "claim_boundary": protocol["dose_claim_boundary"],
            "manifest": None,
            "outputs": {},
            "sources": [],
            "protocol_sources": [],
            "evaluation_records": [],
            "source_tables": [],
            "anchor_rows": [],
            "anchor_criteria_rows": [],
            "criteria_rows": [],
            "heldout_rows": [],
            "rmse_rows": [],
            "bridge_rows": [],
        }
    manifest = _read_json(manifest_path)
    if manifest.get("complete") is not True:
        return _pending_dose(manifest, root, directory, protocol)
    _require_keys(
        manifest,
        {
            "schema_version",
            "status",
            "complete",
            "claimability",
            "supported",
            "falsification",
            "local_supported",
            "forward_bridge_supported",
            "criterion",
            "decisions",
            "prediction_protocol",
            "forward_bridge",
            "claim_boundary",
            "prediction_rows",
            "expected_prediction_rows",
            "protocol",
            "evaluation_manifest",
            "evaluation_input",
            "sources",
            "outputs",
        },
        "complete dose manifest",
    )
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["status"] != "complete"
        or manifest["claimability"] != "claimable"
        or manifest["criterion"] != "at_least_8_of_10_anchors"
        or manifest["prediction_protocol"]
        != "separate OLS predictors over task+transition baseline with leave-one-run/LR-out folds"
        or manifest["claim_boundary"] != protocol["dose_claim_boundary"]
        or _json_int(manifest["prediction_rows"], "dose.prediction_rows") != 84
        or _json_int(manifest["expected_prediction_rows"], "dose.expected_prediction_rows") != 84
    ):
        raise _fail("Complete dose manifest has contradictory status/protocol fields")
    protocol_record = _require_keys(
        manifest["protocol"], {"protocol", "source_bindings"}, "dose.protocol"
    )
    _validate_record(
        protocol_record["protocol"],
        root=root,
        manifest_dir=directory,
        label="dose protocol",
        expected_path=protocol["path"],
    )
    protocol_sources = _validate_source_records(
        protocol_record["source_bindings"],
        root=root,
        manifest_dir=directory,
        label="dose protocol source bindings",
    )
    if protocol_sources != protocol["source_binding_records"]:
        raise _fail("Dose manifest protocol bindings differ from the frozen source ledger")
    evaluation_manifest = _validate_record(
        manifest["evaluation_manifest"],
        root=root,
        manifest_dir=directory,
        label="dose.evaluation_manifest",
    )
    evaluation_input = _validate_record(
        manifest["evaluation_input"],
        root=root,
        manifest_dir=directory,
        label="dose.evaluation_input",
    )
    sources = _validate_source_records(
        manifest["sources"], root=root, manifest_dir=directory, label="dose.sources"
    )
    expected_canonical_paths = tuple(
        str((root / path).resolve()) for path in DOSE_CANONICAL_SOURCE_PATHS
    )
    if (
        len(sources) != 14
        or tuple(record["path"] for record in sources[:4]) != expected_canonical_paths
        or evaluation_input != sources[2]
        or evaluation_manifest != sources[3]
    ):
        raise _fail(
            "Dose sources differ from the canonical four inputs plus ten raw anchor sources"
        )
    outputs = _validate_output_records(
        manifest,
        root=root,
        directory=directory,
        names=("anchor_tests.csv", "heldout_predictions.csv", "report.md"),
        label="dose",
    )
    anchors = _parse_anchors(
        _read_csv(directory / "anchor_tests.csv", ANCHOR_FIELDS, 10, "dose anchor tests")
    )
    basis_margins = _validate_anchor_source(root, sources, anchors)
    for row in anchors:
        row.update(basis_margins[row["anchor"]])
    heldout = _parse_heldout(
        _read_csv(
            directory / "heldout_predictions.csv", HELDOUT_FIELDS, 84, "dose held-out predictions"
        )
    )
    decisions = manifest["decisions"]
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise _fail("Dose decisions must contain exactly one Dense row")
    decision = _require_keys(
        decisions[0],
        {
            "family",
            "anchors",
            "supporting_anchors",
            "loss_dose_monotone_anchors",
            "margin_dose_monotone_anchors",
            "basis_control_anchors",
            "prespecified_band",
            "tail_band_anchors",
            "threshold",
            "local_supported",
        },
        "dose.decisions[0]",
    )
    counts = {
        "anchors": 10,
        "supporting_anchors": sum(row["anchor_passed"] for row in anchors),
        "loss_dose_monotone_anchors": sum(row["loss_dose_monotone"] for row in anchors),
        "margin_dose_monotone_anchors": sum(row["margin_dose_monotone"] for row in anchors),
        "basis_control_anchors": sum(row["basis_swap_negative_control"] for row in anchors),
        "tail_band_anchors": sum(row["tail_band_best_both_metrics"] for row in anchors),
    }
    if decision.get("family") != "dense" or decision.get("prespecified_band") != "tail":
        raise _fail("Dose decision family/band differs from the frozen contract")
    for key, expected in counts.items():
        if _json_int(decision.get(key), f"dose.decisions[0].{key}") != expected:
            raise _fail(f"Dose decision count contradicts anchor table: {key}")
    if _json_int(decision.get("threshold"), "dose.decisions[0].threshold") != 8:
        raise _fail("Dose support threshold must be 8")
    local_supported = (
        min(
            counts["loss_dose_monotone_anchors"],
            counts["margin_dose_monotone_anchors"],
            counts["basis_control_anchors"],
            counts["tail_band_anchors"],
        )
        >= 8
    )
    if (
        _json_bool(decision.get("local_supported"), "dose.decisions[0].local_supported")
        != local_supported
    ):
        raise _fail("Dose local decision contradicts the four 8/10 criteria")
    if _json_bool(manifest["local_supported"], "dose.local_supported") != local_supported:
        raise _fail("Dose top-level local decision contradicts anchor evidence")
    rmse, improvements = _dose_rmse(heldout)
    bridge = _require_keys(
        manifest["forward_bridge"], {"rmse", "rmse_improvement", "supported"}, "dose.forward_bridge"
    )
    declared_rmse = _require_keys(
        bridge["rmse"], {"baseline", *DOSE_PREDICTORS}, "dose.forward_bridge.rmse"
    )
    declared_improvement = _require_keys(
        bridge["rmse_improvement"], set(DOSE_PREDICTORS), "dose.forward_bridge.rmse_improvement"
    )
    for name, expected in rmse.items():
        observed = _finite(declared_rmse[name], f"dose RMSE {name}")
        if observed < 0:
            raise _fail(f"Dose RMSE {name} is negative")
        _assert_close(observed, expected, f"dose RMSE {name}")
    for name, expected in improvements.items():
        _assert_close(
            _finite(declared_improvement[name], f"dose RMSE improvement {name}"),
            expected,
            f"dose RMSE improvement {name}",
        )
    forward_supported = _bridge_supported(improvements)
    if _json_bool(bridge["supported"], "dose.forward_bridge.supported") != forward_supported:
        raise _fail("Dose forward bridge contradicts the matched-control rule")
    if (
        _json_bool(manifest["forward_bridge_supported"], "dose.forward_bridge_supported")
        != forward_supported
    ):
        raise _fail("Dose top-level forward bridge contradicts held-out evidence")
    supported = local_supported and forward_supported
    if _json_bool(manifest["supported"], "dose.supported") != supported:
        raise _fail("Dose overall decision contradicts local and forward evidence")
    expected_falsification = (
        "overall_chain_supported" if supported else "overall_chain_not_supported"
    )
    if manifest["falsification"] != expected_falsification:
        raise _fail("Dose falsification label contradicts its overall decision")
    anchor_criteria = [
        {
            "family": row["family"],
            "anchor": row["anchor"],
            "loss_dose_monotone": row["loss_dose_monotone"],
            "margin_dose_monotone": row["margin_dose_monotone"],
            "tail_band_best_both_metrics": row["tail_band_best_both_metrics"],
            "basis_swap_negative_control": row["basis_swap_negative_control"],
            "anchor_passed": row["anchor_passed"],
        }
        for row in anchors
    ]
    criteria_rows = [
        {
            "criterion": name,
            "supporting_anchors": counts[count_key],
            "anchors": 10,
            "threshold": 8,
            "passed": counts[count_key] >= 8,
            "evidence_source": "canonical_spectral_contrasts_recomputed",
        }
        for name, count_key in (
            ("loss_dose_monotone", "loss_dose_monotone_anchors"),
            ("margin_dose_monotone", "margin_dose_monotone_anchors"),
            ("tail_band_best_both_metrics", "tail_band_anchors"),
            ("basis_swap_negative_control", "basis_control_anchors"),
        )
    ]
    rmse_rows = [
        {
            "predictor": "baseline",
            "predictor_kind": "baseline",
            "rmse": rmse["baseline"],
            "rmse_improvement": 0.0,
            "matched_control": None,
        }
    ]
    for name in DOSE_PREDICTORS:
        rmse_rows.append(
            {
                "predictor": name,
                "predictor_kind": "spectrum"
                if name.startswith("spectrum_")
                else "basis_negative_control",
                "rmse": rmse[name],
                "rmse_improvement": improvements[name],
                "matched_control": MATCHED_CONTROLS.get(name),
            }
        )
    bridge_rows = [
        {
            "spectrum_predictor": spectrum,
            "spectrum_rmse_improvement": improvements[spectrum],
            "matched_basis_control": control,
            "matched_basis_rmse_improvement": improvements[control],
            "improves_baseline": improvements[spectrum] > 0,
            "exceeds_matched_control": improvements[spectrum] > improvements[control],
            "passed": improvements[spectrum] > 0 and improvements[spectrum] > improvements[control],
        }
        for spectrum, control in MATCHED_CONTROLS.items()
    ]
    return {
        "status": "supported" if supported else "negative",
        "complete": True,
        "claimable": True,
        "supported": supported,
        "local_supported": local_supported,
        "forward_bridge_supported": forward_supported,
        "pending_reason": None,
        "missing_inputs": [],
        "claim_boundary": protocol["dose_claim_boundary"],
        "manifest": _identity(manifest_path),
        "outputs": outputs,
        "sources": sources,
        "protocol_sources": protocol_sources,
        "evaluation_records": [evaluation_manifest, evaluation_input],
        "source_tables": [outputs["anchor_tests.csv"], outputs["heldout_predictions.csv"]],
        "anchor_rows": anchors,
        "anchor_criteria_rows": anchor_criteria,
        "criteria_rows": criteria_rows,
        "heldout_rows": heldout,
        "rmse_rows": rmse_rows,
        "bridge_rows": bridge_rows,
        "decision_counts": counts,
    }


def load_causal_chain_evidence(root: str | Path, *, allow_pending: bool = True) -> dict[str, Any]:
    """Load and independently validate causal-chain evidence beneath ``root``.

    Missing or explicitly pending branch manifests yield a non-claimable
    ``pending`` result when ``allow_pending`` is true.  Any purportedly complete
    evidence is always validated strictly; malformed complete evidence never
    degrades silently to pending.
    """

    root = Path(root).resolve()
    if not root.is_dir():
        raise _fail(f"Repository root is not a directory: {root}")
    protocol_path = root / PROTOCOL_PATH
    temporal_manifest = root / TEMPORAL_DIR / "summary_manifest.json"
    dose_manifest = root / DOSE_DIR / "summary_manifest.json"
    if (
        not protocol_path.is_file()
        and not temporal_manifest.is_file()
        and not dose_manifest.is_file()
    ):
        if not allow_pending:
            raise _fail("Causal-chain evidence is pending: protocol, temporal, dose")
        temporal = {
            "status": "pending",
            "complete": False,
            "claimable": False,
            "supported": None,
            "pending_reason": "causal-chain protocol and temporal manifest are missing",
            "missing_inputs": [str(protocol_path), str(temporal_manifest)],
            "claim_boundary": None,
            "manifest": None,
            "outputs": {},
            "source_tables": [],
            "paired_rows": [],
            "loso_rows": [],
            "estimate_rows": [],
            "criteria_rows": [],
            "rmse_rows": [],
        }
        dose = {
            "status": "pending",
            "complete": False,
            "claimable": False,
            "supported": None,
            "pending_reason": "causal-chain protocol and dose manifest are missing",
            "missing_inputs": [str(protocol_path), str(dose_manifest)],
            "claim_boundary": None,
            "manifest": None,
            "outputs": {},
            "sources": [],
            "protocol_sources": [],
            "evaluation_records": [],
            "source_tables": [],
            "anchor_rows": [],
            "anchor_criteria_rows": [],
            "criteria_rows": [],
            "heldout_rows": [],
            "rmse_rows": [],
            "bridge_rows": [],
        }
        boundary = {"temporal_short_branch": None, "dose_band": None}
        reporting_rows = {
            "temporal_estimates": [],
            "temporal_criteria": [],
            "dose_anchor_criteria": [],
            "dose_rmse": [],
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "repository_root": str(root),
            "status": "pending",
            "complete": False,
            "claimable": False,
            "overall_verdict": "pending_not_claimable",
            "verdict": "pending_not_claimable",
            "protocol": None,
            "claim_boundary": boundary,
            "claim_boundaries": boundary,
            "temporal_short_branch": temporal,
            "dose_band": dose,
            "temporal": temporal,
            "dose": dose,
            "reporting_rows": reporting_rows,
            "source_records": [],
            "source_table_records": [],
            "source_tables": [],
        }
    protocol = _validate_protocol(root)
    temporal = _load_temporal(root, protocol)
    dose = _load_dose(root, protocol)
    complete = temporal["complete"] and dose["complete"]
    if not complete and not allow_pending:
        pending = [
            name
            for name, branch in (("temporal", temporal), ("dose", dose))
            if not branch["complete"]
        ]
        raise _fail(f"Causal-chain evidence is pending: {', '.join(pending)}")
    if not complete:
        verdict = "pending_not_claimable"
        claimable = False
    else:
        claimable = True
        verdict = (
            "supported"
            if temporal["supported"] and dose["supported"]
            else "not_supported_claimable_negative"
        )
    reporting_rows = {
        "temporal_estimates": temporal["estimate_rows"],
        "temporal_criteria": temporal["criteria_rows"],
        "dose_anchor_criteria": dose["anchor_criteria_rows"],
        "dose_rmse": dose["rmse_rows"],
    }
    source_table_records = [*temporal["source_tables"], *dose["source_tables"]]
    source_records = [protocol["identity"]]
    source_records.extend(temporal.get("sources", []))
    source_records.extend(dose.get("protocol_sources", []))
    source_records.extend(dose.get("evaluation_records", []))
    source_records.extend(dose.get("sources", []))
    claim_boundary = {
        "temporal_short_branch": protocol["temporal_claim_boundary"],
        "dose_band": protocol["dose_claim_boundary"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "repository_root": str(root),
        "status": "complete" if complete else "pending",
        "complete": complete,
        "claimable": claimable,
        "overall_verdict": verdict,
        "verdict": verdict,
        "protocol": protocol["identity"],
        "claim_boundary": claim_boundary,
        "claim_boundaries": claim_boundary,
        "temporal_short_branch": temporal,
        "dose_band": dose,
        "temporal": temporal,
        "dose": dose,
        "reporting_rows": reporting_rows,
        "source_records": source_records,
        "source_table_records": source_table_records,
        "source_tables": source_table_records,
    }


__all__ = ["CausalChainReportingError", "load_causal_chain_evidence"]
