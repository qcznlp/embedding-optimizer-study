"""Causal dose/band checks and held-out BEIR prediction for spectral transplants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_matrix
from .decontamination import DECONTAMINATED_TASK_NAMES
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
BANDS = ("head", "middle", "tail")
MIN_SUPPORT = 8
CAUSAL_PROTOCOL_SHA256 = "813538910dbd32d6fcfb65ee8736816cd573ae122ec907d25d7208229720cd7f"
DENSE_SCOPE_SHA256 = "c271f73988507bd95de96a3914c782de65d3fb118cd8385a14ad6d010bfc12b5"
DISCOVERY_CHECKPOINT_STEPS = (782, 1563, 2345, 3126, 3907)
EVALUATION_FIELDS = (
    "model_family",
    "optimizer",
    "learning_rate",
    "aux_learning_rate",
    "run_id",
    "stage",
    "fraction",
    "checkpoint_step",
    "task",
    "ndcg_at_10",
    "subsets",
    "result_path",
)
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
OUTPUTS = ("anchor_tests.csv", "heldout_predictions.csv", "report.md")
SPECTRAL_TABLES = ("anchor_query_tail_effects.csv",)
TAIL_CONDITIONS = {
    "muon-native",
    "adam-basis__spectrum-lambda-0.25",
    "adam-basis__spectrum-lambda-0.50",
    "adam-basis__spectrum-lambda-0.75",
    "adam-basis__muon-spectrum",
    "muon-basis__adam-spectrum",
    "adam-basis__muon-head-spectrum",
    "adam-basis__muon-middle-spectrum",
    "adam-basis__muon-tail-spectrum",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _validate_identity(record: dict[str, Any], path: Path) -> None:
    if (
        not path.is_file()
        or record.get("bytes") != path.stat().st_size
        or record.get("sha256") != _sha256(path)
    ):
        raise ValueError(f"Content identity differs for {path}")


def audit_receipt(
    output: Path,
    *,
    summary_dir: Path,
    evaluation: Path,
    protocol: Path,
    evaluation_manifest: Path,
) -> dict[str, Any]:
    manifest_path = output / "summary_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Dose/band summary manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("status") != "complete":
        raise ValueError("Dose/band analysis is not complete")
    with tempfile.TemporaryDirectory(prefix="dose-band-audit-") as temporary:
        expected_dir = Path(temporary)
        expected = analyze(
            summary_dir,
            evaluation,
            expected_dir,
            protocol=protocol,
            evaluation_manifest=evaluation_manifest,
        )
        for name in OUTPUTS:
            actual_path = output / name
            expected_path = expected_dir / name
            if not actual_path.is_file() or actual_path.read_bytes() != expected_path.read_bytes():
                raise ValueError(f"Dose/band output differs from fresh recomputation: {name}")
        actual_comparable = {key: value for key, value in manifest.items() if key != "outputs"}
        expected_comparable = {key: value for key, value in expected.items() if key != "outputs"}
        if actual_comparable != expected_comparable:
            raise ValueError("Dose/band manifest differs from fresh recomputation")
        for name in OUTPUTS:
            _validate_identity(manifest.get("outputs", {}).get(name, {}), output / name)
    return manifest


def _pending(
    output: Path,
    sources: list[Path],
    missing: list[Path],
    protocol_identity: dict[str, Any] | None = None,
    claim_boundary: str | None = None,
) -> dict[str, Any]:
    _write_csv(output / OUTPUTS[0], [], ["family", "anchor", "criterion", "passed"])
    _write_csv(
        output / OUTPUTS[1],
        [],
        ["family", "held_out_run", "task", "anchor", "observed_increment", "predicted_increment"],
    )
    (output / OUTPUTS[2]).write_text(
        "# Dose/band causal analysis\n\nStatus: **pending**. Missing upstream inputs:\n\n"
        + "".join(f"- `{path}`\n" for path in missing)
        + (f"\n> {claim_boundary}\n" if claim_boundary else ""),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
        "complete": False,
        "claimability": "pending",
        "falsification": "not_tested",
        "missing_inputs": [str(path) for path in missing],
        "available_sources": [_identity(path) for path in sources if path.is_file()],
        "protocol": protocol_identity,
        "claim_boundary": claim_boundary,
        "outputs": {name: _identity(output / name) for name in OUTPUTS},
    }
    _atomic_json(output / "summary_manifest.json", receipt)
    return receipt


def _finite(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}")
    return value


def _normalized_evaluation_row(row: dict[str, str]) -> tuple[Any, ...]:
    if tuple(row) != EVALUATION_FIELDS:
        raise ValueError("Discovery evaluation columns differ from the frozen contract")
    try:
        normalized = (
            row["model_family"],
            row["optimizer"],
            float(row["learning_rate"]),
            float(row["aux_learning_rate"]),
            row["run_id"],
            int(row["stage"]),
            float(row["fraction"]),
            int(row["checkpoint_step"]),
            row["task"],
            float(row["ndcg_at_10"]),
            int(row["subsets"]),
            row["result_path"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Discovery evaluation row is malformed") from error
    if not all(
        math.isfinite(value)
        for value in (normalized[2], normalized[3], normalized[6], normalized[9])
    ):
        raise ValueError("Discovery evaluation row contains a non-finite value")
    return normalized


def _evaluation_identity(row: tuple[Any, ...]) -> tuple[str, str, int, str]:
    return row[0], row[4], row[5], row[8]


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dose = payload.get("dose_band", {})
    if (
        _sha256(path) != CAUSAL_PROTOCOL_SHA256
        or payload.get("schema_version") != 1
        or payload.get("family") != "dense"
        or dose.get("anchor_scope", {}).get("expected_anchors") != 10
        or tuple(dose.get("dose_response", {}).get("lambdas", ())) != DOSES
        or dose.get("band_localization", {}).get("prespecified_band") != "tail"
        or dose.get("forward_retrieval_bridge", {}).get("expected_anchor_run_transitions") != 6
        or dose.get("forward_retrieval_bridge", {}).get("expected_rows") != 84
        or not isinstance(dose.get("claim_boundary"), str)
        or not dose["claim_boundary"]
    ):
        raise ValueError("Dose/band protocol differs from the frozen contract")
    root = path.parent.parent
    source_bindings = payload.get("source_bindings")
    if (
        not isinstance(source_bindings, list)
        or tuple(item.get("path") for item in source_bindings) != CAUSAL_SOURCE_PATHS
    ):
        raise ValueError("Dose/band source ledger differs from the frozen contract")
    bindings = []
    for item in source_bindings:
        source = (root / item["path"]).resolve()
        if not source.is_file() or _sha256(source) != item.get("sha256"):
            raise ValueError(f"Frozen causal-chain source binding differs: {source}")
        bindings.append(_identity(source))
    common_spec = json.loads((root / "configs/common_state_probe.json").read_text(encoding="utf-8"))
    anchor_protocol = common_spec["anchor_protocol"]
    if anchor_protocol["run_ids"] != dose["anchor_scope"]["source_runs"] or anchor_protocol[
        "checkpoint_fractions"
    ] != [0.2, 0.6, 1.0]:
        raise ValueError("Causal-chain anchors differ from the frozen common-state grid")
    configs = load_matrix(root / "configs/experiment.yaml")
    selected = {
        config.run_id: config
        for config in configs
        if config.model_family == "dense" and config.run_id in anchor_protocol["run_ids"]
    }
    if set(selected) != set(anchor_protocol["run_ids"]):
        raise ValueError("Frozen Dense anchor runs are missing from the experiment matrix")
    discovery_run_ids = tuple(
        sorted(config.run_id for config in configs if config.model_family == "dense")
    )
    if len(discovery_run_ids) != 12:
        raise ValueError("Frozen Dense discovery run count differs from 12")
    discovery_run_identities = tuple(
        sorted((config.model_family, config.run_id) for config in configs)
    )
    if (
        len(discovery_run_identities) != 24
        or len(set(discovery_run_identities)) != 24
        or {family for family, _ in discovery_run_identities} != {"dense", "late"}
    ):
        raise ValueError("Frozen full discovery run identities differ from 24")
    anchor_contract = {
        "run_ids": tuple(anchor_protocol["run_ids"]),
        "checkpoint_fractions": tuple(map(float, anchor_protocol["checkpoint_fractions"])),
        "discovery_run_ids": discovery_run_ids,
        "discovery_run_identities": discovery_run_identities,
    }
    return dose, {"protocol": _identity(path), "source_bindings": bindings}, anchor_contract


def _expected_anchor_identities(evaluation: Path, anchor_contract: dict[str, Any]) -> set[str]:
    rows = _read_csv(evaluation)
    expected_tasks = set(DECONTAMINATED_TASK_NAMES)
    anchors = {"dense/pretrained"}
    for run_id in anchor_contract["run_ids"]:
        for fraction in anchor_contract["checkpoint_fractions"]:
            members = [
                row
                for row in rows
                if row["model_family"] == "dense"
                and row["run_id"] == run_id
                and float(row["fraction"]) == fraction
            ]
            steps = {int(row["checkpoint_step"]) for row in members}
            tasks = {row["task"] for row in members}
            if len(members) != len(expected_tasks) or tasks != expected_tasks or len(steps) != 1:
                raise ValueError(
                    f"Discovery evaluation does not bind one complete anchor for {run_id}/{fraction}"
                )
            anchors.add(f"dense/{run_id}/checkpoint-{steps.pop()}")
    if len(anchors) != 10:
        raise ValueError("Frozen Dense anchor identity count differs from 10")
    return anchors


def _validate_evaluation_manifest(
    manifest_path: Path,
    evaluation: Path,
    discovery_run_identities: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = payload.get("outputs", {}).get("evaluation_long", {})
    repository = manifest_path.parent.parent.parent
    declared = Path(record.get("path", ""))
    declared = declared if declared.is_absolute() else repository / declared
    required = {
        "complete": True,
        "contract_complete": True,
        "dataset_complete": True,
        "training_complete": True,
        "evaluation_complete": True,
        "verified_experiment_runs": 12,
        "expected_experiment_runs": 12,
        "verified_training_runs": 12,
        "expected_training_runs": 12,
        "verified_training_checkpoints": 60,
        "expected_training_checkpoints": 60,
        "observed_results": 840,
        "expected_results": 840,
        "families": ["dense"],
        "selected_experiment_runs": 12,
        "selected_training_checkpoints": 60,
    }
    source_full_required = {
        "complete": True,
        "verified_experiment_runs": 24,
        "expected_experiment_runs": 24,
        "verified_training_runs": 24,
        "expected_training_runs": 24,
        "verified_training_checkpoints": 120,
        "expected_training_checkpoints": 120,
        "observed_results": 1680,
        "expected_results": 1680,
        "observed_checkpoint_summaries": 120,
        "expected_checkpoint_summaries": 120,
    }
    scope = payload.get("scope_amendment", {})
    source_full = payload.get("source_full_discovery", {})
    if (
        any(payload.get(key) != value for key, value in required.items())
        or scope.get("path") != "configs/dense_scope_amendment.json"
        or scope.get("sha256") != DENSE_SCOPE_SHA256
        or scope.get("status") != "user_directed_post_hoc_scope_amendment"
        or any(source_full.get(key) != value for key, value in source_full_required.items())
        or declared.resolve() != evaluation.resolve()
        or record.get("rows") != 840
    ):
        raise ValueError("Dense discovery evaluation manifest differs from the frozen contract")
    expected_reports = {
        "coverage": "reports/coverage.json",
        "evaluation_long": "reports/evaluation_long.csv",
        "system_metrics": "reports/system_metrics.csv",
        "training_history": "reports/training_history.csv",
    }
    reports = source_full.get("reports", {})
    if set(reports) != set(expected_reports):
        raise ValueError("Dense discovery source report ledger differs from the frozen contract")
    for label, relative in expected_reports.items():
        source_record = reports[label]
        if source_record.get("path") != relative:
            raise ValueError("Dense discovery source report path differs from the frozen contract")
        _validate_identity(source_record, repository / relative)
    full_coverage = json.loads(
        (repository / expected_reports["coverage"]).read_text(encoding="utf-8")
    )
    full_required = {
        "complete": True,
        "contract_complete": True,
        "dataset_complete": True,
        "training_complete": True,
        "deep_training_artifact_validation": True,
        "evaluation_complete": True,
        "verified_experiment_runs": 24,
        "expected_experiment_runs": 24,
        "verified_training_runs": 24,
        "expected_training_runs": 24,
        "verified_training_checkpoints": 120,
        "expected_training_checkpoints": 120,
        "observed_results": 1680,
        "expected_results": 1680,
        "observed_checkpoint_summaries": 120,
        "expected_checkpoint_summaries": 120,
        "missing": [],
        "unexpected": [],
        "contract_errors": [],
        "dataset_errors": [],
        "training_errors": [],
    }
    if any(full_coverage.get(key) != value for key, value in full_required.items()):
        raise ValueError("Full discovery coverage differs from its strict 1,680-unit contract")
    _validate_identity(record, evaluation)
    rows = _read_csv(evaluation)
    normalized_rows = [_normalized_evaluation_row(row) for row in rows]
    discovery_run_ids = tuple(
        run_id for family, run_id in discovery_run_identities if family == "dense"
    )
    if len(discovery_run_ids) != 12:
        raise ValueError("Dense discovery run identity contract differs from 12")
    expected_identities = {
        (run_id, stage, task)
        for run_id in discovery_run_ids
        for stage in range(1, 6)
        for task in DECONTAMINATED_TASK_NAMES
    }
    observed_identities = []
    checkpoint_steps: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in normalized_rows:
        family, run_id, stage, task = _evaluation_identity(row)
        fraction, checkpoint_step, score = row[6], row[7], row[9]
        if (
            family != "dense"
            or fraction != stage / 5
            or stage not in range(1, 6)
            or checkpoint_step != DISCOVERY_CHECKPOINT_STEPS[stage - 1]
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError("Dense discovery evaluation row differs from the frozen contract")
        identity = (run_id, stage, task)
        observed_identities.append(identity)
        checkpoint_steps[(run_id, stage)].add(checkpoint_step)
    if (
        len(rows) != 840
        or len(observed_identities) != len(set(observed_identities))
        or set(observed_identities) != expected_identities
        or any(len(steps) != 1 for steps in checkpoint_steps.values())
    ):
        raise ValueError("Dense discovery evaluation CSV differs from the exact 12x5x14 grid")
    full_rows = [
        _normalized_evaluation_row(row)
        for row in _read_csv(repository / expected_reports["evaluation_long"])
    ]
    full_identities = [_evaluation_identity(row) for row in full_rows]
    expected_full_identities = {
        (family, run_id, stage, task)
        for family, run_id in discovery_run_identities
        for stage in range(1, 6)
        for task in DECONTAMINATED_TASK_NAMES
    }
    if (
        len(full_rows) != 1680
        or len(full_identities) != len(set(full_identities))
        or set(full_identities) != expected_full_identities
    ):
        raise ValueError("Full discovery evaluation differs from its exact 1,680-unit grid")
    for row in full_rows:
        stage, fraction, checkpoint_step, score = row[5], row[6], row[7], row[9]
        if (
            fraction != stage / 5
            or stage not in range(1, 6)
            or checkpoint_step != DISCOVERY_CHECKPOINT_STEPS[stage - 1]
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError("Full discovery evaluation row differs from the frozen contract")
    dense_projection = {_evaluation_identity(row): row for row in full_rows if row[0] == "dense"}
    selected = {_evaluation_identity(row): row for row in normalized_rows}
    if len(dense_projection) != 840 or selected != dense_projection:
        raise ValueError("Dense discovery evaluation is not the exact full-report Dense projection")
    return _identity(manifest_path)


def _anchor_tests(
    summary: Path, expected_anchors: set[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    records = _read_csv(summary / "anchor_query_tail_effects.csv")
    indexed: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in records:
        key = (row["family"], row["anchor"])
        if row["condition"] in indexed[key]:
            raise ValueError(f"Duplicate tail condition for {row['anchor']}")
        indexed[key][row["condition"]] = row
    if {family for family, _ in indexed} != {"dense"}:
        raise ValueError("Dose/band tail inputs must contain exactly the Dense family")
    observed_anchors = {anchor for _, anchor in indexed}
    if expected_anchors is not None and observed_anchors != expected_anchors:
        raise ValueError("Dose/band tail anchor identities differ from the frozen 10-anchor grid")
    dose_conditions = (
        "adam-basis__spectrum-lambda-0.25",
        "adam-basis__spectrum-lambda-0.50",
        "adam-basis__spectrum-lambda-0.75",
        "adam-basis__muon-spectrum",
    )
    band_conditions = dict(
        zip(
            BANDS,
            (
                "adam-basis__muon-head-spectrum",
                "adam-basis__muon-middle-spectrum",
                "adam-basis__muon-tail-spectrum",
            ),
            strict=True,
        )
    )
    rows, features = [], {}
    for (family, anchor), conditions in sorted(indexed.items()):
        if set(conditions) != TAIL_CONDITIONS:
            raise ValueError(f"Tail conditions differ from the exact frozen set for {anchor}")
        required = {*dose_conditions, *band_conditions.values(), "muon-basis__adam-spectrum"}
        if not required.issubset(conditions):
            raise ValueError(f"Incomplete dose/band/control grid for {anchor}")
        # Lambda zero is AdamW native relative to itself. The summary contract
        # intentionally omits this all-zero identity contrast.
        losses = [
            0.0,
            *[_finite(conditions[name], "p95_pairwise_loss_contrast") for name in dose_conditions],
        ]
        margins = [
            0.0,
            *[
                _finite(conditions[name], "p05_pairwise_margin_contrast")
                for name in dose_conditions
            ],
        ]
        loss_monotone = all(right <= left for left, right in zip(losses, losses[1:]))
        margin_monotone = all(right >= left for left, right in zip(margins, margins[1:]))
        spectrum = conditions["adam-basis__muon-spectrum"]
        basis = conditions["muon-basis__adam-spectrum"]
        control = _finite(spectrum, "p95_pairwise_loss_contrast") < _finite(
            basis, "p95_pairwise_loss_contrast"
        ) and _finite(spectrum, "p05_pairwise_margin_contrast") > _finite(
            basis, "p05_pairwise_margin_contrast"
        )
        band_loss = {
            name: _finite(conditions[value], "p95_pairwise_loss_contrast")
            for name, value in band_conditions.items()
        }
        band_margin = {
            name: _finite(conditions[value], "p05_pairwise_margin_contrast")
            for name, value in band_conditions.items()
        }
        tail_best = all(
            band_loss["tail"] < band_loss[name] and band_margin["tail"] > band_margin[name]
            for name in ("head", "middle")
        )
        rows.append(
            {
                "family": family,
                "anchor": anchor,
                "loss_dose_monotone": loss_monotone,
                "margin_dose_monotone": margin_monotone,
                "tail_band_best_both_metrics": tail_best,
                "basis_swap_negative_control": control,
                **{f"loss_lambda_{value:.2f}": observed for value, observed in zip(DOSES, losses)},
                **{
                    f"margin_lambda_{value:.2f}": observed
                    for value, observed in zip(DOSES, margins)
                },
                **{f"loss_band_{name}": band_loss[name] for name in BANDS},
                **{f"margin_band_{name}": band_margin[name] for name in BANDS},
                "anchor_passed": loss_monotone and margin_monotone and tail_best and control,
            }
        )
        features[anchor] = {
            "spectrum_loss": _finite(spectrum, "mean_pairwise_loss_contrast"),
            "spectrum_margin": _finite(spectrum, "mean_pairwise_margin_contrast"),
            "basis_loss": _finite(basis, "mean_pairwise_loss_contrast"),
            "basis_margin": _finite(basis, "mean_pairwise_margin_contrast"),
        }
    return rows, features


def _task_features(
    upstream: dict[str, Any],
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    output, identities = {}, []
    sources = upstream.get("sources", [])
    labels = [source.get("label") for source in sources if isinstance(source, dict)]
    if len(sources) != 10 or len(labels) != 10 or len(set(labels)) != 10:
        raise ValueError("Raw spectral source labels do not cover ten unique anchors")
    for source in sources:
        anchor = source["label"]
        record = source["sample_metrics"]
        path = Path(record["path"])
        _validate_identity(record, path)
        identities.append(_identity(path))
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                grouped[(str(row["group"]), str(row["condition"]))].append(row)
        tasks = sorted({key[0] for key in grouped})
        task_rows = {}
        for task in tasks:
            adam = grouped[(task, "adamw-native")]
            if not adam:
                raise ValueError(f"Missing Adam reference samples for {anchor}/{task}")
            by_id = {int(row["sample_id"]): row for row in adam}
            if len(by_id) != len(adam):
                raise ValueError(f"Duplicate Adam sample IDs for {anchor}/{task}")
            features = {}
            for prefix, condition in (
                ("spectrum", "adam-basis__muon-spectrum"),
                ("basis", "muon-basis__adam-spectrum"),
            ):
                condition_rows = grouped[(task, condition)]
                challenger = {int(row["sample_id"]): row for row in condition_rows}
                if len(challenger) != len(condition_rows):
                    raise ValueError(f"Duplicate intervention sample IDs for {anchor}/{task}")
                if set(challenger) != set(by_id):
                    raise ValueError(
                        f"Task sample coverage differs for {anchor}/{task}/{condition}"
                    )
                features[f"{prefix}_loss"] = float(
                    np.mean(
                        [
                            float(challenger[i]["contrastive_loss"])
                            - float(by_id[i]["contrastive_loss"])
                            for i in by_id
                        ]
                    )
                )
                features[f"{prefix}_margin"] = float(
                    np.mean(
                        [
                            float(challenger[i]["positive_margin"])
                            - float(by_id[i]["positive_margin"])
                            for i in by_id
                        ]
                    )
                )
            task_rows[task] = features
        output[anchor] = task_rows
    return output, identities


def _forward_bridge_supported(improvements: dict[str, float]) -> bool:
    return any(
        improvements[spectrum] > 0.0 and improvements[spectrum] > improvements[basis]
        for spectrum, basis in (
            ("spectrum_loss", "basis_loss"),
            ("spectrum_margin", "basis_margin"),
        )
    )


def _predictions(
    evaluation: Path,
    features: dict[str, dict[str, dict[str, float]]],
    *,
    expected_runs: tuple[str, ...] | None = None,
    expected_tasks: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eval_rows = _read_csv(evaluation)
    evaluation_keys = [
        (row["model_family"], row["run_id"], int(row["stage"]), row["task"]) for row in eval_rows
    ]
    if len(set(evaluation_keys)) != len(evaluation_keys):
        raise ValueError("Dense discovery evaluation contains duplicate run/stage/task identities")
    indexed = {
        key: _finite(row, "ndcg_at_10") for key, row in zip(evaluation_keys, eval_rows, strict=True)
    }
    examples = []
    for anchor, task_features in features.items():
        parts = anchor.split("/")
        if len(parts) != 3 or not parts[2].startswith("checkpoint-"):
            continue
        family, run_id, checkpoint = parts
        step = int(checkpoint.removeprefix("checkpoint-"))
        candidates = [
            row
            for row in eval_rows
            if row["model_family"] == family
            and row["run_id"] == run_id
            and int(row["checkpoint_step"]) == step
        ]
        for row in candidates:
            stage, task = int(row["stage"]), row["task"]
            if stage not in (1, 3) or task not in task_features:
                continue
            target = indexed.get((family, run_id, stage + 1, task))
            if target is not None:
                examples.append(
                    (
                        family,
                        run_id,
                        task,
                        f"stage{stage}-to-{stage + 1}",
                        anchor,
                        task_features[task],
                        target - _finite(row, "ndcg_at_10"),
                    )
                )
    output = []
    predictor_names = ("spectrum_loss", "spectrum_margin", "basis_loss", "basis_margin")
    tasks = sorted({item[2] for item in examples})
    transitions = sorted({item[3] for item in examples})

    def baseline(item):
        return [
            1.0,
            *(float(item[2] == task) for task in tasks[1:]),
            *(float(item[3] == transition) for transition in transitions[1:]),
        ]

    for held_run in sorted({item[1] for item in examples}):
        training = [item for item in examples if item[1] != held_run]
        held = [item for item in examples if item[1] == held_run]
        base_x = np.asarray([baseline(item) for item in training])
        y = np.asarray([item[6] for item in training])
        base_beta = np.linalg.lstsq(base_x, y, rcond=None)[0]
        models = {}
        for name in predictor_names:
            values = np.asarray([item[5][name] for item in training])
            mean = float(values.mean())
            scale = float(values.std())
            if scale == 0:
                raise ValueError(f"Zero-variance held-out predictor: {name}")
            x = np.asarray([baseline(item) + [(item[5][name] - mean) / scale] for item in training])
            models[name] = (np.linalg.lstsq(x, y, rcond=None)[0], mean, scale)
        for item in held:
            family, run_id, task, transition, anchor, feature, target = item
            base_prediction = float(np.asarray(baseline(item)) @ base_beta)
            row = {
                "family": family,
                "held_out_run": run_id,
                "held_out_learning_rate": next(
                    r["learning_rate"]
                    for r in eval_rows
                    if r["model_family"] == family and r["run_id"] == run_id
                ),
                "task": task,
                "transition": transition,
                "anchor": anchor,
                "observed_increment": target,
                "baseline_prediction": base_prediction,
                "fold": "leave-one-run-and-learning-rate-out",
            }
            for name, (beta, mean, scale) in models.items():
                row[f"{name}_prediction"] = float(
                    np.asarray(baseline(item) + [(feature[name] - mean) / scale]) @ beta
                )
            output.append(row)
    if expected_runs is not None or expected_tasks is not None:
        if expected_runs is None or expected_tasks is None:
            raise ValueError("Expected forward runs and tasks must be supplied together")
        expected_grid = {
            (run_id, task, transition)
            for run_id in expected_runs
            for task in expected_tasks
            for transition in ("stage1-to-2", "stage3-to-4")
        }
        observed_grid = [(row["held_out_run"], row["task"], row["transition"]) for row in output]
        if len(observed_grid) != len(set(observed_grid)) or set(observed_grid) != expected_grid:
            raise ValueError("Held-out prediction identities differ from the frozen 84-row grid")
    rmse = {
        "baseline": float(
            np.sqrt(
                np.mean(
                    [
                        (row["baseline_prediction"] - row["observed_increment"]) ** 2
                        for row in output
                    ]
                )
            )
        )
    }
    for name in predictor_names:
        rmse[name] = float(
            np.sqrt(
                np.mean(
                    [(row[f"{name}_prediction"] - row["observed_increment"]) ** 2 for row in output]
                )
            )
        )
    improvements = {name: rmse["baseline"] - rmse[name] for name in predictor_names}
    bridge_supported = _forward_bridge_supported(improvements)
    return output, {"rmse": rmse, "rmse_improvement": improvements, "supported": bridge_supported}


def analyze(
    summary_dir: Path,
    evaluation: Path,
    output: Path,
    *,
    protocol: Path = Path("configs/causal_chain_analysis.json"),
    evaluation_manifest: Path = Path("reports/dense-discovery/coverage.json"),
    audit: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if audit:
        return audit_receipt(
            output,
            summary_dir=summary_dir,
            evaluation=evaluation,
            protocol=protocol,
            evaluation_manifest=evaluation_manifest,
        )
    protocol = protocol.resolve()
    dose_protocol, protocol_identity, anchor_contract = _load_protocol(protocol)
    evaluation_manifest = evaluation_manifest.resolve()
    inputs = [
        summary_dir / "summary_manifest.json",
        summary_dir / "anchor_query_tail_effects.csv",
        evaluation,
        evaluation_manifest,
    ]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        return _pending(
            output,
            inputs,
            missing,
            protocol_identity,
            dose_protocol["claim_boundary"],
        )
    evaluation_manifest_identity = _validate_evaluation_manifest(
        evaluation_manifest, evaluation, anchor_contract["discovery_run_identities"]
    )
    expected_anchors = _expected_anchor_identities(evaluation, anchor_contract)
    upstream = json.loads(inputs[0].read_text(encoding="utf-8"))
    if upstream.get("complete") is not True:
        return _pending(
            output,
            inputs,
            [inputs[0]],
            protocol_identity,
            dose_protocol["claim_boundary"],
        )
    declared_outputs = upstream.get("outputs", {})
    for filename in SPECTRAL_TABLES:
        key = filename.removesuffix(".csv")
        record = declared_outputs.get(key, {})
        declared = Path(record.get("path", ""))
        expected = summary_dir / filename
        if declared.resolve() != expected.resolve():
            raise ValueError(f"Spectral summary output path differs for {filename}")
        _validate_identity(record, expected)
    tests, _ = _anchor_tests(summary_dir, expected_anchors)
    task_features, raw_sources = _task_features(upstream)
    if set(task_features) != expected_anchors:
        raise ValueError("Raw spectral sample anchors differ from the frozen 10-anchor grid")
    expected_tasks = tuple(DECONTAMINATED_TASK_NAMES)
    if any(set(rows) != set(expected_tasks) for rows in task_features.values()):
        raise ValueError("Raw spectral task identities differ from the frozen 14-task suite")
    source_runs = tuple(dose_protocol["anchor_scope"]["source_runs"])
    predictions, bridge = _predictions(
        evaluation,
        task_features,
        expected_runs=source_runs,
        expected_tasks=expected_tasks,
    )
    by_family = defaultdict(list)
    for row in tests:
        by_family[row["family"]].append(row)
    decisions = []
    for family, rows in sorted(by_family.items()):
        loss_support = sum(bool(row["loss_dose_monotone"]) for row in rows)
        margin_support = sum(bool(row["margin_dose_monotone"]) for row in rows)
        control_support = sum(bool(row["basis_swap_negative_control"]) for row in rows)
        band_support = sum(bool(row["tail_band_best_both_metrics"]) for row in rows)
        supported = sum(bool(row["anchor_passed"]) for row in rows)
        decisions.append(
            {
                "family": family,
                "anchors": len(rows),
                "supporting_anchors": supported,
                "loss_dose_monotone_anchors": loss_support,
                "margin_dose_monotone_anchors": margin_support,
                "basis_control_anchors": control_support,
                "prespecified_band": "tail",
                "tail_band_anchors": band_support,
                "threshold": MIN_SUPPORT,
                "local_supported": len(rows) == 10
                and min(loss_support, margin_support, control_support, band_support) >= MIN_SUPPORT,
            }
        )
    claimable = bool(decisions) and all(row["anchors"] == 10 for row in decisions)
    expected_predictions = len(source_runs) * 2 * len(expected_tasks)
    if len(predictions) != expected_predictions:
        raise ValueError(
            f"Held-out next-stage prediction coverage is {len(predictions)}/{expected_predictions}"
        )
    local_supported = all(row["local_supported"] for row in decisions)
    supported = local_supported and bridge["supported"]
    _write_csv(output / OUTPUTS[0], tests, list(tests[0]))
    prediction_fields = list(predictions[0]) if predictions else ["family"]
    _write_csv(output / OUTPUTS[1], predictions, prediction_fields)
    verdict = "supported" if supported else "not supported (claimable negative result)"
    markdown_lines = [
        "# Spectral dose/band causal analysis",
        "",
        f"Overall frozen spectral-component chain: **{verdict}**.",
        "",
        *(
            f"- {row['family']}: local_supported={str(row['local_supported']).lower()} "
            f"({row['supporting_anchors']}/{row['anchors']} all-criterion anchors)"
            for row in decisions
        ),
        f"- forward_bridge_supported={str(bridge['supported']).lower()} over {len(predictions)} held-out rows",
        "",
        "| Predictor | RMSE | Improvement over task+transition baseline |",
        "| --- | ---: | ---: |",
        f"| baseline | {bridge['rmse']['baseline']:.6g} | 0 |",
    ]
    for name in ("spectrum_loss", "spectrum_margin", "basis_loss", "basis_margin"):
        markdown_lines.append(
            f"| {name} | {bridge['rmse'][name]:.6g} | {bridge['rmse_improvement'][name]:+.6g} |"
        )
    markdown_lines += ["", f"> {dose_protocol['claim_boundary']}", ""]
    (output / OUTPUTS[2]).write_text("\n".join(markdown_lines), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "complete": True,
        "claimability": "claimable" if claimable else "unresolved",
        "supported": supported,
        "falsification": "overall_chain_supported" if supported else "overall_chain_not_supported",
        "local_supported": local_supported,
        "forward_bridge_supported": bridge["supported"],
        "criterion": "at_least_8_of_10_anchors",
        "decisions": decisions,
        "prediction_protocol": "separate OLS predictors over task+transition baseline with leave-one-run/LR-out folds",
        "forward_bridge": bridge,
        "claim_boundary": dose_protocol["claim_boundary"],
        "prediction_rows": len(predictions),
        "expected_prediction_rows": expected_predictions,
        "protocol": protocol_identity,
        "evaluation_manifest": evaluation_manifest_identity,
        "evaluation_input": _identity(evaluation),
        "sources": [_identity(path) for path in inputs] + raw_sources,
        "outputs": {name: _identity(output / name) for name in OUTPUTS},
    }
    _atomic_json(output / "summary_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectral-summary-dir", type=Path, default=Path("reports/spectral-transplant")
    )
    parser.add_argument(
        "--evaluation", type=Path, default=Path("reports/dense-discovery/evaluation_long.csv")
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=Path("reports/dense-discovery/coverage.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/dose-band"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/causal_chain_analysis.json"))
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = analyze(
        args.spectral_summary_dir.resolve(),
        args.evaluation.resolve(),
        args.output_dir.resolve(),
        protocol=args.protocol.resolve(),
        evaluation_manifest=args.evaluation_manifest.resolve(),
        audit=args.audit,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
