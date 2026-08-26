from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from .config import load_matrix, resolve_matrix_path
from .decontamination import DECONTAMINATED_TASK_NAMES
from .geometry import SCHEMA_VERSION, _sha256

HEADLINE_MACROS = (
    "DiscoveryHeadline",
    "CommonStateHeadline",
    "RepresentationHeadline",
    "InterventionHeadline",
    "ConfirmationHeadline",
)
PAPER_RESULT_TABLE_PATHS = (
    Path("paper/generated/discovery.tex"),
    Path("paper/generated/common-state.tex"),
    Path("paper/generated/representation.tex"),
    Path("paper/generated/intervention.tex"),
    Path("paper/generated/confirmation.tex"),
)
PAPER_CLAIM_PROTOCOL_SHA256 = "873e741bafad1bc0b2f8650e3614be1e7bb4af12e92c2d69c79045dad28937bb"
PAPER_CLAIM_SOURCE_PATHS = (
    "configs/experiment.yaml",
    "configs/common_state_probe.json",
    "configs/common_state_spectrum_probe.json",
    "configs/representation_probe.json",
    "configs/beir_representation_probe.json",
    "configs/functional_intervention.json",
    "configs/hybrid_adamw_control.json",
    "configs/short_branch_protocol.json",
    "configs/confirmatory_protocol.json",
    "docs/naacl-paper-plan.md",
)
STRICT_EVIDENCE = {
    "DiscoveryHeadline": (
        Path("reports/coverage.json"),
        Path("reports/training-dynamics/summary_manifest.json"),
        Path("reports/training-dynamics/plot_manifest.json"),
        Path("reports/retrieval-dynamics/summary_manifest.json"),
    ),
    "CommonStateHeadline": (
        Path("reports/common-state/summary_manifest.json"),
        Path("results/common-state-spectra/summary/summary_manifest.json"),
    ),
    "RepresentationHeadline": (
        Path("results/representation-space/training/summary/summary_manifest.json"),
        Path("results/representation-space/decontaminated-beir/summary/summary_manifest.json"),
        Path("reports/mechanism-bridge/summary_manifest.json"),
    ),
    "InterventionHeadline": (
        Path("reports/functional-intervention/manifest.json"),
        Path("reports/hybrid-adamw/summary_manifest.json"),
        Path("reports/short-branch/summary_manifest.json"),
        Path("reports/outcome-summary.manifest.json"),
    ),
    "ConfirmationHeadline": (
        Path("reports/confirmatory/summary_manifest.json"),
        Path("reports/outcome-summary.manifest.json"),
    ),
}
MACRO_PATTERN = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")


def _macros(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = MACRO_PATTERN.fullmatch(raw.strip())
        if match is None:
            continue
        name, value = match.groups()
        if name in result:
            raise ValueError(f"Duplicate paper result macro {name} at {path}:{number}")
        result[name] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_paper_claim_protocol(
    path: str | Path = "configs/paper_claim_protocol.json",
    *,
    repo_root: str | Path = ".",
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    root = Path(repo_root).resolve()
    requested = Path(path)
    protocol_path = requested.resolve() if requested.is_absolute() else (root / requested).resolve()
    protocol = _json(protocol_path)
    freeze = protocol.get("freeze_context", {})
    amendments = protocol.get("amendments")
    amendment = amendments[0] if isinstance(amendments, list) and len(amendments) == 1 else {}
    headlines = protocol.get("headline_contract", {})
    bindings = protocol.get("source_bindings")
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("status") != "prospective_completion_lock"
        or _sha256(protocol_path) != PAPER_CLAIM_PROTOCOL_SHA256
        or freeze.get("strict_beir_valid_units") != 168
        or freeze.get("strict_beir_expected_units") != 1_680
        or freeze.get("complete_retrieval_matrix_visible") is not False
        or freeze.get("retrieval_dynamics_output_visible") is not False
        or freeze.get("training_systems_output_visible") is not True
        or freeze.get("weight_trajectory_output_visible") is not True
        or any(
            freeze.get(field) is not False
            for field in (
                "formal_common_state_output_visible",
                "formal_representation_output_visible",
                "formal_functional_intervention_output_visible",
                "hybrid_adamw_output_visible",
                "short_branch_output_visible",
                "confirmatory_output_visible",
            )
        )
        or set(headlines) != set(HEADLINE_MACROS)
        or amendment.get("scope") != "documentation_only_weight_spectrum_tier_correction"
        or amendment.get("previous_source_sha256")
        != "2d61c1c1a150269986dbc41786f5b10c7304b45d23148278959ef3d75b72c888"
        or amendment.get("updated_source_sha256")
        != "adf12c547e4c337a5acb94657b7f6c4207da550c9f2f46ea3ea5098f3e418ce4"
        or amendment.get("strict_beir_valid_units") != 196
        or amendment.get("strict_beir_expected_units") != 1_680
        or amendment.get("complete_retrieval_matrix_visible") is not False
        or any(
            amendment.get(field) is not False
            for field in (
                "formal_common_state_output_visible",
                "formal_representation_output_visible",
                "formal_functional_intervention_output_visible",
                "hybrid_adamw_output_visible",
                "short_branch_output_visible",
                "confirmatory_output_visible",
                "headline_contract_changed",
                "result_contingent_story_map_changed",
            )
        )
        or not isinstance(bindings, list)
        or len(bindings) != len(PAPER_CLAIM_SOURCE_PATHS)
        or [item.get("path") for item in bindings if isinstance(item, dict)]
        != list(PAPER_CLAIM_SOURCE_PATHS)
        or "does not guarantee that any optimizer wins"
        not in str(protocol.get("claim_boundary", ""))
    ):
        raise ValueError("Paper claim protocol differs from its prospective completion lock")
    confirmation = headlines["ConfirmationHeadline"]
    representation = headlines["RepresentationHeadline"]
    if (
        "interval lower bound is above zero" not in str(confirmation.get("selection_rule", ""))
        or "otherwise inconclusive" not in str(confirmation.get("selection_rule", ""))
        or "descriptive" not in str(representation.get("selection_rule", ""))
    ):
        raise ValueError("Paper claim language no longer respects the frozen evidence boundary")

    source_records = []
    for relative, binding in zip(PAPER_CLAIM_SOURCE_PATHS, bindings, strict=True):
        source = (root / relative).resolve()
        if (
            not source.is_file()
            or not isinstance(binding, dict)
            or binding.get("path") != relative
            or binding.get("sha256") != _sha256(source)
        ):
            raise ValueError(f"Paper claim protocol source binding differs: {relative}")
        source_records.append(
            {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    return protocol_path, protocol, source_records


def _count_value(value: Any, target: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_value(item, target) for item in value.values())
    if isinstance(value, list):
        return sum(_count_value(item, target) for item in value)
    return int(value == target)


def _training_data_contract(
    root: Path,
    dataset_manifest_path: Path,
    contract_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _json(contract_path)
    bindings = contract.get("independent_protocol_bindings")
    expected_bindings = [
        "configs/confirmatory_protocol.json",
        "configs/hybrid_adamw_control.json",
        "configs/representation_probe.json",
        "configs/short_branch_protocol.json",
        "configs/validation_probe.json",
    ]
    declared_manifest = (root / str(contract.get("manifest_path", ""))).resolve()
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("status") != "materialized_training_view_receipt"
        or declared_manifest != dataset_manifest_path
        or contract.get("total_queries") != 500_000
        or contract.get("sampled_negatives") != 7
        or contract.get("seed") != 42
        or contract.get("source_repo") != "lightonai/embeddings-fine-tuning"
        or contract.get("source_revision") != "1ca463331ed637d25c1058567e932e0d3bad2983"
        or not isinstance(contract.get("manifest_sha256"), str)
        or not isinstance(contract.get("row_manifest_sha256"), str)
        or bindings != expected_bindings
    ):
        raise ValueError("Training-data paper receipt differs from the frozen materialized view")
    for relative in expected_bindings:
        binding_path = (root / relative).resolve()
        binding = _json(binding_path)
        if _count_value(binding, contract["manifest_sha256"]) != 1:
            raise ValueError(f"Training-data manifest hash is not bound exactly once by {relative}")

    local_available = dataset_manifest_path.is_file()
    if local_available:
        dataset = _json(dataset_manifest_path)
        expected_fields = {
            "total_queries": contract["total_queries"],
            "sampled_negatives": contract["sampled_negatives"],
            "seed": contract["seed"],
            "source_repo": contract["source_repo"],
            "source_revision": contract["source_revision"],
            "row_manifest_sha256": contract["row_manifest_sha256"],
            "dataset_fingerprint": contract["dataset_fingerprint"],
            "materialized_dataset_fingerprint": contract["materialized_dataset_fingerprint"],
        }
        if _sha256(dataset_manifest_path) != contract["manifest_sha256"] or any(
            dataset.get(name) != value for name, value in expected_fields.items()
        ):
            raise ValueError("Local training-data manifest differs from its distributable receipt")
    sources = {
        "dataset_contract": {
            "path": str(contract_path),
            "sha256": _sha256(contract_path),
        },
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": contract["manifest_sha256"],
            "local_byte_verification": local_available,
        },
    }
    return contract, sources


def _weight_constants(weight_dir: Path) -> dict[str, str]:
    manifest_path = weight_dir / "summary_manifest.json"
    manifest = _json(manifest_path)
    item = manifest.get("outputs", {}).get("optimizer_pair_contrast_trajectory.csv", {})
    table = weight_dir / "optimizer_pair_contrast_trajectory.csv"
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("expected_runs") != 24
        or manifest.get("observed_runs") != 24
        or manifest.get("checkpoint_rows") != 120
        or item.get("rows") != 40
        or item.get("bytes") != table.stat().st_size
        or item.get("sha256") != _sha256(table)
    ):
        raise ValueError("Weight-space paper source failed its strict manifest contract")
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 40:
        raise ValueError("Weight-space pair trajectory must contain exactly 40 rows")

    def value_range(field: str) -> str:
        values = [float(row[field]) for row in rows]
        return f"{min(values):.4f}--{max(values):.4f}"

    return {
        "NumWeightPairs": str(len(rows)),
        "DisplacementRatioRange": value_range("normuon_to_muon_displacement_ratio"),
        "RowCVRatioRange": value_range("normuon_to_muon_row_cv_ratio"),
        "TopRowEnergyRatioRange": value_range("normuon_to_muon_top_1pct_row_energy_ratio"),
    }


def _training_constants(training_dir: Path) -> dict[str, str]:
    manifest_path = training_dir / "summary_manifest.json"
    manifest = _json(manifest_path)
    coverage = manifest.get("coverage", {})
    item = manifest.get("outputs", {}).get("optimizer_systems", {})
    table = training_dir / "optimizer_system_summary.csv"
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("complete") is not True
        or coverage.get("runs") != 24
        or coverage.get("checkpoints") != 120
        or coverage.get("history_rows") != 9_384
        or coverage.get("optimizer_family_groups") != 6
        or item.get("rows") != 6
        or item.get("bytes") != table.stat().st_size
        or item.get("sha256") != _sha256(table)
    ):
        raise ValueError("Training-dynamics paper source failed its strict manifest contract")
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["model_family"], row["optimizer"]): row for row in rows}
    expected = {
        (family, optimizer)
        for family in ("dense", "late")
        for optimizer in ("adamw", "muon", "normuon")
    }
    if (
        len(rows) != 6
        or set(indexed) != expected
        or any(int(row["learning_rate_points"]) != 4 for row in rows)
    ):
        raise ValueError("Training systems table does not cover the six frozen sweep groups")

    def value_range(field: str) -> str:
        values = [float(row[field]) for row in rows if row["optimizer"] in {"muon", "normuon"}]
        if len(values) != 4 or not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(f"Training systems table has invalid {field} values")
        return f"{min(values):.4f}--{max(values):.4f}"

    return {
        "MuonFamilyThroughputRatioRange": value_range("throughput_to_adamw_ratio"),
        "MuonFamilyStateRatioRange": value_range("optimizer_state_to_adamw_ratio"),
    }


def expected_constant_macros(
    matrix: str | Path = "configs/experiment.yaml",
    weight_dir: str | Path = "reports/weight-space",
    training_dir: str | Path = "reports/training-dynamics",
    dataset_contract: str | Path = "configs/training_data_contract.json",
    *,
    repo_root: str | Path = ".",
) -> tuple[dict[str, str], dict[str, Any]]:
    matrix_path = resolve_matrix_path(matrix).resolve()
    root = Path(repo_root).resolve()
    configs = load_matrix(matrix_path)
    if len(configs) != 24 or {config.model_family for config in configs} != {"dense", "late"}:
        raise ValueError("Paper constants require the frozen 24-run, two-family discovery matrix")
    datasets = {
        (
            Path(config.dataset_path)
            if Path(config.dataset_path).is_absolute()
            else root / config.dataset_path
        ).resolve()
        for config in configs
    }
    max_lengths = {config.max_length for config in configs}
    checkpoint_counts = {len(config.checkpoint_fractions) for config in configs}
    if len(datasets) != 1 or max_lengths != {8192} or checkpoint_counts != {5}:
        raise ValueError(
            "Paper constants differ from the frozen dataset/context/checkpoint contract"
        )
    dataset_manifest_path = next(iter(datasets)) / "manifest.json"
    dataset, dataset_sources = _training_data_contract(
        root,
        dataset_manifest_path,
        (root / dataset_contract).resolve(),
    )
    expected = {
        "NumDiscoveryRuns": str(len(configs)),
        "NumDiscoveryCheckpoints": str(sum(len(c.checkpoint_fractions) for c in configs)),
        "NumBEIRTasks": str(len(DECONTAMINATED_TASK_NAMES)),
        "NumDiscoveryUnits": str(
            len(configs)
            * len(next(iter(configs)).checkpoint_fractions)
            * len(DECONTAMINATED_TASK_NAMES)
        ),
        "NumTrainingQueries": "500{,}000",
        "NumHardNegatives": str(dataset["sampled_negatives"]),
        "ContextLength": "8{,}192",
        **_training_constants(Path(training_dir).resolve()),
        **_weight_constants(Path(weight_dir).resolve()),
    }
    sources = {
        "matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
        **dataset_sources,
        "weight_manifest": {
            "path": str((Path(weight_dir).resolve() / "summary_manifest.json")),
            "sha256": _sha256(Path(weight_dir).resolve() / "summary_manifest.json"),
        },
        "training_manifest": {
            "path": str((Path(training_dir).resolve() / "summary_manifest.json")),
            "sha256": _sha256(Path(training_dir).resolve() / "summary_manifest.json"),
        },
    }
    return expected, sources


def _complete_manifest(path: Path) -> bool:
    try:
        payload = _json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if path.name == "coverage.json":
        return (
            payload.get("complete") is True
            and payload.get("observed_results") == 1680
            and payload.get("expected_results") == 1680
            and payload.get("observed_checkpoint_summaries") == 120
            and payload.get("expected_checkpoint_summaries") == 120
            and payload.get("missing") == []
            and payload.get("unexpected") == []
        )
    if path.name == "summary_manifest.json" and path.parent.name == "retrieval-dynamics":
        return _retrieval_dynamics_complete(path, payload)
    if path.name == "outcome-summary.manifest.json":
        return _outcome_report_complete(path, payload)
    if path.name == "paper-results.manifest.json":
        return _paper_results_complete(path, payload)
    return payload.get("schema_version") == SCHEMA_VERSION and payload.get("complete") is True


def _declared_path(root: Path, record: Any) -> Path | None:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    declared = Path(record["path"])
    return declared.resolve() if declared.is_absolute() else (root / declared).resolve()


def _hashed_file_complete(
    root: Path,
    record: Any,
    *,
    expected_path: Path | None = None,
    expected_rows: int | None = None,
) -> bool:
    path = _declared_path(root, record)
    return bool(
        path is not None
        and (expected_path is None or path == expected_path.resolve())
        and path.is_file()
        and not isinstance(record.get("bytes"), bool)
        and record.get("bytes") == path.stat().st_size
        and record.get("sha256") == _sha256(path)
        and (expected_rows is None or record.get("rows") == expected_rows)
    )


def _retrieval_dynamics_complete(path: Path, payload: dict[str, Any]) -> bool:
    root = path.parents[2]
    coverage = payload.get("coverage", {})
    sources = payload.get("sources", {})
    outputs = payload.get("outputs", {})
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("complete") is not True
        or coverage
        != {
            "runs": 24,
            "checkpoints": 120,
            "tasks": 14,
            "evaluation_units": 1_680,
            "optimizer_family_groups": 6,
        }
        or not isinstance(sources, dict)
        or not isinstance(outputs, dict)
    ):
        return False

    expected_outputs = {
        "checkpoint_dynamics": ("checkpoint_dynamics.csv", 120),
        "run_first_passage": ("run_first_passage.csv", 24),
        "optimizer_first_passage": ("optimizer_first_passage.csv", 6),
        "quality_vs_useful_wall_time": ("quality_vs_useful_wall_time.svg", None),
    }
    if set(outputs) != set(expected_outputs) or any(
        not _hashed_file_complete(
            root,
            outputs[name],
            expected_path=path.parent / filename,
            expected_rows=rows,
        )
        for name, (filename, rows) in expected_outputs.items()
    ):
        return False

    expected_sources = {
        "frozen_protocol": root / "configs/retrieval_dynamics_protocol.json",
        "matrix": root / "configs/experiment.yaml",
        "strict_coverage": root / "reports/coverage.json",
        "training_summary": root / "reports/training-dynamics/summary_manifest.json",
        "training_run_table": root / "reports/training-dynamics/run_summary.csv",
    }
    if any(
        not _hashed_file_complete(root, sources.get(name), expected_path=expected)
        for name, expected in expected_sources.items()
    ):
        return False
    if not _complete_manifest(expected_sources["strict_coverage"]):
        return False

    try:
        protocol = _json(expected_sources["frozen_protocol"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if (
        protocol.get("status") != "prospective_completion_lock"
        or protocol.get("freeze_context", {}).get("strict_beir_valid_units") != 160
        or protocol.get("freeze_context", {}).get("strict_beir_expected_units") != 1_680
        or protocol.get("freeze_context", {}).get("complete_retrieval_matrix_visible") is not False
        or protocol.get("reference_target", {}).get("uses_muon_or_normuon_outcomes") is not False
        or protocol.get("reference_target", {}).get("uses_confirmation_outcomes") is not False
        or protocol.get("matrix", {}).get("sha256") != sources["matrix"].get("sha256")
        or protocol.get("training_summary", {}).get("sha256")
        != sources["training_summary"].get("sha256")
    ):
        return False

    result_records = sources.get("evaluation_results")
    if not isinstance(result_records, list) or len(result_records) != 1_680:
        return False
    result_paths = [_declared_path(root, record) for record in result_records]
    return len(set(result_paths)) == 1_680 and all(
        _hashed_file_complete(root, record) for record in result_records
    )


def _outcome_report_complete(path: Path, payload: dict[str, Any]) -> bool:
    root = path.parents[1]
    sources = payload.get("sources", {})
    expected_sources = {
        "mechanism_report": root / "reports/mechanism-summary.manifest.json",
        "functional_intervention": root / "reports/functional-intervention/manifest.json",
        "hybrid_adamw": root / "reports/hybrid-adamw/summary_manifest.json",
        "short_branch": root / "reports/short-branch/summary_manifest.json",
        "confirmation": root / "reports/confirmatory/summary_manifest.json",
    }
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("complete") is not True
        or not isinstance(sources, dict)
        or set(sources) != set(expected_sources)
        or any(
            not _hashed_file_complete(root, sources[name], expected_path=source_path)
            or not _complete_manifest(source_path)
            for name, source_path in expected_sources.items()
        )
    ):
        return False

    expected_tables = (
        root / "reports/functional-intervention/family_summary.csv",
        root / "reports/hybrid-adamw/final_summary.csv",
        root / "reports/short-branch/paired_dynamics_summary.csv",
        root / "reports/confirmatory/paired_summary.csv",
    )
    tables = payload.get("source_tables")
    if (
        not isinstance(tables, list)
        or len(tables) != len(expected_tables)
        or any(
            not _hashed_file_complete(root, record, expected_path=expected)
            for record, expected in zip(tables, expected_tables, strict=True)
        )
    ):
        return False

    report_path = root / "reports/outcome-summary.md"
    blog_path = root / "docs/blog.md"
    if (
        not _hashed_file_complete(root, payload.get("output"), expected_path=report_path)
        or not _hashed_file_complete(root, payload.get("blog"), expected_path=blog_path)
        or payload.get("blog", {}).get("markers")
        != ["<!-- OUTCOMES:BEGIN -->", "<!-- OUTCOMES:END -->"]
    ):
        return False
    try:
        blog = blog_path.read_text(encoding="utf-8")
        outcome = report_path.read_text(encoding="utf-8").strip()
        mechanism_manifest = _json(expected_sources["mechanism_report"])
        mechanism_path = _declared_path(root, mechanism_manifest.get("output"))
        mechanism = mechanism_path.read_text(encoding="utf-8").strip()
    except (OSError, AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    outcome_begin, outcome_end = "<!-- OUTCOMES:BEGIN -->", "<!-- OUTCOMES:END -->"
    mechanism_begin, mechanism_end = "<!-- MECHANISM:BEGIN -->", "<!-- MECHANISM:END -->"
    return (
        blog.count(outcome_begin) == blog.count(outcome_end) == 1
        and blog.count(mechanism_begin) == blog.count(mechanism_end) == 1
        and blog.split(outcome_begin, 1)[1].split(outcome_end, 1)[0].strip() == outcome
        and blog.split(mechanism_begin, 1)[1].split(mechanism_end, 1)[0].strip() == mechanism
    )


def _paper_results_complete(path: Path, payload: dict[str, Any]) -> bool:
    root = path.parents[1]
    expected_evidence = {
        (root / relative).resolve() for paths in STRICT_EVIDENCE.values() for relative in paths
    }
    claim = payload.get("claim_protocol")
    evidence = payload.get("evidence_manifests")
    tables = payload.get("source_tables")
    result_tables = payload.get("result_tables")
    headlines = payload.get("headlines")
    results = payload.get("results_tex")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("complete") is not True
        or not _hashed_file_complete(
            root,
            claim,
            expected_path=root / "configs/paper_claim_protocol.json",
        )
        or claim.get("sha256") != PAPER_CLAIM_PROTOCOL_SHA256
        or claim.get("status") != "prospective_completion_lock"
        or not isinstance(evidence, list)
        or len(evidence) != len(expected_evidence)
        or {_declared_path(root, record) for record in evidence} != expected_evidence
        or any(not _hashed_file_complete(root, record) for record in evidence)
        or not isinstance(tables, list)
        or len(tables) != 10
        or any(not _hashed_file_complete(root, record) for record in tables)
        or not _paper_result_tables_complete(root, result_tables)
        or not isinstance(headlines, dict)
        or set(headlines) != set(HEADLINE_MACROS)
        or any(not isinstance(value, str) or not value for value in headlines.values())
        or not _hashed_file_complete(
            root,
            results,
            expected_path=root / "paper/results.tex",
        )
        or "do not convert descriptive checkpoint associations into causal evidence"
        not in str(payload.get("claim_boundary", ""))
    ):
        return False
    try:
        macros = _macros(root / "paper/results.tex")
    except (OSError, TypeError, ValueError):
        return False
    return all(
        "\\ResultPending" not in value and macros.get(name) == value
        for name, value in headlines.items()
    )


def _paper_result_tables_complete(root: Path, records: Any) -> bool:
    expected = tuple((root / relative).resolve() for relative in PAPER_RESULT_TABLE_PATHS)
    if not isinstance(records, list) or len(records) != len(expected):
        return False
    if any(
        not _hashed_file_complete(root, record, expected_path=path)
        for record, path in zip(records, expected, strict=True)
    ):
        return False
    try:
        return all("\\ResultPending" not in path.read_text(encoding="utf-8") for path in expected)
    except (OSError, UnicodeDecodeError):
        return False


def audit_paper(
    paper_dir: str | Path = "paper",
    *,
    repo_root: str | Path = ".",
    matrix: str | Path = "configs/experiment.yaml",
    weight_dir: str | Path = "reports/weight-space",
    training_dir: str | Path = "reports/training-dynamics",
    strict: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    paper = (root / paper_dir).resolve()
    results_path = paper / "results.tex"
    claim_path, claim_protocol, claim_sources = load_paper_claim_protocol(repo_root=root)
    macros = _macros(results_path)
    expected, sources = expected_constant_macros(
        root / matrix,
        root / weight_dir,
        root / training_dir,
        repo_root=root,
    )
    mismatches = {
        name: {"expected": value, "observed": macros.get(name)}
        for name, value in expected.items()
        if macros.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Paper constant macros differ from audited sources: {mismatches}")
    missing_headlines = sorted(set(HEADLINE_MACROS) - set(macros))
    if missing_headlines:
        raise ValueError(f"Paper headline macros are missing: {missing_headlines}")
    pending = sorted(name for name in HEADLINE_MACROS if "\\ResultPending" in macros.get(name, ""))
    evidence = {}
    for headline, relative_paths in STRICT_EVIDENCE.items():
        items = []
        for relative in relative_paths:
            path = root / relative
            items.append(
                {
                    "path": str(path),
                    "complete": _complete_manifest(path),
                    "sha256": _sha256(path) if path.is_file() else None,
                }
            )
        evidence[headline] = items
    incomplete_evidence = sorted(
        headline
        for headline, items in evidence.items()
        if not items or not all(item["complete"] for item in items)
    )
    paper_results_path = root / "reports/paper-results.manifest.json"
    paper_results_complete = _complete_manifest(paper_results_path)
    complete = not pending and not incomplete_evidence and paper_results_complete
    result = {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "strict": strict,
        "paper_dir": str(paper),
        "results_path": str(results_path),
        "constant_macros": expected,
        "constant_sources": sources,
        "claim_protocol": {
            "path": str(claim_path),
            "bytes": claim_path.stat().st_size,
            "sha256": _sha256(claim_path),
            "status": claim_protocol["status"],
            "frozen_at": claim_protocol["frozen_at"],
            "amendments": claim_protocol["amendments"],
            "source_bindings": claim_sources,
        },
        "pending_headlines": pending,
        "incomplete_evidence": incomplete_evidence,
        "evidence": evidence,
        "paper_results": {
            "path": str(paper_results_path),
            "complete": paper_results_complete,
            "sha256": _sha256(paper_results_path) if paper_results_path.is_file() else None,
        },
    }
    if strict and not complete:
        raise ValueError(
            "Paper is not final: "
            f"pending_headlines={pending}, incomplete_evidence={incomplete_evidence}, "
            f"paper_results_complete={paper_results_complete}"
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit manuscript constants, pending headlines, and strict evidence gates"
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--weight-dir", type=Path, default=Path("reports/weight-space"))
    parser.add_argument("--training-dir", type=Path, default=Path("reports/training-dynamics"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_paper(
        args.paper_dir,
        repo_root=args.repo_root,
        matrix=args.matrix,
        weight_dir=args.weight_dir,
        training_dir=args.training_dir,
        strict=args.strict,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
