from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .causal_chain_rendering import (
    CAUSAL_HEADLINE_PREFIX,
    causal_chain_display_contract,
    causal_chain_paper_contract,
    render_causal_chain_headline_fragment,
    render_causal_chain_latex,
    render_causal_chain_markdown,
)
from .causal_chain_reporting import load_causal_chain_evidence
from .config import load_matrix, resolve_matrix_path
from .decontamination import DECONTAMINATED_TASK_NAMES
from .dense_retrieval_dynamics_publication import (
    DYNAMICS_EXTENSION_CSV,
    DYNAMICS_EXTENSION_MANIFEST,
    DYNAMICS_EXTENSION_PDF,
    DYNAMICS_EXTENSION_SVG,
    DYNAMICS_EXTENSION_TEX,
    load_publication_rows,
    render_publication_latex,
    summarize_publication_rows,
)
from .geometry import SCHEMA_VERSION, _sha256
from .outcome_report import (
    FINAL_CONCLUSION_MARKERS,
    FINAL_CONCLUSION_PENDING,
    _confirmation_rows,
    _hybrid_rows,
    _tail_stability_rows,
    build_final_conclusion_contract,
)
from .paper_layout import MAIN_END_LABEL
from .scope import ALL_FAMILIES, resolve_scope
from .state_operator_factorial_publication import (
    PAPER_LATEX as STATE_OPERATOR_LATEX,
)
from .state_operator_factorial_publication import (
    PUBLICATION_MANIFEST as STATE_OPERATOR_PUBLICATION_MANIFEST,
)
from .state_operator_factorial_publication import (
    SUMMARY_MANIFEST as STATE_OPERATOR_SUMMARY_MANIFEST,
)
from .state_operator_factorial_publication import audit_state_operator_publication

FAMILIES = ALL_FAMILIES
CausalEvidenceCache = dict[Path, tuple[dict[str, Any] | None, Exception | None]]
LEGACY_CHECKOUT_NAME = "embedding-optimizer-study"
PORTABLE_EVIDENCE_MANIFEST = Path("configs/portable_paper_evidence.json")
PORTABLE_EVIDENCE_SOURCE_MANIFESTS = (
    Path("reports/retrieval-dynamics/summary_manifest.json"),
    Path("reports/tail-stability/summary_manifest.json"),
    Path("reports/spectral-transplant/summary_manifest.json"),
    Path("reports/dense-retrieval-dynamics/summary_manifest.json"),
)

HEADLINE_MACROS = (
    "DiscoveryHeadline",
    "CommonStateHeadline",
    "RepresentationHeadline",
    "InterventionHeadline",
    "ConfirmationHeadline",
)
PAPER_RESULT_TABLE_PATHS = (
    Path("paper/generated/discovery.tex"),
    Path("paper/generated/per-task.tex"),
    Path("paper/generated/common-state.tex"),
    Path("paper/generated/representation.tex"),
    Path("paper/generated/intervention.tex"),
    Path("paper/generated/confirmation.tex"),
    Path("paper/generated/diagnostics.tex"),
    Path("paper/generated/causal-chain.tex"),
)
PAPER_MAIN_RESULT_TABLE_PATHS = (
    Path("paper/generated/discovery.tex"),
    Path("paper/generated/common-state.tex"),
    Path("paper/generated/intervention.tex"),
    Path("paper/generated/confirmation.tex"),
)
PAPER_APPENDIX_RESULT_TABLE_PATHS = (
    Path("paper/generated/diagnostics.tex"),
    Path("paper/generated/representation.tex"),
    Path("paper/generated/per-task.tex"),
)
PAPER_DEFINITION_RESULT_TABLE_PATHS = (Path("paper/generated/causal-chain.tex"),)
PAPER_MAIN_GENERATED_INPUTS = tuple(
    rf"\input{{{path.relative_to('paper').with_suffix('').as_posix()}}}"
    for path in PAPER_MAIN_RESULT_TABLE_PATHS
)
PAPER_APPENDIX_GENERATED_INPUTS = tuple(
    rf"\input{{{path.relative_to('paper').with_suffix('').as_posix()}}}"
    for path in PAPER_APPENDIX_RESULT_TABLE_PATHS
)
PAPER_DEFINITION_GENERATED_INPUTS = tuple(
    rf"\input{{{path.relative_to('paper').with_suffix('').as_posix()}}}"
    for path in PAPER_DEFINITION_RESULT_TABLE_PATHS
)
PAPER_MAIN_REQUIRED_ONCE = (
    r"\input{results}",
    *PAPER_MAIN_GENERATED_INPUTS,
    *PAPER_APPENDIX_GENERATED_INPUTS,
    *PAPER_DEFINITION_GENERATED_INPUTS,
    rf"\input{{{DYNAMICS_EXTENSION_TEX.relative_to('paper').with_suffix('').as_posix()}}}",
    r"\input{generated/candidate-breadth}",
    r"\input{generated/corrected-no-packing}",
    r"\input{generated/state-operator-factorial}",
    r"\CorrectedAbstractFinding",
    r"\CorrectedMainSection",
    r"\CorrectedConclusionFinding",
    r"\StateOperatorAbstractFinding",
    r"\StateOperatorMechanismFinding",
    r"\StateOperatorConclusionFinding",
    r"\StateOperatorAppendixTable",
    r"\CausalChainSummaryTable",
    r"\CausalChainDiagnostics",
    r"\section{Conclusion}",
    r"\ResultConclusion",
    r"\CandidateBreadthDiscussion",
    r"\CandidateBreadthConclusion",
    r"\CandidateBreadthFigure",
    rf"\label{{{MAIN_END_LABEL}}}",
    r"\section{Limitations}",
    r"\section{Ethical Considerations}",
    r"\bibliography{references}",
    r"\appendix",
    r"\section{Artifact and Reproducibility}",
    r"\CorrectedGeometryBridgeTable",
    r"\CorrectedExecutionSensitivityTable",
)
PAPER_DISCOVERY_FIGURE_INCLUDES = (
    r"\centering\includegraphics[width=\linewidth]{../reports/dense-discovery/figures/dense-training-dynamics-by-run.png}",
    r"\centering\includegraphics[width=\linewidth]{../reports/dense-discovery/figures/dense-lr-sensitivity.png}",
)
PAPER_DISCOVERY_FIGURE_CAPTION = (
    r"\caption{DenseOn discovery dynamics. Panel (a) retains all 12 runs and all five checkpoints; "
    r"panel (b) shows the final nDCG@10 response over each optimizer's frozen four-point "
    r"learning-rate grid. These one-seed discovery curves are exploratory and do not replace the "
    r"validation-frozen confirmation.}"
)
PAPER_DISCOVERY_FIGURE_LABEL = r"\label{fig:dense-discovery-dynamics}"
PAPER_SOURCE_TABLE_PATHS = (
    Path("reports/retrieval-dynamics/checkpoint_dynamics.csv"),
    Path("reports/retrieval-dynamics/optimizer_first_passage.csv"),
    Path("reports/retrieval-dynamics/best_config_task_comparison.csv"),
    Path("reports/retrieval-dynamics/task_delta_stability.csv"),
    Path("reports/training-dynamics/optimizer_system_summary.csv"),
    Path("reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.csv"),
    Path("reports/common-state/anchor_contrasts.csv"),
    Path("reports/basis-sensitivity/summary.csv"),
    Path("results/common-state-spectra/summary/spectrum_metrics.csv"),
    Path("reports/mechanism-bridge/checkpoint_bridge.csv"),
    Path("reports/mechanism-bridge/descriptive_correlations.csv"),
    Path("reports/functional-intervention/family_summary.csv"),
    Path("reports/hybrid-adamw/final_summary.csv"),
    Path("reports/short-branch/paired_dynamics_summary.csv"),
    Path("reports/tail-stability/discovery_cross_tail_summary.csv"),
    Path("reports/tail-stability/short_branch_final_summary.csv"),
    Path("reports/spectral-transplant/family_factorial_summary.csv"),
    Path("reports/spectral-transplant/family_query_tail_summary.csv"),
    Path("reports/confirmatory/paired_summary.csv"),
    Path("reports/temporal-short-branch/paired_contrasts.csv"),
    Path("reports/temporal-short-branch/loso_predictions.csv"),
    Path("reports/temporal-short-branch/estimates.csv"),
    Path("reports/dose-band/anchor_tests.csv"),
    Path("reports/dose-band/heldout_predictions.csv"),
)
PAPER_CLAIM_PROTOCOL_SHA256 = "0ddff916eccedbe493b41a07538d0e4e9e058a784f5440e10d688f5270609949"
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
    "configs/basis_sensitivity.json",
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
        Path("reports/basis-sensitivity/summary_manifest.json"),
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
        Path("reports/tail-stability/summary_manifest.json"),
        Path("reports/spectral-transplant/summary_manifest.json"),
        Path("reports/outcome-summary.manifest.json"),
    ),
    "ConfirmationHeadline": (
        Path("reports/confirmatory/summary_manifest.json"),
        Path("reports/outcome-summary.manifest.json"),
    ),
}
SCOPED_DENSE_DISCOVERY_EVIDENCE = (
    Path("reports/dense-discovery/coverage.json"),
    Path("reports/training-dynamics/summary_manifest.json"),
    Path("reports/training-dynamics/plot_manifest.json"),
    Path("reports/retrieval-dynamics-dense/summary_manifest.json"),
    DYNAMICS_EXTENSION_MANIFEST,
)
MACRO_PATTERN = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")
FINAL_DOCUMENT_STALE_PHRASES = {
    Path("README.md"): (
        "remaining DenseOn confirmation is running",
        "Once complete, the supplemental",
    ),
    Path("paper/main.tex"): (
        "The final analysis will report",
        "The practical recommendation will therefore",
        "intentionally left unresolved",
    ),
}

_LATEX_SECTION_COMMAND = re.compile(r"\\section(?![A-Za-z@])")
_LATEX_FILE_INPUT_COMMAND = re.compile(r"\\(?:input|include)(?![A-Za-z@])")


def _strip_latex_comments(source: str) -> str:
    """Return the active TeX source after removing unescaped comments."""
    active: list[str] = []
    index = 0
    while index < len(source):
        character = source[index]
        if character != "%":
            active.append(character)
            index += 1
            continue

        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and source[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2:
            active.append(character)
            index += 1
            continue

        while index < len(source) and source[index] not in "\r\n":
            index += 1
        if index < len(source) and source[index] == "\r":
            index += 1
        if index < len(source) and source[index] == "\n":
            index += 1
    return "".join(active)


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


def _latex_escape_value(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _causal_evidence_snapshot(
    root: Path,
    cache: CausalEvidenceCache | None,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Load causal evidence once per audit so every consumer sees one snapshot."""

    root = root.resolve()
    if cache is None:
        evidence = load_causal_chain_evidence(root, allow_pending=True)
    else:
        if root not in cache:
            try:
                cache[root] = (load_causal_chain_evidence(root, allow_pending=True), None)
            except (OSError, TypeError, ValueError) as error:
                cache[root] = (None, error)
        evidence, error = cache[root]
        if error is not None:
            raise ValueError(f"Invalid causal-chain evidence: {error}") from error
        if evidence is None:  # pragma: no cover - defensive type narrowing
            raise ValueError("Invalid empty causal-chain evidence snapshot")
    if require_complete and evidence.get("complete") is not True:
        raise ValueError("Causal-chain evidence is pending")
    return evidence


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
    if not isinstance(amendments, list) or len(amendments) != 3:
        raise ValueError("Paper claim protocol differs from its prospective completion lock")
    documentation_amendment, inference_amendment, basis_amendment = amendments
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
        or documentation_amendment.get("scope")
        != "documentation_only_weight_spectrum_tier_correction"
        or documentation_amendment.get("previous_source_sha256")
        != "2d61c1c1a150269986dbc41786f5b10c7304b45d23148278959ef3d75b72c888"
        or documentation_amendment.get("updated_source_sha256")
        != "adf12c547e4c337a5acb94657b7f6c4207da550c9f2f46ea3ea5098f3e418ce4"
        or documentation_amendment.get("strict_beir_valid_units") != 196
        or documentation_amendment.get("strict_beir_expected_units") != 1_680
        or documentation_amendment.get("complete_retrieval_matrix_visible") is not False
        or any(
            documentation_amendment.get(field) is not False
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
        or inference_amendment.get("scope") != "prospective_confirmatory_inference_correction"
        or inference_amendment.get("previous_source_sha256")
        != "adf12c547e4c337a5acb94657b7f6c4207da550c9f2f46ea3ea5098f3e418ce4"
        or inference_amendment.get("updated_source_sha256")
        != "f77c22170144adcab3364f9e19167984727ba6edf8899dcf421158ef0588cdf0"
        or inference_amendment.get("strict_beir_valid_units") != 322
        or inference_amendment.get("strict_beir_expected_units") != 1_680
        or inference_amendment.get("complete_retrieval_matrix_visible") is not False
        or any(
            inference_amendment.get(field) is not False
            for field in (
                "formal_common_state_output_visible",
                "formal_representation_output_visible",
                "formal_functional_intervention_output_visible",
                "hybrid_adamw_output_visible",
                "short_branch_output_visible",
                "confirmatory_output_visible",
                "result_contingent_story_map_changed",
            )
        )
        or inference_amendment.get("headline_contract_changed") is not True
        or basis_amendment.get("scope") != "prospective_rope_basis_symmetry_correction"
        or basis_amendment.get("previous_source_sha256")
        != "f77c22170144adcab3364f9e19167984727ba6edf8899dcf421158ef0588cdf0"
        or basis_amendment.get("updated_source_sha256")
        != "3296f4882f1a68e96a0ee4a1608bc47155b776d5078dc40d6cfb654e096cc0c3"
        or basis_amendment.get("strict_beir_valid_units") != 340
        or basis_amendment.get("strict_beir_expected_units") != 1_680
        or basis_amendment.get("complete_retrieval_matrix_visible") is not False
        or basis_amendment.get("formal_basis_output_visible") is not False
        or any(
            basis_amendment.get(field) is not False
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
        "familywise interval for headline sign language"
        not in str(confirmation.get("selection_rule", ""))
        or "lower bound is above zero" not in str(confirmation.get("selection_rule", ""))
        or "otherwise inconclusive" not in str(confirmation.get("selection_rule", ""))
        or "both nominal and familywise intervals"
        not in str(confirmation.get("selection_rule", ""))
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


def _weight_constants(
    weight_dir: Path,
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, str]:
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
    selected = [row for row in rows if row.get("model_family") in families]
    if len(selected) != 20 * len(families):
        raise ValueError("Weight-space pair trajectory does not cover the active family scope")

    def value_range(field: str) -> str:
        values = [float(row[field]) for row in selected]
        return f"{min(values):.4f}--{max(values):.4f}"

    return {
        "NumWeightPairs": str(len(selected)),
        "DisplacementRatioRange": value_range("normuon_to_muon_displacement_ratio"),
        "RowCVRatioRange": value_range("normuon_to_muon_row_cv_ratio"),
        "TopRowEnergyRatioRange": value_range("normuon_to_muon_top_1pct_row_energy_ratio"),
    }


def _training_constants(
    training_dir: Path,
    families: tuple[str, ...] = FAMILIES,
) -> dict[str, str]:
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
        values = [
            float(row[field])
            for row in rows
            if row["model_family"] in families and row["optimizer"] in {"muon", "normuon"}
        ]
        if len(values) != 2 * len(families) or not all(
            math.isfinite(value) and value > 0 for value in values
        ):
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
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: str | Path | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    families, scope = resolve_scope(families, scope_amendment)
    matrix_path = resolve_matrix_path(matrix).resolve()
    root = Path(repo_root).resolve()
    configs = load_matrix(matrix_path)
    if len(configs) != 24 or {config.model_family for config in configs} != {"dense", "late"}:
        raise ValueError("Paper constants require the frozen 24-run, two-family discovery matrix")
    selected_configs = [config for config in configs if config.model_family in families]
    if len(selected_configs) != 12 * len(families):
        raise ValueError("Discovery matrix does not cover the active family scope")
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
        "NumDiscoveryRuns": str(len(selected_configs)),
        "NumDiscoveryCheckpoints": str(sum(len(c.checkpoint_fractions) for c in selected_configs)),
        "NumBEIRTasks": str(len(DECONTAMINATED_TASK_NAMES)),
        "NumDiscoveryUnits": str(
            len(selected_configs)
            * len(next(iter(selected_configs)).checkpoint_fractions)
            * len(DECONTAMINATED_TASK_NAMES)
        ),
        "NumTrainingQueries": "500{,}000",
        "NumHardNegatives": str(dataset["sampled_negatives"]),
        "ContextLength": "8{,}192",
        **_training_constants(Path(training_dir).resolve(), families),
        **_weight_constants(Path(weight_dir).resolve(), families),
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
    if scope is not None:
        sources["scope_amendment"] = scope
    return expected, sources


def _scope_matches(
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope_amendment: dict[str, Any] | None,
) -> bool:
    return families == FAMILIES or (
        payload.get("families") == list(families)
        and payload.get("scope_amendment") == scope_amendment
    )


def _strict_evidence_paths(
    families: tuple[str, ...],
) -> dict[str, tuple[Path, ...]]:
    evidence = dict(STRICT_EVIDENCE)
    if families == ("dense",):
        evidence["DiscoveryHeadline"] = SCOPED_DENSE_DISCOVERY_EVIDENCE
    return evidence


def _active_result_summary_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope_amendment: dict[str, Any] | None,
) -> bool:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("complete") is not True
        or not _scope_matches(payload, families, scope_amendment)
    ):
        return False
    root = path.parents[2]
    family_count = len(families)
    if path.parent.name == "hybrid-adamw":
        evaluations = payload.get("evaluations", {})
        return bool(
            evaluations.get("native_five_stage_units") == 280 * family_count
            and evaluations.get("native_final_units") == 56 * family_count
            and evaluations.get("hybrid_final_units") == 56 * family_count
            and evaluations.get("tasks") == 14
            and _hashed_file_complete(
                root,
                payload.get("outputs", {}).get("final_summary"),
                expected_path=path.parent / "final_summary.csv",
                expected_rows=4 * family_count,
            )
        )
    if path.parent.name == "short-branch":
        coverage = payload.get("coverage", {})
        return bool(
            coverage.get("runs") == 9 * family_count
            and coverage.get("checkpoints") == 45 * family_count
            and coverage.get("paired_checkpoint_contrasts") == 45 * family_count
            and coverage.get("paired_dynamics_summaries") == 60 * family_count
            and _hashed_file_complete(
                root,
                payload.get("outputs", {}).get("paired_summary"),
                expected_path=path.parent / "paired_dynamics_summary.csv",
                expected_rows=60 * family_count,
            )
        )
    if path.parent.name == "confirmatory":
        coverage = payload.get("coverage", {})
        inference = payload.get("inference", {})
        return bool(
            coverage
            == {
                "seeds": 3,
                "runs": 9 * family_count,
                "tasks": 14,
                "evaluation_units": 126 * family_count,
                "paired_contrast_units": 126 * family_count,
            }
            and inference.get("familywise_method") == "bonferroni"
            and inference.get("familywise_contrasts") == 6
            and _hashed_file_complete(
                root,
                payload.get("outputs", {}).get("paired_summary"),
                expected_path=path.parent / "paired_summary.csv",
                expected_rows=3 * family_count,
            )
        )
    return False


def _complete_manifest(
    path: Path,
    *,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: str | Path | None = None,
    causal_cache: CausalEvidenceCache | None = None,
    diagnostics: list[str] | None = None,
) -> bool:
    try:
        families, scope = resolve_scope(families, scope_amendment)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    try:
        payload = _json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if path.name == "coverage.json":
        if path.parent.name == "dense-discovery":
            return _dense_discovery_coverage_complete(path, payload, families, scope)
        return (
            payload.get("complete") is True
            and payload.get("observed_results") == 1680
            and payload.get("expected_results") == 1680
            and payload.get("observed_checkpoint_summaries") == 120
            and payload.get("expected_checkpoint_summaries") == 120
            and payload.get("missing") == []
            and payload.get("unexpected") == []
        )
    if path.name == "summary_manifest.json" and path.parent.name in {
        "retrieval-dynamics",
        "retrieval-dynamics-dense",
    }:
        return _retrieval_dynamics_complete(path, payload, families, scope)
    if path.name == "summary_manifest.json" and path.parent.name == "dense-retrieval-dynamics":
        if families != ("dense",) or scope is None:
            return False
        try:
            from .dense_retrieval_dynamics_summary import (
                audit_dense_retrieval_dynamics_summary,
            )

            root = path.parents[2]
            receipt = audit_dense_retrieval_dynamics_summary(
                root / "configs/dense_retrieval_dynamics_extension.json",
                path.parent,
            )
        except (OSError, TypeError, ValueError):
            # A source checkout contains the original model checkpoints and must pass the full
            # reconstruction above.  A clean distribution intentionally omits hundreds of GB of
            # checkpoints, so it instead verifies every content-addressed evaluation input in the
            # checked-in portable closure.
            if (root / "outputs").exists():
                return False
            return _portable_dense_retrieval_dynamics_complete(path, payload, families, scope)
        return bool(
            receipt.get("complete") is True
            and receipt.get("read_only") is True
            and receipt.get("coverage")
            == {
                "dynamics_units": 728,
                "formal_hybrid_stage5_units": 56,
                "formal_confirmatory_stage5_units": 126,
                "task_units": 910,
                "trajectory_rows": 65,
            }
            and receipt.get("formal_inference_reads_joined_outputs") is False
            and _hashed_file_complete(root, receipt.get("manifest"), expected_path=path)
        )
    if path.name == "summary_manifest.json" and path.parent.name in {
        "hybrid-adamw",
        "short-branch",
        "confirmatory",
    }:
        return _active_result_summary_complete(path, payload, families, scope)
    if path.name == "outcome-summary.manifest.json":
        return _outcome_report_complete(
            path,
            payload,
            families,
            scope,
            scope_amendment,
            causal_cache=causal_cache,
            diagnostics=diagnostics,
        )
    if path.name == "paper-results.manifest.json":
        return _paper_results_complete(path, payload, families, scope, causal_cache=causal_cache)
    if path.name == "mechanism-summary.manifest.json":
        return _mechanism_report_complete(path, payload, families, scope, causal_cache=causal_cache)
    if path.name == "summary_manifest.json" and path.parent.name == "tail-stability":
        return _tail_stability_complete(path, payload, families, scope)
    if path.name == "summary_manifest.json" and path.parent.name == "spectral-transplant":
        return _spectral_transplant_complete(path, payload, families, scope)
    return payload.get("schema_version") == SCHEMA_VERSION and payload.get("complete") is True


@lru_cache(maxsize=None)
def _repository_root(root: Path) -> Path:
    """Find the active checkout even when an audit helper receives a report directory."""

    resolved = root.resolve()
    start = resolved.parent if resolved.is_file() else resolved
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def _declared_path(root: Path, record: Any) -> Path | None:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return None
    declared = Path(record["path"])
    if not declared.is_absolute():
        return (root / declared).resolve()

    # Historical receipts intentionally preserve the producer checkout.  Rebase only paths whose
    # ancestry explicitly names this project; all other absolute paths keep their original meaning.
    # The caller still checks byte size and SHA-256, so relocation never substitutes by filename.
    parts = declared.parts
    project_indexes = [index for index, part in enumerate(parts) if part == LEGACY_CHECKOUT_NAME]
    if not project_indexes:
        return declared.resolve()
    relative = Path(*parts[project_indexes[-1] + 1 :])
    if not relative.parts:
        return None
    return (_repository_root(root) / relative).resolve()


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


def _marked_section_complete(
    path: Path,
    record: Any,
    markers: tuple[str, str],
    *,
    repository_root: Path,
) -> bool:
    if not isinstance(record, dict) or record.get("markers") != list(markers):
        return False
    declared = Path(str(record.get("path", "")))
    declared_path = (
        declared.resolve() if declared.is_absolute() else (repository_root / declared).resolve()
    )
    if declared_path != path.resolve():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    begin, end = markers
    if text.count(begin) != 1 or text.count(end) != 1:
        return False
    start = text.index(begin)
    stop = text.index(end, start + len(begin)) + len(end)
    block = text[start:stop].encode("utf-8")
    return bool(
        record.get("block_bytes") == len(block)
        and record.get("block_sha256") == hashlib.sha256(block).hexdigest()
    )


def _hash_only_file_complete(
    root: Path,
    record: Any,
    *,
    expected_path: Path | None = None,
) -> bool:
    path = _declared_path(root, record)
    return bool(
        isinstance(record, dict)
        and path is not None
        and (expected_path is None or path == expected_path.resolve())
        and path.is_file()
        and record.get("sha256") == _sha256(path)
    )


def _hashed_csv_complete(
    root: Path,
    record: Any,
    *,
    expected_path: Path,
    expected_rows: int,
) -> bool:
    if not _hashed_file_complete(
        root,
        record,
        expected_path=expected_path,
        expected_rows=expected_rows,
    ):
        return False
    try:
        path = _declared_path(root, record)
        if path is None:
            return False
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    return len(rows) == expected_rows


def _hashed_reference_tree_complete(value: Any, *, root: Path) -> bool:
    """Rehash every path/sha256 source identity nested below ``value``."""

    if isinstance(value, list):
        return all(_hashed_reference_tree_complete(item, root=root) for item in value)
    if not isinstance(value, dict):
        return True
    if "path" in value or "sha256" in value:
        if not _hashed_file_complete(root, value):
            return False
    return all(
        _hashed_reference_tree_complete(item, root=root)
        for name, item in value.items()
        if name not in {"path", "bytes", "sha256", "rows"}
    )


def _declared_reference_paths(value: Any, *, root: Path) -> list[Path]:
    if isinstance(value, list):
        return [path for item in value for path in _declared_reference_paths(item, root=root)]
    if not isinstance(value, dict):
        return []
    paths = []
    if "path" in value and "sha256" in value:
        path = _declared_path(root, value)
        if path is not None:
            paths.append(path)
    return paths + [
        path
        for name, item in value.items()
        if name not in {"path", "bytes", "sha256", "rows"}
        for path in _declared_reference_paths(item, root=root)
    ]


def _portable_evidence_records(root: Path) -> dict[Path, dict[str, Any]] | None:
    try:
        payload = _json(_repository_root(root) / PORTABLE_EVIDENCE_MANIFEST)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    repository = _repository_root(root)
    expected_bindings = {
        "generator": repository / "scripts/portable_evidence.py",
        "audit_implementation": repository / "src/embed_optim/paper_audit.py",
    }
    source_manifests = payload.get("source_manifests")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "portable-paper-evidence-closure"
        or payload.get("complete") is not True
        or any(
            not _hashed_file_complete(repository, payload.get(name), expected_path=path)
            for name, path in expected_bindings.items()
        )
        or not isinstance(source_manifests, list)
        or len(source_manifests) != len(PORTABLE_EVIDENCE_SOURCE_MANIFESTS)
        or any(
            not _hashed_file_complete(
                repository,
                record,
                expected_path=repository / relative,
            )
            for record, relative in zip(
                source_manifests, PORTABLE_EVIDENCE_SOURCE_MANIFESTS, strict=True
            )
        )
        or not isinstance(payload.get("files"), list)
    ):
        return None
    records: dict[Path, dict[str, Any]] = {}
    for record in payload["files"]:
        path = _declared_path(repository, record)
        if (
            path is None
            or path in records
            or not _hashed_file_complete(repository, record, expected_path=path)
        ):
            return None
        records[path] = record
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary != {
        "files": len(records),
        "bytes": sum(path.stat().st_size for path in records),
    }:
        return None
    return records


def _portable_evidence_covers(root: Path, paths: list[Path]) -> bool:
    records = _portable_evidence_records(root)
    return bool(records is not None and set(paths).issubset(records))


def _portable_reference_tree_complete(value: Any, *, root: Path) -> bool:
    """Rehash a tree whose legacy records may omit byte counts but always bind SHA-256."""

    if isinstance(value, list):
        return all(_portable_reference_tree_complete(item, root=root) for item in value)
    if not isinstance(value, dict):
        return True
    if "path" in value or "sha256" in value:
        validator = _hashed_file_complete if "bytes" in value else _hash_only_file_complete
        if not validator(root, value):
            return False
    return all(
        _portable_reference_tree_complete(item, root=root)
        for name, item in value.items()
        if name not in {"path", "bytes", "sha256", "rows"}
    )


def _portable_dense_retrieval_dynamics_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope: dict[str, Any] | None,
) -> bool:
    """Verify the complete 910-result dynamics closure without local model checkpoints."""

    repository = path.parents[2]
    coverage = {
        "runs": 13,
        "stages_per_run": 5,
        "trajectory_rows": 65,
        "tasks_per_stage": 14,
        "task_units": 910,
        "dynamics_units": 728,
        "formal_stage5_units": 182,
    }
    outputs = payload.get("outputs")
    expected_outputs = {
        "trajectory_csv": ("five_stage_retrieval_dynamics.csv", 65),
        "figure_svg": ("five_stage_retrieval_dynamics.svg", None),
        "figure_pdf": ("five_stage_retrieval_dynamics.pdf", None),
    }
    sources = payload.get("sources")
    if (
        families != ("dense",)
        or scope is None
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("complete") is not True
        or payload.get("families") != ["dense"]
        or payload.get("coverage") != coverage
        or payload.get("inference_boundary", {}).get("formal_inference_reads_joined_outputs")
        is not False
        or not isinstance(outputs, dict)
        or set(outputs) != set(expected_outputs)
        or not isinstance(sources, dict)
    ):
        return False
    for name, (filename, rows) in expected_outputs.items():
        record = outputs[name]
        if not _hashed_file_complete(
            repository,
            record,
            expected_path=path.parent / filename,
            expected_rows=rows,
        ):
            return False
        if rows is None and "rows" in record:
            return False
    expected_source_files = {
        "implementation": repository / "src/embed_optim/dense_retrieval_dynamics_summary.py",
        "contract": repository / "configs/dense_retrieval_dynamics_extension.json",
        "confirmatory_protocol": repository / "configs/confirmatory_protocol.json",
    }
    if any(
        not _hashed_file_complete(repository, sources.get(name), expected_path=source_path)
        for name, source_path in expected_source_files.items()
    ):
        return False
    dynamics_audit = sources.get("dynamics_audit")
    formal_audit = sources.get("formal_confirmatory_audit")
    if dynamics_audit != {
        "complete": True,
        "expected_units": 728,
        "valid_units": 728,
        "contract_sha256": sources["contract"]["sha256"],
    } or formal_audit != {
        "complete": True,
        "expected_units": 126,
        "valid_units": 126,
        "protocol_sha256": sources["confirmatory_protocol"]["sha256"],
        "matrix_manifest_sha256": sources.get("confirmatory_matrix_manifest_sha256"),
    }:
        return False
    partitions = sources.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != 8:
        return False
    expected_partitions = {
        ("hybrid", None, "dynamics-stage1-4"): 224,
        ("hybrid", None, "formal-stage5"): 56,
        **{("confirmatory", seed, "dynamics-stage1-4"): 168 for seed in (314159, 271828, 161803)},
    }
    expected_partitions.update(
        {("confirmatory", seed, "formal-stage5"): 42 for seed in (314159, 271828, 161803)}
    )
    observed_partitions: dict[tuple[str, int | None, str], int] = {}
    for partition in partitions:
        if not isinstance(partition, dict):
            return False
        identity = (partition.get("suite"), partition.get("seed"), partition.get("partition"))
        if identity in observed_partitions or partition.get("valid_units") != partition.get(
            "expected_units"
        ):
            return False
        result_sources = partition.get("result_sources")
        if not isinstance(result_sources, list) or len(result_sources) != partition.get(
            "expected_units"
        ):
            return False
        observed_partitions[identity] = len(result_sources)
    source_paths = _declared_reference_paths(partitions, root=repository)
    return bool(
        observed_partitions == expected_partitions
        and len(source_paths) == 926
        and len(set(source_paths)) == len(source_paths)
        and _portable_reference_tree_complete(partitions, root=repository)
        and _portable_evidence_covers(repository, source_paths)
    )


def _csv_identity_set(
    path: Path,
    fields: tuple[str, ...],
) -> set[tuple[str, ...]] | None:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(fields).issubset(reader.fieldnames or ()):
                return None
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    identities = {tuple(row[field] for field in fields) for row in rows}
    return identities if len(identities) == len(rows) else None


def _dense_discovery_coverage_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope: dict[str, Any] | None,
) -> bool:
    if families != ("dense",) or scope is None:
        return False
    root = path.parents[2]
    required = {
        "complete": True,
        "contract_complete": True,
        "dataset_complete": True,
        "training_complete": True,
        "deep_training_artifact_validation": True,
        "evaluation_complete": True,
        "verified_experiment_runs": 12,
        "expected_experiment_runs": 12,
        "verified_training_examples": 500_000,
        "expected_training_examples": 500_000,
        "verified_training_runs": 12,
        "expected_training_runs": 12,
        "verified_training_checkpoints": 60,
        "expected_training_checkpoints": 60,
        "observed_results": 840,
        "expected_results": 840,
        "observed_checkpoint_summaries": 60,
        "expected_checkpoint_summaries": 60,
        "selected_experiment_runs": 12,
        "selected_training_checkpoints": 60,
        "missing": [],
        "unexpected": [],
        "contract_errors": [],
        "dataset_errors": [],
        "training_errors": [],
    }
    source = payload.get("source_full_discovery")
    reports = source.get("reports") if isinstance(source, dict) else None
    expected_reports = {
        "coverage": root / "reports/coverage.json",
        "evaluation_long": root / "reports/evaluation_long.csv",
        "system_metrics": root / "reports/system_metrics.csv",
        "training_history": root / "reports/training_history.csv",
    }
    expected_outputs = {
        "evaluation_long": ("evaluation_long.csv", 840),
        "checkpoint_summary": ("checkpoint_summary.csv", 60),
        "optimizer_summary": ("optimizer_summary.csv", 3),
        "best_config_dynamics": ("best_config_dynamics.csv", 15),
        "best_config_task_comparison": ("best_config_task_comparison.csv", 14),
        "paired_comparison": ("paired_comparison.csv", 2),
        "training_history": ("training_history.csv", 4_692),
        "system_metrics": ("system_metrics.csv", 12),
        "system_summary": ("system_summary.csv", 3),
        "dense_training_dynamics": ("figures/dense-training-dynamics.png", None),
        "dense_training_dynamics_by_run": (
            "figures/dense-training-dynamics-by-run.png",
            None,
        ),
        "dense_lr_sensitivity": ("figures/dense-lr-sensitivity.png", None),
    }
    outputs = payload.get("outputs")
    outputs_complete = isinstance(outputs, dict) and set(outputs) == set(expected_outputs)
    if outputs_complete:
        for name, (filename, rows) in expected_outputs.items():
            expected_path = path.parent / filename
            if rows is None:
                valid = (
                    _hashed_file_complete(root, outputs[name], expected_path=expected_path)
                    and "rows" not in outputs[name]
                )
            else:
                valid = _hashed_csv_complete(
                    root,
                    outputs[name],
                    expected_path=expected_path,
                    expected_rows=rows,
                )
            if not valid:
                outputs_complete = False
                break
    return bool(
        all(payload.get(name) == value for name, value in required.items())
        and payload.get("families") == ["dense"]
        and payload.get("scope_amendment") == scope
        and isinstance(payload.get("training_row_manifest_sha256"), str)
        and len(payload["training_row_manifest_sha256"]) == 64
        and isinstance(payload.get("training_dataset_view_fingerprint"), str)
        and source is not None
        and source.get("complete") is True
        and source.get("observed_results") == 1_680
        and source.get("expected_results") == 1_680
        and source.get("observed_checkpoint_summaries") == 120
        and source.get("expected_checkpoint_summaries") == 120
        and source.get("verified_experiment_runs") == 24
        and source.get("expected_experiment_runs") == 24
        and source.get("verified_training_runs") == 24
        and source.get("expected_training_runs") == 24
        and source.get("verified_training_checkpoints") == 120
        and source.get("expected_training_checkpoints") == 120
        and isinstance(reports, dict)
        and set(reports) == set(expected_reports)
        and all(
            _hashed_file_complete(root, reports[name], expected_path=expected)
            for name, expected in expected_reports.items()
        )
        and _complete_manifest(expected_reports["coverage"])
        and outputs_complete
    )


def _dense_retrieval_dynamics_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope: dict[str, Any] | None,
) -> bool:
    if families != ("dense",) or scope is None:
        return False
    root = path.parents[2]
    coverage = payload.get("coverage")
    expected_coverage = {
        "runs": 12,
        "checkpoints": 60,
        "tasks": 14,
        "evaluation_units": 840,
        "optimizer_family_groups": 3,
        "best_config_task_delta_rows": 140,
        "adjacent_stage_task_stability_rows": 8,
    }
    expected_outputs = {
        "checkpoint_dynamics": ("checkpoint_dynamics.csv", 60),
        "run_first_passage": ("run_first_passage.csv", 12),
        "optimizer_first_passage": ("optimizer_first_passage.csv", 3),
        "best_config_task_comparison": ("best_config_task_comparison.csv", 14),
        "best_config_task_delta_dynamics": (
            "best_config_task_delta_dynamics.csv",
            140,
        ),
        "task_delta_stability": ("task_delta_stability.csv", 8),
        "quality_vs_useful_wall_time": ("quality_vs_useful_wall_time.svg", None),
    }
    outputs = payload.get("outputs")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("complete") is not True
        or payload.get("families") != ["dense"]
        or payload.get("scope_amendment") != scope
        or coverage != expected_coverage
        or not isinstance(outputs, dict)
        or set(outputs) != set(expected_outputs)
    ):
        return False
    for name, (filename, rows) in expected_outputs.items():
        expected_path = path.parent / filename
        if rows is None:
            if (
                not _hashed_file_complete(
                    root,
                    outputs[name],
                    expected_path=expected_path,
                )
                or "rows" in outputs[name]
            ):
                return False
        elif not _hashed_csv_complete(
            root,
            outputs[name],
            expected_path=expected_path,
            expected_rows=rows,
        ):
            return False

    source_full = payload.get("source_full_discovery")
    if not isinstance(source_full, dict) or source_full != {
        "complete": True,
        "runs": 24,
        "checkpoints": 120,
        "tasks": 14,
        "evaluation_units": 1_680,
        "optimizer_family_groups": 6,
        "best_config_task_delta_rows": 280,
        "adjacent_stage_task_stability_rows": 16,
    }:
        return False
    sources = payload.get("sources")
    expected_sources = {
        "full_retrieval_dynamics": root / "reports/retrieval-dynamics/summary_manifest.json",
        "strict_coverage": root / "reports/coverage.json",
    }
    if (
        not isinstance(sources, dict)
        or set(sources) != set(expected_sources)
        or any(
            not _hashed_file_complete(root, sources[name], expected_path=expected)
            for name, expected in expected_sources.items()
        )
        or sources["strict_coverage"].get("observed_results") != 1_680
        or not _complete_manifest(expected_sources["full_retrieval_dynamics"])
        or not _complete_manifest(expected_sources["strict_coverage"])
    ):
        return False
    return True


def _retrieval_dynamics_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope: dict[str, Any] | None,
) -> bool:
    if path.parent.name == "retrieval-dynamics-dense":
        return _dense_retrieval_dynamics_complete(path, payload, families, scope)
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
            "best_config_task_delta_rows": 280,
            "adjacent_stage_task_stability_rows": 16,
        }
        or not isinstance(sources, dict)
        or not isinstance(outputs, dict)
    ):
        return False

    expected_outputs = {
        "checkpoint_dynamics": ("checkpoint_dynamics.csv", 120),
        "run_first_passage": ("run_first_passage.csv", 24),
        "optimizer_first_passage": ("optimizer_first_passage.csv", 6),
        "best_config_task_comparison": ("best_config_task_comparison.csv", 28),
        "best_config_task_delta_dynamics": (
            "best_config_task_delta_dynamics.csv",
            280,
        ),
        "task_delta_stability": ("task_delta_stability.csv", 16),
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


def _tail_stability_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope: dict[str, Any] | None,
) -> bool:
    report_dir = path.parent
    repository_root = report_dir.parents[1]
    family_count = len(families)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("complete") is not True
        or payload.get("discovery_complete") is not True
        or payload.get("short_branch_confirmation_complete") is not True
        or payload.get("analysis_status")
        != "post_hoc_discovery_with_prospective_short_branch_confirmation"
        or payload.get("families") != list(families)
        or (families != FAMILIES and payload.get("scope_amendment") != scope)
        or payload.get("pending_reason") is not None
        or payload.get("discovery_anchors") != 10 * family_count
        or payload.get("discovery_anchor_operator_rows") != 30 * family_count
        or payload.get("discovery_contrasts") != 2 * family_count
        or payload.get("discovery_cross_tail_rows") != 20 * family_count
        or payload.get("discovery_cross_tail_summaries") != 2 * family_count
        or payload.get("short_branch_checkpoint_rows") != 45 * family_count
        or payload.get("short_branch_contrast_rows") != 30 * family_count
        or payload.get("short_branch_final_rows") != 2 * family_count
        or not isinstance(payload.get("claim_boundary"), str)
        or not payload["claim_boundary"]
    ):
        return False
    expected_outputs = {
        "discovery_anchor_tail": ("discovery_anchor_tail.csv", 30 * family_count),
        "discovery_family_contrasts": (
            "discovery_family_contrasts.csv",
            2 * family_count,
        ),
        "discovery_cross_tail": ("discovery_cross_tail.csv", 20 * family_count),
        "discovery_cross_tail_summary": (
            "discovery_cross_tail_summary.csv",
            2 * family_count,
        ),
        "short_branch_checkpoint_tail": (
            "short_branch_checkpoint_tail.csv",
            45 * family_count,
        ),
        "short_branch_checkpoint_contrasts": (
            "short_branch_checkpoint_contrasts.csv",
            30 * family_count,
        ),
        "short_branch_final_summary": (
            "short_branch_final_summary.csv",
            2 * family_count,
        ),
        "readme": ("README.md", None),
    }
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(expected_outputs):
        return False
    for name, (filename, rows) in expected_outputs.items():
        expected_path = report_dir / filename
        if rows is None:
            if (
                not _hashed_file_complete(
                    report_dir,
                    outputs[name],
                    expected_path=expected_path,
                )
                or "rows" in outputs[name]
            ):
                return False
        elif not _hashed_csv_complete(
            report_dir,
            outputs[name],
            expected_path=expected_path,
            expected_rows=rows,
        ):
            return False

    discovery_sources = payload.get("discovery_sources")
    short_sources = payload.get("short_branch_sources")
    protocol = payload.get("protocol")
    validation_sources = (
        [item for item in short_sources[1:] if "validation_manifest" in item]
        if isinstance(short_sources, list)
        else []
    )
    unseen_sources = (
        [item for item in short_sources[1:] if "unseen_metrics" in item]
        if isinstance(short_sources, list)
        else []
    )
    if (
        not isinstance(discovery_sources, list)
        or len(discovery_sources) != 10 * family_count
        or any(
            not isinstance(item, dict) or set(item) != {"label", "manifest", "sample_metrics"}
            for item in discovery_sources
        )
        or not isinstance(short_sources, list)
        or len(short_sources) != 1 + 90 * family_count
        or not isinstance(short_sources[0], dict)
        or set(short_sources[0]) != {"short_branch_summary"}
        or len(validation_sources) != 45 * family_count
        or any(
            not isinstance(item, dict)
            or set(item) != {"label", "validation_manifest", "validation_samples"}
            for item in validation_sources
        )
        or len(unseen_sources) != 45 * family_count
        or any(
            not isinstance(item, dict) or set(item) != {"label", "unseen_metrics"}
            for item in unseen_sources
        )
        or not isinstance(protocol, dict)
        or not _hashed_file_complete(repository_root, protocol)
        or not _hashed_reference_tree_complete(discovery_sources, root=repository_root)
        or not _hashed_reference_tree_complete(short_sources, root=repository_root)
    ):
        return False
    source_paths = _declared_reference_paths(
        [*discovery_sources, *short_sources], root=repository_root
    )
    if len(source_paths) != 20 * family_count + 1 + 135 * family_count or len(
        set(source_paths)
    ) != len(source_paths):
        return False

    expected_identities = {
        (family, challenger, "adamw") for family in families for challenger in ("muon", "normuon")
    }
    discovery_identities = _csv_identity_set(
        report_dir / "discovery_cross_tail_summary.csv",
        ("family", "challenger", "reference"),
    )
    final_identities = _csv_identity_set(
        report_dir / "short_branch_final_summary.csv",
        ("family", "challenger", "reference"),
    )
    return discovery_identities == expected_identities and final_identities == expected_identities


def _spectral_transplant_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope: dict[str, Any] | None,
) -> bool:
    report_dir = path.parent
    family_count = len(families)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("complete") is not True
        or payload.get("analysis_status") != "post_hoc_explanatory_intervention"
        or payload.get("families") != list(families)
        or (families != FAMILIES and payload.get("scope_amendment") != scope)
        or payload.get("anchors") != 10 * family_count
        or payload.get("anchor_effect_records") != 100 * family_count
        or payload.get("anchor_tail_effect_records") != 90 * family_count
        or not isinstance(payload.get("claim_boundary"), str)
        or not payload["claim_boundary"]
    ):
        return False
    tail_protocol = payload.get("tail_protocol")
    if (
        not isinstance(tail_protocol, dict)
        or tail_protocol.get("status") != "frozen-before-spectral-transplant-output"
        or tail_protocol.get("tail_fraction") != 0.05
        or tail_protocol.get("tail_count") != 12
    ):
        return False
    expected_outputs = {
        "anchor_condition_effects": ("anchor_condition_effects.csv", 100 * family_count),
        "family_condition_summary": ("family_condition_summary.csv", 60 * family_count),
        "anchor_factorial_effects": ("anchor_factorial_effects.csv", 60 * family_count),
        "family_factorial_summary": ("family_factorial_summary.csv", 6 * family_count),
        "anchor_spectral_path": ("anchor_spectral_path.csv", 300 * family_count),
        "family_spectral_path": ("family_spectral_path.csv", 30 * family_count),
        "anchor_band_effects": ("anchor_band_effects.csv", 180 * family_count),
        "family_band_summary": ("family_band_summary.csv", 18 * family_count),
        "anchor_query_tail_effects": (
            "anchor_query_tail_effects.csv",
            90 * family_count,
        ),
        "family_query_tail_summary": (
            "family_query_tail_summary.csv",
            9 * family_count,
        ),
    }
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(expected_outputs):
        return False
    if any(
        not _hashed_csv_complete(
            report_dir,
            outputs[name],
            expected_path=report_dir / filename,
            expected_rows=rows,
        )
        for name, (filename, rows) in expected_outputs.items()
    ):
        return False
    sources = payload.get("sources")
    spectral_spec = payload.get("spectral_transplant_spec")
    common_state_spec = payload.get("common_state_spec")
    if (
        not isinstance(sources, list)
        or len(sources) != 10 * family_count
        or any(
            not isinstance(item, dict) or set(item) != {"label", "manifest", "sample_metrics"}
            for item in sources
        )
        or not isinstance(spectral_spec, dict)
        or not _hashed_file_complete(report_dir, spectral_spec)
        or not isinstance(common_state_spec, dict)
        or not _hashed_file_complete(report_dir, common_state_spec)
        or not _hashed_reference_tree_complete(sources, root=report_dir)
    ):
        return False
    source_paths = _declared_reference_paths(sources, root=report_dir)
    if len(source_paths) != 20 * family_count or len(set(source_paths)) != len(source_paths):
        return False

    factorial = _csv_identity_set(report_dir / "family_factorial_summary.csv", ("family", "metric"))
    tail = _csv_identity_set(report_dir / "family_query_tail_summary.csv", ("family", "condition"))
    expected_metrics = {
        "contrastive_loss",
        "positive_score",
        "hardest_negative_score",
        "positive_margin",
        "reciprocal_rank",
        "top1_accuracy",
    }
    expected_conditions = {
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
    return factorial == {
        (family, metric) for family in families for metric in expected_metrics
    } and tail == {(family, condition) for family in families for condition in expected_conditions}


def _causal_chain_source_complete(
    root: Path,
    record: Any,
    expected_path: Path,
    label: str,
    *,
    causal_cache: CausalEvidenceCache | None = None,
) -> bool:
    """Verify a rendered causal branch against independently recomputed evidence."""
    if label not in {"temporal_short_branch", "dose_band"}:
        return False
    try:
        evidence = _causal_evidence_snapshot(root, causal_cache, require_complete=True)
    except (OSError, TypeError, ValueError):
        return False
    branch = evidence[label]
    identity = branch.get("manifest")
    if (
        not isinstance(record, dict)
        or set(record)
        != {
            "path",
            "bytes",
            "sha256",
            "status",
            "claimable",
            "supported",
            "claim_boundary",
        }
        or not isinstance(identity, dict)
        or Path(str(identity.get("path", ""))).resolve() != expected_path.resolve()
        or not _hashed_file_complete(root, record, expected_path=expected_path)
        or record.get("bytes") != identity.get("bytes")
        or record.get("sha256") != identity.get("sha256")
    ):
        return False
    return bool(
        evidence["complete"] is True
        and branch["complete"] is True
        and record.get("status") == branch["status"]
        and record.get("claimable") == branch["claimable"]
        and record.get("supported") == branch["supported"]
        and record.get("claim_boundary") == branch["claim_boundary"]
    )


def _mechanism_report_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope: dict[str, Any] | None,
    *,
    causal_cache: CausalEvidenceCache | None = None,
) -> bool:
    root = path.parents[1]
    try:
        causal_evidence = _causal_evidence_snapshot(root, causal_cache, require_complete=True)
        causal_display = causal_chain_display_contract(causal_evidence)
        causal_markdown = render_causal_chain_markdown(
            causal_evidence, detailed=True, heading_level=3
        )
    except (OSError, TypeError, ValueError):
        return False
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("complete") is not True
        or not _scope_matches(payload, families, scope)
        or payload.get("causal_chain") != causal_display
    ):
        return False
    expected_sources = {
        "common_state": root / "reports/common-state/summary_manifest.json",
        "retrieval_dynamics": root / "reports/retrieval-dynamics/summary_manifest.json",
        "exact_spectra": root / "results/common-state-spectra/summary/summary_manifest.json",
        "basis_sensitivity": root / "reports/basis-sensitivity/summary_manifest.json",
        "mechanism_bridge": root / "reports/mechanism-bridge/summary_manifest.json",
        "temporal_short_branch": root / "reports/temporal-short-branch/summary_manifest.json",
        "dose_band": root / "reports/dose-band/summary_manifest.json",
    }
    sources = payload.get("sources")
    if (
        not isinstance(sources, dict)
        or set(sources) != set(expected_sources)
        or any(
            not _hash_only_file_complete(root, sources[name], expected_path=source_path)
            for name, source_path in expected_sources.items()
        )
    ):
        return False
    if any(
        not _causal_chain_source_complete(
            root,
            sources[label],
            expected_sources[label],
            label,
            causal_cache=causal_cache,
        )
        for label in ("temporal_short_branch", "dose_band")
    ):
        return False
    family_count = len(families)
    if (
        sources["common_state"].get("anchors") != 10 * family_count
        or sources["retrieval_dynamics"].get("evaluation_units") != 840 * family_count
        or sources["exact_spectra"].get("spectra") != 180 * family_count
        or sources["basis_sensitivity"].get("records") != 270 * family_count
        or sources["basis_sensitivity"].get("head_records") != 1_620 * family_count
        or sources["mechanism_bridge"].get("checkpoints") != 60 * family_count
    ):
        return False
    if families != FAMILIES and (
        sources["basis_sensitivity"].get("summary_rows") != 3 * family_count
        or sources["mechanism_bridge"].get("within_run_transitions") != 48 * family_count
        or sources["mechanism_bridge"].get("correlations") != 96 * family_count
    ):
        return False

    expected_tables = (
        root / "reports/retrieval-dynamics/optimizer_first_passage.csv",
        root / "reports/common-state/anchor_contrasts.csv",
        root / "reports/basis-sensitivity/summary.csv",
        root / "results/common-state-spectra/summary/spectrum_metrics.csv",
        root / "reports/mechanism-bridge/checkpoint_bridge.csv",
        root / "reports/mechanism-bridge/descriptive_correlations.csv",
        root / "reports/mechanism-bridge/within_run_changes.csv",
        *(Path(record["path"]) for record in causal_evidence["source_table_records"]),
    )
    tables = payload.get("source_tables")
    if (
        not isinstance(tables, list)
        or len(tables) != len(expected_tables)
        or any(
            not _hash_only_file_complete(root, record, expected_path=expected)
            for record, expected in zip(tables, expected_tables, strict=True)
        )
    ):
        return False
    figures = payload.get("figures")
    expected_figure = root / "reports/retrieval-dynamics/quality_vs_useful_wall_time.svg"
    if (
        not isinstance(figures, dict)
        or "retrieval_dynamics" not in figures
        or not _hash_only_file_complete(
            root,
            figures["retrieval_dynamics"],
            expected_path=expected_figure,
        )
    ):
        return False

    report_path = root / "reports/mechanism-summary.md"
    if not _hashed_file_complete(root, payload.get("output"), expected_path=report_path):
        return False
    try:
        report = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return report.count(causal_markdown) == 1


def _outcome_report_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope: dict[str, Any] | None,
    scope_amendment: str | Path | None,
    *,
    causal_cache: CausalEvidenceCache | None = None,
    diagnostics: list[str] | None = None,
) -> bool:
    def fail(reason: str) -> bool:
        if diagnostics is not None:
            diagnostics.append(reason)
        return False

    root = path.parents[1]
    try:
        causal_evidence = _causal_evidence_snapshot(root, causal_cache, require_complete=True)
        causal_display = causal_chain_display_contract(causal_evidence)
        causal_markdown = render_causal_chain_markdown(
            causal_evidence, detailed=False, heading_level=3
        )
    except (OSError, TypeError, ValueError) as error:
        return fail(f"causal_evidence:{type(error).__name__}:{error}")
    sources = payload.get("sources", {})
    expected_sources = {
        "mechanism_report": root / "reports/mechanism-summary.manifest.json",
        "functional_intervention": root / "reports/functional-intervention/manifest.json",
        "hybrid_adamw": root / "reports/hybrid-adamw/summary_manifest.json",
        "short_branch": root / "reports/short-branch/summary_manifest.json",
        "tail_stability": root / "reports/tail-stability/summary_manifest.json",
        "spectral_transplant": root / "reports/spectral-transplant/summary_manifest.json",
        "confirmation": root / "reports/confirmatory/summary_manifest.json",
        "temporal_short_branch": root / "reports/temporal-short-branch/summary_manifest.json",
        "dose_band": root / "reports/dose-band/summary_manifest.json",
    }
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("complete") is not True
        or not _scope_matches(payload, families, scope)
        or payload.get("causal_chain") != causal_display
        or not isinstance(sources, dict)
        or set(sources) != set(expected_sources)
        or any(
            not _hashed_file_complete(root, sources[name], expected_path=source_path)
            or not _complete_manifest(
                source_path,
                families=families,
                scope_amendment=scope_amendment,
                causal_cache=causal_cache,
            )
            for name, source_path in expected_sources.items()
        )
    ):
        return fail("metadata_sources_or_recursive_source_contract")
    if any(
        not _causal_chain_source_complete(
            root,
            sources[label],
            expected_sources[label],
            label,
            causal_cache=causal_cache,
        )
        for label in ("temporal_short_branch", "dose_band")
    ):
        return fail("causal_chain_source_contract")

    family_count = len(families)
    source_coverage = {
        "functional_intervention": {"anchors": 20},
        "hybrid_adamw": {"hybrid_units": 56 * family_count},
        "short_branch": {"runs": 9 * family_count},
        "tail_stability": {
            "anchors": 10 * family_count,
            "final_contrasts": 2 * family_count,
        },
        "spectral_transplant": {
            "anchors": 10 * family_count,
            "anchor_effect_records": 100 * family_count,
        },
        "confirmation": {"units": 126 * family_count},
    }
    if any(
        any(sources[name].get(field) != value for field, value in fields.items())
        for name, fields in source_coverage.items()
    ):
        return fail("source_coverage_contract")

    expected_tables = (
        root / "reports/functional-intervention/family_summary.csv",
        root / "reports/hybrid-adamw/final_summary.csv",
        root / "reports/short-branch/paired_dynamics_summary.csv",
        root / "reports/tail-stability/discovery_cross_tail_summary.csv",
        root / "reports/tail-stability/short_branch_final_summary.csv",
        root / "reports/spectral-transplant/family_factorial_summary.csv",
        root / "reports/spectral-transplant/family_query_tail_summary.csv",
        root / "reports/confirmatory/paired_summary.csv",
        *(Path(record["path"]) for record in causal_evidence["source_table_records"]),
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
        return fail("source_table_contract")

    report_path = root / "reports/outcome-summary.md"
    readme_path = root / "README.md"
    if not _hashed_file_complete(root, payload.get("output"), expected_path=report_path):
        return fail("rendered_output_contract")
    try:
        outcome = report_path.read_text(encoding="utf-8").strip()
        readme = readme_path.read_text(encoding="utf-8")
        confirmation_rows, _confirmation_table, _confirmation_manifest = _confirmation_rows(
            root / "reports/confirmatory", families, scope
        )
        hybrid_rows, _hybrid_table, _hybrid_manifest = _hybrid_rows(
            root / "reports/hybrid-adamw", families, scope
        )
        _tail_discovery, tail_final_rows, _tail_tables, _tail_manifest = _tail_stability_rows(
            root / "reports/tail-stability", families, scope
        )
        expected_conclusion = build_final_conclusion_contract(
            confirmation_rows,
            hybrid_rows,
            tail_final_rows,
            causal_evidence,
            families=families,
        )
    except (OSError, AttributeError, json.JSONDecodeError, TypeError, ValueError) as error:
        return fail(f"conclusion_reconstruction:{type(error).__name__}:{error}")
    conclusion_begin, conclusion_end = FINAL_CONCLUSION_MARKERS
    final_checks = {
        "causal_markdown_once": outcome.count(causal_markdown) == 1,
        "conclusion_complete": expected_conclusion.get("status") == "complete",
        "conclusion_not_pending": FINAL_CONCLUSION_PENDING
        not in expected_conclusion.get("plain", ""),
        "conclusion_payload_matches": payload.get("conclusion") == expected_conclusion,
        "readme_conclusion_record": _marked_section_complete(
            readme_path,
            payload.get("readme_conclusion"),
            FINAL_CONCLUSION_MARKERS,
            repository_root=root,
        ),
        "readme_conclusion_matches": readme.split(conclusion_begin, 1)[1]
        .split(conclusion_end, 1)[0]
        .strip()
        == expected_conclusion["markdown"],
    }
    failed = [name for name, complete in final_checks.items() if not complete]
    if failed:
        return fail("final_render_contract:" + ",".join(failed))
    return True


def _paper_main_topology_complete(root: Path) -> bool:
    try:
        main_text = _strip_latex_comments((root / "paper/main.tex").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return False
    if any(main_text.count(token) != 1 for token in PAPER_MAIN_REQUIRED_ONCE):
        return False
    conclusion = main_text.find(r"\section{Conclusion}")
    corrected_abstract = main_text.find(r"\CorrectedAbstractFinding")
    state_operator_abstract = main_text.find(r"\StateOperatorAbstractFinding")
    corrected_main = main_text.find(r"\CorrectedMainSection")
    state_operator_main = main_text.find(r"\StateOperatorMechanismFinding")
    corrected_conclusion = main_text.find(r"\CorrectedConclusionFinding")
    state_operator_conclusion = main_text.find(r"\StateOperatorConclusionFinding")
    historical_conclusion = main_text.find(r"\ResultConclusion")
    main_end_label = main_text.find(rf"\label{{{MAIN_END_LABEL}}}")
    limitations = main_text.find(r"\section{Limitations}")
    ethics = main_text.find(r"\section{Ethical Considerations}")
    bibliography = main_text.find(r"\bibliography{references}")
    appendix = main_text.find(r"\appendix")
    artifact = main_text.find(r"\section{Artifact and Reproducibility}")
    dynamics_input = (
        rf"\input{{{DYNAMICS_EXTENSION_TEX.relative_to('paper').with_suffix('').as_posix()}}}"
    )
    corrected_input = r"\input{generated/corrected-no-packing}"
    corrected_input_position = main_text.find(corrected_input)
    state_operator_input = r"\input{generated/state-operator-factorial}"
    state_operator_input_position = main_text.find(state_operator_input)
    corrected_bridge = r"\CorrectedGeometryBridgeTable"
    corrected_sensitivity = r"\CorrectedExecutionSensitivityTable"
    main_inputs = (r"\input{results}", *PAPER_MAIN_GENERATED_INPUTS)
    appendix_inputs = (*PAPER_APPENDIX_GENERATED_INPUTS, dynamics_input)
    exempt_region_start = main_end_label + len(rf"\label{{{MAIN_END_LABEL}}}")
    exempt_region = main_text[exempt_region_start:appendix]
    exempt_section_positions = [
        exempt_region_start + match.start()
        for match in _LATEX_SECTION_COMMAND.finditer(exempt_region)
    ]
    return (
        0
        <= corrected_input_position
        < state_operator_input_position
        < corrected_abstract
        < state_operator_abstract
        < corrected_main
        < state_operator_main
        < conclusion
        < corrected_conclusion
        < state_operator_conclusion
        < main_end_label
        < limitations
        < ethics
        < bibliography
        < appendix
        < artifact
        < main_text.find(corrected_bridge)
        < main_text.find(corrected_sensitivity)
        < main_text.find(r"\StateOperatorAppendixTable")
        < historical_conclusion
        and exempt_section_positions == [limitations, ethics]
        and _LATEX_FILE_INPUT_COMMAND.search(exempt_region) is None
        and all(main_text.find(token) < main_end_label for token in main_inputs)
        and all(
            main_text.find(token) < main_end_label for token in PAPER_DEFINITION_GENERATED_INPUTS
        )
        and main_text.find(corrected_input) == corrected_input_position
        and main_text.find(state_operator_input) == state_operator_input_position
        and main_text.find(r"\CausalChainSummaryTable") < main_end_label
        and all(main_text.find(token) > appendix for token in appendix_inputs)
        and main_text.find(r"\CandidateBreadthFigure") > appendix
        and main_text.find(r"\CausalChainDiagnostics") > appendix
    )


def _paper_results_complete(
    path: Path,
    payload: dict[str, Any],
    families: tuple[str, ...],
    scope: dict[str, Any] | None,
    *,
    causal_cache: CausalEvidenceCache | None = None,
) -> bool:
    root = path.parents[1]
    try:
        causal_evidence = _causal_evidence_snapshot(root, causal_cache, require_complete=True)
        causal_display_expected = causal_chain_display_contract(causal_evidence)
        causal_headline_expected = render_causal_chain_headline_fragment(causal_evidence)
        causal_latex_expected = render_causal_chain_latex(causal_evidence)
        confirmation_rows, _confirmation_table, _confirmation_manifest = _confirmation_rows(
            root / "reports/confirmatory", families, scope
        )
        hybrid_rows, _hybrid_table, _hybrid_manifest = _hybrid_rows(
            root / "reports/hybrid-adamw", families, scope
        )
        _tail_discovery, tail_final_rows, _tail_tables, _tail_manifest = _tail_stability_rows(
            root / "reports/tail-stability", families, scope
        )
        conclusion_expected = build_final_conclusion_contract(
            confirmation_rows,
            hybrid_rows,
            tail_final_rows,
            causal_evidence,
            families=families,
        )
        main_text = (root / "paper/main.tex").read_text(encoding="utf-8")
        main_text_normalized = " ".join(main_text.split())
        causal_paper_contract = causal_chain_paper_contract()
        causal_latex_observed = (root / PAPER_RESULT_TABLE_PATHS[-1]).read_text(encoding="utf-8")
    except (KeyError, OSError, UnicodeDecodeError, TypeError, ValueError):
        return False
    expected_evidence = {
        (root / relative).resolve()
        for paths in _strict_evidence_paths(families).values()
        for relative in paths
    }
    claim = payload.get("claim_protocol")
    evidence = payload.get("evidence_manifests")
    tables = payload.get("source_tables")
    result_tables = payload.get("result_tables")
    headlines = payload.get("headlines")
    results = payload.get("results_tex")
    causal_chain = payload.get("causal_chain")
    causal_display = payload.get("causal_chain_display")
    conclusion = payload.get("conclusion")
    conclusion_macro = payload.get("conclusion_macro")
    causal_paths = {
        "temporal_short_branch": root / "reports/temporal-short-branch/summary_manifest.json",
        "dose_band": root / "reports/dose-band/summary_manifest.json",
    }
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("complete") is not True
        or not _scope_matches(payload, families, scope)
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
        or not _paper_source_tables_complete(root, tables)
        or not _paper_result_tables_complete(root, result_tables)
        or not _paper_figures_complete(root, payload.get("figures"))
        or not _paper_dynamics_extension_complete(root, payload.get("dynamics_extension"))
        or not _paper_systems_complete(root, families)
        or not _paper_main_topology_complete(root)
        or not isinstance(headlines, dict)
        or set(headlines) != set(HEADLINE_MACROS)
        or any(not isinstance(value, str) or not value for value in headlines.values())
        or headlines["InterventionHeadline"].count(CAUSAL_HEADLINE_PREFIX) != 1
        or not headlines["InterventionHeadline"].endswith(causal_headline_expected)
        or not _hashed_file_complete(
            root,
            results,
            expected_path=root / "paper/results.tex",
        )
        or not isinstance(causal_chain, dict)
        or set(causal_chain) != set(causal_paths)
        or any(
            not _causal_chain_source_complete(
                root,
                causal_chain[label],
                path,
                label,
                causal_cache=causal_cache,
            )
            for label, path in causal_paths.items()
        )
        or causal_display != causal_display_expected
        or causal_latex_observed != causal_latex_expected
        or conclusion_expected.get("status") != "complete"
        or FINAL_CONCLUSION_PENDING in conclusion_expected.get("plain", "")
        or conclusion != conclusion_expected
        or not isinstance(conclusion_macro, dict)
        or conclusion_macro.get("name") != "ResultConclusion"
        or not isinstance(conclusion_macro.get("value"), str)
        or not conclusion_macro["value"]
        or conclusion_macro["value"] != _latex_escape_value(conclusion_expected["plain"])
        or FINAL_CONCLUSION_PENDING in conclusion_macro["value"]
        or any(main_text.count(token) != 1 for token in causal_paper_contract["required_once"])
        or any(
            boundary not in main_text_normalized
            for boundary in causal_paper_contract["required_boundary_substrings"]
        )
        or any(
            overclaim in main_text_normalized.lower()
            for overclaim in causal_paper_contract["forbidden_overclaim_substrings"]
        )
        or "do not convert descriptive checkpoint associations into causal evidence"
        not in str(payload.get("claim_boundary", ""))
    ):
        return False
    try:
        macros = _macros(root / "paper/results.tex")
    except (OSError, TypeError, ValueError):
        return False
    return (
        all(
            "\\ResultPending" not in value and macros.get(name) == value
            for name, value in headlines.items()
        )
        and macros.get("ResultConclusion") == conclusion_macro["value"]
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


def _paper_figures_complete(root: Path, contract: Any) -> bool:
    coverage_path = root / "reports/dense-discovery/coverage.json"
    if not isinstance(contract, dict) or not _hashed_file_complete(
        root, contract.get("source_manifest"), expected_path=coverage_path
    ):
        return False
    try:
        outputs = _json(coverage_path).get("outputs", {})
        discovery_tex = (root / "paper/generated/discovery.tex").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    expected = (
        (
            "dense_training_dynamics_by_run",
            root / "reports/dense-discovery/figures/dense-training-dynamics-by-run.png",
        ),
        (
            "dense_lr_sensitivity",
            root / "reports/dense-discovery/figures/dense-lr-sensitivity.png",
        ),
    )
    panels = contract.get("panels")
    if not isinstance(panels, list) or len(panels) != len(expected):
        return False
    for panel, (name, path) in zip(panels, expected, strict=True):
        source = outputs.get(name)
        if (
            not isinstance(panel, dict)
            or panel.get("name") != name
            or not isinstance(source, dict)
            or not _hashed_file_complete(root, panel, expected_path=path)
            or any(panel.get(field) != source.get(field) for field in ("path", "bytes", "sha256"))
        ):
            return False
    return bool(
        all(discovery_tex.count(command) == 1 for command in PAPER_DISCOVERY_FIGURE_INCLUDES)
        and discovery_tex.count(PAPER_DISCOVERY_FIGURE_CAPTION) == 1
        and discovery_tex.count(PAPER_DISCOVERY_FIGURE_LABEL) == 1
    )


def _paper_dynamics_extension_complete(root: Path, contract: Any) -> bool:
    if not isinstance(contract, dict):
        return False
    try:
        rows, _manifest = load_publication_rows(root)
        summary_rows = summarize_publication_rows(rows)
        expected_latex = render_publication_latex(summary_rows)
        generated_tex = (root / DYNAMICS_EXTENSION_TEX).read_text(encoding="utf-8")
        main = (root / "paper/main.tex").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, TypeError, ValueError):
        return False
    expected_records = {
        "manifest": root / DYNAMICS_EXTENSION_MANIFEST,
        "trajectory_csv": root / DYNAMICS_EXTENSION_CSV,
        "figure_svg": root / DYNAMICS_EXTENSION_SVG,
        "figure_pdf": root / DYNAMICS_EXTENSION_PDF,
        "generated_tex": root / DYNAMICS_EXTENSION_TEX,
    }
    if any(
        not _hashed_file_complete(root, contract.get(name), expected_path=path)
        for name, path in expected_records.items()
    ):
        return False
    return bool(
        contract.get("summary_rows") == summary_rows
        and contract.get("role") == "descriptive-only"
        and contract.get("formal_inference_reads_joined_outputs") is False
        and generated_tex == expected_latex
        and "\\ResultPending" not in generated_tex
        and generated_tex.count("five_stage_retrieval_dynamics.pdf") == 1
        and generated_tex.count("not an inference input") == 1
        and main.count(r"\input{generated/retrieval-dynamics-extension}") == 1
    )


def _paper_systems_complete(root: Path, families: tuple[str, ...]) -> bool:
    table = root / "reports/training-dynamics/optimizer_system_summary.csv"
    try:
        with table.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        diagnostics_tex = (root / "paper/generated/diagnostics.tex").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    indexed = {(row.get("model_family"), row.get("optimizer")): row for row in rows}
    expected = {
        (family, optimizer) for family in FAMILIES for optimizer in ("adamw", "muon", "normuon")
    }
    if len(rows) != 6 or set(indexed) != expected:
        return False
    for family in families:
        family_label = "DenseOn" if family == "dense" else "LateOn"
        for optimizer, optimizer_label in (
            ("adamw", "AdamW"),
            ("muon", "Muon"),
            ("normuon", "NorMuon"),
        ):
            row = indexed[(family, optimizer)]
            try:
                token = " & ".join(
                    (
                        family_label,
                        optimizer_label,
                        f"{float(row['wall_time_hours_median']):.2f}",
                        f"{float(row['samples_per_second_median']):.2f}",
                        f"{float(row['throughput_to_adamw_ratio']):.2f}x",
                        f"{float(row['peak_allocated_gib_median']):.2f}",
                        f"{float(row['optimizer_state_gib_median']):.2f}",
                        f"{float(row['checkpoint_gib_median']):.2f}",
                    )
                )
            except (KeyError, TypeError, ValueError):
                return False
            if diagnostics_tex.count(token) != 1:
                return False
        try:
            muon = float(indexed[(family, "muon")]["throughput_to_adamw_ratio"])
            normuon = float(indexed[(family, "normuon")]["throughput_to_adamw_ratio"])
        except (KeyError, TypeError, ValueError):
            return False
        ratio_fragment = (
            f"{family_label} Muon/NorMuon throughput ratios were {muon:.2f}x/{normuon:.2f}x AdamW"
        )
        speed_fragment = (
            f"neither was faster for {family_label}"
            if muon <= 1 and normuon <= 1
            else f"at least one was faster for {family_label}"
        )
        if diagnostics_tex.count(ratio_fragment) != 1 or diagnostics_tex.count(speed_fragment) != 1:
            return False
    return True


def _paper_source_tables_complete(root: Path, records: Any) -> bool:
    expected = tuple((root / relative).resolve() for relative in PAPER_SOURCE_TABLE_PATHS)
    return bool(
        isinstance(records, list)
        and len(records) == len(expected)
        and all(
            _hashed_file_complete(root, record, expected_path=path)
            for record, path in zip(records, expected, strict=True)
        )
    )


def _final_document_language_problems(root: Path) -> list[str]:
    problems = []
    for relative, phrases in FINAL_DOCUMENT_STALE_PHRASES.items():
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            problems.append(f"{relative}: unreadable ({type(error).__name__})")
            continue
        problems.extend(f"{relative}: {phrase}" for phrase in phrases if phrase in text)
    return problems


def _state_operator_publication_status(root: Path) -> dict[str, Any]:
    """Require exact paper rendering once the complete factorial summary exists."""

    summary = root / STATE_OPERATOR_SUMMARY_MANIFEST
    paper = root / STATE_OPERATOR_LATEX
    publication = root / STATE_OPERATOR_PUBLICATION_MANIFEST
    if not summary.is_file():
        try:
            fixture = paper.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            return {
                "complete": False,
                "status": "missing_topology_fixture",
                "error": f"{type(error).__name__}: {error}",
            }
        return {
            "complete": "\\ResultPending" in fixture and not publication.exists(),
            "status": "prospective_pending",
            "summary_present": False,
        }
    try:
        manifest = audit_state_operator_publication(root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "complete": False,
            "status": "invalid_or_stale",
            "summary_present": True,
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "complete": True,
        "status": "complete",
        "summary_present": True,
        "summary_sha256": _sha256(summary),
        "publication_sha256": _sha256(publication),
        "paper_latex_sha256": _sha256(paper),
        "interpretation": manifest["interpretation"],
    }


def _causal_chain_evidence(
    root: Path, causal_cache: CausalEvidenceCache | None = None
) -> dict[str, dict[str, Any]]:
    paths = {
        "temporal_short_branch": root / "reports/temporal-short-branch/summary_manifest.json",
        "dose_band": root / "reports/dose-band/summary_manifest.json",
    }
    try:
        strict = _causal_evidence_snapshot(root, causal_cache, require_complete=False)
    except (OSError, TypeError, ValueError) as error:
        return {
            label: {
                "path": str(path),
                "complete": False,
                "claimable": False,
                "status": "invalid",
                "sha256": _sha256(path) if path.is_file() else None,
                "error": f"{type(error).__name__}: {error}",
            }
            for label, path in paths.items()
        }
    return {
        label: {
            "path": str(path),
            "complete": branch["complete"] is True,
            "claimable": branch["claimable"],
            "status": branch["status"],
            "supported": branch["supported"],
            "sha256": _sha256(path) if path.is_file() else None,
            "claim_boundary": branch["claim_boundary"],
        }
        for label, path in paths.items()
        for branch in (strict[label],)
    }


def _causal_snapshot_still_current(root: Path, evidence: dict[str, Any]) -> bool:
    """Rehash the snapshot inputs at audit exit to close the read/use window."""

    declared_root = Path(str(evidence.get("repository_root", "")))
    if not declared_root.is_absolute() or declared_root.resolve() != root.resolve():
        return False
    records: list[Any] = [
        evidence.get("protocol"),
        *evidence.get("source_records", []),
        *evidence.get("source_table_records", []),
        evidence.get("temporal_short_branch", {}).get("manifest"),
        evidence.get("dose_band", {}).get("manifest"),
        *evidence.get("temporal_short_branch", {}).get("outputs", {}).values(),
        *evidence.get("dose_band", {}).get("outputs", {}).values(),
    ]
    return bool(records and all(_hashed_file_complete(root, record) for record in records))


def audit_paper(
    paper_dir: str | Path = "paper",
    *,
    repo_root: str | Path = ".",
    matrix: str | Path = "configs/experiment.yaml",
    weight_dir: str | Path = "reports/weight-space",
    training_dir: str | Path = "reports/training-dynamics",
    strict: bool = False,
    families: tuple[str, ...] = FAMILIES,
    scope_amendment: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if scope_amendment is not None:
        requested_scope = Path(scope_amendment)
        scope_amendment = (
            requested_scope.resolve()
            if requested_scope.is_absolute()
            else (root / requested_scope).resolve()
        )
    families, scope = resolve_scope(families, scope_amendment)
    causal_cache: CausalEvidenceCache = {}
    paper = (root / paper_dir).resolve()
    results_path = paper / "results.tex"
    claim_path, claim_protocol, claim_sources = load_paper_claim_protocol(repo_root=root)
    macros = _macros(results_path)
    expected, sources = expected_constant_macros(
        root / matrix,
        root / weight_dir,
        root / training_dir,
        repo_root=root,
        families=families,
        scope_amendment=scope_amendment,
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
    for headline, relative_paths in _strict_evidence_paths(families).items():
        items = []
        for relative in relative_paths:
            path = root / relative
            diagnostics: list[str] = []
            item = {
                "path": str(path),
                "complete": _complete_manifest(
                    path,
                    families=families,
                    scope_amendment=scope_amendment,
                    causal_cache=causal_cache,
                    diagnostics=diagnostics,
                ),
                "sha256": _sha256(path) if path.is_file() else None,
            }
            if diagnostics:
                item["problems"] = diagnostics
            items.append(item)
        evidence[headline] = items
    incomplete_evidence = sorted(
        headline
        for headline, items in evidence.items()
        if not items or not all(item["complete"] for item in items)
    )
    paper_results_path = root / "reports/paper-results.manifest.json"
    paper_results_complete = _complete_manifest(
        paper_results_path,
        families=families,
        scope_amendment=scope_amendment,
        causal_cache=causal_cache,
    )
    document_language_problems = _final_document_language_problems(root)
    causal_chain = _causal_chain_evidence(root, causal_cache)
    snapshot, snapshot_error = causal_cache.get(root, (None, None))
    causal_chain_complete = bool(
        all(item["complete"] for item in causal_chain.values())
        and snapshot_error is None
        and snapshot is not None
        and _causal_snapshot_still_current(root, snapshot)
    )
    state_operator_publication = _state_operator_publication_status(root)
    complete = (
        not pending
        and not incomplete_evidence
        and paper_results_complete
        and not document_language_problems
        and causal_chain_complete
        and state_operator_publication["complete"]
    )
    portable_manifest_path = root / PORTABLE_EVIDENCE_MANIFEST
    if (root / "outputs").exists():
        evidence_mode = {
            "mode": "checkpoint-backed-full-source",
            "checkpoint_tree_present": True,
        }
    else:
        try:
            portable_payload = _json(portable_manifest_path)
            portable_summary = portable_payload["summary"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            portable_summary = {}
        evidence_mode = {
            "mode": "portable-evaluation-closure",
            "checkpoint_tree_present": False,
            "manifest": {
                "path": str(portable_manifest_path),
                "sha256": (
                    _sha256(portable_manifest_path) if portable_manifest_path.is_file() else None
                ),
            },
            "files": portable_summary.get("files"),
            "bytes": portable_summary.get("bytes"),
        }
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
        "document_language_problems": document_language_problems,
        "causal_chain": causal_chain,
        "state_operator_publication": state_operator_publication,
        "evidence_mode": evidence_mode,
    }
    if scope is not None:
        result["families"] = list(families)
        result["scope_amendment"] = scope
    if strict and not complete:
        raise ValueError(
            "Paper is not final: "
            f"pending_headlines={pending}, incomplete_evidence={incomplete_evidence}, "
            f"paper_results_complete={paper_results_complete}, "
            f"document_language_problems={document_language_problems}"
            f", causal_chain_complete={causal_chain_complete}, "
            f"state_operator_publication={state_operator_publication['status']}"
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
    parser.add_argument(
        "--families", nargs="+", choices=("dense", "late"), default=["dense", "late"]
    )
    parser.add_argument("--scope-amendment", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = audit_paper(
        args.paper_dir,
        repo_root=args.repo_root,
        matrix=args.matrix,
        weight_dir=args.weight_dir,
        training_dir=args.training_dir,
        strict=False,
        families=tuple(args.families),
        scope_amendment=args.scope_amendment,
    )
    result["strict"] = args.strict
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict and not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
