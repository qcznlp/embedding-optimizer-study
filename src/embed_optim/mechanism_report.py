from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

MECHANISM_MARKERS = ("<!-- MECHANISM:BEGIN -->", "<!-- MECHANISM:END -->")
FAMILIES = ("dense", "late")
OPTIMIZERS = ("adamw", "muon", "normuon")
FAMILY_LABELS = {"dense": "DenseOn", "late": "LateOn"}
OPTIMIZER_LABELS = {"adamw": "AdamW", "muon": "Muon", "normuon": "NorMuon"}


def _resolve_declared(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {path}")
    return payload


def _rehash_references(value: Any, *, context: str) -> int:
    if isinstance(value, list):
        return sum(
            _rehash_references(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    if not isinstance(value, dict):
        return 0
    verified = 0
    path_value = value.get("path")
    digest = value.get("sha256")
    if isinstance(path_value, str) and isinstance(digest, str):
        path = Path(path_value).resolve()
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Referenced source differs in {context}: {path}")
        verified += 1
    return verified + sum(
        _rehash_references(item, context=f"{context}.{name}")
        for name, item in value.items()
        if name not in {"path", "sha256"}
    )


def _read_declared_csv(
    root: Path,
    manifest: dict[str, Any],
    name: str,
    *,
    required_fields: set[str],
) -> tuple[list[dict[str, str]], Path]:
    declared = manifest.get("outputs", {}).get(name)
    if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
        raise ValueError(f"Manifest does not declare {name}")
    path = _resolve_declared(root, declared["path"])
    if (
        not path.is_file()
        or path.stat().st_size != declared.get("bytes")
        or _sha256(path) != declared.get("sha256")
    ):
        raise ValueError(f"Declared table differs from its manifest: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not required_fields.issubset(fields):
            raise ValueError(f"Required fields are absent from {path}")
        rows = list(reader)
    if len(rows) != declared.get("rows"):
        raise ValueError(f"Declared row count differs for {path}")
    return rows, path


def _finite(row: dict[str, str], field: str, *, allow_empty: bool = False) -> float | None:
    value = row.get(field, "")
    if allow_empty and value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite numeric value for {field}: {value!r}")
    return parsed


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty metric group")
    return float(statistics.median(values))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| "
            + " | ".join("---" if index == 0 else "---:" for index in range(len(headers)))
            + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def _format(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _common_state_rows(summary_dir: Path) -> tuple[list[list[str]], dict[str, Any], Path]:
    summary_dir = summary_dir.resolve()
    manifest_path = summary_dir / "summary_manifest.json"
    manifest = _load_manifest(manifest_path)
    expected_rows = {
        "gradient_tensor_metrics": 1_760,
        "update_tensor_metrics": 5_280,
        "pairwise_tensor_cosines": 5_280,
        "gradient_anchor_metrics": 20,
        "anchor_metrics": 60,
        "pairwise_anchor_cosines": 60,
        "update_gradient_contrasts": 60,
        "anchor_contrasts": 40,
    }
    if (
        manifest.get("complete") is not True
        or manifest.get("allow_partial") is not False
        or manifest.get("expected_anchors") != 20
        or manifest.get("valid_anchors") != 20
        or manifest.get("missing_labels") != []
        or {name: item.get("rows") for name, item in manifest.get("outputs", {}).items()}
        != expected_rows
    ):
        raise ValueError("Common-state summary is not the complete frozen 20-anchor matrix")
    fields = {
        "family",
        "label",
        "update_operator",
        "row_norm_cv_parameter_weighted_to_adamw_ratio",
        "top_1pct_row_energy_parameter_weighted_to_adamw_ratio",
        "approx_stable_rank_parameter_weighted_to_adamw_ratio",
        "spectral_norm_parameter_weighted_to_adamw_ratio",
        "cosine_with_adamw_parameter_weighted",
    }
    rows, path = _read_declared_csv(
        summary_dir, manifest, "anchor_contrasts", required_fields=fields
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("family", ""), row.get("update_operator", ""))
        if key[0] not in FAMILIES or key[1] not in {"muon", "normuon"}:
            raise ValueError(f"Unexpected common-state contrast identity: {key}")
        grouped[key].append(row)
    if set(grouped) != {
        (family, optimizer) for family in FAMILIES for optimizer in ("muon", "normuon")
    } or any(len(values) != 10 for values in grouped.values()):
        raise ValueError("Common-state contrasts do not cover ten anchors per family/operator")
    output: list[list[str]] = []
    metrics = (
        "row_norm_cv_parameter_weighted_to_adamw_ratio",
        "top_1pct_row_energy_parameter_weighted_to_adamw_ratio",
        "approx_stable_rank_parameter_weighted_to_adamw_ratio",
        "spectral_norm_parameter_weighted_to_adamw_ratio",
        "cosine_with_adamw_parameter_weighted",
    )
    for family in FAMILIES:
        for optimizer in ("muon", "normuon"):
            values = grouped[(family, optimizer)]
            medians = [_median([float(_finite(row, field)) for row in values]) for field in metrics]
            output.append(
                [
                    FAMILY_LABELS[family],
                    OPTIMIZER_LABELS[optimizer],
                    *[_format(value, 3) for value in medians],
                ]
            )
    return output, manifest, path


def _spectrum_rows(summary_dir: Path) -> tuple[list[list[str]], dict[str, Any], Path]:
    summary_dir = summary_dir.resolve()
    manifest_path = summary_dir / "summary_manifest.json"
    manifest = _load_manifest(manifest_path)
    if (
        manifest.get("complete") is not True
        or manifest.get("allow_partial") is not False
        or manifest.get("expected_anchors") != 20
        or manifest.get("valid_anchors") != 20
        or manifest.get("expected_spectra") != 360
        or manifest.get("valid_spectra") != 360
        or manifest.get("missing_labels") != []
    ):
        raise ValueError("Exact-spectrum summary is not the frozen 360-spectrum matrix")
    fields = {
        "family",
        "label",
        "update_operator",
        "tensor",
        "rank",
        "stable_rank",
        "entropy_effective_rank",
        "condition_number",
    }
    rows, path = _read_declared_csv(
        summary_dir, manifest, "spectrum_metrics", required_fields=fields
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("family", ""), row.get("update_operator", ""))
        if key[0] not in FAMILIES or key[1] not in OPTIMIZERS:
            raise ValueError(f"Unexpected exact-spectrum identity: {key}")
        grouped[key].append(row)
    if set(grouped) != {
        (family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS
    } or any(len(values) != 60 for values in grouped.values()):
        raise ValueError("Exact spectra do not cover 60 selected matrices per family/operator")
    output: list[list[str]] = []
    for family in FAMILIES:
        for optimizer in OPTIMIZERS:
            values = grouped[(family, optimizer)]
            normalized_stable = []
            normalized_entropy = []
            conditions = []
            for row in values:
                rank = _finite(row, "rank")
                stable = _finite(row, "stable_rank")
                entropy = _finite(row, "entropy_effective_rank")
                assert rank is not None and stable is not None and entropy is not None
                if rank <= 0:
                    raise ValueError("Exact spectrum has a non-positive rank")
                normalized_stable.append(stable / rank)
                normalized_entropy.append(entropy / rank)
                condition = _finite(row, "condition_number", allow_empty=True)
                if condition is not None:
                    conditions.append(condition)
            output.append(
                [
                    FAMILY_LABELS[family],
                    OPTIMIZER_LABELS[optimizer],
                    _format(_median(normalized_stable)),
                    _format(_median(normalized_entropy)),
                    _format(_median(conditions) if conditions else None, 2),
                ]
            )
    return output, manifest, path


def _bridge_rows(
    bridge_dir: Path,
) -> tuple[list[list[str]], list[list[str]], dict[str, Any], list[Path]]:
    bridge_dir = bridge_dir.resolve()
    manifest_path = bridge_dir / "summary_manifest.json"
    manifest = _load_manifest(manifest_path)
    if (
        manifest.get("complete") is not True
        or manifest.get("checkpoints") != 120
        or manifest.get("within_run_transitions") != 96
        or manifest.get("correlations") != 200
    ):
        raise ValueError("Mechanism bridge is not the strict 120-checkpoint join")
    if _rehash_references(manifest.get("sources"), context="mechanism_bridge.sources") < 6:
        raise ValueError("Mechanism bridge does not bind all strict source manifests and tables")
    checkpoint_fields = {
        "model_family",
        "optimizer",
        "stage",
        "training_margin_mean",
        "unseen_margin_mean",
        "unseen_query_normalized_effective_rank",
        "unseen_reference_top1_agreement",
        "unseen_document_token_coverage_mean",
        "mean_beir_ndcg_at_10",
    }
    checkpoints, checkpoint_path = _read_declared_csv(
        bridge_dir, manifest, "checkpoint_bridge", required_fields=checkpoint_fields
    )
    final: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in checkpoints:
        family = row.get("model_family", "")
        optimizer = row.get("optimizer", "")
        try:
            stage = int(row.get("stage", ""))
        except ValueError as error:
            raise ValueError(f"Invalid mechanism checkpoint stage: {row}") from error
        if family not in FAMILIES or optimizer not in OPTIMIZERS or not 1 <= stage <= 5:
            raise ValueError(f"Invalid mechanism checkpoint identity: {row}")
        if stage == 5:
            final[(family, optimizer)].append(row)
    if set(final) != {
        (family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS
    } or any(len(values) != 4 for values in final.values()):
        raise ValueError("Final representation bridge does not cover four learning rates")
    representation: list[list[str]] = []
    fields = (
        "training_margin_mean",
        "unseen_margin_mean",
        "unseen_query_normalized_effective_rank",
        "unseen_reference_top1_agreement",
        "mean_beir_ndcg_at_10",
    )
    for family in FAMILIES:
        for optimizer in OPTIMIZERS:
            values = final[(family, optimizer)]
            summaries = [
                _median([float(_finite(row, field)) for row in values]) for field in fields
            ]
            coverage: float | None = None
            if family == "late":
                coverage = _median(
                    [float(_finite(row, "unseen_document_token_coverage_mean")) for row in values]
                )
            representation.append(
                [
                    FAMILY_LABELS[family],
                    OPTIMIZER_LABELS[optimizer],
                    *[_format(value) for value in summaries[:4]],
                    _format(coverage),
                    _format(summaries[4]),
                ]
            )

    correlation_fields = {
        "model_family",
        "scope",
        "optimizer",
        "analysis",
        "predictor",
        "outcome",
        "observations",
        "spearman_rho",
    }
    correlations, correlation_path = _read_declared_csv(
        bridge_dir, manifest, "descriptive_correlations", required_fields=correlation_fields
    )
    indexed = {
        (
            row["model_family"],
            row["scope"],
            row["optimizer"],
            row["analysis"],
            row["predictor"],
            row["outcome"],
        ): row
        for row in correlations
    }
    selected = [
        (family, predictor, outcome)
        for family in FAMILIES
        for predictor, outcome in (
            ("reference_delta_row_cv_parameter_weighted", "unseen_margin_mean"),
            ("unseen_margin_mean", "mean_beir_ndcg_at_10"),
            ("unseen_query_normalized_effective_rank", "mean_beir_ndcg_at_10"),
        )
    ]
    selected.append(("late", "unseen_document_token_coverage_mean", "mean_beir_ndcg_at_10"))
    labels = {
        "reference_delta_row_cv_parameter_weighted": "weight-delta row CV",
        "unseen_margin_mean": "unseen margin",
        "unseen_query_normalized_effective_rank": "unseen query effective rank",
        "unseen_document_token_coverage_mean": "document-token coverage",
        "mean_beir_ndcg_at_10": "mean BEIR nDCG@10",
    }
    correlation_output: list[list[str]] = []
    for family, predictor, outcome in selected:
        key = (
            family,
            "all_optimizers",
            "all",
            "within_run_first_differences",
            predictor,
            outcome,
        )
        if key not in indexed:
            raise ValueError(f"Prespecified mechanism correlation is absent: {key}")
        row = indexed[key]
        observations = int(row["observations"])
        if observations != 48:
            raise ValueError(f"Unexpected within-run observation count for {key}: {observations}")
        rho = _finite(row, "spearman_rho", allow_empty=True)
        correlation_output.append(
            [
                FAMILY_LABELS[family],
                labels[predictor],
                labels[outcome],
                str(observations),
                _format(rho, 3),
            ]
        )
    return representation, correlation_output, manifest, [checkpoint_path, correlation_path]


def _validate_figure(path: Path, *, spectra: bool = False) -> dict[str, Any]:
    path = path.resolve()
    sidecar = path.with_suffix(".manifest.json")
    manifest = _load_manifest(sidecar)
    output = manifest.get("output") or {}
    if (
        not path.is_file()
        or Path(output.get("path", "")).resolve() != path
        or path.stat().st_size != output.get("bytes")
        or _sha256(path) != output.get("sha256")
        or (not spectra and manifest.get("complete") is not True)
        or (spectra and (manifest.get("anchors") != 20 or manifest.get("spectra") != 360))
    ):
        raise ValueError(f"Mechanism figure differs from its strict manifest: {path}")
    sources = {name: value for name, value in manifest.items() if name != "output"}
    if _rehash_references(sources, context=f"figure.{path.name}") == 0:
        raise ValueError(f"Mechanism figure does not bind a source manifest: {path}")
    return {"path": str(path), "sha256": _sha256(path), "manifest_sha256": _sha256(sidecar)}


def _retrieval_rows(
    retrieval_dir: Path,
) -> tuple[list[list[str]], dict[str, Any], Path, dict[str, Any]]:
    retrieval_dir = retrieval_dir.resolve()
    repository_root = retrieval_dir.parents[1]
    manifest_path = retrieval_dir / "summary_manifest.json"
    manifest = _load_manifest(manifest_path)
    if (
        manifest.get("complete") is not True
        or manifest.get("coverage")
        != {
            "runs": 24,
            "checkpoints": 120,
            "tasks": 14,
            "evaluation_units": 1_680,
            "optimizer_family_groups": 6,
        }
        or _rehash_references(manifest.get("sources"), context="retrieval_dynamics.sources") < 1_685
    ):
        raise ValueError("Retrieval dynamics is not the strict 1,680-unit completion report")
    fields = {
        "model_family",
        "optimizer",
        "learning_rate_points",
        "adamw_median_final_target",
        "points_reaching_target",
        "points_right_censored",
        "fastest_observed_useful_wall_time_hours",
        "median_observed_useful_wall_time_hours",
        "target_definition",
        "interpolation",
    }
    rows, table = _read_declared_csv(
        repository_root,
        manifest,
        "optimizer_first_passage",
        required_fields=fields,
    )
    indexed = {(row.get("model_family", ""), row.get("optimizer", "")): row for row in rows}
    expected = {(family, optimizer) for family in FAMILIES for optimizer in OPTIMIZERS}
    if len(rows) != 6 or set(indexed) != expected:
        raise ValueError("Retrieval first-passage table does not cover six optimizer/family groups")
    output = []
    for family in FAMILIES:
        for optimizer in OPTIMIZERS:
            row = indexed[(family, optimizer)]
            learning_rates = int(row["learning_rate_points"])
            reached = int(row["points_reaching_target"])
            censored = int(row["points_right_censored"])
            target = _finite(row, "adamw_median_final_target")
            fastest = _finite(row, "fastest_observed_useful_wall_time_hours", allow_empty=True)
            median = _finite(row, "median_observed_useful_wall_time_hours", allow_empty=True)
            if (
                learning_rates != 4
                or reached + censored != 4
                or row["target_definition"] != "within-family-median-of-four-adamw-final-points"
                or row["interpolation"] != "none-five-observed-checkpoints-only"
            ):
                raise ValueError(f"Invalid retrieval first-passage group: {(family, optimizer)}")
            output.append(
                [
                    FAMILY_LABELS[family],
                    OPTIMIZER_LABELS[optimizer],
                    _format(target),
                    f"{reached}/4",
                    _format(fastest, 3),
                    _format(median, 3),
                    str(censored),
                ]
            )
    figure_item = manifest.get("outputs", {}).get("quality_vs_useful_wall_time", {})
    figure_path = _resolve_declared(repository_root, str(figure_item.get("path", "")))
    if (
        figure_path != retrieval_dir / "quality_vs_useful_wall_time.svg"
        or not figure_path.is_file()
        or figure_path.stat().st_size != figure_item.get("bytes")
        or _sha256(figure_path) != figure_item.get("sha256")
    ):
        raise ValueError("Retrieval dynamics figure differs from its strict manifest")
    figure = {"path": str(figure_path), "sha256": _sha256(figure_path)}
    return output, manifest, table, figure


def _replace_marked(text: str, content: str) -> str:
    begin, end = MECHANISM_MARKERS
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Expected exactly one mechanism marker pair in the blog")
    before, remainder = text.split(begin)
    _, after = remainder.split(end)
    return f"{before}{begin}\n\n{content}\n\n{end}{after}"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def render_mechanism_report(
    common_state_dir: Path,
    spectrum_dir: Path,
    bridge_dir: Path,
    retrieval_dir: Path,
    blog_path: Path,
    output_path: Path,
    *,
    spectrum_figure: Path,
    representation_figure: Path,
    late_token_figure: Path,
) -> dict[str, Any]:
    common_rows, common_manifest, common_table = _common_state_rows(common_state_dir)
    spectrum_rows, spectrum_manifest, spectrum_table = _spectrum_rows(spectrum_dir)
    representation_rows, correlation_rows, bridge_manifest, bridge_tables = _bridge_rows(bridge_dir)
    retrieval_rows, retrieval_manifest, retrieval_table, retrieval_figure = _retrieval_rows(
        retrieval_dir
    )
    figures = {
        "retrieval_dynamics": retrieval_figure,
        "exact_spectra": _validate_figure(spectrum_figure, spectra=True),
        "representation_dynamics": _validate_figure(representation_figure),
        "late_token_dynamics": _validate_figure(late_token_figure),
    }
    content = "\n\n".join(
        [
            "The formal mechanism tier evaluates every optimizer transform at the same frozen weights "
            "and on the same ordered eight-gradient history. The values below are generated only after "
            "the complete 20-anchor matrix, 360 exact spectra, both 122-job representation tiers, and "
            "the 1,680-unit retrieval matrix pass their content-hash audits.",
            "### Retrieval time to an AdamW reference\n\n"
            "![Retrieval quality versus useful wall time]"
            "(../reports/retrieval-dynamics/quality_vs_useful_wall_time.svg)\n\n"
            + _table(
                [
                    "Family",
                    "Optimizer",
                    "AdamW reference",
                    "LR points reaching",
                    "fastest hours",
                    "median hours",
                    "right-censored",
                ],
                retrieval_rows,
            )
            + "\n\nThe reference is the within-family median final nDCG@10 of the four AdamW "
            "learning-rate points. Passage is observed only at the five saved checkpoints; no "
            "interpolation is used, and non-reaching points remain right-censored. Checkpoint time "
            "is a step-proportional estimate from audited useful terminal wall time. The rule was "
            "locked after 160/1,680 discovery units were visible, so this is exploratory rather "
            "than a preregistration or a substitute for the three-seed confirmation.",
            "### Same-state optimizer fingerprints\n\n"
            + _table(
                [
                    "Family",
                    "Operator",
                    "row CV / AdamW",
                    "top-1% row energy / AdamW",
                    "stable rank / AdamW",
                    "spectral norm / AdamW",
                    "cosine with AdamW",
                ],
                common_rows,
            )
            + "\n\nEach cell is the median over ten frozen anchors. Ratios use raw optimizer "
            "directions but are scale-invariant except for the explicitly reported spectral-norm ratio; "
            "the exact-spectrum intervention below uses per-tensor Frobenius-matched directions. Weight "
            "decay is excluded from this comparison.",
            "### Exact update spectra\n\n"
            "![Exact common-state update spectra](../reports/common-state/exact-update-spectra.svg)\n\n"
            + _table(
                [
                    "Family",
                    "Operator",
                    "stable rank / rank",
                    "entropy rank / rank",
                    "condition number",
                ],
                spectrum_rows,
            )
            + "\n\nThe six matrices were fixed by early/middle/final depth and attention/MLP role "
            "before formal spectra existed. Values are medians over 60 exact spectra per "
            "family/operator; the figure shows the full normalized curves and interquartile bands.",
            "### Representation and score geometry\n\n"
            "![Representation dynamics](../reports/representation-space/representation-dynamics.svg)\n\n"
            + _table(
                [
                    "Family",
                    "Optimizer",
                    "training margin",
                    "unseen margin",
                    "unseen query rank",
                    "pretrained top-1 agreement",
                    "Late document-token coverage",
                    "mean BEIR nDCG@10",
                ],
                representation_rows,
            )
            + "\n\nRows are final-stage medians across all four frozen learning rates, not "
            "test-selected winners. Training and unseen probes remain separate; the latter contains "
            "224 fixed examples balanced over all 14 decontaminated tasks.",
            "### Late-interaction token utilization\n\n"
            "![LateOn token-utilization dynamics](../reports/representation-space/late-token-dynamics.svg)\n\n"
            "This panel reports the four prespecified MaxSim evidence summaries on both probe tiers. "
            "It is kept separate from the shared DenseOn/LateOn figure so a LateOn-only signal cannot "
            "change the cross-architecture metric definition after results are visible.",
            "### Descriptive temporal bridge\n\n"
            + _table(
                ["Family", "Predictor change", "Outcome change", "Transitions", "Spearman ρ"],
                correlation_rows,
            )
            + "\n\nThese seven associations were fixed in the renderer and use within-run first "
            "differences across all optimizers. They are one-seed observational summaries, not a "
            "causal mediation analysis. The same-state fingerprints identify what each update rule "
            "does; causal claims about later retrieval still require matched short branches or "
            "optimizer-switch interventions.",
        ]
    )
    output_path = output_path.resolve()
    _atomic_text(output_path, content + "\n")
    blog_path = blog_path.resolve()
    rendered_blog = _replace_marked(blog_path.read_text(encoding="utf-8"), content)
    _atomic_text(blog_path, rendered_blog)
    source_manifests = {
        "common_state": {
            "path": str((common_state_dir / "summary_manifest.json").resolve()),
            "sha256": _sha256((common_state_dir / "summary_manifest.json").resolve()),
            "anchors": common_manifest["valid_anchors"],
        },
        "retrieval_dynamics": {
            "path": str((retrieval_dir / "summary_manifest.json").resolve()),
            "sha256": _sha256((retrieval_dir / "summary_manifest.json").resolve()),
            "evaluation_units": retrieval_manifest["coverage"]["evaluation_units"],
        },
        "exact_spectra": {
            "path": str((spectrum_dir / "summary_manifest.json").resolve()),
            "sha256": _sha256((spectrum_dir / "summary_manifest.json").resolve()),
            "spectra": spectrum_manifest["valid_spectra"],
        },
        "mechanism_bridge": {
            "path": str((bridge_dir / "summary_manifest.json").resolve()),
            "sha256": _sha256((bridge_dir / "summary_manifest.json").resolve()),
            "checkpoints": bridge_manifest["checkpoints"],
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "sources": source_manifests,
        "source_tables": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in [retrieval_table, common_table, spectrum_table, *bridge_tables]
        ],
        "figures": figures,
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "blog": {
            "path": str(blog_path),
            "bytes": blog_path.stat().st_size,
            "sha256": _sha256(blog_path),
            "markers": list(MECHANISM_MARKERS),
        },
        "aggregation": {
            "retrieval_dynamics": "six-family-optimizer-groups-over-four-learning-rate-points",
            "common_state": "median-over-ten-frozen-anchors-per-family-operator",
            "exact_spectra": "median-over-sixty-prespecified-spectra-per-family-operator",
            "representation": "final-stage-median-over-four-frozen-learning-rates",
            "bridge": "seven-prespecified-within-run-first-difference-spearman-associations",
        },
        "interpretation": (
            "Common-state transforms identify optimizer fingerprints and the bridge remains "
            "descriptive one-seed evidence; causal retrieval claims require short interventions."
        ),
    }
    _atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the strict update-to-retrieval mechanism section into the final blog"
    )
    parser.add_argument("--common-state-dir", type=Path, default=Path("reports/common-state"))
    parser.add_argument(
        "--spectrum-dir", type=Path, default=Path("results/common-state-spectra/summary")
    )
    parser.add_argument("--bridge-dir", type=Path, default=Path("reports/mechanism-bridge"))
    parser.add_argument("--retrieval-dir", type=Path, default=Path("reports/retrieval-dynamics"))
    parser.add_argument("--blog", type=Path, default=Path("docs/blog.md"))
    parser.add_argument("--output", type=Path, default=Path("reports/mechanism-summary.md"))
    parser.add_argument(
        "--spectrum-figure",
        type=Path,
        default=Path("reports/common-state/exact-update-spectra.svg"),
    )
    parser.add_argument(
        "--representation-figure",
        type=Path,
        default=Path("reports/representation-space/representation-dynamics.svg"),
    )
    parser.add_argument(
        "--late-token-figure",
        type=Path,
        default=Path("reports/representation-space/late-token-dynamics.svg"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = render_mechanism_report(
        args.common_state_dir,
        args.spectrum_dir,
        args.bridge_dir,
        args.retrieval_dir,
        args.blog,
        args.output,
        spectrum_figure=args.spectrum_figure,
        representation_figure=args.representation_figure,
        late_token_figure=args.late_token_figure,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
