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
STRICT_EVIDENCE = {
    "DiscoveryHeadline": (
        Path("reports/coverage.json"),
        Path("reports/training-dynamics/summary_manifest.json"),
        Path("reports/training-dynamics/plot_manifest.json"),
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
    ),
    "ConfirmationHeadline": (Path("reports/confirmatory/summary_manifest.json"),),
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
    dataset = _json(dataset_manifest_path)
    if dataset.get("total_queries") != 500_000 or dataset.get("sampled_negatives") != 7:
        raise ValueError("Paper constants differ from the materialized dataset manifest")
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
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": _sha256(dataset_manifest_path),
        },
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
    return payload.get("schema_version") == SCHEMA_VERSION and payload.get("complete") is True


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
    complete = not pending and not incomplete_evidence
    result = {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "strict": strict,
        "paper_dir": str(paper),
        "results_path": str(results_path),
        "constant_macros": expected,
        "constant_sources": sources,
        "pending_headlines": pending,
        "incomplete_evidence": incomplete_evidence,
        "evidence": evidence,
    }
    if strict and not complete:
        raise ValueError(
            "Paper is not final: "
            f"pending_headlines={pending}, incomplete_evidence={incomplete_evidence}"
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
