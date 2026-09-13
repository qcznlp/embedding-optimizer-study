"""Close the small dimension-publication evidence after a full authoring audit.

Portable auditing recomputes statistics and exact manuscript text. It verifies
the recorded model/export provenance by digest, but does not rerun model
encoding or coordinate ablations and does not certify whole-study completion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import dimension_publication as publication

MANIFEST = Path("reports/dimension-utilization-publication/portable_manifest.json")
DIMENSION_DIR = Path("reports/dimension-utilization-primary")
PUBLICATION_DIR = Path("reports/dimension-utilization-publication")
OUTCOME_DIR = Path("reports/dense-no-packing-outcomes")
PAPER_OUTPUT = Path("paper/generated/dimension-utilization.tex")
PROTOCOL = Path("configs/dense_dimension_utilization_protocol.json")
FIXED_SOURCES = (
    "src/embed_optim/dimension_release.py",
    "src/embed_optim/dimension_publication.py",
    "src/embed_optim/dimension_utilization.py",
    "src/embed_optim/config.py",
    "configs/dense_dimension_utilization_protocol.json",
    "configs/dense_primary_dimension_export_protocol.json",
    "configs/dense_no_packing_retrain.yaml",
    "configs/beir_representation_probe.json",
)
BOUNDARY = (
    "Portable publication recomputation only: no checkpoint revalidation, model encoding, "
    "or coordinate-ablation recomputation. Full source inputs were audited before this closure "
    "was generated; their identities are retained. Whole-study scientific completion is separate."
)


def publication_args(repository: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repository=repository,
        protocol=repository / PROTOCOL,
        dimension_dir=repository / DIMENSION_DIR,
        outcomes_dir=repository / OUTCOME_DIR,
        output_dir=repository / PUBLICATION_DIR,
        paper_output=repository / PAPER_OUTPUT,
    )


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dimension evidence object: {path}")
    return payload


def selected_files(repository: Path) -> list[Path]:
    """Derive the closure from validated identity records, not an arbitrary file list."""
    paths = {repository / name for name in FIXED_SOURCES}

    def take(record: dict[str, Any]) -> Path:
        path = publication._resolve_identity(record, repository)
        if not path.is_relative_to(repository.resolve()):
            raise ValueError(f"Portable dimension evidence is outside the repository: {path}")
        paths.add(path)
        return path

    report_path = repository / PUBLICATION_DIR / "summary_manifest.json"
    paths.add(report_path)
    report = _load(report_path)
    dimension_path = take(report["sources"]["dimension"]["manifest"])
    outcome_path = take(report["sources"]["outcome"]["manifest"])
    if dimension_path != (repository / DIMENSION_DIR / "summary_manifest.json").resolve():
        raise ValueError("Portable dimension source is not the canonical primary analysis")
    if outcome_path != (repository / OUTCOME_DIR / "summary_manifest.json").resolve():
        raise ValueError("Portable dimension outcome is not the canonical primary outcome")
    dimension = _load(dimension_path)
    if dimension.get("scope") != "corrected_dense_dimension_utilization":
        raise ValueError("Portable dimension source is not primary")
    for record in report["outputs"].values():
        take(record)
    for record in dimension["outputs"].values():
        take(record)
    take(report["sources"]["outcome"]["run_stage_scores"])
    for record in (
        report["implementation"],
        report["protocol"],
        dimension["implementation"],
        dimension["protocol"],
    ):
        take(record)
    handoff_path = take(dimension["primary_export_handoff"])
    handoff = _load(handoff_path)
    if handoff.get("status") != "complete" or len(handoff.get("outputs", [])) != 61:
        raise ValueError("Portable dimension closure requires the complete export handoff")
    take(handoff["training_audit"])
    for record in handoff["outputs"]:
        # The large exported arrays and checkpoint payloads are intentionally not in Git.
        # Their content identities remain in these original encoder and execution receipts.
        take(record["export_manifest"])
        take(record["receipt"])
    exporter = _load(repository / "configs/dense_primary_dimension_export_protocol.json")
    for group in ("source_bindings", "parent_bindings"):
        for record in exporter[group].values():
            take(record)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(paths)


def _payload(repository: Path) -> dict[str, Any]:
    files = [publication._identity(path, repository) for path in selected_files(repository)]
    return {
        "schema_version": 1,
        "status": "portable_dimension_publication_closure",
        "authoring_audit": "passed_full_checkpoint_backed_dimension_publication_audit",
        "scientific_completion": False,
        "implementation": publication._identity(Path(__file__), repository),
        "files": files,
        "summary": {"files": len(files), "bytes": sum(row["bytes"] for row in files)},
        "boundary": BOUNDARY,
    }


def build_closure(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    publication.audit_report(publication_args(repository))
    payload = _payload(repository)
    publication._atomic_json(repository / MANIFEST, payload)
    return audit_closure(repository)


def audit_closure(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    observed = _load(repository / MANIFEST)
    if observed != _payload(repository):
        raise ValueError("Portable dimension closure differs from its source-bound files")
    return {
        "complete": True,
        "scope": "portable_dimension_publication_closure",
        "scientific_completion": False,
        "checkpoint_revalidated": False,
        "embedding_features_recomputed": False,
        "manifest": publication._identity(repository / MANIFEST, repository),
        **observed["summary"],
        "boundary": BOUNDARY,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    result = audit_closure(args.repository) if args.audit_only else build_closure(args.repository)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
