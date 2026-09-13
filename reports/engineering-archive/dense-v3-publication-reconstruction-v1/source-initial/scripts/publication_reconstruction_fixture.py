"""Extend the accepted full synthetic joint fixture, never forge primary authoring."""

from pathlib import Path

from embed_optim import primary_v3_publication as publication
from embed_optim import primary_v3_publication_archive as archive
from embed_optim import primary_v3_publication_reconstruction as reconstruction
from embed_optim import reconstruction_files as files
from embed_optim.primary_contract import file_identity, read_json, require_same
from embed_optim.primary_v3_reconstruction_sources import recheck_selection, source_selection
from embed_optim.primary_v3_validation_io import write_new

ORIGINAL = Path("/tmp/dense-v3-joint-cold.yR0bGv/fixture-input/archive")
ANCHOR = "246927afda59eeefc6e260709d93a119b9464f3097cd4800e716282e9236fbfc"


def make(work, contract):
    """Reuse authenticated raw bytes; the new cold reader repeats all numerics."""
    work = Path(work)
    if work.exists():
        raise ValueError("Require a new synthetic publication fixture directory")
    previous = files.inspect(ORIGINAL, ANCHOR)
    functional = read_json(ORIGINAL / "inference/evidence.json")
    assert functional["upstream_primary_admission_simulated"] is True
    tables = {}
    for kind, role, counts in (
        ("outcomes", "outcomes", publication.TABLE_COUNTS),
        ("geometry", "geometry", publication.geometry.TABLE_COUNTS),
        ("bridge", "bridge", publication.inference.original.TABLE_COUNTS),
        ("functional", "inference", publication.inference.TABLE_COUNTS),
    ):
        tables[kind] = publication.read_tables(ORIGINAL / role, counts)
    current = publication.inspect_bundle(
        ORIGINAL / "inference",
        previous["metadata"]["authoring_plan"],
        functional,
        tables["functional"],
    )
    evidence = {
        "functional_plan": previous["metadata"]["authoring_plan"],
        "functional_evidence": functional,
        "functional_reader": current,
        "source_bindings": [
            *functional["source_bindings"],
            *publication.inference.original.snapshot(
                ORIGINAL / "inference",
                reconstruction.joint.bundle_names(publication.inference.TABLE_COUNTS),
            ),
        ],
        "publication_rules": publication.RULES,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    outcomes = read_json(ORIGINAL / "outcomes/evidence.json")
    contents = publication.generate(
        contract.primary,
        tables,
        outcomes["validation_selection"],
        outcomes["grid"]["runs"],
        functional["decisions"],
        evidence,
    )
    plan = reconstruction.publication_plan(contract, evidence)
    publication.save(work / "synthetic-publication", plan, contents)
    selected = {}
    for name, row in previous["files"].items():
        role, relative = name.split("/", 1)
        if role not in {"source", "training-source"}:
            files.select(selected, role, relative, ORIGINAL / name, row)
    source_selection(contract.inference, selected)
    accepted = archive.authoring.require_parent(contract.inference)
    files.select(
        selected, "source", archive.authoring.ACCEPTANCE[0], accepted, file_identity(accepted)
    )
    archive.add_publication_sources(contract, selected)
    for name in (*publication.OUTPUTS, "manifest.json"):
        path = work / "synthetic-publication" / name
        files.select(
            selected, "provenance", "primary-publication/" + name, path, file_identity(path)
        )
    metadata = {
        **previous["metadata"],
        archive.META: archive.extension(contract, plan, evidence),
        "publication_fixture": {
            "original_raw_archive_sha256": ANCHOR,
            "raw_input_bytes_reused": True,
            "upstream_primary_admission_simulated": True,
            "actual_primary_authoring_called": False,
            "scientific_completion": False,
        },
    }
    result = files.write(work / "archive", selected, metadata)
    recheck_selection(selected)
    require_same(files.inspect(ORIGINAL, ANCHOR), previous)
    write_new(
        work / "fixture.json",
        {
            "scope": "engineering_complete_synthetic_publication_fixture",
            "original_raw_archive": {"path": str(ORIGINAL), "sha256": ANCHOR},
            "archive": str(work / "archive"),
            **result,
            "all_original_raw_non_source_roles_retained": True,
            "upstream_primary_admission_simulated": True,
            "actual_checkpoint_backed_authoring_verified": False,
            "scientific_completion": False,
        },
    )
    return work / "archive", result["manifest_sha256"]
