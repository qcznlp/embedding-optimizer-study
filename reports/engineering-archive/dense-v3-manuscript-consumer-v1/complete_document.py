"""Exercise only the full document component on an explicitly synthetic paper copy."""

import argparse
import os
from pathlib import Path

from embed_optim import primary_v3_manuscript as document
from embed_optim import primary_v3_manuscript_build as build
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.state_operator_factorial_publication import ESTIMANDS, _render_latex
from scripts.audit_dense_natural_data import handoff


def run(args):
    root = document.ordinary(args.repository, directory=True)
    work = document.ordinary(args.workdir, directory=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or list(work.iterdir()):
        raise ValueError("Require a new empty CPU-only synthetic document directory")
    before = handoff()
    verify_file(
        args.generation, {"bytes": args.generation.stat().st_size, "sha256": args.generation_sha256}
    )
    generation = read_json(args.generation)
    if (
        generation["passed"] is not True
        or generation["upstream_primary_admission_simulated"] is not True
        or generation["actual_checkpoint_backed_positive_authoring_verified"] is not False
    ):
        raise ValueError("Require the named complete synthetic publication fixture")
    for row in generation["sources"]:
        verify_file(row["path"], row)
    originals = {
        str(root / "paper" / name): file_identity(document.ordinary(root / "paper" / name))
        for name in build.BUILD_INPUTS
    }
    sources = {
        str(path): file_identity(path)
        for path in (
            Path(__file__).resolve(),
            Path(document.__file__).resolve(),
            Path(build.__file__).resolve(),
            root / "tests/test_primary_v3_manuscript.py",
            root / "tests/test_primary_v3_manuscript_build.py",
            root / "src/embed_optim/state_operator_factorial_publication.py",
        )
    }
    candidate = root / "reports/engineering-archive/dense-full-identity-v1/candidate-source"
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, candidate
    )
    constants = document.render_constants(primary)
    records = {row["path"]: row for row in generation["generated_artifacts"]}
    includes = {}
    for name in document.RESULT_FILES[:2]:
        path = args.generation.parent / "synthetic-evidence" / Path(name).name
        verify_file(path, records[str(path)])
        includes[name] = path.read_bytes()
    # Pure presentation inputs; these are NOT a factorial summary or numerical acceptance.
    estimands = {
        name: {
            "point_estimate": 0.01,
            "bootstrap_ci_95_lower": 0.001,
            "bootstrap_ci_95_upper": 0.02,
            "decision": "supported_positive",
        }
        for name in ESTIMANDS
    }
    includes[document.RESULT_FILES[2]] = _render_latex(estimands).encode()
    source = work / "synthetic-source"
    for name in build.BUILD_INPUTS:
        destination = source / "paper" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(
                {"results.tex": constants, **includes}.get(
                    name, (root / "paper" / name).read_bytes()
                )
            )
    main = source / "paper/main.tex"
    text = main.read_text()
    if text.count(r"\maketitle") != 1:
        raise ValueError("Unexpected complete-paper title topology")
    main.write_text(
        text.replace(
            r"\maketitle",
            r"\maketitle\begin{center}\textbf{SYNTHETIC DOCUMENT TEST -- NOT EXPERIMENT RESULTS}\end{center}",
        )
    )
    checked = build.compile_document(source, includes, constants, work / "fresh")
    require_same(checked["source_inputs"]["main.tex"], file_identity(main))
    for path, identity in {**originals, **sources}.items():
        verify_file(path, identity)
    for row in generation["sources"]:
        verify_file(row["path"], row)
    verify_file(
        args.generation, {"bytes": args.generation.stat().st_size, "sha256": args.generation_sha256}
    )
    require_same(handoff(), before)
    receipt = {
        "scope": "engineering_complete_synthetic_v3_document_component",
        "passed": True,
        "document": checked,
        "fixture_primary_and_dimension_source": {
            "path": str(args.generation),
            **file_identity(args.generation),
        },
        "fixture_factorial_presentation_values": estimands,
        "fixture_admission": "No primary or factorial checkpoint admission was performed by this document test",
        "actual_primary_results": False,
        "strict_complete_manuscript_consumer_integrated": False,
        "manuscript_installed": False,
        "scientific_completion": False,
        "original_paper_inputs": originals,
        "sources": sources,
        "post_check_dispatchers": before,
    }
    write_new(work / "result.json", receipt)
    print(
        {
            "passed": True,
            "scope": receipt["scope"],
            "main_end_page": checked["layout"]["main_end_page"],
            "actual_abstract_words": checked["document"]["actual_abstract"]["words_conservative"],
            "scientific_completion": False,
            **file_identity(work / "result.json"),
        },
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "workdir", "generation"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generation-sha256", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
