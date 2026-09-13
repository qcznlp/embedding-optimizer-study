"""Independent functional inference/replay audit, explicitly without primary findings."""

import argparse
import copy
import json
import os
import runpy
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from embed_optim.dimension_inference import summarize
from embed_optim.dimension_inference_render import render
from embed_optim.primary_contract import digest, file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import (
    PARENTS,
    SOURCES,
    FunctionalInferenceContract,
    gather,
)
from embed_optim.primary_v3_outcomes import csv_bytes, inspect_bundle, save_bundle
from embed_optim.primary_v3_validation_io import write_new
from embed_optim.runtime import runtime_snapshot
from scripts.audit_dense_natural_data import handoff
from scripts.dimension_inference_reference import exact_predictions, task_inference

EXTRA_SOURCES = (
    "scripts/audit_dense_v3_dimension_inference.py",
    "scripts/dimension_inference_reference.py",
    "tests/test_dimension_inference.py",
    "tests/test_primary_v3_dimension_inference.py",
    "tests/test_primary_v3_bridge.py",
)


def binding(path):
    return {"path": str(Path(path).resolve()), **file_identity(path)}


def numerical(primary, inputs):
    return summarize(
        primary, inputs["tables"], inputs["original_panel"], inputs["protocol"], inputs["tasks"]
    )


def altered_bundles(work, source, plan, evidence, tables):
    results = []
    for name in (
        "primary_contrasts",
        "rotation_contrasts",
        "figure_points",
        "held_out_predictions",
        "feature_prediction_summary",
        "rendered_text",
    ):
        target = work / ("altered-" + name)
        shutil.copytree(source, target)
        if name == "rendered_text":
            changed = copy.deepcopy(evidence)
            changed["rendered_latex"] += "\nUnsupported conclusion.\n"
            filename, content = "evidence.json", json.dumps(changed, sort_keys=True).encode()
        else:
            changed = copy.deepcopy(tables[name])
            key = next(key for key, value in changed[0].items() if type(value) is float)
            changed[0][key] += 0.001
            filename, content = name + ".csv", csv_bytes(changed)
        (target / filename).write_bytes(content)
        manifest = read_json(target / "manifest.json")
        manifest["outputs"][filename] = {"path": filename, **file_identity(target / filename)}
        (target / "manifest.json").write_text(json.dumps(manifest))
        try:
            inspect_bundle(target, plan, evidence, tables)
        except ValueError as error:
            results.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise AssertionError("Rehashed altered inference was accepted")
    return results


def compile_text(work, text):
    document = work / "synthetic-layout.tex"
    document.write_text(
        "\\documentclass[11pt]{article}\n\\usepackage[margin=0.8in]{geometry}\n"
        "\\usepackage{times}\n\\usepackage{booktabs}\n\\usepackage{pgfplots}\n"
        "\\usepgfplotslibrary{groupplots}\n\\pgfplotsset{compat=1.18}\n"
        + text
        + "\n\\begin{document}\n\\section*{Synthetic engineering fixture: not scientific results}\n"
        "\\DimensionUtilizationFinding\n\n\\DimensionRetrievalBridgeFinding\n\n"
        "\\DimensionConclusionFinding\n\\DimensionUtilizationFigure\n"
        "\\DimensionUtilizationAppendixTable\n\\end{document}\n"
    )
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(work),
        str(document),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (work / "latex-stdout.txt").write_text(result.stdout)
    (work / "latex-stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout[-4000:]
    fonts = subprocess.run(
        ["pdffonts", str(work / "synthetic-layout.pdf")], capture_output=True, text=True, timeout=60
    )
    assert fonts.returncode == 0 and "Type 3" not in fonts.stdout
    (work / "fonts.txt").write_text(fonts.stdout)
    return {
        "command": command,
        "exit_code": result.returncode,
        "fonts_exit_code": fonts.returncode,
        "type_3_fonts": False,
        "document": binding(document),
        "full_naacl_layout_acceptance": False,
    }


def cli_refusals(args, work):
    records = []
    for mode in ("build", "inspect"):
        command = [sys.executable, "-m", "embed_optim.primary_v3_dimension_inference", mode]
        supplied = {
            "repository": args.repository,
            "training-root": args.training_root,
            "protocol": args.protocol,
            "experiment-root": work / "absent-primary",
            "output": work / ("forbidden-" + mode),
            "vector-manifest-sha256": "0" * 64,
        }
        for name in (
            "results-root",
            "validation-data",
            "validation-root",
            "outcomes-root",
            "geometry-root",
            "reference",
            "bridge-root",
            "dimension-vectors",
            "dimension-features",
            "probe-root",
        ):
            supplied[name] = work / "absent-later-input"
        for name, value in supplied.items():
            command.extend(["--" + name, str(value)])
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=60, cwd=args.repository
        )
        assert result.returncode == 1 and "ordinary retained run" in result.stderr
        assert not supplied["output"].exists()
        records.append(
            {
                "mode": mode,
                "command": command,
                "exit_code": result.returncode,
                "output_absent": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return records


def run(args):
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError("Require CPU-only assertions and a new empty audit directory")
    primary = PrimaryV3Contract.load(root / PARENTS["primary"][0], root, args.training_root)
    contract = FunctionalInferenceContract.load(args.protocol, primary)
    assert contract.sha256 == args.expected_protocol_sha256
    sources = [binding(root / name) for name in (*SOURCES, *EXTRA_SOURCES)]
    before = handoff()
    parent = None
    if args.replay:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        parent = read_json(args.replay)
        require_same(parent["sources"], sources)
        for row in parent["artifacts"]:
            verify_file(row["path"], row)
        # Copy both fixture and bundle before reading; no old-root input resolver is used.
        shutil.copy2(args.replay.parent / "fixture-input.json", work / "fixture-input.json")
        shutil.copytree(args.replay.parent / "bundle", work / "relocated-bundle")
        inputs = read_json(work / "fixture-input.json")
    else:
        producer = runpy.run_path(str(root / "tests/test_dimension_inference.py"))["fixture"]
        tables, panel, protocol, tasks = producer(primary)
        inputs = {
            "scope": "engineering_synthetic_complete_functional_panel",
            "tables": tables,
            "original_panel": panel,
            "protocol": protocol,
            "tasks": tasks,
            "scientific_completion": False,
        }
        write_new(work / "fixture-input.json", inputs)
    try:
        gather(contract, SimpleNamespace(experiment_root=work / "absent-primary"))
    except ValueError as error:
        assert "ordinary retained run" in str(error)
        primary_refusal = str(error)
    else:
        raise AssertionError("A nonexistent primary population was admitted")
    tables, decisions = numerical(primary, inputs)
    task_reference = task_inference(inputs, tables, decisions)
    bridge_reference = exact_predictions(tables)
    evidence = {
        "scope": inputs["scope"],
        "input_sha256": digest(inputs),
        "decisions": decisions,
        "rendered_latex": render(tables, decisions),
        "scientific_completion": False,
    }
    plan = {
        "scope": "engineering_synthetic_functional_inference_replay",
        "protocol_sha256": contract.sha256,
        "scientific_completion": False,
    }
    if parent:
        inspected = inspect_bundle(work / "relocated-bundle", plan, evidence, tables)
        require_same(task_reference, parent["task_reference"])
        require_same(bridge_reference, parent["bridge_reference"])
    else:
        inspected = None
    saved = save_bundle(work / "bundle", plan, evidence, tables)
    altered = altered_bundles(work, work / "bundle", plan, evidence, tables)
    layout = compile_text(work, evidence["rendered_latex"])
    cli = cli_refusals(args, work)
    for row in sources:
        verify_file(row["path"], row)
    contract.recheck()
    require_same(handoff(), before)
    artifacts = [binding(path) for path in sorted(work.rglob("*")) if path.is_file()]
    result = {
        "scope": "engineering_dense_v3_functional_inference_audit",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "fresh_replay": parent is not None,
        "replay_parent": None if args.replay is None else binding(args.replay),
        "sources": sources,
        "protocol": binding(args.protocol),
        "actual_runtime": runtime_snapshot(["torch", "numpy", "scipy", "sympy"]),
        "task_reference": task_reference,
        "bridge_reference": bridge_reference,
        "actual_primary_refusal": primary_refusal,
        "actual_cli_refusals": cli,
        "synthetic_bundle": saved,
        "relocated_replay": inspected,
        "altered_cases": altered,
        "synthetic_layout": layout,
        "artifacts": artifacts,
        "post_execution_dispatchers": handoff(),
        "raw_primary_vectors_used": False,
        "primary_runs_admitted": 0,
        "upstream_primary_sources_independently_reconstructed": False,
        "full_primary_publication_pipeline_verified": False,
        "manuscript_installed": False,
        "scientific_completion": False,
    }
    write_new(work / "result.json", result)
    print(
        {"passed": True, "scientific_completion": False, **file_identity(work / "result.json")},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "protocol", "workdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
