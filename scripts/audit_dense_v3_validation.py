"""Admit the complete revised validation data, then run a declared one-GPU score diagnostic."""

import argparse
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.gpu_lease import acquire_gpu_lease
from embed_optim.primary_contract import (
    digest,
    file_identity,
    inspect_sealed_checkpoint,
    read_json,
    require_same,
    verify_file,
)
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation import SETTINGS, SOURCES, ValidationContract, row_identity
from embed_optim.primary_v3_validation_io import inspect_saved, write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_revised_natural import run_owned

PROTOCOL_SHA = "96a8aa2fc684d29ec67649c2e637c0cd4c7dcdd0f31f2e354b7d0955a18fb1c3"
NATURAL_SHA = "bf6e893e3b9e244242871227e28a3729acc17477dd3c989f7c647308bf332d36"
DATA_SOURCES = ("fiqa", "hotpotqa", "nq", "squadv2", "trivia", "msmarco", "fever")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or work.parent != Path("/tmp")
        or not work.name.startswith("dense-v3-validation.")
        or not work.is_dir()
        or work.is_symlink()
        or any(work.iterdir())
    ):
        raise ValueError("Require a fresh explicit CPU-parent diagnostic namespace")
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    protocol = root / "configs/dense_primary_v3_validation_protocol.json"
    if file_identity(protocol)["sha256"] != PROTOCOL_SHA:
        raise ValueError("The declared validation protocol differs")
    validation = ValidationContract.load(protocol, primary)
    amendment = read_json(root / "configs/dense_primary_v3_data_amendment.json")
    data_root = Path(amendment["datasets"]["validation"]["root"])
    dataset, admitted = validation.data(data_root)
    mandatory = sorted(amendment["datasets"]["validation"]["changed_sample_ids"])
    if len(mandatory) != 52 or len(set(mandatory)) != 52:
        raise ValueError("The diagnostic must include all 52 declared validation replacements")
    positions = set(mandatory)
    source_values = list(dataset["source"])
    for source in DATA_SOURCES:
        positions.add(
            next(i for i, name in enumerate(source_values) if name == source and i not in positions)
        )
    positions = sorted(positions)
    if len(positions) != 59:
        raise ValueError("The declared diagnostic coverage count differs")
    subset = dataset.select(positions)
    identities = [row_identity(subset[i], i) for i in range(len(subset))]
    subset.save_to_disk(str(work / "dataset"))
    from datasets import Dataset
    from transformers import AutoTokenizer

    restored = Dataset.load_from_disk(str(work / "dataset"))
    require_same(list(restored), list(subset))
    tokenizer = AutoTokenizer.from_pretrained(
        primary.inputs["initial_model"]["cached_root"], local_files_only=True
    )
    lengths = []
    for column in ("query", "positive", *(f"negative_{i}" for i in range(7))):
        prompt = "query: " if column == "query" else "document: "
        encoded = tokenizer(
            [prompt + r[column] for r in restored], truncation=True, max_length=8192, padding=False
        )
        lengths.extend(len(x) for x in encoded["input_ids"])
    selection = {
        "positions": positions,
        "mandatory_replacement_positions": mandatory,
        "rows": len(subset),
        "source_counts": dict(Counter(subset["source"])),
        "rule": "all 52 amended validation groups, plus first remaining row per source, sorted by original position",
        "row_identities": identities,
        "row_identities_sha256": digest(identities),
        "parent_validation_identity_sha256": admitted["row_identities_sha256"],
        "tokenized_texts": len(lengths),
        "max_truncated_token_length": max(lengths),
        "boundary": "Deterministic coverage, not proportional sampling, recipe selection or scientific validation results.",
    }
    natural_path = root / "reports/engineering-archive/dense-revised-natural-v1/natural-result.json"
    if file_identity(natural_path)["sha256"] != NATURAL_SHA:
        raise ValueError("The actual natural diagnostic producer differs")
    natural = read_json(natural_path)
    models, model_files = [], []
    for row in natural["records"]:
        checkpoint = Path(row["run_root"]) / "checkpoint-3"
        expected = read_json(checkpoint / "dense_run_contract.json")
        if digest(expected) != row["checked"]["run_identity_sha256"]:
            raise ValueError("An actual diagnostic model identity differs")
        checked = inspect_sealed_checkpoint(checkpoint, expected, 3)
        require_same(checked, row["checked"]["checkpoints"][-1])
        models.append(
            {"algorithm": row["algorithm"], "checkpoint": str(checkpoint), "checked": checked}
        )
        model_files.extend(
            {"path": str(checkpoint / f["path"]), **{k: f[k] for k in ("bytes", "sha256")}}
            for f in checked["files"]
        )
    if [m["algorithm"] for m in models] != ["adamw", "muon", "normuon"]:
        raise ValueError("The three diagnostic algorithm controls are incomplete")
    extra_sources = (
        "scripts/audit_dense_v3_validation.py",
        "scripts/run_dense_v3_validation_diagnostic.py",
        "src/embed_optim/functional_intervention.py",
        "src/embed_optim/gradient_probe.py",
        "src/embed_optim/probe_export.py",
        "src/embed_optim/primary_contract.py",
        "scripts/audit_dense_natural_data.py",
        "scripts/audit_dense_revised_natural.py",
        "scripts/pause_dense_evaluation_for_audit.py",
    )
    sources = [
        {"path": str(root / name), **file_identity(root / name)}
        for name in dict.fromkeys((*SOURCES, *extra_sources))
    ]
    inputs = [
        *model_files,
        *amendment["datasets"]["validation"]["files"],
        {"path": str(protocol), **file_identity(protocol)},
        {"path": str(natural_path), **file_identity(natural_path)},
        *[
            {"path": str(p), **file_identity(p)}
            for p in sorted((work / "dataset").rglob("*"))
            if p.is_file()
        ],
    ]
    plan = {
        "scope": "engineering_dense_v3_validation_diagnostic_plan",
        "repository": str(root),
        "settings": SETTINGS,
        "selection": selection,
        "complete_validation_admission": admitted,
        "models": models,
        "source_bindings": sources,
        "input_bindings": inputs,
        "pre_execution_dispatchers": handoff(),
        "scientific_completion": False,
        "formal_validation_executed": False,
        "model_updates": 0,
    }
    write_new(work / "plan.json", plan)
    command = [
        sys.executable,
        "-B",
        str(root / "scripts/run_dense_v3_validation_diagnostic.py"),
        "--plan",
        str(work / "plan.json"),
        "--plan-sha256",
        file_identity(work / "plan.json")["sha256"],
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "4",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{root / 'src'}:{root}",
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    print(
        {
            "stage": "start_owned_gpu_score_comparison",
            "rows": 59,
            "models": 3,
            "max_length_observed": max(lengths),
            "scientific_completion": False,
        },
        flush=True,
    )
    with acquire_gpu_lease(
        ("4",),
        lock_dir="/tmp/embedding-optimizer-primary-gpu-leases",
        timeout_seconds=30,
        purpose="engineering-v3-validation-scorer",
        ledger_path=work / "lease.json",
    ):
        worker_code = run_owned(command, root, env, work)
    worker, checked, failure = None, [], None
    try:
        if worker_code:
            raise ValueError(f"Actual diagnostic worker exited {worker_code}")
        worker = read_json(work / "worker-result.json")
        if worker.get("scoring_comparison_passed") is not True or len(worker["records"]) != 3:
            raise ValueError("Incomplete actual GPU score comparison")
        for row in worker["records"]:
            checked.append(inspect_saved(row["output"], row["plan"], identities))
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    for row in sources + inputs:
        verify_file(row["path"], row)
    value = {
        "scope": "engineering_dense_v3_validation_scorer_readiness",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "readiness_passed": failure is None,
        "failure": failure,
        "worker_returncode": worker_code,
        "worker": worker,
        "cpu_readback": checked,
        "plan": {"path": str(work / "plan.json"), **file_identity(work / "plan.json")},
        "post_execution_dispatchers": handoff(),
        "model_updates": 0,
        "formal_validation_executed": False,
        "scientific_completion": False,
        "formal_replication_ready": False,
        "artifacts": [
            {"path": str(p), **file_identity(p)} for p in sorted(work.rglob("*")) if p.is_file()
        ],
    }
    write_new(work / "result.json", value)
    print(
        {
            "readiness_passed": failure is None,
            "worker_returncode": worker_code,
            "diagnostic_models": len(checked),
            "formal_validation_executed": False,
            **file_identity(work / "result.json"),
        },
        flush=True,
    )
    if failure:
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
