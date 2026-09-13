"""Bounded real-data readiness check, never a primary training release.

The parent authenticates the fixed 500k data and original base, materializes an
unchosen 288-row prefix, leases four devices, and launches the unchanged prepared
training entrypoint. No old checkpoint, source, dispatcher or primary output is
modified. --verify only checks an existing, explicitly trusted diagnostic receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.gpu_lease import acquire_gpu_lease
from embed_optim.primary_completion import inspect_complete_run, number
from embed_optim.primary_contract import (
    INPUT_SHA,
    digest,
    file_identity,
    read_json,
    require_same,
    verify_file,
)
from scripts.pause_dense_evaluation_for_audit import CHAIN, LEDGER, LEDGER_SHA
from scripts.pause_dense_evaluation_for_audit import inspect as inspect_process

SOURCE_SHA = "1c6163c5aeee9d0735a6859c0e2f56a701c59d5111b553de6fb5c758da138dfd"
INPUT_PATH = "reports/dense-primary-v2/input-bindings.json"
MANIFEST_PATH = "reports/engineering-archive/dense-full-identity-v1/prepared-source.json"
ALGORITHMS = ("adamw", "muon", "normuon")
TEXT_COLUMNS = ("query", "positive", *(f"negative_{i}" for i in range(7)))
SOURCE_NAMES = (
    "scripts/audit_dense_natural_data.py",
    "scripts/run_dense_natural_diagnostic.py",
    "scripts/audit_dense_identity_entrypoint.py",
    "scripts/audit_prepared_dense_gpu.py",
    "scripts/pause_dense_evaluation_for_audit.py",
    "src/embed_optim/primary_completion.py",
    "src/embed_optim/primary_contract.py",
    "src/embed_optim/primary_io.py",
    "src/embed_optim/gpu_lease.py",
    "configs/formal_runtime.json",
)


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def handoff():
    verify_file(LEDGER, {"bytes": LEDGER.stat().st_size, "sha256": LEDGER_SHA})
    rows = [
        inspect_process(pid, start, name, CHAIN[i - 1][0] if i else 1)
        for i, (pid, start, name) in enumerate(CHAIN)
    ]
    if any(row["state"] != "T" for row in rows):
        raise ValueError("Original evaluation dispatcher is not stopped")
    return rows


def input_files(root):
    path = root / INPUT_PATH
    if file_identity(path)["sha256"] != INPUT_SHA:
        raise ValueError("The trusted primary input receipt changed")
    inputs = read_json(path)
    data = Path(inputs["data_linkage_audit"]["dataset_path"])
    cached = Path(inputs["initial_model"]["cached_root"])
    files = []
    for directory, entries, links in (
        (data / "dataset", inputs["common_identity"]["data"]["files"], False),
        (cached, inputs["common_identity"]["model"]["files"], True),
    ):
        actual = sorted(
            p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
        )
        if actual != [row["path"] for row in entries] or directory.is_symlink():
            raise ValueError("The original data/base inventory differs")
        for row in entries:
            source = directory / row["path"]
            if source.is_symlink():
                if (
                    not links
                    or source.resolve().parent != (directory.parent.parent / "blobs").resolve()
                ):
                    raise ValueError("Untrusted cache link")
            resolved = source.resolve() if links else source
            verify_file(resolved, row)
            files.append({"path": str(resolved), **file_identity(resolved)})
    for key in ("data_manifest", "row_manifest"):
        row = inputs[key]
        verify_file(Path(row["path"]), row)
        files.append(row)
    return inputs, files


def select_prefix(dataset, count=288):
    """An outcome-independent prefix, preserving every existing negative verbatim."""
    if count != 288 or len(dataset) < count:
        raise ValueError("Require the predeclared 288-row readiness prefix")
    subset = dataset.select(range(count))
    rows = list(subset)
    for row in rows:
        if any(not isinstance(row.get(c), str) or not row[c].strip() for c in TEXT_COLUMNS):
            raise ValueError("A diagnostic row lacks one of its nine original texts")
        if type(row.get("length")) is not int or row["length"] <= 0:
            raise ValueError("The original length feature is missing or invalid")
    return subset, {
        "selection": "original_dataset_indices_0_through_287_no_outcome_selection",
        "original_row_count": len(dataset),
        "selected_row_count": len(rows),
        "selected_rows_sha256": digest(rows),
        "row_sha256": [digest(row) for row in rows],
        "selected_columns": list(subset.column_names),
        "micro_batches_per_rank": 9,
        "expected_global_query_groups": [128, 128, 32],
    }


def logged_observations(run_root):
    history = read_json(run_root / "trainer_state_final.json")["log_history"]
    first = [r for r in history if r.get("step") == 1 and "grad_norm" in r and "loss" in r]
    if len(first) != 1:
        raise ValueError("The actual first-step loss/gradient-norm log is missing or duplicated")
    row = first[0]
    if number(row["loss"]) < 0 or number(row["grad_norm"], positive=True) <= 0:
        raise ValueError("Invalid first-step training observation")
    if number(row["learning_rate"]) != 0:
        raise ValueError("The declared one-step warmup does not start at zero LR")
    return row


def inspect_outputs(work, prepared):
    producer = read_json(work / "worker-result.json")
    if producer.get("natural_entrypoint_passed") is not True or len(producer["records"]) != 3:
        raise ValueError("The actual three-algorithm training worker did not finish")
    records = []
    for algorithm, row in zip(ALGORITHMS, producer["records"], strict=True):
        run_root = work / "runs/dense" / f"diagnostic-natural-{algorithm}"
        expected = prepared["expected_runs"][algorithm]
        if row["algorithm"] != algorithm or row["identity_sha256"] != digest(expected):
            raise ValueError("Actual producer differs from its prepared natural-data identity")
        require_same(expected["model"], prepared["original_model_identity"])
        if (
            expected["data"]["rows"] != 288
            or expected["recipe"]["model_name"] != "lightonai/DenseOn-unsupervised"
        ):
            raise ValueError("Wrong natural-data/base diagnostic")
        checked = inspect_complete_run(run_root, expected, [1, 2, 3])
        records.append(
            {
                "algorithm": algorithm,
                "run_root": str(run_root),
                "checked": checked,
                "first_step": logged_observations(run_root),
            }
        )
    return records


def verify_receipt(path, sha):
    if file_identity(path)["sha256"] != sha:
        raise ValueError("Existing diagnostic receipt is not the requested trusted digest")
    result = read_json(path)
    if (
        result.get("natural_data_readiness_passed") is not True
        or result.get("scientific_completion") is not False
    ):
        raise ValueError("Receipt is not a successful bounded diagnostic")
    for row in result["source_bindings"] + result["input_bindings"] + result["artifacts"]:
        verify_file(Path(row["path"]), row)
    work = Path(result["workdir"])
    prepared = read_json(work / "prepared.json")
    records = inspect_outputs(work, prepared)
    require_same(records, result["records"])
    handoff()
    return {
        "verified": True,
        "natural_runs": len(records),
        "scientific_completion": False,
        "receipt": file_identity(path),
    }


def launch(root, work):
    if (
        work.parent != Path("/tmp")
        or not work.name.startswith("dense-natural-readiness.")
        or work.is_symlink()
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError("Require a fresh ordinary named mktemp diagnostic namespace")
    sources = [{"path": str(root / name), **file_identity(root / name)} for name in SOURCE_NAMES]
    manifest_path = root / MANIFEST_PATH
    if file_identity(manifest_path)["sha256"] != SOURCE_SHA:
        raise ValueError("Prepared candidate source manifest differs")
    manifest = read_json(manifest_path)
    candidate = Path(manifest["candidate_root"])
    for row in manifest["candidate_sources"]:
        verify_file(candidate / row["relative_path"], row["identity"])
        sources.append(row["identity"])
    sources.append({"path": str(manifest_path), **file_identity(manifest_path)})
    inputs, bindings = input_files(root)
    bindings.append({"path": str(root / INPUT_PATH), **file_identity(root / INPUT_PATH)})
    original_dispatchers = handoff()
    from datasets import Dataset
    from transformers import AutoTokenizer

    original = Dataset.load_from_disk(
        str(Path(inputs["data_linkage_audit"]["dataset_path"]) / "dataset")
    )
    if len(original) != 500000:
        raise ValueError("The authenticated original dataset is not 500k rows")
    subset, selection = select_prefix(original)
    subset.save_to_disk(str(work / "dataset"))
    restored = Dataset.load_from_disk(str(work / "dataset"))
    if digest(list(restored)) != selection["selected_rows_sha256"]:
        raise ValueError("Diagnostic materialization changed an original row")
    tokenizer = AutoTokenizer.from_pretrained(
        inputs["initial_model"]["cached_root"], local_files_only=True
    )
    lengths = []
    for column in TEXT_COLUMNS:
        prefix = "query: " if column == "query" else "document: "
        encoded = tokenizer(
            [prefix + r[column] for r in restored], truncation=True, max_length=8192, padding=False
        )
        lengths.extend(len(row) for row in encoded["input_ids"])
    selection["tokenized_texts"] = len(lengths)
    selection["observed_max_truncated_token_length"] = max(lengths)
    selection["observed_min_token_length"] = min(lengths)
    write_new(work / "selection.json", selection)
    command = [
        sys.executable,
        "-B",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=4",
        "--log-dir",
        str(work / "rank-logs"),
        "--redirects",
        "3",
        "--tee",
        "0",
        str(root / "scripts/run_dense_natural_diagnostic.py"),
        "--candidate-root",
        str(candidate),
        "--source-manifest",
        str(manifest_path),
        "--source-sha256",
        SOURCE_SHA,
        "--workdir",
        str(work),
        "--input-receipt",
        str(root / INPUT_PATH),
        "--input-sha256",
        INPUT_SHA,
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": f"{candidate / 'src'}:{root}:{candidate}",
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "EMBED_OPTIM_MAX_STEPS": "3",
        "EMBED_OPTIM_STOP_AFTER_STEP": "-1",
        "OPENBLAS_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    print(
        json.dumps(
            {
                "stage": "actual_natural_data_launch",
                "rows": 288,
                "algorithms": list(ALGORITHMS),
                "selection": selection["selected_rows_sha256"],
                "max_observed_tokens": max(lengths),
                "scientific_completion": False,
            }
        ),
        flush=True,
    )
    result = {
        "scope": "engineering_dense_natural_data_readiness",
        "workdir": str(work),
        "natural_data_readiness_passed": False,
        "scientific_completion": False,
        "production_deployed": False,
        "primary_training_executed": False,
        "source_published": False,
        "original_dispatchers": original_dispatchers,
        "source_bindings": sources,
        "input_bindings": bindings,
        "selection": selection,
        "command": command,
        "records": [],
    }
    try:
        with acquire_gpu_lease(
            ("4", "5", "6", "7"),
            lock_dir="/tmp/embedding-optimizer-primary-gpu-leases",
            timeout_seconds=30,
            purpose="engineering-dense-natural-readiness",
            ledger_path=work / "lease.json",
        ):
            handoff()
            with (work / "launcher.log").open("x") as log:
                worker = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
        result["worker_returncode"] = worker.returncode
        if worker.returncode:
            raise RuntimeError(f"Actual natural-data diagnostic worker exited {worker.returncode}")
        result["records"] = inspect_outputs(work, read_json(work / "prepared.json"))
        for row in sources + bindings:
            verify_file(Path(row["path"]), row)
        result["post_execution_dispatchers"] = handoff()
        result["natural_data_readiness_passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["observed_at_utc"] = datetime.now(UTC).isoformat()
        result["artifacts"] = [
            {"path": str(p), **file_identity(p)} for p in sorted(work.rglob("*")) if p.is_file()
        ]
        result["boundary"] = (
            "Three unpatched prepared run_training calls start from the immutable original untrained DenseOn base and the same fixed 288-row prefix of the authenticated 500k dataset. Three scheduled diagnostic steps preserve 128/128/32 expected global groups, BF16/FA2, four ranks, eight workers/rank and explicit negatives. First-step logged gradient norm/loss and all completed checkpoint states are checked; this is not an independent gradient oracle on every natural batch, a 500k horizon, all learning rates, primary CLI release, cross-host replay, evaluation or scientific finding. All changes are diagnostic namespaces and declared short recipe identity; no formal lock or source is altered. W&B disabled."
        )
        write_new(work / "result.json", result)
    print(
        json.dumps(
            {
                "passed": result["natural_data_readiness_passed"],
                "error": result.get("error"),
                "receipt": str(work / "result.json"),
                **file_identity(work / "result.json"),
            }
        ),
        flush=True,
    )
    if not result["natural_data_readiness_passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Parent/verification process must be explicitly CPU-only")
    if args.verify:
        if not args.receipt_sha256 or args.workdir or args.repository:
            parser.error("Verification requires only an existing receipt and its trusted SHA")
        print(json.dumps(verify_receipt(args.verify, args.receipt_sha256)))
    else:
        if not args.repository or not args.workdir or args.receipt_sha256:
            parser.error("Launch requires repository and fresh workdir")
        launch(args.repository.resolve(), args.workdir.absolute())


if __name__ == "__main__":
    main()
