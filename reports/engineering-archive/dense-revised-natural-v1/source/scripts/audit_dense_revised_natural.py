"""Three real-data diagnostic runs under an explicit data amendment, never a primary launch."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from embed_optim.gpu_lease import acquire_gpu_lease
from embed_optim.primary_contract import digest, file_identity, read_json, verify_file
from embed_optim.primary_data_revision import TEXT_COLUMNS, load_amendment, select_natural_coverage
from scripts.audit_dense_natural_data import (
    MANIFEST_PATH,
    SOURCE_SHA,
    handoff,
    input_files,
    inspect_outputs,
    write_new,
)


def run_owned(command, root, environment, work):
    """A new isolated process group contains only this diagnostic's owned descendants."""
    with (work / "launcher.log").open("x") as log:
        worker = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        start = Path(f"/proc/{worker.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        if os.getpgid(worker.pid) != worker.pid:
            raise RuntimeError("Diagnostic launcher is not its own isolated group")
        write_new(
            work / "launcher.json",
            {
                "pid": worker.pid,
                "start_ticks": int(start),
                "process_group": worker.pid,
                "command": command,
            },
        )
        try:
            return worker.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            if worker.poll() is None:
                current = Path(f"/proc/{worker.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                if current != start or os.getpgid(worker.pid) != worker.pid:
                    raise RuntimeError("Owned launcher identity changed; refuse a signal")
                # Torch elastic owns separate rank process groups. Ask its verified
                # launcher to shut them down and join them; do not kill the launcher
                # group and assume that separate worker groups also exited.
                worker.terminate()
                while worker.poll() is None:
                    try:
                        worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(
                            "Waiting for owned torchrun shutdown; GPU lease remains held",
                            flush=True,
                        )
            raise RuntimeError("Owned diagnostic exceeded its declared 1800-second limit")


def launch(root, amendment_path, work):
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or work.parent != Path("/tmp")
        or not work.name.startswith("dense-natural-readiness.")
        or not work.is_dir()
        or work.is_symlink()
        or any(work.iterdir())
    ):
        raise ValueError("Require CPU-only parent and a fresh natural diagnostic namespace")
    amendment, inputs = load_amendment(amendment_path, root)
    _, original_bindings = input_files(root)
    sources = [
        {"path": str(root / name), **binding}
        for name, binding in amendment["source_bindings"].items()
    ]
    sources += [{"path": str(amendment_path), **file_identity(amendment_path)}]
    source_manifest = root / MANIFEST_PATH
    verify_file(source_manifest, inputs["source_manifest"])
    manifest = read_json(source_manifest)
    candidate = Path(manifest["candidate_root"])
    for row in manifest["candidate_sources"]:
        verify_file(candidate / row["relative_path"], row["identity"])
        sources.append(row["identity"])
    sources.append({"path": str(source_manifest), **file_identity(source_manifest)})
    bindings = original_bindings + [r for p in amendment["datasets"].values() for r in p["files"]]
    input_path = root / amendment["inputs"]["path"]
    bindings.append({"path": str(input_path), **file_identity(input_path)})
    dispatchers = handoff()
    original = Dataset.load_from_disk(
        str(Path(amendment["datasets"]["training"]["root"]) / "dataset")
    )
    subset, selection = select_natural_coverage(
        original, amendment["datasets"]["training"]["changed_sample_ids"]
    )
    subset.save_to_disk(str(work / "dataset"))
    restored = Dataset.load_from_disk(str(work / "dataset"))
    if digest(list(restored)) != selection["selected_rows_sha256"]:
        raise ValueError("Diagnostic serialization changed a selected value")
    tokenizer = AutoTokenizer.from_pretrained(
        inputs["initial_model"]["cached_root"], local_files_only=True
    )
    lengths = []
    for column in TEXT_COLUMNS:
        prefix = "query: " if column == "query" else "document: "
        encoded = tokenizer(
            [prefix + row[column] for row in restored],
            truncation=True,
            max_length=8192,
            padding=False,
        )
        lengths.extend(len(row) for row in encoded["input_ids"])
    selection.update(
        tokenized_texts=len(lengths),
        observed_max_truncated_token_length=max(lengths),
        observed_min_token_length=min(lengths),
        amendment_sha256=file_identity(amendment_path)["sha256"],
    )
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
        str(source_manifest),
        "--source-sha256",
        SOURCE_SHA,
        "--workdir",
        str(work),
        "--input-receipt",
        str(input_path),
        "--input-sha256",
        amendment["inputs"]["sha256"],
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
    result = {
        "scope": "engineering_dense_revised_natural_readiness_v3",
        "workdir": str(work),
        "scientific_completion": False,
        "primary_training_executed": False,
        "production_deployed": False,
        "source_published": False,
        "natural_data_readiness_passed": False,
        "source_bindings": sources,
        "input_bindings": bindings,
        "original_dispatchers": dispatchers,
        "selection": selection,
        "command": command,
        "records": [],
    }
    print(
        json.dumps(
            {
                "stage": "declared_revised_natural_launch",
                "source_quotas": selection["source_quotas"],
                "mandatory_replacements": selection["mandatory_replacement_positions"],
                "max_tokens": max(lengths),
                "formal_training": False,
            }
        ),
        flush=True,
    )
    try:
        with acquire_gpu_lease(
            ("4", "5", "6", "7"),
            lock_dir="/tmp/embedding-optimizer-primary-gpu-leases",
            timeout_seconds=30,
            purpose="engineering-revised-natural-readiness",
            ledger_path=work / "lease.json",
        ):
            handoff()
            result["worker_returncode"] = run_owned(command, root, environment, work)
        if result["worker_returncode"] != 0:
            raise RuntimeError(f"Actual natural-data worker exited {result['worker_returncode']}")
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
            "Actual unpatched prepared run_training, immutable untrained base, same 288 source-balanced natural groups including all six revisions, three algorithms/steps and full diagnostic checkpoint inspection. Not a full epoch, every LR, an independent natural-gradient oracle, cross-host resume, formal protocol acceptance, or optimizer-quality result. The old failed prefix remains unchanged; these are declared new inputs and new outputs, with W&B disabled."
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
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    launch(args.repository.resolve(), args.amendment.resolve(), args.workdir.absolute())


if __name__ == "__main__":
    main()
