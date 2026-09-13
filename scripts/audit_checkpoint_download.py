"""Verify a pinned HF checkpoint download and exercise CPU-only restoration.

This tool never downloads, uploads, repairs files, or starts training. Download with
the HF client first. The independently trusted audit SHA-256 is mandatory; pickle
payloads are not opened until every selected file passes content verification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from embed_optim.artifact_inventory import compare_inventories, file_inventory

REPO_ID = "qcz/embedding-optimizer-study-checkpoints"
PREFIX = "corrected-dense-no-packing-v1/dense"
STEPS = (782, 1563, 2345, 3126, 3907)
REQUIRED = {
    "config.json",
    "model.safetensors",
    "modules.json",
    "1_Pooling/config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
    "tokenizer.json",
    "tokenizer_config.json",
}


def _sha256(path: Path) -> str:
    return file_inventory(path)["sha256"]


def _relative(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or str(path) != name
        or any(part in {".", ".."} or part.startswith(".") for part in path.parts)
        or "\\" in name
        or "\0" in name
    ):
        raise ValueError("Unsafe inventory path")
    return name


def select_download(audit: Path, trusted_sha256: str, run_id: str, step: int) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", trusted_sha256):
        raise ValueError("A trusted audit SHA-256 is required")
    raw = audit.read_bytes()
    if hashlib.sha256(raw).hexdigest() != trusted_sha256:
        raise ValueError("Audit content differs from the trusted SHA-256")
    data = json.loads(raw)
    if (
        data.get("schema_version") != 1
        or data.get("audit_scope") != "immutable remote file content versus completed local runs"
        or data.get("complete_for_audited_runs") is not True
        or data.get("scientific_completion") is not False
    ):
        raise ValueError("Require a completed content audit, not a study-completion claim")
    if not re.fullmatch(r"padded-(adamw|muon|normuon)-[13]e-[1-6]", run_id):
        raise ValueError("Require a current primary run ID")
    if type(step) is not int or step not in STEPS:
        raise ValueError("Require a scheduled checkpoint step")
    matches = [r for r in data["records"] if r["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("Run must occur exactly once in the trusted audit")
    record = matches[0]
    prefix = f"{PREFIX}/{run_id}"
    if (
        record["repo_id"] != REPO_ID
        or record["prefix"] != prefix
        or not re.fullmatch(r"[0-9a-f]{40}", record["revision"])
    ):
        raise ValueError("Unexpected repository, prefix or immutable revision")
    local, remote = record["local_inventory"], record["remote_inventory"]
    for name in set(local) | set(remote):
        _relative(name)
    for item in local.values():
        if (
            type(item["size"]) is not int
            or item["size"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or not re.fullmatch(r"[0-9a-f]{40}", item["git_blob_sha1"])
        ):
            raise ValueError("Invalid local digest inventory")
    if not compare_inventories(local, remote)["complete"]:
        raise ValueError("Trusted audit contains inconsistent inventories")
    checkpoint = f"checkpoint-{step}/"
    selected = {k: v for k, v in local.items() if k.startswith(checkpoint)}
    if not REQUIRED <= {k.removeprefix(checkpoint) for k in selected}:
        raise ValueError("Checkpoint lacks required reconstruction payloads")
    if "run_config.json" not in local:
        raise ValueError("Missing run configuration")
    selected["run_config.json"] = local["run_config.json"]
    return {
        "repo_id": REPO_ID,
        "revision": record["revision"],
        "prefix": prefix,
        "run_id": run_id,
        "step": step,
        "audit_sha256": trusted_sha256,
        "files": selected,
    }


def verify_download(selection: dict, download_root: Path) -> dict:
    root = download_root.absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Download root may not traverse symlinks")
    run_root = root / selection["prefix"]
    checkpoint = run_root / f"checkpoint-{selection['step']}"
    for path in (run_root, checkpoint, *checkpoint.parents):
        if path.is_symlink():
            raise ValueError("Downloaded payload may not traverse symlinks")
    found = set()
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError("Downloaded payload may not contain symlinks")
        if path.is_file():
            found.add(str(path.relative_to(run_root)))
    found.add("run_config.json")
    if found != set(selection["files"]):
        raise ValueError("Downloaded checkpoint file coverage differs from the trusted audit")
    verified = {}
    for relative, expected in selection["files"].items():
        path = run_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked downloaded file")
        actual = file_inventory(path)
        if actual != expected:
            raise ValueError(f"Downloaded file content mismatch: {relative}")
        verified[relative] = actual
    return {
        "complete_for_selected_download": True,
        "checkpoint_root": str(checkpoint),
        "files": len(verified),
        "bytes": sum(x["size"] for x in verified.values()),
        "inventory": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--download-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if any(
        os.environ.get(k) != v
        for k, v in {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }.items()
    ):
        raise ValueError("Require disabled CUDA and offline HF/Transformers")
    output = args.output.absolute()
    download_root = args.download_root.absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(download_root):
        raise ValueError("Choose a new diagnostic receipt outside the download root")
    if args.step not in STEPS[:-1]:
        raise ValueError("The two-update diagnostic requires a nonterminal scheduled checkpoint")
    selection = select_download(args.audit, args.audit_sha256, args.run_id, args.step)
    before = verify_download(selection, download_root)
    # Only digest-authenticated local files are opened by the model/optimizer loaders below.
    import torch
    from sentence_transformers import SentenceTransformer

    from embed_optim.config import load_matrix
    from embed_optim.corrected_input_execution import require_independently_padded_dense
    from embed_optim.optimizers import build_optimizer
    from scripts.audit_checkpoint_optimizer_resume import exercise_resume

    repository = args.repository.resolve()
    for function, name in [(load_matrix, "config"), (build_optimizer, "optimizers")]:
        if (
            Path(inspect.getsourcefile(function)).resolve()
            != repository / f"src/embed_optim/{name}.py"
        ):
            raise ValueError("Set PYTHONPATH to the audited checkout's absolute src directory")
    torch.set_num_threads(2)
    matrix = repository / "configs/dense_no_packing_retrain.yaml"
    configs = {c.run_id: c for c in load_matrix(matrix)}
    config = configs[args.run_id]
    run_root = download_root / selection["prefix"]
    downloaded_config = json.loads((run_root / "run_config.json").read_text())
    if json.loads(json.dumps(config.as_dict())) != downloaded_config:
        raise ValueError("Downloaded run configuration differs from the frozen matrix")
    execution_path = repository / "configs/dense_no_packing_execution_protocol.json"
    execution = json.loads(execution_path.read_text())
    if _sha256(matrix) != execution["source_bindings"]["formal_matrix"]["sha256"]:
        raise ValueError("Matrix differs from its execution lock")
    total = execution["training"]["expected_optimizer_steps"]
    checkpoint = Path(before["checkpoint_root"])
    if json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] != args.step:
        raise ValueError("Downloaded trainer state has the wrong step")
    os.chdir(download_root)

    def factory():
        model = SentenceTransformer(
            str(checkpoint),
            device="cpu",
            local_files_only=True,
            model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
        )
        require_independently_padded_dense(model)
        return model

    model = factory()
    if (
        model.prompts != {"query": "query: ", "document": "document: "}
        or model.max_seq_length != 8192
    ):
        raise ValueError("Unexpected restored prompts or configured context length")
    vectors = torch.stack(
        [
            model.encode(
                "How do dense retrievers represent questions?",
                prompt_name="query",
                normalize_embeddings=True,
                convert_to_tensor=True,
            ),
            model.encode(
                "A dense retriever embeds questions and documents into a shared vector space.",
                prompt_name="document",
                normalize_embeddings=True,
                convert_to_tensor=True,
            ),
        ]
    )
    if vectors.shape != (2, 768) or not torch.isfinite(vectors).all():
        raise ValueError("Restored encoder produced invalid vectors")
    torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(2), rtol=0, atol=1e-6)
    encoding = {
        "shape": list(vectors.shape),
        "finite": True,
        "unit_norms": vectors.norm(dim=-1).tolist(),
        "texts": "two short synthetic query/document strings; not a benchmark",
    }
    del model, vectors
    gc.collect()
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    if scheduler["last_epoch"] != args.step:
        raise ValueError("Downloaded scheduler has the wrong step")
    continuation = exercise_resume(
        factory,
        config.optimizer,
        optimizer,
        scheduler,
        total,
        math.ceil(config.warmup_ratio * total),
    )
    if verify_download(selection, download_root) != before:
        raise ValueError("Downloaded evidence changed during restoration")
    source_paths = [
        Path(__file__).resolve(),
        Path(inspect.getsourcefile(exercise_resume)),
        repository / "src/embed_optim/optimizers.py",
        repository / "src/embed_optim/config.py",
        repository / "src/embed_optim/corrected_input_execution.py",
        matrix,
        execution_path,
    ]
    report = {
        "schema_version": 1,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "scope": "downloaded_primary_checkpoint_cpu_restoration",
        "scientific_completion": False,
        "selection": selection,
        "download_verification": before,
        "encoding": encoding,
        "cpu_continuation": continuation,
        "source_bindings": [{"path": str(p), **file_inventory(p)} for p in source_paths],
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "downloaded_payload_unchanged": True,
        "boundary": "One independently downloaded nonterminal checkpoint. CPU FP32 encoding and two synthetic-gradient optimizer/scheduler continuation steps only. No live output checkpoint read, GPU/distributed training, data-loader/rank-RNG restoration, full BEIR or maximum-length execution. This is not a physical second host or optimizer-quality result. No source payload or HF artifact was modified.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "complete_for_selected_download": True,
                "run_id": args.run_id,
                "step": args.step,
                "files": before["files"],
                "bytes": before["bytes"],
                "parameters": continuation["parameter_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
