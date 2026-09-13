"""Prepare, never activate, the new source-bound primary execution/consumer proposal."""

import argparse
import ast
import json
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import DRAFT, INPUT_SHA, SCOPE, STEPS, file_identity, read_json

CONSUMERS = (
    "src/embed_optim/primary_contract.py",
    "src/embed_optim/primary_training.py",
    "src/embed_optim/primary_io.py",
    "src/embed_optim/gpu_lease.py",
    "src/embed_optim/runtime.py",
    "src/embed_optim/evaluate_matrix.py",
    "src/embed_optim/evaluation_utils.py",
    "src/embed_optim/corrected_input_execution.py",
    "src/embed_optim/decontamination.py",
    "src/embed_optim/aggregate.py",
    "src/embed_optim/evaluation_source_provenance.py",
    "scripts/eval/dense_no_packing_parallel.py",
    "scripts/eval/dense_parallel.py",
    "scripts/eval/dense_sequential.py",
    "configs/formal_runtime.json",
    "configs/formal_runtime_constraints.txt",
    "requirements-formal.lock",
    "requirements-formal-flash.txt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, training = args.repository.resolve(), args.training_root.resolve()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Use a new protocol path; previous locks are immutable")
    input_path = root / "reports/dense-primary-v2/input-bindings.json"
    if file_identity(input_path)["sha256"] != INPUT_SHA:
        raise ValueError("Primary actual-input receipt differs")
    inputs = read_json(input_path)
    sources = read_json(Path(inputs["source_manifest"]["path"]))
    training_sources = {}
    for row in sources["candidate_sources"]:
        expected = {k: row["identity"][k] for k in ("bytes", "sha256")}
        if file_identity(training / row["relative_path"]) != expected:
            raise ValueError("Numerical candidate changed")
        training_sources[row["relative_path"]] = expected
    tree = ast.parse((root / "src/embed_optim/decontamination.py").read_text())
    tasks = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "DECONTAMINATED_BEIR"
    )
    if list(tasks) != inputs["original_evaluation"]["tasks"]:
        raise ValueError("Original 14-task evaluation order differs")
    payload = {
        "schema_version": 1,
        "scope": SCOPE,
        "status": DRAFT,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Prospective revised primary matrix using the verified numerical and full-run identity implementations; not a retrospectively preregistered original study.",
        "input_bindings": {
            "path": input_path.relative_to(root).as_posix(),
            **file_identity(input_path),
        },
        "training_sources": training_sources,
        "consumer_sources": {p: file_identity(root / p) for p in CONSUMERS},
        "world_size": 4,
        "checkpoint_steps": list(STEPS),
        "data_path": "data/denseon-sft-500k-seed42",
        "output_root": "outputs/dense-correctness-v2",
        "gpu_pools": [["0", "1", "2", "3"], ["4", "5", "6", "7"]],
        "gpu_lease_root": "/tmp/embedding-optimizer-primary-gpu-leases",
        "wandb": {"project": "embedding-optimizer-study", "entity": "stevezenguom"},
        "checkpoint_repository": "qcz/embedding-optimizer-study-checkpoints",
        "checkpoint_prefix": "corrected-dense-correctness-v2/dense",
        "worker_python": "/usr/bin/python3",
        "evaluation": inputs["original_evaluation"],
        "beir_task_revisions": {
            name: {"repo": repo, "revision": revision} for name, (repo, revision) in tasks.items()
        },
        "analysis": {
            k: v for k, v in inputs["original_analysis"].items() if k != "historical_bridge"
        },
        "historical_artifacts": "Preserve outside scientific inference; no old score/cache/checkpoint can supply revised primary results. No implementation incident or historical implementation comparison enters any manuscript section.",
        "pending_release_requirements": [
            "owner development-publication/deployment direction",
            "single assembled committed source checkout",
            "reviewed evidence-preserving retirement/transition of old paused controllers",
            "natural-data execution readiness",
            "new complete-run/outcome/dimension/factorial/publication consumers",
        ],
        "scientific_completion": False,
        "production_deployed": False,
    }
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": DRAFT, "protocol": str(args.output), **file_identity(args.output)}))


if __name__ == "__main__":
    main()
