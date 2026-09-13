"""Re-read the same immutable real task output; never rerun a GPU evaluator."""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_completion import task_score
from embed_optim.primary_contract import file_identity, read_json, verify_file

PRODUCER_SHA = "878ede1e95a0539d297bed7e6fd9dabf99f641b0dbe5faa03540ede35008353b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Use a new reader receipt path")
    if file_identity(args.producer)["sha256"] != PRODUCER_SHA:
        raise ValueError("Different actual task producer receipt")
    producer = read_json(args.producer)
    if producer["worker_returncode"] != 0 or producer["real_task_reader_passed"] is not False:
        raise ValueError("Not the retained successful-worker/failed-reader case")
    for row in producer["artifacts"]:
        verify_file(Path(row["path"]), row)
    path = next(
        Path(row["path"])
        for row in producer["artifacts"]
        if Path(row["path"]).name == "SciFactDecontaminated.json"
    )
    root = args.repository.resolve()
    versions = read_json(root / "configs/formal_runtime.json")["packages"]
    inspected = task_score(
        path,
        "SciFact",
        "verified-muon-3e-4",
        3,
        8192,
        "0729fa34af49875724d18ace64ce07f3e1dc0587",
        versions,
    )
    # Independent source-level control for MTEB's supported undefined auxiliary
    # metric. It is not a reconstruction of this task's per-query confidence data.
    import inspect
    import numpy as np
    from mteb._evaluators.retrieval_metrics import nauc

    control = nauc(np.arange(20, dtype=np.float64), np.ones(20, dtype=np.float64))
    if not math.isnan(control):
        raise ValueError("Pinned upstream undefined-nAUC control did not reproduce")
    for row in producer["artifacts"]:
        verify_file(Path(row["path"]), row)
    result = {
        "scope": "engineering_dense_primary_real_task_reader_replay",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "real_task_reader_passed": True,
        "scientific_completion": False,
        "production_deployed": False,
        "model_updates_executed": 0,
        "evaluation_workers_executed": 0,
        "hf_writes_executed": 0,
        "raw_task_artifacts_unchanged": True,
        "producer": {"path": str(args.producer), **file_identity(args.producer)},
        "inspected": inspected,
        "upstream_control": {
            "constant_metric_instances": 20,
            "undefined_nauc_returned": True,
            "source": {
                "path": inspect.getsourcefile(nauc),
                **file_identity(Path(inspect.getsourcefile(nauc))),
            },
        },
        "source_bindings": {
            name: file_identity(root / name)
            for name in (
                "src/embed_optim/primary_completion.py",
                "src/embed_optim/primary_contract.py",
                "scripts/replay_dense_primary_task_reader.py",
            )
        },
        "boundary": "One real SciFact full-corpus diagnostic output parsed again on CPU. Undefined unused auxiliary nAUC values are recorded, not zero-imputed or used by the predeclared nDCG estimand. The original failed reader, raw NaN bytes and successful worker are preserved. This is not a primary result, another evaluation, or proof of the cause of each actual per-query auxiliary degeneracy.",
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "passed": True,
                "undefined_auxiliary_metrics": len(
                    inspected["undefined_auxiliary_metrics_not_used"]
                ),
                "raw_files_unchanged": True,
                "output": str(args.output),
                **file_identity(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
