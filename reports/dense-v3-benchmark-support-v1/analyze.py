"""Descriptive benchmark support and endpoint decomposition; no new inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROTECTED = "f20353ce500807a7bfe4ddf30169560bfe6ef415b082d2291060102946e37f66"
READOUT = "592c902f94b8f190f3165f4075dfddc1e8c3f52add760805da639b541d27716f"
READER = "52baa3d3fca7ff9b31574e48d2817d2b9c81a5642fa8822426b73e815fa82ab5"
PRIMARY = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
OPTIMIZERS = ("adamw", "muon", "normuon")
CONTRASTS = (("muon", "adamw"), ("normuon", "adamw"), ("normuon", "muon"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_bytes(path, digest, size=None):
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "Require ordinary input: " + str(path))
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, "Input digest changed: " + str(path))
    require(size is None or len(raw) == size, "Input length changed: " + str(path))
    return raw


def identity(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def query_counts(query_ids, qrel_ids, relevance, recorded_ids):
    require(all(v is not None for v in (*query_ids, *qrel_ids, *recorded_ids)), "Null query ID")
    queries = [str(v) for v in query_ids]
    rels = [str(v) for v in qrel_ids]
    recorded = [str(v) for v in recorded_ids]
    require(len(queries) == len(set(queries)), "Duplicate query-file IDs")
    require(len(recorded) == len(set(recorded)), "Duplicate recorded query IDs")
    require(len(rels) == len(relevance) and all(math.isfinite(v) for v in relevance),
            "Invalid relevance rows")
    covered = set(rels)
    require(covered <= set(queries), "Qrel query absent from query file")
    require(covered == set(recorded), "Original protected-query membership differs")
    positive = {q for q, value in zip(rels, relevance, strict=True) if value > 0}
    return {
        "query_file_rows": len(queries),
        "qrel_rows": len(rels),
        "qrel_covered_queries": len(covered),
        "queries_with_positive_qrel": len(positive),
        "queries_without_positive_qrel": len(covered - positive),
    }


def decompose(deltas):
    require(len(deltas) == 14, "Require every one of the fourteen tasks")
    total = sum(deltas, Fraction())
    mean = total / 14
    contributions = [d / 14 for d in deltas]
    require(sum(contributions, Fraction()) == mean, "Contribution conservation failed")
    return mean, contributions, {
        "positive_tasks": sum(d > 0 for d in deltas),
        "negative_tasks": sum(d < 0 for d in deltas),
        "tied_tasks": sum(d == 0 for d in deltas),
    }


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists()
            and not any(p.is_symlink() for p in args.output.parents), "Use a fresh ordinary output path")
    old = args.repository / "reports/dense-v3-final-inference-v1"
    reader_path = old / "source/run_endpoint_inference.py"
    bound_bytes(reader_path, READER)
    spec = importlib.util.spec_from_file_location("bound_endpoint_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    # Reuse only the unchanged input readers, never the bootstrap or its main().
    manifest = reader.authenticate_snapshot(args.evaluation_root)
    primary_path = args.repository / "configs/dense_primary_v3_protocol.json"
    primary = json.loads(bound_bytes(primary_path, PRIMARY))
    tasks = primary["evaluation"]["tasks"]
    require(len(tasks) == len(set(tasks)) == 14, "Wrong task population")
    indexed, configs, selected = reader.scored_population(args.evaluation_root, tasks)
    readout_path = old / "tables/readout.json"
    readout = json.loads(bound_bytes(readout_path, READOUT))
    require(selected == readout["selected"], "Original validation selection differs")
    protected_path = args.repository / "reports/engineering-archive/dense-data-partition-v1/protected-queries.json"
    protected = json.loads(bound_bytes(protected_path, PROTECTED))
    require([r["task"] for r in protected["beir"]] == tasks, "Different benchmark audit")

    counts, data_bindings = [], []
    for record in protected["beir"]:
        task = record["task"]
        revision = primary["beir_task_revisions"][task]
        require((record["repo"], record["revision"]) == (revision["repo"], revision["revision"]),
                "Dataset revision mismatch")
        split = "dev" if task == "MSMARCO" else "test"
        require(record["evaluation_split"] == split, "Different evaluation split")
        files = {f["repo_path"]: f for f in record["files"]}
        qrels_name = "qrels_validation.parquet" if split == "dev" else "qrels_test.parquet"
        require(set(files) == {"queries.parquet", qrels_name}, "Unexpected query/qrel files")
        arrays = {}
        for name, columns in (("queries.parquet", ["_id"]), (qrels_name, ["query-id", "score"])):
            f = files[name]
            require(f["sha256"] == f["independent_hf_lfs_sha256"], "Missing independent HF binding")
            path = args.hf_cache / ("datasets--" + record["repo"].replace("/", "--")) / "blobs" / f["sha256"]
            raw = bound_bytes(path, f["sha256"], f["bytes"])
            arrays[name] = pq.ParquetFile(pa.BufferReader(raw)).read(columns=columns).to_pydict()
            data_bindings.append({"task": task, "repo": record["repo"], "revision": record["revision"],
                                  "repo_path": name, "bytes": len(raw), "sha256": f["sha256"]})
        values = query_counts(arrays["queries.parquet"]["_id"], arrays[qrels_name]["query-id"],
                              arrays[qrels_name]["score"], [q["query_id"] for q in record["queries"]])
        require(values["qrel_covered_queries"] == record["query_count"], "Original query count differs")
        counts.append({"task": task, "split": split, **values, "dataset_revision": record["revision"]})

    native_count = 0
    for name in manifest["files"]:
        if name.startswith("native/beir-final/") and name.endswith("Decontaminated.json"):
            payload = reader.read(args.evaluation_root / name)
            task = payload["task_name"].removesuffix("Decontaminated")
            require(task in tasks and payload["dataset_revision"] == primary["beir_task_revisions"][task]["revision"],
                    "Native result revision differs from query/qrel revision")
            native_count += 1
    require(native_count == 168, "Incomplete native revision population")

    by_task = {r["task"]: r for r in counts}
    contribution_rows, summary_rows, input_bindings = [], [], {}
    for family in ("primary", "secondary"):
        tables = {}
        for suffix in ("task_effects", "summary"):
            name = family + "_" + suffix + ".csv"
            binding = readout["outputs"][name]
            raw = bound_bytes(old / "tables" / name, binding["sha256"], binding["bytes"])
            input_bindings[name] = binding
            tables[suffix] = list(csv.DictReader(io.StringIO(raw.decode())))
        require([r["task"] for r in tables["task_effects"]] == tasks, "Incomplete original task table")
        original_summary = {(r["treatment"], r["baseline"]): r for r in tables["summary"]}
        require(len(tables["summary"]) == 3 and set(original_summary) == set(CONTRASTS), "Wrong contrast family")
        means = {}
        for task, row in zip(tasks, tables["task_effects"], strict=True):
            for optimizer in OPTIMIZERS:
                members = [r for r in configs if r.optimizer.name == optimizer
                           and (family == "primary" or r.run_id == selected[optimizer])]
                require(len(members) == (4 if family == "primary" else 1), "Missing rate")
                mean = sum((Fraction(str(indexed[(r.run_id, 5, task)]["ndcg_at_10"]))
                            for r in members), Fraction()) / len(members)
                require(abs(float(mean) - float(row[optimizer + "_ndcg_at_10"])) < 2e-15,
                        "Original task mean reconstruction differs")
                means[task, optimizer] = mean
        for treatment, baseline in CONTRASTS:
            deltas = [(means[t, treatment] - means[t, baseline]) * 100 for t in tasks]
            mean, contributions, signs = decompose(deltas)
            original = original_summary[treatment, baseline]
            require(abs(float(mean) - 100 * float(original["mean_delta_ndcg_at_10"])) < 2e-13,
                    "Original endpoint mean differs")
            summary_rows.append({"family": family, "treatment": treatment, "baseline": baseline,
                                 **signs, "macro_delta_points": float(mean), "exact_macro_delta": str(mean),
                                 "original_simultaneous_lower_points": 100 * float(original["simultaneous_ci_95_lower"]),
                                 "original_simultaneous_upper_points": 100 * float(original["simultaneous_ci_95_upper"]),
                                 "original_support": original["support"]})
            for task, delta, contribution in zip(tasks, deltas, contributions, strict=True):
                contribution_rows.append({"family": family, "treatment": treatment, "baseline": baseline,
                                          "task": task, "qrel_covered_queries": by_task[task]["qrel_covered_queries"],
                                          "task_delta_points": float(delta), "exact_task_delta": str(delta),
                                          "macro_contribution_points": float(contribution),
                                          "exact_macro_contribution": str(contribution),
                                          "fraction_of_net_macro_delta": "" if mean == 0 else float(contribution / mean)})

    args.output.mkdir()
    outputs = {"benchmark_sizes.csv": counts, "task_contributions.csv": contribution_rows,
               "contrast_summary.csv": summary_rows}
    for name, rows in outputs.items():
        write_csv(args.output / name, rows)
    receipt = {
        "scope": "posthoc_descriptive_benchmark_support_and_endpoint_decomposition",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": identity(Path(__file__)), "unchanged_input_reader": identity(reader_path),
        "endpoint_readout": identity(readout_path), "protected_query_audit": identity(protected_path),
        "primary_protocol": identity(primary_path), "endpoint_manifest_sha256": reader.MANIFEST,
        "recovered_endpoint_files_authenticated": 660, "native_result_revisions_checked": native_count,
        "query_qrel_file_bindings": data_bindings, "original_table_bindings": input_bindings,
        "query_qrel_files_checked": len(data_bindings), "qrel_covered_queries": sum(r["qrel_covered_queries"] for r in counts),
        "selected": selected, "pyarrow_version": pa.__version__,
        "evaluation_root": str(args.evaluation_root), "hf_cache": str(args.hf_cache),
        "outputs": {name: identity(args.output / name) for name in outputs},
        "new_inference": False, "bootstrap_rerun": False, "task_reweighting": False,
        "task_exclusion": False, "per_query_ranks_available": False,
        "model_execution": False, "full_840_cell_admission": False,
        "manuscript_installation": False, "source_release": False, "scientific_completion": False,
    }
    with (args.output / "readout.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "observed_at_utc": receipt["observed_at_utc"],
                      "query_count": receipt["qrel_covered_queries"], "contrasts": summary_rows,
                      "new_inference": False}, sort_keys=True))


if __name__ == "__main__":
    main()
