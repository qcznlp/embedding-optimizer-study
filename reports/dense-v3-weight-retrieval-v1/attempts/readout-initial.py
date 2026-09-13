"""Actual immutable-data numerical readout, not a formal primary consumer."""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

PROTOCOLS = {
    "dense_primary_v3_bridge_protocol.json": "540c3c3041ea13d9029fb5534283187b4f4141dfc66eba668938a7dc7d8ae803",
    "dense_primary_v3_exact_bridge_protocol.json": "270a908d4242a831ce239f6d6258a19b3884329646f076c0af43ab968e5c9fdc",
    "dense_no_packing_analysis_protocol.json": "91d7fa09ebb7e609afc5eb584f499b97baf9718d8cc23ff67eb295adfdd9f27a",
}
WEIGHT_MANIFEST = "9f53cdb8260f7b9d8f4c29db04159d02dfb7ff4e272cf9c9ba49cba842a1a841"
INDEX_SHA = "0418c68b9a78cc80173c9cf6af5d5bb57d223d7e763bc6da03d2fe4f682bbf96"
OUTCOME_SHA = "c21ac61b21fc00907011758ca64d92f1e27478942644462d834477e88054b4bc"
PRIMARY_SHA = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
STEPS = [782, 1563, 2345, 3126, 3907]
BOUNDARY = (
    "Actual sixty-state fixed-grid numerical prediction readout using unchanged pure kernels. "
    "Not formal whole-primary/source admission, significance, mediation, causal explanation, "
    "functional dimension utility, seed replication, source publication or paper completion."
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    need(not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked input/output")
    need(path.is_file(), f"Missing ordinary input: {path.name}")
    before = path.stat()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    need(all(getattr(before, k) == getattr(after, k) for k in
             ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "Input changed")
    return {"bytes": after.st_size, "sha256": sha}


def authenticated(path, expected):
    actual = identity(path)
    if isinstance(expected, str):
        need(actual["sha256"] == expected, f"SHA mismatch: {Path(path).name}")
    else:
        need(actual == {k: expected[k] for k in actual}, f"Identity mismatch: {Path(path).name}")
    return actual


def read_json(path):
    def unique(pairs):
        value = {}
        for k, v in pairs:
            need(k not in value, "Duplicate JSON field")
            value[k] = v
        return value
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=invalid)


def within(root, name):
    need(isinstance(name, str), "Non-string path")
    p = Path(name)
    need(not p.is_absolute() and ".." not in p.parts and p.as_posix() == name
         and name not in ("", ".") and "\\" not in name, "Noncanonical relative path")
    return Path(root) / p


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    need(bool(rows), "Empty result table")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pure_kernels(repository):
    """Load original AST nodes without importing model or contract entry points."""
    bindings, protocols, nodes_seen = {}, {}, {}
    for name, sha in PROTOCOLS.items():
        p = Path(repository) / "configs" / name
        bindings[str(p)] = authenticated(p, sha)
        protocols[name] = read_json(p)
    original = protocols["dense_primary_v3_bridge_protocol.json"]
    exact = protocols["dense_primary_v3_exact_bridge_protocol.json"]
    need(original["scientific_rules"] == protocols["dense_no_packing_analysis_protocol.json"]["retrieval_bridge"],
         "Scientific parent differs")
    need(original["formal_execution_authorized"] is False and exact["formal_execution_authorized"] is False,
         "Unexpected formal authority; this is the separately scoped data readout")
    source_bindings = exact["sources"]

    def load(filename, names, deps):
        relative = "src/embed_optim/" + filename + ".py"
        path = Path(repository) / relative
        bindings[str(path)] = authenticated(path, source_bindings[relative])
        tree = ast.parse(path.read_text())
        picked = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                picked.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                picked.append(node)
        found = set()
        for node in picked:
            keys = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [t.id for t in node.targets]
            found.update(keys)
            for key in keys:
                nodes_seen[filename + ":" + key] = hashlib.sha256(
                    ast.dump(node, include_attributes=False).encode()).hexdigest()
        need(found == set(names), f"Pure node inventory differs: {filename}")
        module = types.ModuleType("_actual_weight_bridge_" + filename)
        sys.modules[module.__name__] = module
        module.__dict__.update(dict(np=np, math=math, csv=csv, json=json, copy=copy,
             itertools=itertools, defaultdict=defaultdict, dataclass=dataclass,
             Decimal=Decimal, localcontext=localcontext, Q=Fraction, Any=object, **deps))
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        selected = ast.fix_missing_locations(ast.Module(body=[future, *picked], type_ignores=[]))
        exec(compile(selected, str(path), "exec"), module.__dict__)
        return {name: module.__dict__[name] for name in names}

    contract = load("primary_contract", ["canonical", "require_same"], {})
    integer = load("primary_completion", ["integer"], {})
    views = load("primary_v3_outcomes", ["OptimizerView", "RecipeView"], {})
    arithmetic = load("bridge_exact_arithmetic", ["rational", "dot", "inverse", "decimal",
         "displayed", "root", "rmse_reduction", "mse", "average_ranks", "correlation", "design_health", "Baseline"], {})
    join = load("corrected_retrieval_bridge", ["OPTIMIZERS", "DISPLACEMENT_KINDS", "FEATURES",
         "_finite", "_validate_matrix", "_index_checkpoint_geometry", "_index_scores",
         "_adamw_overlap_index", "assemble_bridge_rows", "_baseline_design"], {})
    numerical = load("bridge_numerics", ["NUMERICAL_POLICY", "validate_panel", "error_columns", "evaluate"],
                     {**arithmetic, **join})
    named = load("bridge_named_features", ["MISSING_POLICY", "empty_health", "evaluate_named"],
                 {**arithmetic, **join, **numerical})
    measurements = load("exact_bridge_measurements", ["CHECKPOINT_FEATURES", "PAIR_FEATURES", "FEATURES",
         "MEASUREMENT_RULES", "bounded", "checkpoint_features", "pair_features", "assemble_exact_panel"],
                        {**contract, **integer, **arithmetic, **numerical})
    parser = load("primary_v3_exact_bridge", ["typed_csv"], {})
    need(original["features"] == list(join["FEATURES"]), "Original feature inventory differs")
    need(exact["amendment"]["additional_named_features"] == list(measurements["FEATURES"]), "Exact inventory differs")
    need(original["numerical_policy"] == exact["numerical_policy"] == numerical["NUMERICAL_POLICY"], "Numerical policy differs")
    need(exact["measurement_rules"] == measurements["MEASUREMENT_RULES"]
         and exact["undefined_policy"] == named["MISSING_POLICY"], "Exact measurement policy differs")
    return dict(views=views, arithmetic=arithmetic, join=join, numerical=numerical, named=named,
                measurements=measurements, parser=parser, protocols=protocols,
                bindings=bindings, ast_sha256=nodes_seen)


def inspect_inputs(args, kernels):
    bindings = dict(kernels["bindings"])
    def take(path, expected):
        bindings[str(path)] = authenticated(path, expected)
        return read_json(path)
    weight = take(args.weight_root / "artifact_manifest.json", WEIGHT_MANIFEST)
    index = take(args.index, INDEX_SHA)
    accepted = take(args.outcome_root / "verification.json", OUTCOME_SHA)
    need(index["primary_protocol_sha256"] == PRIMARY_SHA and index["steps"] == STEPS,
         "Outcome index protocol/steps differ")
    need(index["checkpoint_count"] == 60 and index["raw_task_score_count"] == 840, "Incomplete raw index")
    parse = kernels["parser"]["typed_csv"]
    def table(root, name, expected):
        path = within(root, name)
        bindings[str(path)] = authenticated(path, expected)
        return parse(path)
    score_name = "tables/run_stage_scores.csv"
    scores = table(args.outcome_root, score_name, accepted["payloads"][score_name])
    all_name = "tables/all_task_scores.csv"
    task_scores = table(args.outcome_root, all_name, accepted["payloads"][all_name])
    need(len(scores) == 60 and len(task_scores) == 840, "Incomplete outcome tables")
    score_map = {(r["run_id"], r["stage"]): r for r in scores}
    task_map = {(r["run_id"], r["step"], r["task"]): r["ndcg_at_10"] for r in task_scores}
    need(len(score_map) == 60 and len(task_map) == 840, "Duplicate outcome rows")
    raw_index = {}
    need(len(index["snapshots"]) == 5 and [s["step"] for s in index["snapshots"]] == STEPS, "Index stage inventory differs")
    for snapshot in index["snapshots"]:
        need(len(snapshot["checkpoints"]) == 12, "Incomplete indexed stage")
        for row in snapshot["checkpoints"]:
            key = (row["run_id"], row["step"])
            need(key not in raw_index and row["step"] == snapshot["step"], "Duplicate/wrong indexed step")
            raw_index[key] = row
            need(len(row["task_files"]) == 14 and {t["task"] for t in row["task_files"]} == set(index["tasks"]), "Incomplete task identity")
            for task in row["task_files"]:
                need(task_map[(*key, task["task"])] == task["ndcg_at_10"], "Raw indexed score differs")
            stage = STEPS.index(row["step"]) + 1
            need(score_map[(row["run_id"], stage)]["mean_ndcg_at_10"] == row["macro_ndcg_at_10"], "Macro outcome differs")
    need(set(raw_index) == {(r, s) for r in index["runs"] for s in STEPS}, "Incomplete indexed grid")
    recipes, joined, tables, plan_by_kind = {}, [], {}, {}
    for kind, checkpoint_name, pair_name, protocol_sha in [
        ("approximate", "checkpoint_geometry.csv", "run_pair_subspace_overlap.csv", "9b6dad688072c08e8994e45f80c897a7889ab5fbcbccab27b643944c85ae625b"),
        ("exact", "checkpoint_exact_geometry.csv", "run_pair_exact_subspace_overlap.csv", "b13351152e6bd4b8ec2562d46c4e8bd2c1c2496bfd4bae7d4dabd093c47cb9f5"),
    ]:
        checkpoints = table(args.weight_root, kind + "/" + checkpoint_name, weight["files"][kind + "/" + checkpoint_name])
        pairs = table(args.weight_root, kind + "/" + pair_name, weight["files"][kind + "/" + pair_name])
        need(len(checkpoints) == 60 and len(pairs) == 660, "Incomplete geometry tables")
        panel = {(r["run_id"], r["step"]): r for r in checkpoints}
        need(set(panel) == set(raw_index) and len(panel) == len(checkpoints), "Geometry identity grid differs")
        summary_name = kind + "/summary.json"
        summary = take(within(args.weight_root, summary_name), weight["files"][summary_name])
        need(summary["primary_protocol_sha256"] == PRIMARY_SHA, "Geometry primary identity differs")
        protocol_key = "geometry_protocol_sha256" if kind == "approximate" else "exact_geometry_protocol_sha256"
        need(summary[protocol_key] == protocol_sha, "Geometry numerical definition differs")
        raw_bindings = {r["path"]: r for r in summary["raw_bindings"]}
        plan_by_kind[kind] = {}
        for run_id in index["runs"]:
            relative = f"{kind}/runs/{run_id}/manifest.json"
            manifest = take(within(args.weight_root, relative), weight["files"][relative])
            plan = manifest["plan"]
            need(plan["geometry_protocol_sha256"] == protocol_sha and plan["steps"] == STEPS,
                 "Native plan definition or steps differ")
            need(plan["recipe"]["run_id"] == run_id, "Recipe name differs")
            plan_by_kind[kind][run_id] = plan
            if run_id in recipes:
                need(recipes[run_id] == plan["recipe"], "Approximate/exact recipe differs")
                prior = plan_by_kind["approximate"][run_id]
                need(prior["checkpoint_files"] == plan["checkpoint_files"]
                     and prior["reference"] == plan["reference"], "Weight sources differ across branches")
            recipes[run_id] = plan["recipe"]
            need([c["step"] for c in plan["checkpoint_files"]] == STEPS
                 and [c["step"] for c in manifest["outputs"]] == STEPS, "Native checkpoint inventory differs")
            outputs = {r["step"]: r for r in manifest["outputs"]}
            for stage, ckpt in enumerate(plan["checkpoint_files"], 1):
                step = ckpt["step"]
                indexed = raw_index[(run_id, step)]
                files = {f["path"]: f for f in ckpt["files"]}
                seal = files["dense_checkpoint_seal.json"]
                need(plan["run_identity_sha256"] == indexed["run_identity_sha256"], "Weight/retrieval run identity differs")
                need({k: seal[k] for k in ("bytes", "sha256")} == indexed["checkpoint_seal"], "Weight/retrieval checkpoint seal differs")
                rel = f"{kind}/runs/{run_id}/checkpoint-{step}.json"
                raw = take(within(args.weight_root, rel), weight["files"][rel])
                need(bindings[str(within(args.weight_root, rel))] == {k: outputs[step]["records"][k] for k in ("bytes", "sha256")}, "Raw producer record differs")
                native_rel = f"runs/{run_id}/checkpoint-{step}.json"
                need(bindings[str(within(args.weight_root, rel))] == {k: raw_bindings[native_rel][k] for k in ("bytes", "sha256")}, "Summary raw binding differs")
                need(raw["checkpoint_row"] == panel[(run_id, step)], "CSV differs from original raw checkpoint row")
                need(raw["step"] == step and raw["previous_step"] == (0 if stage == 1 else STEPS[stage-2])
                     and raw["cumulative_anchor_step"] == 0, "Displacement anchors differ")
                need(panel[(run_id, step)]["stage"] == stage and panel[(run_id, step)]["progress_fraction"] == stage/5, "Stage label differs")
                need(score_map[(run_id, stage)]["tasks"] == 14, "Outcome task count differs")
                if kind == "approximate":
                    joined.append(dict(run_id=run_id, stage=stage, step=step,
                         run_identity_sha256=plan["run_identity_sha256"], checkpoint_seal=indexed["checkpoint_seal"],
                         model_safetensors={k: files["model.safetensors"][k] for k in ("bytes", "sha256")},
                         mean_ndcg_at_10=indexed["macro_ndcg_at_10"]))
        tables[kind] = dict(checkpoints=checkpoints, pairs=pairs)
    v = kernels["views"]
    configs = [v["RecipeView"](run_id=r["run_id"], optimizer=v["OptimizerView"](name=r["optimizer"]["name"], lr=r["optimizer"]["lr"]),
               model_family=r["model_family"], dense_can_flatten_inputs=r["dense_can_flatten_inputs"],
               checkpoint_fractions=tuple(r["checkpoint_fractions"])) for r in recipes.values()]
    need(len(joined) == 60, "Incomplete shared checkpoint evidence")
    return dict(bindings=bindings, joined=joined, recipes=recipes, configs=configs, tables=tables, scores=scores)


def calculate(inputs, kernels):
    a = inputs["tables"]["approximate"]
    rows = kernels["join"]["assemble_bridge_rows"](a["checkpoints"], a["pairs"], inputs["scores"], inputs["configs"])
    original, original_diagnostics = kernels["numerical"]["evaluate"](rows)
    exact_data = inputs["tables"]["exact"]
    extra, comparisons, coverage = kernels["measurements"]["assemble_exact_panel"](
         original["bridge_rows"], exact_data["checkpoints"], exact_data["pairs"], inputs["configs"], STEPS)
    exact, exact_diagnostics = kernels["named"]["evaluate_named"](extra, kernels["measurements"]["FEATURES"])
    exact["measurement_comparison"] = comparisons
    prior = {r["feature"]: r for r in original["feature_prediction_summary"]}
    contrasted = []
    for row in exact["feature_prediction_summary"]:
        f = row["feature"]
        mapping = kernels["measurements"]
        old_name = mapping["CHECKPOINT_FEATURES"][f][2] if f in mapping["CHECKPOINT_FEATURES"] else mapping["PAIR_FEATURES"][f][1]
        old = prior[old_name]
        need(row["pooled_baseline_mse_exact"] == old["pooled_baseline_mse_exact"], "Baselines differ")
        a, b = old["pooled_feature_mse_exact"], row["pooled_feature_mse_exact"]
        defined = a is not None and b is not None
        difference = Fraction(a) - Fraction(b) if defined else None
        contrasted.append(dict(original_feature=old_name, exact_feature=f,
            original_predictively_useful=old["predictively_useful"], exact_predictively_useful=row["predictively_useful"],
            original_pooled_feature_rmse=old["pooled_feature_rmse"], exact_pooled_feature_rmse=row["pooled_feature_rmse"],
            original_minus_exact_feature_mse=float(difference) if defined else None,
            original_minus_exact_feature_mse_rational=str(difference) if defined else None,
            original_minus_exact_feature_rmse=kernels["arithmetic"]["rmse_reduction"](Fraction(a), Fraction(b)) if defined else None,
            defined=defined))
    exact["predictive_sensitivity_summary"] = contrasted
    for family, tables, name in [("original", original, "dense_primary_v3_bridge_protocol.json"),
                                 ("exact", exact, "dense_primary_v3_exact_bridge_protocol.json")]:
        need({k: len(v) for k, v in tables.items()} == kernels["protocols"][name]["table_counts"], f"{family} output coverage differs")
    return {"original": original, "exact": exact}, dict(original=original_diagnostics, exact=exact_diagnostics, measurement_coverage=coverage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "weight-root", "outcome-root", "index", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args(argv)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only readout")
    for p in (args.repository, args.weight_root, args.outcome_root, args.index, args.output):
        need(p.is_absolute() and not any(q.is_symlink() for q in (p, *p.parents)), "Require absolute ordinary paths")
    need(not args.output.exists(), "Refuse existing output")
    kernels = pure_kernels(args.repository)
    inputs = inspect_inputs(args, kernels)
    result, diagnostics = ({}, {}) if args.inputs_only else calculate(inputs, kernels)
    for path, binding in inputs["bindings"].items():
        authenticated(path, binding)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "joined-checkpoints.json", inputs["joined"])
    write_json(args.output / "recipes.json", inputs["recipes"])
    if not args.inputs_only:
        for family, tables in result.items():
            directory = args.output / family
            directory.mkdir()
            for name, rows in tables.items():
                write_csv(directory / (name + ".csv"), rows)
        write_json(args.output / "diagnostics.json", diagnostics)
        write_json(args.output / "tables.json", result)
        lines = ["# Complete weight-to-retrieval held-dose numerical readout", "", BOUNDARY, "",
                 "RMSE is on nDCG@10 × 100. The same 60 states, eight-column baseline and",
                 "four held-dose folds are used for every feature. Support requires lower",
                 "pooled error and improvements in at least three of four folds.", ""]
        def fmt(x):
            return "undefined" if x is None else f"{100*x:.6f}"
        for family, tables in result.items():
            lines += ["## " + family.capitalize(), "", "| Feature | Baseline RMSE | Added-feature RMSE | Improved folds | Defined folds | Predictive support |",
                      "| --- | ---: | ---: | ---: | ---: | --- |"]
            for r in tables["feature_prediction_summary"]:
                lines.append(f"| {r['feature']} | {fmt(r['pooled_baseline_rmse'])} | {fmt(r['pooled_feature_rmse'])} | {r['folds_improved']}/4 | {r['folds_defined']}/4 | {r['predictively_useful']} |")
            lines.append("")
        lines += ["No family-wise significance test is defined for these predictive flags.",
                  "Residual associations are descriptive, and the omitted dose still belongs",
                  "to the same optimizer families, model, data and training seed. This is not",
                  "a held-task, held-seed or external-model validation. Exact full-spectrum",
                  "entropy and original truncated entropy are distinct measurements.", ""]
        with (args.output / "summary.md").open("x") as stream:
            stream.write("\n".join(lines))
    outputs = {p.relative_to(args.output).as_posix(): identity(p) for p in sorted(args.output.rglob("*")) if p.is_file()}
    receipt = dict(scope="actual-complete-weight-retrieval-pure-numerical-readout", completed_at_utc=datetime.now(timezone.utc).isoformat(),
         inputs_only=args.inputs_only, boundary=BOUNDARY, input_bindings=inputs["bindings"], source_ast_sha256=kernels["ast_sha256"],
         native_weight_retrieval_identity_seal_joins=60, original_raw_checkpoint_rows_matched=120, raw_index_task_scores_matched=840,
         primary_protocol_sha256=PRIMARY_SHA, outputs=outputs, numerical_analysis_executed=not args.inputs_only,
         frozen_numerical_logic_changed=False, formal_consumer_called=False, formal_primary_admission=False,
         model_or_retrieval_executed=False, functional_recovery=False, source_release=False, scientific_completion=False)
    write_json(args.output / "readout.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("scope", "completed_at_utc", "inputs_only", "native_weight_retrieval_identity_seal_joins", "original_raw_checkpoint_rows_matched", "raw_index_task_scores_matched", "formal_consumer_called", "scientific_completion")}))


if __name__ == "__main__":
    main()
