"""Independent exact-OLS/complete-v3 bridge rehearsal; all positive panels are synthetic."""

import argparse
import copy
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import sympy as sp

from embed_optim import primary_v3_bridge as bridge
from embed_optim.bridge_numerics import evaluate
from embed_optim.corrected_retrieval_bridge import FEATURES
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import csv_bytes, inspect_bundle, save_bundle
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), **file_identity(path)}


def independent_ranks(vector):
    return sp.Matrix(
        [
            1
            + sum(1 for value in vector if value < x)
            + sp.Rational(sum(1 for value in vector if value == x) - 1, 2)
            for x in vector
        ]
    )


def independent_full_systems(tables):
    """SymPy full augmented normal equations, not the production rational FWL solver."""
    rank_controls = [
        ([3, 1, 1, 2], [4, sp.Rational(3, 2), sp.Rational(3, 2), 3]),
        ([sp.Rational(-1, 3), 0, sp.Rational(1, 3)], [1, 2, 3]),
        ([sp.Rational(1, 7)] * 4, [sp.Rational(5, 2)] * 4),
    ]
    for values, expected in rank_controls:
        assert independent_ranks(sp.Matrix(values)) == sp.Matrix(expected)
    rows = tables["bridge_rows"]
    baseline = sp.Matrix(
        [
            [
                1,
                int(r["optimizer"] == "muon"),
                int(r["optimizer"] == "normuon"),
                *[int(r["stage"] == s) for s in range(2, 6)],
                sp.Rational(r["centered_log10_learning_rate"]),
            ]
            for r in rows
        ]
    )
    outcome = sp.Matrix([sp.Rational(r["mean_ndcg_at_10"]) for r in rows])
    expected_predictions = {
        (r["feature"], r["run_id"], r["stage"]): r for r in tables["held_out_predictions"]
    }
    controls, comparisons, mse_checks = [], 0, 0
    for fold in range(1, 5):
        train = [i for i, r in enumerate(rows) if r["dose_index"] != fold]
        test = [i for i, r in enumerate(rows) if r["dose_index"] == fold]
        b = baseline.extract(train, range(8))
        yt = outcome.extract(train, [0])
        baseline_beta = (b.T * b).inv(method="DM") * b.T * yt
        base_prediction = baseline.extract(test, range(8)) * baseline_beta
        ytest = outcome.extract(test, [0])
        for feature in FEATURES:
            x = sp.Matrix([sp.Rational(r[feature]) for r in rows])
            full = baseline.row_join(x)
            design = full.extract(train, range(9))
            assert design.rank() == 9
            beta = (design.T * design).inv(method="DM") * design.T * yt
            predicted = full.extract(test, range(9)) * beta
            for k, i in enumerate(test):
                row = expected_predictions[(feature, rows[i]["run_id"], rows[i]["stage"])]
                assert predicted[k] == sp.Rational(row["feature_prediction_exact"])
                assert base_prediction[k] == sp.Rational(row["baseline_prediction_exact"])
                comparisons += 1
            errors, base_errors = ytest - predicted, ytest - base_prediction
            feature_mse, base_mse = errors.dot(errors) / 15, base_errors.dot(base_errors) / 15
            stored = next(
                r
                for r in tables["leave_dose_fold_metrics"]
                if r["feature"] == feature and r["held_out_dose_index"] == fold
            )
            assert sp.Rational(stored["feature_mse_exact"]) == feature_mse
            assert sp.Rational(stored["baseline_mse_exact"]) == base_mse
            assert sp.Rational(stored["mse_reduction_exact"]) == base_mse - feature_mse
            assert stored["feature_improves"] == bool(base_mse > feature_mse)
            mse_checks += 1
            controls.append(
                {
                    "feature": feature,
                    "fold": fold,
                    "full_augmented_rank": 9,
                    "prediction_count": 15,
                    "exact_equal": True,
                }
            )
    projection = baseline * (baseline.T * baseline).inv(method="DM") * baseline.T
    residual_y = outcome - projection * outcome
    association_controls = []

    def corr(a, b):
        ac, bc = a - sp.ones(60, 1) * sum(a) / 60, b - sp.ones(60, 1) * sum(b) / 60
        return float((ac.dot(bc) / sp.sqrt(ac.dot(ac) * bc.dot(bc))).evalf(80))

    for feature in FEATURES:
        x = sp.Matrix([sp.Rational(r[feature]) for r in rows])
        residual_x = x - projection * x
        actual = next(r for r in tables["residual_associations"] if r["feature"] == feature)
        assert actual["status"] == "resolved"
        pearson, spearman = (
            corr(residual_x, residual_y),
            corr(independent_ranks(residual_x), independent_ranks(residual_y)),
        )
        assert math.isclose(
            actual["pearson_residual_association"], pearson, rel_tol=2e-12, abs_tol=2e-14
        )
        assert math.isclose(
            actual["spearman_residual_association"], spearman, rel_tol=2e-12, abs_tol=2e-14
        )
        association_controls.append(
            {"feature": feature, "pearson": pearson, "spearman": spearman, "passed": True}
        )
        folds = [r for r in tables["leave_dose_fold_metrics"] if r["feature"] == feature]
        baseline_mse = sum(sp.Rational(r["baseline_mse_exact"]) for r in folds) / 4
        feature_mse = sum(sp.Rational(r["feature_mse_exact"]) for r in folds) / 4
        summary = next(r for r in tables["feature_prediction_summary"] if r["feature"] == feature)
        assert sp.Rational(summary["pooled_mse_reduction_exact"]) == baseline_mse - feature_mse
        assert summary["predictively_useful"] == (
            bool(baseline_mse > feature_mse) and sum(r["feature_improves"] for r in folds) >= 3
        )
    return {
        "independent_exact_rank_controls": len(rank_controls),
        "full_systems": controls,
        "exact_predictions": comparisons,
        "exact_fold_mse_comparisons": mse_checks,
        "residual_associations": association_controls,
        "exact_pooled_decisions": 9,
    }


def corrupt_and_refuse(work, original, plan, evidence, tables):
    cases = []
    for name in ("prediction", "support", "association", "diagnostic"):
        target = work / f"altered-{name}"
        shutil.copytree(original, target)
        if name == "diagnostic":
            changed = copy.deepcopy(evidence)
            changed["numerical_diagnostics"]["feature_designs"][0]["rank_cutoff"] = 0
            filename = "evidence.json"
            content = json.dumps(changed, sort_keys=True).encode()
        else:
            table, field = {
                "prediction": ("held_out_predictions", "feature_prediction"),
                "support": ("feature_prediction_summary", "predictively_useful"),
                "association": ("residual_associations", "pearson_residual_association"),
            }[name]
            changed = copy.deepcopy(tables[table])
            changed[0][field] = not changed[0][field] if name == "support" else 0.99123
            filename, content = f"{table}.csv", csv_bytes(changed)
        (target / filename).write_bytes(content)
        manifest = read_json(target / "manifest.json")
        manifest["outputs"][filename] = {"path": filename, **file_identity(target / filename)}
        (target / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        try:
            inspect_bundle(target, plan, evidence, tables)
        except ValueError as error:
            cases.append({"case": name, "rejected": True, "error": str(error)})
        else:
            raise AssertionError("Rehashed numerical corruption accepted")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "training-root", "workdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError("Require CPU-only execution, assertions and an explicit empty workdir")
    primary = PrimaryV3Contract.load(root / bridge.PARENTS["primary"][0], root, args.training_root)
    contract = bridge.BridgeContract.load(
        root / "configs/dense_primary_v3_bridge_protocol.json", primary
    )
    assert contract.sha256 == args.expected_protocol_sha256
    before = handoff()
    initial = [binding(root / name) for name in bridge.SOURCES]
    initial += [binding(root / name) for name, _ in bridge.PARENTS.values()]
    initial += [
        binding(contract.path),
        binding(Path(__file__)),
        binding(root / "tests/test_primary_v3_bridge.py"),
        binding(root / "tests/test_bridge_numerics.py"),
    ]
    if args.replay:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        parent = read_json(args.replay)
        for row in (*parent["inputs"], *parent["artifacts"]):
            verify_file(row["path"], row)
        origin = args.replay.parent
        fixture = read_json(origin / "fixture-input.json")
    else:
        generator = runpy.run_path(str(root / "tests/test_primary_v3_bridge.py"))["inputs"]
        checkpoints, pairs, scores = generator(primary)
        fixture = {
            "scope": "engineering_synthetic_bridge_inputs",
            "not_primary_evidence": True,
            "checkpoints": checkpoints,
            "pairs": pairs,
            "scores": scores,
        }
        origin = work
    assert (
        fixture["scope"] == "engineering_synthetic_bridge_inputs"
        and fixture["not_primary_evidence"] is True
    )
    tables, numerical = bridge.bridge_tables(
        primary, fixture["checkpoints"], fixture["pairs"], fixture["scores"]
    )
    evidence = {"fixture": fixture, "numerical_diagnostics": numerical}
    plan = {
        "scope": "engineering_synthetic_bridge_fixture",
        "bridge_protocol_sha256": contract.sha256,
        "scientific_completion": False,
        "not_primary_evidence": True,
    }
    if args.replay:
        saved = inspect_bundle(origin / "fixture-bundle", plan, evidence, tables)
        oracle = None
    else:
        write_new(work / "fixture-input.json", fixture)
        saved = save_bundle(work / "fixture-bundle", plan, evidence, tables)
        oracle = independent_full_systems(tables)
    corrupted = corrupt_and_refuse(work, origin / "fixture-bundle", plan, evidence, tables)
    original_cases = read_json(root / bridge.PARENTS["null_counterexample"][0])["cases"]
    null_results = []
    for case in original_cases:
        repaired, diagnostics = evaluate(case["rows"])
        assert all(
            r["pooled_mse_reduction_exact"] == "0" and r["predictively_useful"] is False
            for r in repaired["feature_prediction_summary"]
        )
        assert all(
            r["pearson_residual_association"] is None and r["spearman_residual_association"] is None
            for r in repaired["residual_associations"]
        )
        null_results.append({"case": case["case"], "tables": repaired, "diagnostics": diagnostics})
    changes = []
    for key in (
        "features",
        "table_counts",
        "numerical_policy",
        "amendment",
        "parents",
        "sources",
        "status",
        "scientific_completion",
    ):
        value = copy.deepcopy(contract.payload)
        if key == "status":
            value[key] = "released"
        elif key == "scientific_completion":
            value[key] = True
        elif key == "features":
            value[key].pop()
        else:
            value[key].pop(next(iter(value[key])))
        path = work / f"altered-protocol-{key}.json"
        write_new(path, value)
        try:
            bridge.BridgeContract.load(path, primary)
        except (ValueError, KeyError) as error:
            changes.append({"case": key, "rejected": True, "error": str(error)})
        else:
            raise AssertionError("Altered bridge protocol accepted")
    cli = []
    for action in ("build", "inspect"):
        output = work / f"missing-primary-{action}"
        command = [sys.executable, "-B", "-m", "embed_optim.primary_v3_bridge", action]
        options = {
            "repository": root,
            "training-root": args.training_root,
            "primary-protocol": primary.path,
            "bridge-protocol": contract.path,
            "experiment-root": "/root/embedding-optimizer-study",
            "results-root": work,
            "validation-data": work,
            "validation-root": work,
            "outcomes-root": work,
            "geometry-root": work,
            "reference": work,
            "output": output,
        }
        for key, value in options.items():
            command.extend([f"--{key}", str(value)])
        result = subprocess.run(
            command, cwd=root, env=os.environ.copy(), capture_output=True, text=True, check=False
        )
        assert (
            result.returncode != 0
            and "ordinary retained run" in result.stderr
            and not output.exists()
        )
        cli.append(
            {
                "action": action,
                "returncode": result.returncode,
                "stderr": result.stderr,
                "output_created": False,
            }
        )
    for row in initial:
        verify_file(row["path"], row)
    require_same(handoff(), before)
    result = {
        "scope": "engineering_dense_v3_bridge_numerical_rehearsal",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "rehearsal_passed": True,
        "all_positive_panels_are_synthetic": True,
        "formal_bridge_produced": False,
        "scientific_completion": False,
        "model_updates": 0,
        "gpu_workers": 0,
        "protocol": binding(contract.path),
        "fresh_replay": bool(args.replay),
        "replay_parent": None if args.replay is None else binding(args.replay),
        "complete_bundle_readback": saved,
        "independent_full_system_controls": oracle,
        "null_counterexample_corrections": null_results,
        "altered_bundle_cases": corrupted,
        "altered_protocol_cases": changes,
        "actual_missing_primary_cli": cli,
        "inputs": initial,
        "artifacts": [binding(path) for path in sorted(work.rglob("*")) if path.is_file()],
        "post_execution_dispatchers": handoff(),
    }
    write_new(work / "result.json", result)
    print(
        {
            "rehearsal_passed": True,
            "fresh_replay": bool(args.replay),
            "scientific_completion": False,
            **file_identity(work / "result.json"),
        }
    )


if __name__ == "__main__":
    main()
