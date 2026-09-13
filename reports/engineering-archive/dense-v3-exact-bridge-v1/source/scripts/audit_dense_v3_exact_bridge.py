"""CPU-only exact-feature sensitivity audit; positive bridge panels remain synthetic."""

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
from types import SimpleNamespace

import sympy as sp

from embed_optim import primary_v3_exact_bridge as exact
from embed_optim.bridge_named_features import evaluate_named
from embed_optim.bridge_numerics import evaluate
from embed_optim.dimension_publication import BRIDGE_FEATURES as DIMENSION_FEATURES
from embed_optim.dimension_publication import _evaluate_bridge as legacy_dimension_bridge
from embed_optim.exact_bridge_measurements import (
    CHECKPOINT_FEATURES,
    FEATURES,
    checkpoint_features,
    pair_features,
)
from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_outcomes import csv_bytes, inspect_bundle, save_bundle
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff
from scripts.audit_dense_v3_bridge import independent_ranks


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), **file_identity(path)}


def independent_systems(tables):
    rows = tables["bridge_rows"]
    b = sp.Matrix(
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
    y = sp.Matrix([sp.Rational(r["mean_ndcg_at_10"]) for r in rows])
    saved = {(r["feature"], r["run_id"], r["stage"]): r for r in tables["held_out_predictions"]}
    counts, errors, systems = 0, 0, []
    for fold in range(1, 5):
        train = [i for i, r in enumerate(rows) if r["dose_index"] != fold]
        test = [i for i, r in enumerate(rows) if r["dose_index"] == fold]
        base = b.extract(train, range(8))
        target = y.extract(train, [0])
        beta_b = (base.T * base).inv(method="DM") * base.T * target
        baseline = b.extract(test, range(8)) * beta_b
        for feature in FEATURES:
            design = b.row_join(sp.Matrix([sp.Rational(r[feature]) for r in rows]))
            training = design.extract(train, range(9))
            assert training.rank() == 9
            beta = (training.T * training).inv(method="DM") * training.T * target
            predicted = design.extract(test, range(9)) * beta
            for position, i in enumerate(test):
                expected = saved[(feature, rows[i]["run_id"], rows[i]["stage"])]
                assert predicted[position] == sp.Rational(expected["feature_prediction_exact"])
                assert baseline[position] == sp.Rational(expected["baseline_prediction_exact"])
                counts += 1
            residual = y.extract(test, [0]) - predicted
            base_residual = y.extract(test, [0]) - baseline
            am, bm = residual.dot(residual) / 15, base_residual.dot(base_residual) / 15
            stored = next(
                r
                for r in tables["leave_dose_fold_metrics"]
                if r["feature"] == feature and r["held_out_dose_index"] == fold
            )
            assert sp.Rational(stored["feature_mse_exact"]) == am
            assert sp.Rational(stored["baseline_mse_exact"]) == bm
            assert sp.Rational(stored["mse_reduction_exact"]) == bm - am
            assert stored["feature_improves"] == bool(bm > am)
            errors += 1
            systems.append(
                {"feature": feature, "fold": fold, "exact_equal": True, "predictions": 15}
            )
    projection = b * (b.T * b).inv(method="DM") * b.T
    yr = y - projection * y

    def corr(a, c):
        a, c = a - sp.ones(60, 1) * sum(a) / 60, c - sp.ones(60, 1) * sum(c) / 60
        return float((a.dot(c) / sp.sqrt(a.dot(a) * c.dot(c))).evalf(80))

    associations = []
    for feature in FEATURES:
        vector = sp.Matrix([sp.Rational(r[feature]) for r in rows])
        residual = vector - projection * vector
        pearson, spearman = (
            corr(residual, yr),
            corr(independent_ranks(residual), independent_ranks(yr)),
        )
        row = next(r for r in tables["residual_associations"] if r["feature"] == feature)
        assert math.isclose(
            row["pearson_residual_association"], pearson, rel_tol=2e-12, abs_tol=2e-14
        )
        assert math.isclose(
            row["spearman_residual_association"], spearman, rel_tol=2e-12, abs_tol=2e-14
        )
        associations.append(
            {"feature": feature, "pearson": pearson, "spearman": spearman, "passed": True}
        )
        folds = [r for r in tables["leave_dose_fold_metrics"] if r["feature"] == feature]
        delta = sum(sp.Rational(r["mse_reduction_exact"]) for r in folds) / 4
        summary = next(r for r in tables["feature_prediction_summary"] if r["feature"] == feature)
        assert sp.Rational(summary["pooled_mse_reduction_exact"]) == delta
        assert summary["predictively_useful"] == (
            bool(delta > 0) and sum(r["feature_improves"] for r in folds) >= 3
        )
    return {
        "exact_predictions": counts,
        "exact_fold_mse_comparisons": errors,
        "exact_pooled_decisions": 5,
        "systems": systems,
        "residual_associations": associations,
    }


def real_geometry_mapping(root):
    receipt_path = root / "reports/engineering-archive/dense-v3-exact-geometry-v1/validation.json"
    assert (
        file_identity(receipt_path)["sha256"]
        == "63076c3fcabbca857601b72f5dd55f83698d25134980b54f71304dbf517fa993"
    )
    accepted = read_json(receipt_path)
    source = Path("/tmp/dense-v3-exact-geometry.CBVst9/checkpoint_exact_geometry.csv")
    bound = next(row for row in accepted["external_payload_checks"] if row["path"] == str(source))
    verify_file(source, bound)
    rows = exact.typed_csv(source)
    # This geometry-only projection uses already independently accepted diagnostic row identities.
    configs = [
        SimpleNamespace(
            run_id=row["run_id"],
            optimizer=SimpleNamespace(name=row["optimizer"], lr=row["learning_rate"]),
        )
        for row in rows
        if row["stage"] == 1
    ]
    result = checkpoint_features(rows, configs, [1, 2, 3])
    comparisons = []
    for row in rows:
        extracted = result[(row["run_id"], row["stage"])]
        for feature, (column, _, _) in CHECKPOINT_FEATURES.items():
            assert extracted["values"][feature] == row[column]
            comparisons.append(
                {
                    "run_id": row["run_id"],
                    "stage": row["stage"],
                    "feature": feature,
                    "value": extracted["values"][feature],
                    "verified": True,
                }
            )
    try:
        pair_features([], configs, [1, 2, 3])
    except ValueError as error:
        refused = str(error)
    else:
        raise AssertionError("Diagnostic population accepted as full primary comparator population")
    assert len(comparisons) == 27 and sum(r["value"] is None for r in comparisons) == 9
    verify_file(source, bound)
    return {
        "scope": "engineering_geometry_only_projection",
        "source": binding(source),
        "comparisons": comparisons,
        "undefined": 9,
        "defined": 18,
        "diagnostic_pair_population_rejected": refused,
        "retrieval_outcomes_added": False,
        "primary_identity_assigned": False,
    }


def dimension_control(root):
    source = (
        root / "reports/engineering-archive/dense-v3-exact-geometry-v1/bridge-null-feature.json"
    )
    assert (
        file_identity(source)["sha256"]
        == "5c8d8e3fbd0f9a13c3bd6a4fbcd23a2bcc6f2f05d40ac42aa12fca07e77067b0"
    )
    rows = copy.deepcopy(read_json(source)["cases"][1]["rows"])
    for row in rows:
        row.update({feature: row["stage"] / 8 for feature in DIMENSION_FEATURES})
    old_folds, old_summary = legacy_dimension_bridge(rows)
    new_tables, new_diagnostics = evaluate_named(rows, DIMENSION_FEATURES)
    assert all(r["predictively_useful"] is True for r in old_summary)
    assert all(
        r["predictively_useful"] is False and r["pooled_mse_reduction_exact"] == "0"
        for r in new_tables["feature_prediction_summary"]
    )
    return {
        "scope": "engineering_synthetic_dimension_numeric_control",
        "rows": rows,
        "legacy_folds": old_folds,
        "legacy_summary": old_summary,
        "named_tables": new_tables,
        "named_diagnostics": new_diagnostics,
        "primary_dimension_pipeline_integrated": False,
        "four_feature_slots_repeat_one_counterexample": True,
    }


def corruptions(work, origin, plan, evidence, tables):
    result = []
    for name in (
        "measurement_comparison",
        "predictive_sensitivity_summary",
        "held_out_predictions",
        "coverage",
    ):
        output = work / f"altered-{name}"
        shutil.copytree(origin, output)
        if name == "coverage":
            value = copy.deepcopy(evidence)
            value["measurement_coverage"][0]["checkpoint"]["saved_segment"][
                "nonzero_parameters"
            ] -= 1
            filename, content = "evidence.json", json.dumps(value, sort_keys=True).encode()
        else:
            value = copy.deepcopy(tables[name])
            key = next(k for k, v in value[0].items() if type(v) is float)
            value[0][key] += 0.0001
            filename, content = name + ".csv", csv_bytes(value)
        (output / filename).write_bytes(content)
        manifest = read_json(output / "manifest.json")
        manifest["outputs"][filename] = {"path": filename, **file_identity(output / filename)}
        (output / "manifest.json").write_text(json.dumps(manifest))
        try:
            inspect_bundle(output, plan, evidence, tables)
        except ValueError as error:
            result.append({"case": name, "rejected": True, "error": str(error)})
        else:
            raise AssertionError("Rehashed sensitivity artifact accepted")
    return result


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
        raise ValueError("Require CPU-only assertions and an explicit empty workdir")
    primary = PrimaryV3Contract.load(root / exact.PARENTS["primary"][0], root, args.training_root)
    contract = exact.ExactBridgeContract.load(
        root / "configs/dense_primary_v3_exact_bridge_protocol.json", primary
    )
    assert contract.sha256 == args.expected_protocol_sha256
    before = handoff()
    initial = [binding(root / name) for name in exact.SOURCES]
    initial += [binding(root / name) for name, _ in exact.PARENTS.values()]
    initial += [
        binding(root / name)
        for name in (
            "tests/test_exact_bridge_measurements.py",
            "tests/test_primary_v3_bridge.py",
            "tests/test_primary_v3_exact_bridge.py",
            "src/embed_optim/dimension_publication.py",
            "configs/dense_dimension_utilization_protocol.json",
            "scripts/audit_dense_v3_bridge.py",
            "reports/engineering-archive/dense-v3-exact-geometry-v1/bridge-null-feature.json",
        )
    ]
    initial += [binding(contract.path), binding(Path(__file__))]
    if args.replay:
        assert file_identity(args.replay)["sha256"] == args.replay_sha256
        parent = read_json(args.replay)
        for row in (*parent["inputs"], *parent["artifacts"]):
            verify_file(row["path"], row)
        origin = args.replay.parent
        fixture = read_json(origin / "fixture-input.json")
    else:
        generate = runpy.run_path(str(root / "tests/test_exact_bridge_measurements.py"))["fixture"]
        original, checkpoints, pairs, _, _ = generate(primary)
        fixture = {
            "scope": "engineering_synthetic_exact_bridge_inputs",
            "not_primary_evidence": True,
            "original_rows": original,
            "checkpoints": checkpoints,
            "pairs": pairs,
        }
        origin = work
    assert (
        fixture["scope"] == "engineering_synthetic_exact_bridge_inputs"
        and fixture["not_primary_evidence"] is True
    )
    original_tables, _ = evaluate(fixture["original_rows"])
    tables, metadata = exact.sensitivity_tables(
        primary, original_tables, fixture["checkpoints"], fixture["pairs"]
    )
    evidence = {"fixture": fixture, **metadata}
    plan = {
        "scope": "engineering_synthetic_exact_bridge_fixture",
        "exact_bridge_protocol_sha256": contract.sha256,
        "scientific_completion": False,
        "not_primary_evidence": True,
    }
    if args.replay:
        saved = inspect_bundle(origin / "fixture-bundle", plan, evidence, tables)
        independent = None
    else:
        write_new(work / "fixture-input.json", fixture)
        saved = save_bundle(work / "fixture-bundle", plan, evidence, tables)
        independent = independent_systems(tables)
    altered = corruptions(work, origin / "fixture-bundle", plan, evidence, tables)
    real = real_geometry_mapping(root)
    initial.append(real["source"])
    dimension = dimension_control(root)
    changes = []
    for key in (
        "checkpoint_feature_mapping",
        "pair_feature_mapping",
        "numerical_policy",
        "undefined_policy",
        "measurement_rules",
        "table_counts",
        "parents",
        "sources",
        "amendment",
        "status",
    ):
        value = copy.deepcopy(contract.payload)
        if key == "status":
            value[key] = "released"
        else:
            value[key].pop(next(iter(value[key])))
        path = work / f"altered-protocol-{key}.json"
        write_new(path, value)
        try:
            exact.ExactBridgeContract.load(path, primary)
        except (ValueError, KeyError) as error:
            changes.append({"case": key, "rejected": True, "error": str(error)})
        else:
            raise AssertionError("Changed exact sensitivity protocol accepted")
    cli = []
    for action in ("build", "inspect"):
        output = work / f"missing-primary-{action}"
        command = [sys.executable, "-B", "-m", "embed_optim.primary_v3_exact_bridge", action]
        options = {
            "repository": root,
            "training-root": args.training_root,
            "primary-protocol": primary.path,
            "exact-bridge-protocol": contract.path,
            "experiment-root": "/root/embedding-optimizer-study",
            "results-root": work,
            "validation-data": work,
            "validation-root": work,
            "outcomes-root": work,
            "geometry-root": work,
            "bridge-root": work,
            "exact-geometry-root": work,
            "reference": work,
            "output": output,
        }
        for key, value in options.items():
            command.extend([f"--{key}", str(value)])
        called = subprocess.run(
            command, cwd=root, env=os.environ.copy(), capture_output=True, text=True, check=False
        )
        assert (
            called.returncode != 0
            and "ordinary retained run" in called.stderr
            and not output.exists()
        )
        cli.append(
            {
                "action": action,
                "returncode": called.returncode,
                "stderr": called.stderr,
                "output_created": False,
            }
        )
    for row in initial:
        verify_file(row["path"], row)
    require_same(handoff(), before)
    result = {
        "scope": "engineering_dense_v3_exact_bridge_rehearsal",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "rehearsal_passed": True,
        "all_positive_retrieval_panels_are_synthetic": True,
        "formal_exact_bridge_produced": False,
        "scientific_completion": False,
        "model_updates": 0,
        "gpu_workers": 0,
        "protocol": binding(contract.path),
        "fresh_replay": bool(args.replay),
        "replay_parent": None if args.replay is None else binding(args.replay),
        "complete_bundle_readback": saved,
        "independent_full_system_controls": independent,
        "real_diagnostic_geometry_mapping": real,
        "dimension_numeric_control": dimension,
        "altered_bundle_cases": altered,
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
