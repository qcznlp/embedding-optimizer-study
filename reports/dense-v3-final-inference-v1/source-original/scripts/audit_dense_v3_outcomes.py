"""CPU-only v3 outcome rehearsal. Every positive statistical table is synthetic, not a model result."""

import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embed_optim import primary_v3_outcomes as outcome
from embed_optim.corrected_outcome_summary import CONTRASTS, paired_max_t_intervals
from embed_optim.decontamination import DECONTAMINATED_TASK_NAMES
from embed_optim.primary_contract import (
    RELEASED,
    file_identity,
    read_json,
    require_same,
    verify_file,
)
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation import ValidationContract, select_recipes
from embed_optim.primary_v3_validation_io import write_new
from scripts.audit_dense_natural_data import handoff


def binding(path):
    return {"path": str(Path(path).resolve()), **file_identity(path)}


def independent_max_t(effects):
    points = [statistics.fmean(effects[k]) for k in CONTRASTS]
    errors = [statistics.stdev(effects[k]) / math.sqrt(14) for k in CONTRASTS]
    indices = np.random.default_rng(20260903).integers(0, 14, size=(50_000, 14)).tolist()
    maxima, samples = [], [[], [], []]
    for positions in indices:
        means = [statistics.fmean(effects[k][i] for i in positions) for k in CONTRASTS]
        for j, value in enumerate(means):
            samples[j].append(value)
        maxima.append(max(abs((means[j] - points[j]) / errors[j]) for j in range(3)))

    def quantile(values, p):
        ordered = sorted(values)
        where = (len(ordered) - 1) * p
        left, weight = math.floor(where), where % 1
        return ordered[left] * (1 - weight) + ordered[min(left + 1, len(ordered) - 1)] * weight

    critical = quantile(maxima, 0.95)
    return {
        k: {
            "mean_delta_ndcg_at_10": points[j],
            "across_task_standard_error": errors[j],
            "simultaneous_critical_value": critical,
            "simultaneous_ci_95_lower": points[j] - critical * errors[j],
            "simultaneous_ci_95_upper": points[j] + critical * errors[j],
            "nominal_bootstrap_ci_95_lower": quantile(samples[j], 0.025),
            "nominal_bootstrap_ci_95_upper": quantile(samples[j], 0.975),
        }
        for j, k in enumerate(CONTRASTS)
    }


def fixture(primary):
    rows, metrics = [], []
    for index, run in enumerate(primary.inputs["runs"]):
        opt = primary.expected_identity(run["run_id"])["recipe"]["optimizer"]
        for stage, step in enumerate(primary.payload["checkpoint_steps"], 1):
            for t, task in enumerate(DECONTAMINATED_TASK_NAMES):
                rows.append(
                    {
                        "run_id": run["run_id"],
                        "optimizer": opt["name"],
                        "learning_rate": opt["lr"],
                        "model_family": "dense",
                        "stage": stage,
                        "step": step,
                        "fraction": stage / 5,
                        "task": task,
                        "ndcg_at_10": 0.25
                        + 0.013 * stage
                        + 0.001 * index
                        + 0.000013 * index**2 * t,
                    }
                )
        metrics.append(
            {
                "run_id": run["run_id"],
                "optimizer": opt["name"],
                "learning_rate": opt["lr"],
                "contrastive_loss": 0.9 - 0.01 * (index % 4),
                "positive_margin": -index,
            }
        )
    selection = {
        "scope": "dense_primary_v3_validation_selection",
        "primary_protocol_sha256": primary.sha256,
        "validation_protocol_sha256": outcome.PARENTS["validation"][1],
        "scientific_completion": False,
        "selected": select_recipes(metrics, primary.inputs["runs"]),
        "run_metrics": metrics,
    }
    return {
        "scope": "engineering_synthetic_outcome_inputs",
        "scientific_completion": False,
        "primary_labels_used_only_to_exercise_adapter": True,
        "score_rows": rows,
        "selection": selection,
    }


def load(root, training_root):
    primary = PrimaryV3Contract.load(root / outcome.PARENTS["primary"][0], root, training_root)
    validation = ValidationContract.load(root / outcome.PARENTS["validation"][0], primary)
    path = root / "configs/dense_primary_v3_outcome_protocol_v2.json"
    if (
        file_identity(root / "configs/dense_primary_v3_outcome_protocol.json")["sha256"]
        != "914996b3423ba01329c03844419cca61e2a33452fe026665bb940d5b82e2cffe"
    ):
        raise ValueError("The failed predecessor was not preserved")
    return outcome.OutcomeContract.load(path, validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--replay-fixture", action="store_true")
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    if sys.flags.optimize or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not work.is_dir():
        raise ValueError("Require CPU-only explicit existing workdir and enabled assertions")
    contract = load(root, args.training_root)
    if contract.sha256 != args.expected_protocol_sha256:
        raise ValueError("Actual v2 protocol differs from the declared invocation")
    if args.replay_fixture:
        expected = read_json(work / "fixture-input.json")
        assert expected["scope"] == "engineering_synthetic_outcome_inputs"
        assert expected["scientific_completion"] is False
        tables = outcome.outcome_tables(
            contract.primary, expected["score_rows"], expected["selection"]
        )
        result = outcome.inspect_bundle(
            work / "fixture-bundle", read_json(work / "fixture-plan.json"), expected, tables
        )
        write_new(work / "fresh-cpu-fixture-readback.json", result)
        print({"fixture_recomputed": True, "scientific_completion": False})
        return
    if any(work.iterdir()):
        raise ValueError("Use a fresh outcome rehearsal directory")
    state_before = handoff()
    initial = [binding(root / p) for p in outcome.SOURCES]
    initial += [binding(root / p) for p, _ in outcome.PARENTS.values()]
    initial += [binding(contract.path), binding(Path(__file__).resolve())]
    evidence = fixture(contract.primary)
    plan = {
        "scope": "engineering_synthetic_outcome_fixture",
        "scientific_completion": False,
        "outcome_protocol_sha256": contract.sha256,
        "not_primary_evidence": True,
    }
    write_new(work / "fixture-input.json", evidence)
    write_new(work / "fixture-plan.json", plan)
    tables = outcome.outcome_tables(contract.primary, evidence["score_rows"], evidence["selection"])
    saved = outcome.save_bundle(work / "fixture-bundle", plan, evidence, tables)
    changed = copy.deepcopy(evidence["selection"])
    for row in changed["run_metrics"]:
        row["contrastive_loss"] = 1 - row["contrastive_loss"]
    changed["selected"] = select_recipes(changed["run_metrics"], contract.primary.inputs["runs"])
    second = outcome.outcome_tables(contract.primary, evidence["score_rows"], changed)
    require_same(tables["primary_summary"], second["primary_summary"])
    require_same(tables["primary_task_effects"], second["primary_task_effects"])
    assert tables["secondary_task_effects"] != second["secondary_task_effects"]
    effects = {
        key: [r[f"{key[0]}_minus_{key[1]}"] for r in tables["primary_task_effects"]]
        for key in CONTRASTS
    }
    scalar, original = independent_max_t(effects), paired_max_t_intervals(effects)
    comparisons = []
    for key in CONTRASTS:
        for field, expected in scalar[key].items():
            actual = original[key][field]
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    "Fixed bootstrap kernel disagrees with independent scalar reference"
                )
            comparisons.append(
                {"contrast": list(key), "field": field, "absolute_error": abs(actual - expected)}
            )
    for path in (work / "fixture-input.json", work / "fixture-plan.json"):
        initial.append(binding(path))
    replay_argv = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--repository",
        str(root),
        "--training-root",
        str(args.training_root),
        "--workdir",
        str(work),
        "--replay-fixture",
        "--expected-protocol-sha256",
        args.expected_protocol_sha256,
    ]
    replay = subprocess.run(replay_argv, capture_output=True, text=True, timeout=180, check=False)
    if replay.returncode != 0:
        raise RuntimeError(f"Fresh fixture readback failed: {replay.stderr}")
    cli_argv = [
        sys.executable,
        "-B",
        "-m",
        "embed_optim.primary_v3_outcomes",
        "build",
        "--protocol",
        str(contract.path),
        "--primary-protocol",
        str(contract.primary.path),
        "--validation-protocol",
        str(contract.validation.path),
        "--repository",
        str(root),
        "--training-root",
        str(args.training_root),
        "--experiment-root",
        "/root/embedding-optimizer-study",
        "--results-root",
        str(work / "absent-beir"),
        "--validation-data",
        "/tmp/dense-partition-candidate.kHXoGW/validation",
        "--validation-root",
        str(work / "absent-validation"),
        "--output",
        str(work / "must-not-be-created"),
    ]
    missing = subprocess.run(cli_argv, capture_output=True, text=True, timeout=180, check=False)
    assert (
        missing.returncode != 0 and "Require an ordinary retained run directory" in missing.stderr
    )
    assert not (work / "must-not-be-created").exists()
    rejections = []

    def refused(name, action, path):
        try:
            action()
        except (ValueError, KeyError) as error:
            rejections.append(
                {"case": name, "rejected": True, "error": str(error), "artifact": binding(path)}
            )
        else:
            raise AssertionError(f"Changed case accepted: {name}")

    changes = [
        ("missing_sources", ("sources",), {}),
        ("old_primary", ("parents", "primary", "sha256"), "0" * 64),
        ("old_validation", ("parents", "validation", "sha256"), "0" * 64),
        ("missing_validation_acceptance", ("parents", "validation_acceptance", "sha256"), "0" * 64),
        (
            "best_rate_estimand",
            ("scientific_rules", "inference", "primary_estimand"),
            "best BEIR rate",
        ),
        ("run_bootstrap", ("scientific_rules", "inference", "resampling_unit"), "run"),
        ("other_seed", ("scientific_rules", "inference", "bootstrap_seed"), 42),
        ("fewer_resamples", ("scientific_rules", "inference", "bootstrap_samples"), 1000),
        ("nominal_support", ("scientific_rules", "inference", "support_rule"), "nominal CI"),
        (
            "impute_initial_score",
            ("scientific_rules", "dynamics", "observed_auc"),
            "include imputed zero-stage",
        ),
        ("missing_high_rate", ("table_counts", "all_task_scores"), 756),
        ("scientific_complete", ("scientific_completion",), True),
        ("claim_execution_authority", ("formal_execution_authorized",), True),
    ]
    for name, keys, value in changes:
        payload = copy.deepcopy(contract.payload)
        node = payload
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
        path = work / f"invalid-{name}.json"
        write_new(path, payload)
        refused(name, lambda p=path: outcome.OutcomeContract.load(p, contract.validation), path)
    # A reviewed release is not a status-only edit. Preserve the actual rejection;
    # never patch loaders, commit source, or claim authority from this fixture.
    projected = copy.deepcopy(contract.primary.payload)
    projected["status"] = RELEASED
    path = work / "invalid-status-only-primary-release.json"
    write_new(path, projected)
    projected_primary = PrimaryV3Contract.load(path, root, args.training_root)
    refused(
        "status_only_release_changes_bound_parent",
        lambda: ValidationContract.load(contract.validation.path, projected_primary),
        path,
    )
    original_bundle = work / "fixture-bundle"
    for name in (
        "primary_summary.csv",
        "secondary_summary.csv",
        "run_stage_scores.csv",
        "evidence.json",
    ):
        import shutil

        target = work / f"invalid-bundle-{name.replace('.', '-')}"
        shutil.copytree(original_bundle, target)
        changed_path = target / name
        # Append a byte rather than regenerate a statistic. Update only this invalid copy's hash.
        changed_path.write_bytes(changed_path.read_bytes() + b" ")
        manifest_path = target / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["outputs"][name] = {"path": name, **file_identity(changed_path)}
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        refused(
            f"rehashed_{name}",
            lambda p=target: outcome.inspect_bundle(p, plan, evidence, tables),
            manifest_path,
        )
    for row in initial:
        verify_file(row["path"], row)
    state_after = handoff()
    require_same(state_before, state_after)
    artifacts = [binding(p) for p in sorted(work.rglob("*")) if p.is_file()]
    result = {
        "scope": "engineering_dense_v3_outcome_rehearsal",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "rehearsal_passed": True,
        "synthetic_grid_cells": 840,
        "all_positive_tables_are_synthetic": True,
        "original_kernel_scalar_comparisons": comparisons,
        "scalar_atol": 1e-12,
        "scalar_rtol": 1e-12,
        "numpy_version": np.__version__,
        "bootstrap_samples": 50_000,
        "bootstrap_seed": 20260903,
        "primary_unchanged_when_validation_selection_changes": True,
        "secondary_changes_when_selection_changes": True,
        "saved_fixture": saved,
        "fresh_cpu_fixture_readback": read_json(work / "fresh-cpu-fixture-readback.json"),
        "replay_cli": {
            "argv": replay_argv,
            "returncode": replay.returncode,
            "stdout": replay.stdout,
            "stderr": replay.stderr,
        },
        "actual_missing_primary_cli": {
            "argv": cli_argv,
            "returncode": missing.returncode,
            "stdout": missing.stdout,
            "stderr": missing.stderr,
        },
        "changed_cases": rejections,
        "all_changed_cases_rejected": True,
        "unchanged_inputs_and_sources": initial,
        "artifacts": artifacts,
        "post_execution_dispatchers": state_after,
        "formal_outcome_produced": False,
        "primary_recipe_selection_produced": False,
        "formal_replication_ready": False,
        "source_published": False,
        "scientific_completion": False,
        "model_updates": 0,
        "gpu_workers": 0,
        "network_uploads": 0,
        "boundary": "Synthetic arithmetic/readback checks and an actual missing-input rejection, not complete primary model evidence. Draft parent hashes cannot become a runnable release through a status edit; an explicit reviewed source/runtime/parent transition remains required.",
    }
    write_new(work / "result.json", result)
    print(
        {
            "rehearsal_passed": True,
            "changed_cases": len(rejections),
            "scalar_comparisons": len(comparisons),
            "formal_outcome_produced": False,
            **file_identity(work / "result.json"),
        }
    )


if __name__ == "__main__":
    main()
