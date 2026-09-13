"""Complete five-feature exact measurement sensitivity, separate from the original bridge."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import primary_v3_bridge as original
from . import primary_v3_exact_geometry as exact_geometry
from .bridge_exact_arithmetic import displayed, rmse_reduction
from .bridge_named_features import MISSING_POLICY, evaluate_named
from .bridge_numerics import NUMERICAL_POLICY
from .exact_bridge_measurements import (
    CHECKPOINT_FEATURES,
    FEATURES,
    MEASUREMENT_RULES,
    PAIR_FEATURES,
    assemble_exact_panel,
)
from .primary_contract import DRAFT, digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_outcomes import inspect_bundle, recipe_views, save_bundle

SCOPE = "dense_primary_v3_exact_bridge_sensitivity"
PARENTS = {
    "primary": original.PARENTS["primary"],
    "original_bridge": (
        "configs/dense_primary_v3_bridge_protocol.json",
        "540c3c3041ea13d9029fb5534283187b4f4141dfc66eba668938a7dc7d8ae803",
    ),
    "exact_geometry": (
        "configs/dense_primary_v3_exact_geometry_protocol.json",
        "b13351152e6bd4b8ec2562d46c4e8bd2c1c2496bfd4bae7d4dabd093c47cb9f5",
    ),
    "bridge_acceptance": (
        "reports/engineering-archive/dense-v3-bridge-v1/validation.json",
        "f8a36ff600ba431fb71dee4c82edfa5aa87dae0513bd1e3b3ec186f81250110f",
    ),
}
SOURCES = tuple(
    sorted(
        set(original.SOURCES)
        | set(exact_geometry.SOURCES)
        | {
            "src/embed_optim/primary_v3_exact_bridge.py",
            "src/embed_optim/bridge_named_features.py",
            "src/embed_optim/exact_bridge_measurements.py",
            "scripts/prepare_dense_v3_exact_bridge.py",
        }
    )
)
TABLE_COUNTS = {
    "bridge_rows": 60,
    "leave_dose_fold_metrics": 20,
    "feature_prediction_summary": 5,
    "residual_associations": 5,
    "held_out_predictions": 300,
    "measurement_comparison": 300,
    "predictive_sensitivity_summary": 5,
}
AMENDMENT = {
    "additional_named_features": list(FEATURES),
    "original_features_overwritten": False,
    "numerical_policy_changed": False,
    "undefined_measurement_domain_explicit": True,
    "entropy_and_nonzero_means_are_distinct_estimands": True,
    "four_dimension_features_only_numerically_tested_not_primary_admitted": True,
    "full_primary_results_observed_when_prepared": False,
    "boundary": "A predeclared post-diagnosis measurement sensitivity on this one-seed grid, not feature selection, causal mediation, functional dimension utility or an optimizer verdict. Report every original/exact correspondence and every undefined state.",
}


@dataclass(frozen=True)
class ExactBridgeContract:
    path: Path
    payload: dict
    sha256: str
    bridge: original.BridgeContract
    exact: exact_geometry.ExactGeometryContract

    @property
    def primary(self):
        return self.bridge.primary

    @classmethod
    def load(cls, path, primary):
        path, root = Path(path).resolve(), primary.repository
        value = read_json(path)
        if (
            value.get("scope") != SCOPE
            or value.get("status") != DRAFT
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("scientific_completion") is not False
            or value.get("formal_execution_authorized") is not False
            or primary.sha256 != PARENTS["primary"][1]
            or set(value.get("parents", {})) != set(PARENTS)
            or set(value.get("sources", {})) != set(SOURCES)
        ):
            raise ValueError("Not the complete preparation-only exact bridge sensitivity")
        for key, (name, sha) in PARENTS.items():
            binding = value["parents"][key]
            if binding["path"] != name or binding["sha256"] != sha:
                raise ValueError("Changed immutable exact bridge parent")
            verify_file(root / name, binding)
        for key, expected in (
            ("numerical_policy", NUMERICAL_POLICY),
            ("undefined_policy", MISSING_POLICY),
            ("measurement_rules", MEASUREMENT_RULES),
            ("amendment", AMENDMENT),
            ("table_counts", TABLE_COUNTS),
        ):
            require_same(value[key], expected)
        require_same(value["checkpoint_feature_mapping"], CHECKPOINT_FEATURES)
        require_same(value["pair_feature_mapping"], PAIR_FEATURES)
        for name, binding in value["sources"].items():
            verify_file(root / name, binding)
        verify_file(
            Path(__file__).resolve(), value["sources"]["src/embed_optim/primary_v3_exact_bridge.py"]
        )
        parent_bridge = original.BridgeContract.load(root / PARENTS["original_bridge"][0], primary)
        exact = exact_geometry.ExactGeometryContract.load(
            root / PARENTS["exact_geometry"][0], primary
        )
        if (
            parent_bridge.sha256 != PARENTS["original_bridge"][1]
            or exact.sha256 != PARENTS["exact_geometry"][1]
        ):
            raise ValueError("Actual exact bridge upstream parent differs")
        return cls(path, value, file_identity(path)["sha256"], parent_bridge, exact)

    def recheck(self):
        primary = PrimaryV3Contract.load(
            self.primary.path, self.primary.repository, self.primary.training_root
        )
        current = type(self).load(self.path, primary)
        require_same(current.sha256, self.sha256)
        return current


def typed_csv(path):
    """Decode only a table whose bytes the upstream numeric reader has already checked."""

    def bad_constant(value):
        raise ValueError(f"Non-finite source CSV value: {value}")

    def parse(value):
        if value == "":
            return None
        if value in ("True", "False"):
            return value == "True"
        try:
            return json.loads(value, parse_constant=bad_constant)
        except json.JSONDecodeError:
            return value

    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Invalid or duplicate CSV field names")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed source CSV row")
    return [{key: parse(value) for key, value in row.items()} for row in rows]


def sensitivity_tables(primary, original_tables, exact_checkpoints, exact_pairs):
    rows, comparisons, coverage = assemble_exact_panel(
        original_tables["bridge_rows"],
        exact_checkpoints,
        exact_pairs,
        recipe_views(primary),
        primary.payload["checkpoint_steps"],
    )
    tables, numerical = evaluate_named(rows, FEATURES)
    previous = {row["feature"]: row for row in original_tables["feature_prediction_summary"]}
    contrasted = []
    for row in tables["feature_prediction_summary"]:
        feature = row["feature"]
        old_name = (
            CHECKPOINT_FEATURES[feature][2]
            if feature in CHECKPOINT_FEATURES
            else PAIR_FEATURES[feature][1]
        )
        old = previous[old_name]
        require_same(row["pooled_baseline_mse_exact"], old["pooled_baseline_mse_exact"])
        a, b = old["pooled_feature_mse_exact"], row["pooled_feature_mse_exact"]
        defined = a is not None and b is not None
        difference = Fraction(a) - Fraction(b) if defined else None
        contrasted.append(
            {
                "original_feature": old_name,
                "exact_feature": feature,
                "original_predictively_useful": old["predictively_useful"],
                "exact_predictively_useful": row["predictively_useful"],
                "original_pooled_feature_rmse": old["pooled_feature_rmse"],
                "exact_pooled_feature_rmse": row["pooled_feature_rmse"],
                "original_minus_exact_feature_mse": displayed(difference) if defined else None,
                "original_minus_exact_feature_mse_rational": str(difference) if defined else None,
                "original_minus_exact_feature_rmse": rmse_reduction(Fraction(a), Fraction(b))
                if defined
                else None,
                "defined": defined,
            }
        )
    tables["measurement_comparison"] = comparisons
    tables["predictive_sensitivity_summary"] = contrasted
    require_same({key: len(value) for key, value in tables.items()}, TABLE_COUNTS)
    return tables, {"numerical_diagnostics": numerical, "measurement_coverage": coverage}


def gather(contract, args):
    contract = contract.recheck()
    # The original complete reader admits all primary runs before any score bundle/output.
    old_plan, old_evidence, old_tables = original.gather(contract.bridge, args)
    source_bindings = original.snapshot(
        args.bridge_root,
        ["manifest.json", "evidence.json", *(f"{name}.csv" for name in original.TABLE_COUNTS)],
    )
    old_reader = inspect_bundle(args.bridge_root, old_plan, old_evidence, old_tables)
    source_bindings += original.snapshot(
        args.exact_geometry_root,
        ["summary.json", *(f"{name}.csv" for name in exact_geometry.TABLE_COUNTS)],
    )
    exact_reader = exact_geometry.operate(
        contract.exact,
        args.experiment_root,
        args.reference,
        args.exact_geometry_root,
        produce=False,
    )
    original.verify_snapshot(source_bindings)
    require_same(read_json(Path(args.exact_geometry_root) / "summary.json"), exact_reader)
    tables, measurements = sensitivity_tables(
        contract.primary,
        old_tables,
        typed_csv(Path(args.exact_geometry_root) / "checkpoint_exact_geometry.csv"),
        typed_csv(Path(args.exact_geometry_root) / "run_pair_exact_subspace_overlap.csv"),
    )
    source_bindings += old_evidence["source_bindings"]
    source_bindings += [
        {**row, "path": str(Path(args.exact_geometry_root) / row["path"])}
        for row in exact_reader["raw_bindings"]
    ]
    original.verify_snapshot(source_bindings)
    contract.recheck()
    evidence = {
        "source_bindings": source_bindings,
        "original_bridge_reader": old_reader,
        "original_bridge_evidence": old_evidence,
        "exact_geometry_reader": exact_reader,
        **measurements,
    }
    plan = {
        "scope": SCOPE,
        "exact_bridge_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "original_bridge_protocol_sha256": contract.bridge.sha256,
        "exact_geometry_protocol_sha256": contract.exact.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "boundary": AMENDMENT["boundary"],
    }
    return plan, evidence, tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "inspect"))
    for name in (
        "repository",
        "training-root",
        "primary-protocol",
        "exact-bridge-protocol",
        "experiment-root",
        "results-root",
        "validation-data",
        "validation-root",
        "outcomes-root",
        "geometry-root",
        "bridge-root",
        "exact-geometry-root",
        "reference",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Exact bridge sensitivity is CPU-only")
    primary = PrimaryV3Contract.load(args.primary_protocol, args.repository, args.training_root)
    contract = ExactBridgeContract.load(args.exact_bridge_protocol, primary)
    plan, evidence, tables = gather(contract, args)
    action = save_bundle if args.mode == "build" else inspect_bundle
    result = action(args.output, plan, evidence, tables)
    original.verify_snapshot(evidence["source_bindings"])
    contract.recheck()
    print(
        {
            "exact_bridge_operation_verified": result["recomputed_bytes_verified"],
            "scientific_completion": False,
        }
    )


if __name__ == "__main__":
    main()
