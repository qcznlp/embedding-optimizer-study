"""Complete v3 retrieval bridge with explicit exact-OLS numerical amendment.

No source bundle is trusted just because its manifest hashes match: outcomes
and geometry are freshly recomputed through their own unchanged v3 readers.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

from .bridge_numerics import NUMERICAL_POLICY, evaluate
from .corrected_retrieval_bridge import FEATURES, assemble_bridge_rows
from .primary_contract import DRAFT, digest, file_identity, read_json, require_same, verify_file
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_geometry import GeometryContract, admit_primary, operate
from .primary_v3_outcomes import (
    TABLE_COUNTS as OUTCOME_COUNTS,
)
from .primary_v3_outcomes import (
    OutcomeContract,
    collect_evidence,
    inspect_bundle,
    recipe_views,
    save_bundle,
)
from .primary_v3_validation import ValidationContract

SCOPE = "dense_primary_v3_retrieval_bridge"
PARENTS = {
    "primary": (
        "configs/dense_primary_v3_protocol.json",
        "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b",
    ),
    "validation": (
        "configs/dense_primary_v3_validation_protocol.json",
        "96a8aa2fc684d29ec67649c2e637c0cd4c7dcdd0f31f2e354b7d0955a18fb1c3",
    ),
    "outcomes": (
        "configs/dense_primary_v3_outcome_protocol_v2.json",
        "2a12b78a1a309cd6cc4a56593e4b1b1030370fc579045d1708d572eb5577d68e",
    ),
    "geometry": (
        "configs/dense_primary_v3_geometry_protocol_v2.json",
        "9b6dad688072c08e8994e45f80c897a7889ab5fbcbccab27b643944c85ae625b",
    ),
    "analysis": (
        "configs/dense_no_packing_analysis_protocol.json",
        "91d7fa09ebb7e609afc5eb584f499b97baf9718d8cc23ff67eb295adfdd9f27a",
    ),
    "legacy_bridge": (
        "configs/dense_no_packing_bridge_implementation_protocol_v2.json",
        "88fb013be3fc69c7477750314af264df78080ba2573c578c95e886bb0712ec46",
    ),
    "exact_geometry_acceptance": (
        "reports/engineering-archive/dense-v3-exact-geometry-v1/validation.json",
        "63076c3fcabbca857601b72f5dd55f83698d25134980b54f71304dbf517fa993",
    ),
    "null_counterexample": (
        "reports/engineering-archive/dense-v3-exact-geometry-v1/bridge-null-feature.json",
        "5c8d8e3fbd0f9a13c3bd6a4fbcd23a2bcc6f2f05d40ac42aa12fca07e77067b0",
    ),
}
SOURCES = (
    "src/embed_optim/primary_v3_bridge.py",
    "src/embed_optim/bridge_numerics.py",
    "src/embed_optim/bridge_exact_arithmetic.py",
    "scripts/prepare_dense_v3_bridge.py",
    "src/embed_optim/corrected_retrieval_bridge.py",
    "src/embed_optim/primary_contract.py",
    "src/embed_optim/primary_v3_contract.py",
    "src/embed_optim/primary_v3_outcomes.py",
    "src/embed_optim/primary_v3_geometry.py",
    "src/embed_optim/primary_v3_geometry_io.py",
    "src/embed_optim/primary_v3_validation.py",
    "src/embed_optim/primary_v3_validation_io.py",
)
TABLE_COUNTS = {
    "bridge_rows": 60,
    "leave_dose_fold_metrics": 36,
    "feature_prediction_summary": 9,
    "residual_associations": 9,
    "held_out_predictions": 540,
}
AMENDMENT = {
    "old_source_and_protocol_preserved": True,
    "features_baseline_folds_and_mathematical_support_rule_changed": False,
    "numerical_implementation_changed": True,
    "undefined_domain_explicitly_extended": True,
    "full_primary_results_observed_when_prepared": False,
    "exact_geometry_feature_family_added": False,
    "boundary": "One-seed fixed-grid descriptive prediction/association, not significance, mediation, causal explanation or useful embedding dimensions. Original approximate features retain their measured limitations; exact sensitivity remains separate work.",
}


@dataclass(frozen=True)
class BridgeContract:
    path: Path
    payload: dict
    sha256: str
    outcomes: OutcomeContract
    geometry: GeometryContract

    @property
    def primary(self):
        return self.outcomes.primary

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
            raise ValueError("Not the complete preparation-only v3 bridge contract")
        for key, (name, sha) in PARENTS.items():
            binding = value["parents"][key]
            if binding["path"] != name or binding["sha256"] != sha:
                raise ValueError("Changed immutable bridge parent")
            verify_file(root / name, binding)
        require_same(value["numerical_policy"], NUMERICAL_POLICY)
        require_same(value["amendment"], AMENDMENT)
        require_same(value["features"], list(FEATURES))
        require_same(value["table_counts"], TABLE_COUNTS)
        require_same(
            value["scientific_rules"], read_json(root / PARENTS["analysis"][0])["retrieval_bridge"]
        )
        for name, binding in value["sources"].items():
            verify_file(root / name, binding)
        verify_file(
            Path(__file__).resolve(), value["sources"]["src/embed_optim/primary_v3_bridge.py"]
        )
        validation = ValidationContract.load(root / PARENTS["validation"][0], primary)
        outcomes = OutcomeContract.load(root / PARENTS["outcomes"][0], validation)
        geometry = GeometryContract.load(root / PARENTS["geometry"][0], primary)
        if outcomes.sha256 != PARENTS["outcomes"][1] or geometry.sha256 != PARENTS["geometry"][1]:
            raise ValueError("Actual bridge downstream parent differs")
        return cls(path, value, file_identity(path)["sha256"], outcomes, geometry)

    def recheck(self):
        primary = PrimaryV3Contract.load(
            self.primary.path, self.primary.repository, self.primary.training_root
        )
        current = type(self).load(self.path, primary)
        require_same(current.sha256, self.sha256)
        return current


def snapshot(directory, names):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Require an ordinary admitted source bundle")
    return [{"path": str(directory / name), **file_identity(directory / name)} for name in names]


def verify_snapshot(bindings):
    for row in bindings:
        verify_file(row["path"], row)


def bridge_tables(primary, checkpoints, pairs, scores):
    rows = assemble_bridge_rows(checkpoints, pairs, scores, recipe_views(primary))
    tables, numerical = evaluate(rows)
    require_same({key: len(value) for key, value in tables.items()}, TABLE_COUNTS)
    return tables, numerical


def gather(contract, args):
    contract = contract.recheck()
    # Fail on missing primary runs before opening scores, diagnostic exports or output paths.
    admit_primary(contract.geometry, args.experiment_root)
    source_bindings = snapshot(
        args.outcomes_root,
        ["manifest.json", "evidence.json", *(f"{name}.csv" for name in OUTCOME_COUNTS)],
    )
    source_bindings += snapshot(
        args.geometry_root,
        ["summary.json", *(f"{name}.csv" for name in contract.geometry.payload["table_counts"])],
    )
    plan, evidence, outcome_tables = collect_evidence(
        contract.outcomes,
        args.experiment_root,
        args.results_root,
        args.validation_data,
        args.validation_root,
    )
    outcome_reader = inspect_bundle(args.outcomes_root, plan, evidence, outcome_tables)
    geometry_reader = operate(
        contract.geometry, args.experiment_root, args.reference, args.geometry_root, produce=False
    )
    verify_snapshot(source_bindings)
    require_same(read_json(Path(args.geometry_root) / "summary.json"), geometry_reader)
    # Stored CSVs are used only after all raw geometry and every table byte were recomputed.
    geometry_tables = {}
    for name in ("checkpoint_geometry", "run_pair_subspace_overlap"):
        with (Path(args.geometry_root) / f"{name}.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            geometry_tables[name] = list(csv.DictReader(stream))
    tables, numerical = bridge_tables(
        contract.primary,
        geometry_tables["checkpoint_geometry"],
        geometry_tables["run_pair_subspace_overlap"],
        outcome_tables["run_stage_scores"],
    )
    source_bindings += [
        {**row, "path": str(Path(args.geometry_root) / row["path"])}
        for row in geometry_reader["raw_bindings"]
    ]
    verify_snapshot(source_bindings)
    contract.recheck()
    bound = {
        "source_bindings": source_bindings,
        "outcome_reader": outcome_reader,
        "outcome_evidence": evidence,
        "geometry_reader": geometry_reader,
        "numerical_diagnostics": numerical,
    }
    output_plan = {
        "scope": SCOPE,
        "bridge_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "outcome_protocol_sha256": contract.outcomes.sha256,
        "geometry_protocol_sha256": contract.geometry.sha256,
        "evidence_sha256": digest(bound),
        "scientific_completion": False,
        "boundary": AMENDMENT["boundary"],
    }
    return output_plan, bound, tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "inspect"))
    for name in (
        "repository",
        "training-root",
        "primary-protocol",
        "bridge-protocol",
        "experiment-root",
        "results-root",
        "validation-data",
        "validation-root",
        "outcomes-root",
        "geometry-root",
        "reference",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Bridge is a CPU-only consumer")
    primary = PrimaryV3Contract.load(args.primary_protocol, args.repository, args.training_root)
    contract = BridgeContract.load(args.bridge_protocol, primary)
    plan, evidence, tables = gather(contract, args)
    action = save_bundle if args.mode == "build" else inspect_bundle
    result = action(args.output, plan, evidence, tables)
    verify_snapshot(evidence["source_bindings"])
    contract.recheck()
    print(
        {
            "bridge_operation_verified": result["recomputed_bytes_verified"],
            "scientific_completion": False,
        }
    )


if __name__ == "__main__":
    main()
