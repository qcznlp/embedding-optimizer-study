"""V3 outcome integration: complete primary evidence, unchanged statistical estimands.

This adapter reads authenticated primary/validation artifacts. It cannot train,
select using BEIR, fill missing cells, or publish a manuscript. Pure table helpers
also support explicitly labelled fixtures; they do not authenticate model results.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .corrected_outcome_summary import summarize_score_rows
from .decontamination import DECONTAMINATED_TASK_NAMES
from .primary_completion import integer, number
from .primary_contract import (
    DRAFT,
    canonical,
    digest,
    file_identity,
    read_json,
    relative_path,
    require_same,
    verify_file,
)
from .primary_v3_completion import inspect_matrix
from .primary_v3_contract import PrimaryV3Contract
from .primary_v3_validation import ValidationContract, select_recipes
from .primary_v3_validation_io import collect_selection, write_new

SCOPE = "dense_primary_v3_outcomes"
PARENTS = {
    "primary": (
        "configs/dense_primary_v3_protocol.json",
        "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b",
    ),
    "validation": (
        "configs/dense_primary_v3_validation_protocol.json",
        "96a8aa2fc684d29ec67649c2e637c0cd4c7dcdd0f31f2e354b7d0955a18fb1c3",
    ),
    "validation_acceptance": (
        "reports/engineering-archive/dense-v3-validation-v1/validation.json",
        "1967ca78e216df0934e0b379808fd10c98598b8054e936f3f2ae2f3806f363b4",
    ),
    "statistical_parent": (
        "configs/dense_no_packing_outcome_protocol.json",
        "7f321f44b73bf18e88321204781814ff143e4072acf1f26780db17c341188b35",
    ),
    "outcome_predecessor": (
        "configs/dense_primary_v3_outcome_protocol.json",
        "914996b3423ba01329c03844419cca61e2a33452fe026665bb940d5b82e2cffe",
    ),
}
IMPLEMENTATION_REVISION = {
    "version": 2,
    "change": "CSV columns use sorted field names, independent of input dictionary insertion order",
    "reason": "Fresh JSON readback preserves every value but changes dictionary key order",
    "statistical_kernel_changed": False,
    "estimands_or_thresholds_changed": False,
    "previous_failure_overridden": False,
}
SOURCES = (
    "src/embed_optim/primary_v3_outcomes.py",
    "scripts/prepare_dense_v3_outcomes.py",
    "src/embed_optim/corrected_outcome_summary.py",
    "src/embed_optim/aggregate.py",
    "src/embed_optim/decontamination.py",
    "src/embed_optim/config.py",
    "src/embed_optim/primary_contract.py",
    "src/embed_optim/primary_completion.py",
    "src/embed_optim/primary_v3_completion.py",
    "src/embed_optim/primary_v3_validation.py",
    "src/embed_optim/primary_v3_validation_io.py",
)
SCIENCE_KEYS = ("inference", "dynamics", "systems", "expected_outputs", "claim_boundary")
TABLE_COUNTS = {
    "all_task_scores": 840,
    "primary_task_effects": 14,
    "primary_summary": 3,
    "secondary_task_effects": 14,
    "secondary_summary": 3,
    "run_stage_scores": 60,
    "optimizer_stage_scores": 15,
    "run_observed_auc": 12,
    "validation_run_metrics": 12,
    "system_metrics": 12,
}
ROWS_KEYS = {
    "run_id",
    "model_family",
    "optimizer",
    "learning_rate",
    "stage",
    "step",
    "fraction",
    "task",
    "ndcg_at_10",
}
BOUNDARY = (
    "Statistics describe this pinned DenseOn model, one training seed, revised common data and "
    "declared learning-rate grids. Paired-task intervals do not measure seed variability; learning "
    "rates are not independent seeds. Dynamics and systems are descriptive. Outcome acceptance "
    "does not certify mechanisms, cross-host reproducibility, formal release or paper completion."
)


@dataclass(frozen=True)
class OptimizerView:
    name: str
    lr: float


@dataclass(frozen=True)
class RecipeView:
    """Only fields consumed by the unchanged pure summarizer, never a training config."""

    run_id: str
    optimizer: OptimizerView
    model_family: str
    dense_can_flatten_inputs: bool
    checkpoint_fractions: tuple


def recipe_views(primary):
    return [
        RecipeView(
            run_id=row["run_id"],
            optimizer=OptimizerView(
                **{
                    key: primary.expected_identity(row["run_id"])["recipe"]["optimizer"][key]
                    for key in ("name", "lr")
                }
            ),
            model_family=primary.expected_identity(row["run_id"])["recipe"]["model_family"],
            dense_can_flatten_inputs=primary.expected_identity(row["run_id"])["recipe"][
                "dense_can_flatten_inputs"
            ],
            checkpoint_fractions=tuple(
                primary.expected_identity(row["run_id"])["recipe"]["checkpoint_fractions"]
            ),
        )
        for row in primary.inputs["runs"]
    ]


def validate_scores(primary, rows):
    """Reject coercible or mislabelled cells before the retained numeric kernel."""
    recipes = {r.run_id: r for r in recipe_views(primary)}
    wanted = {(r, s, t) for r in recipes for s in range(1, 6) for t in DECONTAMINATED_TASK_NAMES}
    seen = set()
    if len(rows) != 840 or len(recipes) != 12:
        raise ValueError("Require the full distinct 12-by-5-by-14 score grid")
    for row in rows:
        if set(row) != ROWS_KEYS:
            raise ValueError("A score row has missing or extra fields")
        stage = integer(row["stage"], minimum=1)
        config = recipes.get(row["run_id"])
        key = (row["run_id"], stage, row["task"])
        if config is None or key not in wanted or key in seen:
            raise ValueError("Missing, duplicated, historical or undeclared score cell")
        if (
            row["model_family"] != config.model_family
            or row["optimizer"] != config.optimizer.name
            or number(row["learning_rate"], positive=True) != config.optimizer.lr
            or integer(row["step"], minimum=1) != primary.payload["checkpoint_steps"][stage - 1]
            or number(row["fraction"], positive=True) != config.checkpoint_fractions[stage - 1]
            or not 0 <= number(row["ndcg_at_10"]) <= 1
        ):
            raise ValueError("Score identity, stage, recipe or bounded nDCG differs")
        seen.add(key)
    if seen != wanted:
        raise ValueError("Incomplete primary score grid")


def outcome_tables(primary, score_rows, selection):
    """Pure calculation; caller must separately authenticate all source artifacts."""
    if (
        selection.get("scope") != "dense_primary_v3_validation_selection"
        or selection.get("primary_protocol_sha256") != primary.sha256
        or selection.get("validation_protocol_sha256") != PARENTS["validation"][1]
        or selection.get("scientific_completion") is not False
    ):
        raise ValueError("Selection does not belong to this v3 primary/validation revision")
    selected = select_recipes(selection["run_metrics"], primary.inputs["runs"])
    require_same(selected, selection["selected"])
    validate_scores(primary, score_rows)
    tables = summarize_score_rows(
        score_rows,
        recipe_views(primary),
        selected,
        bootstrap_samples=50_000,
        bootstrap_seed=20260903,
    )
    tables["all_task_scores"] = sorted(
        score_rows, key=lambda r: (r["run_id"], r["stage"], r["task"])
    )
    tables["validation_run_metrics"] = sorted(selection["run_metrics"], key=lambda r: r["run_id"])
    for name, rows in tables.items():
        if len(rows) != TABLE_COUNTS[name]:
            raise ValueError(f"Outcome table coverage differs: {name}")
    return tables


def system_rows(primary, runs):
    """Use only already deep-admitted run metadata; never silently add timing adjustments."""
    expected_ids = [r["run_id"] for r in primary.inputs["runs"]]
    if Counter(r["run_id"] for r in runs) != Counter(expected_ids):
        raise ValueError("Systems table requires all twelve distinct complete runs")
    output = []
    for run in runs:
        expected = primary.expected_identity(run["run_id"])
        if run.get("whole_run_artifacts_verified") is not True or run.get(
            "run_identity_sha256"
        ) != digest(expected):
            raise ValueError("Systems input lacks authenticated whole-run identity")
        require_same(run["steps"], primary.payload["checkpoint_steps"])
        timing = run["accepted_timing"]
        if integer(timing["segments"], minimum=1) != 5:
            raise ValueError("System timing does not cover all five segments")
        wall = number(timing["total_wall_time_seconds_max_rank"], positive=True)
        metrics = run["system_metrics"]
        checkpoint_names = [f"checkpoint-{s}" for s in run["steps"]]
        sizes = {}
        for key in ("checkpoint_bytes", "optimizer_state_bytes"):
            if set(metrics[key]) != set(checkpoint_names):
                raise ValueError("Missing or extra system payload sizes")
            sizes[key] = [integer(metrics[key][s], minimum=1) for s in checkpoint_names]
        if (
            metrics["world_size"] != expected["execution"]["world_size"]
            or not isinstance(metrics["gpu_name"], str)
            or not metrics["gpu_name"].strip()
        ):
            raise ValueError("Invalid actual hardware identity")
        allocated = number(metrics["peak_allocated_bytes_max_rank"], positive=True)
        reserved = number(metrics["peak_reserved_bytes_max_rank"], positive=True)
        if allocated > reserved:
            raise ValueError("Allocated memory exceeds reserved memory")
        opt = expected["recipe"]["optimizer"]
        output.append(
            {
                "run_id": run["run_id"],
                "optimizer": opt["name"],
                "learning_rate": opt["lr"],
                "wall_time_seconds": wall,
                "samples_per_second": expected["data"]["rows"] / wall,
                "steps_per_second": run["steps"][-1] / wall,
                "trainer_reported_samples_per_second": number(
                    metrics["trainer"]["train_samples_per_second"], positive=True
                ),
                "trainer_reported_steps_per_second": number(
                    metrics["trainer"]["train_steps_per_second"], positive=True
                ),
                "peak_allocated_gib": allocated / 2**30,
                "peak_reserved_gib": reserved / 2**30,
                "maximum_checkpoint_gib": max(sizes["checkpoint_bytes"]) / 2**30,
                "maximum_optimizer_state_gib": max(sizes["optimizer_state_bytes"]) / 2**30,
                "gpu_name": metrics["gpu_name"],
                "world_size": metrics["world_size"],
            }
        )
    return sorted(output, key=lambda r: r["run_id"])


@dataclass(frozen=True)
class OutcomeContract:
    path: Path
    validation: ValidationContract
    payload: dict
    sha256: str

    @property
    def primary(self):
        return self.validation.primary

    @classmethod
    def load(cls, path, validation):
        path = Path(path).resolve()
        primary, root = validation.primary, validation.primary.repository
        value = read_json(path)
        if (
            value.get("scope") != SCOPE
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("status") != DRAFT
            or value.get("scientific_completion") is not False
            or value.get("formal_execution_authorized") is not False
            or set(value.get("parents", {})) != set(PARENTS)
        ):
            raise ValueError("Not the preparation-only v3 outcome contract")
        trusted = {}
        for key, (name, sha) in PARENTS.items():
            binding = value["parents"][key]
            if binding["path"] != name or binding["sha256"] != sha:
                raise ValueError("An immutable outcome parent binding differs")
            verify_file(root / name, binding)
            trusted[key] = read_json(root / name)
        if primary.sha256 != PARENTS["primary"][1] or validation.sha256 != PARENTS["validation"][1]:
            raise ValueError("Actual outcome primary/validation parent differs")
        require_same(
            value["scientific_rules"], {k: trusted["statistical_parent"][k] for k in SCIENCE_KEYS}
        )
        require_same(value["table_counts"], TABLE_COUNTS)
        require_same(value["implementation_revision"], IMPLEMENTATION_REVISION)
        if set(value["sources"]) != set(SOURCES):
            raise ValueError("Incomplete outcome source closure")
        for name, binding in value["sources"].items():
            verify_file(root / relative_path(name), binding)
        verify_file(
            Path(__file__).resolve(), value["sources"]["src/embed_optim/primary_v3_outcomes.py"]
        )
        return cls(path, validation, value, file_identity(path)["sha256"])

    def recheck(self):
        primary = PrimaryV3Contract.load(
            self.primary.path, self.primary.repository, self.primary.training_root
        )
        validation = ValidationContract.load(self.validation.path, primary)
        current = type(self).load(self.path, validation)
        require_same(current.sha256, self.sha256)
        return current


def collect_evidence(contract, experiment_root, results_root, validation_data, validation_root):
    contract = contract.recheck()
    # Validation has no BEIR input and is completed before the BEIR grid is read.
    selected = collect_selection(
        contract.validation, experiment_root, validation_data, validation_root
    )
    grid = inspect_matrix(contract.primary, experiment_root, results_root)
    tables = outcome_tables(contract.primary, grid["score_rows"], selected)
    tables["system_metrics"] = system_rows(contract.primary, grid["runs"])
    contract.recheck()
    evidence = {"grid": grid, "validation_selection": selected}
    plan = {
        "scope": SCOPE,
        "outcome_protocol_sha256": contract.sha256,
        "primary_protocol_sha256": contract.primary.sha256,
        "validation_protocol_sha256": contract.validation.sha256,
        "evidence_sha256": digest(evidence),
        "scientific_completion": False,
        "scientific_rules": contract.payload["scientific_rules"],
        "boundary": BOUNDARY,
    }
    return plan, evidence, tables


def csv_bytes(rows):
    if not rows:
        raise ValueError("Cannot serialize an empty outcome table")
    fields = sorted(rows[0])
    if any(set(r) != set(fields) for r in rows):
        raise ValueError("Outcome rows have inconsistent fields")
    canonical(rows)  # Reject every non-finite value before serialization.
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def inspect_bundle(output, plan, evidence, tables):
    """Compare every stored byte with a freshly recomputed expected bundle.

    This pure reader does not authenticate expected evidence. The public CLI
    obtains its expected values only via collect_evidence, never from the bundle.
    """
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Require an ordinary outcome bundle")
    manifest = read_json(output / "manifest.json")
    if set(manifest) != {"status", "plan", "outputs"} or manifest["status"] != "complete":
        raise ValueError("Incomplete outcome manifest")
    require_same(manifest["plan"], plan)
    files = {"evidence.json": canonical(evidence) + b"\n"}
    files.update({f"{name}.csv": csv_bytes(rows) for name, rows in tables.items()})
    if set(manifest["outputs"]) != set(files) or set(p.name for p in output.iterdir()) != {
        *files,
        "manifest.json",
    }:
        raise ValueError("Outcome inventory differs")
    for name, expected in files.items():
        binding = manifest["outputs"][name]
        if binding["path"] != name:
            raise ValueError("Unexpected outcome file path")
        verify_file(output / name, binding)
        if (output / name).read_bytes() != expected:
            raise ValueError(
                f"Outcome differs from fresh evidence/statistical recomputation: {name}"
            )
        verify_file(output / name, binding)
    return {
        "plan": plan,
        "outputs": manifest["outputs"],
        "recomputed_bytes_verified": True,
        "scientific_completion": False,
    }


def save_bundle(output, plan, evidence, tables):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Use a new outcome bundle directory; do not overwrite prior evidence")
    # Complete calculation/serialization precedes every output side effect.
    contents = {"evidence.json": canonical(evidence) + b"\n"}
    contents.update({f"{name}.csv": csv_bytes(rows) for name, rows in tables.items()})
    output.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    outputs = {name: {"path": name, **file_identity(output / name)} for name in contents}
    write_new(output / "manifest.json", {"status": "complete", "plan": plan, "outputs": outputs})
    return inspect_bundle(output, plan, evidence, tables)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "build"))
    for name in (
        "protocol",
        "primary-protocol",
        "validation-protocol",
        "repository",
        "training-root",
        "experiment-root",
        "results-root",
        "validation-data",
        "validation-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    primary = PrimaryV3Contract.load(args.primary_protocol, args.repository, args.training_root)
    validation = ValidationContract.load(args.validation_protocol, primary)
    contract = OutcomeContract.load(args.protocol, validation)
    plan, evidence, tables = collect_evidence(
        contract,
        args.experiment_root,
        args.results_root,
        args.validation_data,
        args.validation_root,
    )
    action = save_bundle if args.mode == "build" else inspect_bundle
    result = action(args.output, plan, evidence, tables)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
