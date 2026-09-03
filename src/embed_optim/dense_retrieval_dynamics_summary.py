"""Build the strict descriptive five-stage DenseOn retrieval-dynamics artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .aggregate import collect_evaluations
from .config import RunConfig
from .confirmatory_evaluation import audit_confirmatory_evaluations
from .confirmatory_summary import _atomic_csv
from .decontamination import DECONTAMINATED_TASK_NAMES
from .dense_retrieval_dynamics_evaluation import (
    DYNAMICS_STAGES,
    FINAL_INFERENCE_STAGE,
    _audit_input_manifest,
    _confirmatory_configs,
    _confirmatory_context,
    _hybrid_configs,
    _relative,
    audit_dense_retrieval_dynamics,
    load_dynamics_contract,
)
from .geometry import SCHEMA_VERSION, _atomic_json, _sha256

PARTITIONS = ("dynamics-stage1-4", "formal-stage5")
EXPECTED_RUNS = 13
EXPECTED_STAGES = 5
EXPECTED_TRAJECTORY_ROWS = EXPECTED_RUNS * EXPECTED_STAGES
EXPECTED_TASK_UNITS = EXPECTED_TRAJECTORY_ROWS * len(DECONTAMINATED_TASK_NAMES)
OPTIMIZER_COLORS = {
    "adamw": "#4C78A8",
    "muon": "#F58518",
    "normuon": "#54A24B",
    "hybrid_adamw": "#7A5195",
}
OPTIMIZER_LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
    "normuon": "NorMuon",
    "hybrid_adamw": "Hybrid AdamW",
}


def _file_identity(path: Path, repository: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": _relative(path, repository),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        **extra,
    }


def _expected_partition(
    configs: Sequence[RunConfig], stages: Sequence[int]
) -> set[tuple[str, str, int, str]]:
    return {
        (config.model_family, config.run_id, stage, task)
        for config in configs
        for stage in stages
        for task in DECONTAMINATED_TASK_NAMES
    }


def _collect_partition(
    root: Path,
    configs: Sequence[RunConfig],
    *,
    suite: str,
    partition: str,
    repository: Path,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect one exact result partition and bind every row to a file hash."""

    if partition not in PARTITIONS:
        raise ValueError(f"Unknown retrieval-dynamics partition: {partition}")
    stages = DYNAMICS_STAGES if partition == PARTITIONS[0] else (FINAL_INFERENCE_STAGE,)
    rows = collect_evaluations(root, list(configs))
    identities = [
        (
            str(row.get("model_family")),
            str(row.get("run_id")),
            int(row.get("stage", 0)),
            str(row.get("task")),
        )
        for row in rows
    ]
    expected = _expected_partition(configs, stages)
    counts = Counter(identities)
    if (
        len(rows) != len(expected)
        or set(counts) != expected
        or any(count != 1 for count in counts.values())
    ):
        raise ValueError(
            f"{suite} {partition} coverage is {len(rows)}/{len(expected)} exact task units"
        )
    candidates = {path.resolve() for path in root.rglob("*Decontaminated.json")}
    selected_paths = {Path(str(row["result_path"])).resolve() for row in rows}
    if candidates != selected_paths:
        raise ValueError(f"{suite} {partition} contains unrecognized result files")

    input_manifest = _audit_input_manifest(
        root,
        configs,
        repository=repository,
        stages=stages,
    )
    runtime_path = root / "evaluation_runtime.json"
    runtime_manifest = _file_identity(runtime_path, repository)
    config_by_id = {(config.model_family, config.run_id): config for config in configs}
    annotated: list[dict[str, Any]] = []
    result_sources = []
    for row in rows:
        result_path = Path(str(row["result_path"])).resolve()
        result_sha256 = _sha256(result_path)
        config = config_by_id[(str(row["model_family"]), str(row["run_id"]))]
        annotated.append(
            {
                **row,
                "suite": suite,
                "training_seed": int(config.seed),
                "partition": partition,
                "result_path": _relative(result_path, repository),
                "result_sha256": result_sha256,
            }
        )
        result_sources.append(
            _file_identity(
                result_path,
                repository,
                task=str(row["task"]),
                stage=int(row["stage"]),
            )
        )
    if seed is not None and {int(row["training_seed"]) for row in annotated} != {seed}:
        raise ValueError(f"Confirmatory partition seed differs from its matrix: {seed}")
    return annotated, {
        "suite": suite,
        "seed": seed,
        "partition": partition,
        "results_root": _relative(root, repository),
        "expected_units": len(expected),
        "valid_units": len(rows),
        "input_manifest": input_manifest,
        "runtime_manifest": runtime_manifest,
        "result_sources": sorted(result_sources, key=lambda item: item["path"]),
    }


def _finite_score(row: dict[str, Any]) -> float:
    try:
        value = float(row["ndcg_at_10"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid trajectory task score: {row}") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Non-finite/out-of-range trajectory task score: {row}")
    return value


def summarize_five_stage_trajectories(
    task_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate 910 exact task units into 65 descriptive run-stage rows."""

    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        try:
            suite = str(row["suite"])
            seed = int(row["training_seed"])
            run_id = str(row["run_id"])
            stage = int(row["stage"])
            partition = str(row["partition"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid trajectory identity: {row}") from error
        expected_partition = PARTITIONS[0] if stage in DYNAMICS_STAGES else PARTITIONS[1]
        if (
            suite not in {"hybrid", "confirmatory"}
            or not run_id
            or stage not in {*DYNAMICS_STAGES, FINAL_INFERENCE_STAGE}
            or partition != expected_partition
        ):
            raise ValueError(
                f"Trajectory row crossed the stage/partition inference boundary: "
                f"{suite}/{seed}/{run_id}/stage-{stage}/{partition}"
            )
        result_hash = row.get("result_sha256")
        if not isinstance(result_hash, str) or len(result_hash) != 64:
            raise ValueError("Trajectory task row lacks a result-file SHA-256")
        grouped[(suite, seed, run_id, stage)].append(dict(row))

    output: list[dict[str, Any]] = []
    for identity, rows in sorted(
        grouped.items(),
        key=lambda item: (
            0 if item[0][0] == "hybrid" else 1,
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    ):
        suite, seed, run_id, stage = identity
        tasks = [str(row.get("task")) for row in rows]
        if len(rows) != len(DECONTAMINATED_TASK_NAMES) or set(tasks) != set(
            DECONTAMINATED_TASK_NAMES
        ):
            raise ValueError(f"Trajectory stage lacks exact 14-task coverage: {identity}")
        invariant_fields = (
            "model_family",
            "optimizer",
            "learning_rate",
            "aux_learning_rate",
            "fraction",
            "checkpoint_step",
            "partition",
        )
        for field in invariant_fields:
            if len({str(row.get(field)) for row in rows}) != 1:
                raise ValueError(f"Trajectory stage has inconsistent {field}: {identity}")
        scores = [_finite_score(row) for row in rows]
        source_records = sorted(
            (
                str(row["task"]),
                str(row["result_path"]),
                str(row["result_sha256"]),
            )
            for row in rows
        )
        source_digest = hashlib.sha256(
            json.dumps(source_records, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        first = rows[0]
        output.append(
            {
                "suite": suite,
                "model_family": str(first["model_family"]),
                "optimizer": str(first["optimizer"]),
                "learning_rate": float(first["learning_rate"]),
                "aux_learning_rate": (
                    None
                    if first.get("aux_learning_rate") is None
                    else float(first["aux_learning_rate"])
                ),
                "run_id": run_id,
                "training_seed": seed,
                "stage": stage,
                "fraction": float(first["fraction"]),
                "checkpoint_step": int(first["checkpoint_step"]),
                "tasks_completed": len(scores),
                "mean_ndcg_at_10": statistics.mean(scores),
                "median_ndcg_at_10": statistics.median(scores),
                "standard_deviation_ndcg_at_10": statistics.pstdev(scores),
                "minimum_ndcg_at_10": min(scores),
                "maximum_ndcg_at_10": max(scores),
                "source_partition": str(first["partition"]),
                "formal_source_stage5": stage == FINAL_INFERENCE_STAGE,
                "joined_summary_role": "descriptive-only",
                "joined_summary_used_for_formal_inference": False,
                "source_result_set_sha256": source_digest,
            }
        )

    run_stages: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for row in output:
        run_stages[(row["suite"], row["training_seed"], row["run_id"])].add(row["stage"])
    hybrid_runs = {identity for identity in run_stages if identity[0] == "hybrid"}
    confirmatory_runs = {identity for identity in run_stages if identity[0] == "confirmatory"}
    if (
        len(task_rows) != EXPECTED_TASK_UNITS
        or len(output) != EXPECTED_TRAJECTORY_ROWS
        or len(run_stages) != EXPECTED_RUNS
        or len(hybrid_runs) != 4
        or len(confirmatory_runs) != 9
        or any(stages != {1, 2, 3, 4, 5} for stages in run_stages.values())
    ):
        raise ValueError(
            "Five-stage Dense retrieval dynamics requires exactly 13 runs, 65 stages, "
            "and 910 task units"
        )
    return output


def _atomic_figure(figure: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_format = path.suffix.removeprefix(".")
    if image_format not in {"svg", "pdf"}:
        raise ValueError(f"Unsupported Dense retrieval-dynamics figure format: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{image_format}")
    metadata = (
        {"Date": None, "Creator": "embedding-optimizer-study"}
        if image_format == "svg"
        else {
            "CreationDate": None,
            "ModDate": None,
            "Creator": "embedding-optimizer-study",
        }
    )
    figure.savefig(temporary, format=image_format, bbox_inches="tight", metadata=metadata)
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _trajectory_figure(
    rows: Sequence[dict[str, Any]], output_dir: Path
) -> dict[str, dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.hashsalt"] = "dense-retrieval-dynamics-extension-v1"
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(15.5, 4.25), sharex=True, sharey=True)
    panels = (
        ("hybrid", "hybrid_adamw", "Hybrid AdamW routing control"),
        ("confirmatory", "adamw", "Confirmatory AdamW"),
        ("confirmatory", "muon", "Confirmatory Muon"),
        ("confirmatory", "normuon", "Confirmatory NorMuon"),
    )
    for axis, (suite, optimizer, title) in zip(axes, panels, strict=True):
        subset = [row for row in rows if row["suite"] == suite and row["optimizer"] == optimizer]
        by_run: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            by_run[(int(row["training_seed"]), str(row["run_id"]))].append(dict(row))
        colors = plt.get_cmap("viridis")(
            [0.10, 0.37, 0.64, 0.91] if suite == "hybrid" else [0.15, 0.52, 0.88]
        )
        for color, ((seed, _run_id), values) in zip(colors, sorted(by_run.items()), strict=True):
            ordered = sorted(values, key=lambda row: int(row["stage"]))
            label = (
                f"LR {float(ordered[0]['learning_rate']):.0e}"
                if suite == "hybrid"
                else f"seed {seed}"
            )
            x = [float(row["fraction"]) for row in ordered]
            axis.plot(
                x,
                [float(row["mean_ndcg_at_10"]) for row in ordered],
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=label,
            )
            axis.plot(
                x,
                [float(row["median_ndcg_at_10"]) for row in ordered],
                color=color,
                linestyle=":",
                linewidth=1.25,
                alpha=0.82,
            )
        axis.axvspan(0.18, 0.82, color="#7F7F7F", alpha=0.055, linewidth=0)
        axis.axvline(1.0, color="#B22222", linestyle="--", linewidth=1.0, alpha=0.70)
        axis.set_title(title, fontsize=10.5)
        axis.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20", "40", "60", "80", "100"])
        axis.set_xlabel("Training progress (%)")
        axis.grid(alpha=0.20)
        axis.legend(frameon=False, fontsize=7.5, title_fontsize=8)
    axes[0].set_ylabel("Decontaminated BEIR nDCG@10")
    figure.suptitle(
        "DenseOn five-stage retrieval dynamics (solid: task mean; dotted: task median)",
        fontsize=12.5,
    )
    figure.text(
        0.5,
        0.005,
        "Stages 1–4 are descriptive dynamics; stage 5 is copied from the isolated formal root. "
        "This joined figure is not an inference input.",
        ha="center",
        fontsize=8.2,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    records = {
        suffix: _atomic_figure(
            figure,
            output_dir / f"five_stage_retrieval_dynamics.{suffix}",
        )
        for suffix in ("svg", "pdf")
    }
    plt.close(figure)
    return records


def _verify_outputs(outputs: dict[str, dict[str, Any]], repository: Path) -> None:
    if set(outputs) != {"trajectory_csv", "figure_svg", "figure_pdf"}:
        raise ValueError("Dense retrieval-dynamics output ledger is incomplete")
    for name, record in outputs.items():
        path = repository / str(record.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("sha256")
            or (name == "trajectory_csv" and record.get("rows") != EXPECTED_TRAJECTORY_ROWS)
        ):
            raise ValueError(f"Dense retrieval-dynamics output differs: {name} ({path})")


def write_summary_outputs(
    trajectory_rows: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    repository: Path,
) -> dict[str, dict[str, Any]]:
    """Write and hash the 65-row table plus publication-ready SVG/PDF."""

    if len(trajectory_rows) != EXPECTED_TRAJECTORY_ROWS:
        raise ValueError("Refusing to render an incomplete five-stage trajectory table")
    output = output_dir.resolve()
    table = _atomic_csv(output / "five_stage_retrieval_dynamics.csv", list(trajectory_rows))
    figures = _trajectory_figure(trajectory_rows, output)
    records = {
        "trajectory_csv": table,
        "figure_svg": figures["svg"],
        "figure_pdf": figures["pdf"],
    }
    for record in records.values():
        record["path"] = _relative(Path(record["path"]), repository)
    _verify_outputs(records, repository)
    return records


def _collect_summary_material(contract: Any) -> dict[str, Any]:
    """Recompute every dynamics and formal-stage-5 source bound into the join."""

    dynamics_audit = audit_dense_retrieval_dynamics(contract.path, verify_results=True)
    if (
        dynamics_audit.get("complete") is not True
        or dynamics_audit.get("valid_units") != 728
        or dynamics_audit.get("formal_inference_uses_dynamics_rows") is not False
    ):
        raise ValueError("Five-stage summary requires the strict 728-unit dynamics audit")

    task_rows: list[dict[str, Any]] = []
    partition_sources: list[dict[str, Any]] = []
    hybrid_configs = _hybrid_configs(contract)
    for partition, root in (
        (PARTITIONS[0], contract.result_root("hybrid")),
        (PARTITIONS[1], contract.formal_result_root("hybrid")),
    ):
        rows, source = _collect_partition(
            root,
            hybrid_configs,
            suite="hybrid",
            partition=partition,
            repository=contract.repository,
        )
        task_rows.extend(rows)
        partition_sources.append(source)

    protocol_path, _, matrix_paths, matrix_manifest_sha256 = _confirmatory_context(contract)
    formal_confirmatory = audit_confirmatory_evaluations(
        protocol_path,
        experiment_matrix=contract.repository / "configs/experiment.yaml",
        validation_spec=contract.repository / "configs/validation_probe.json",
        matrix_dir=contract.source_path("confirmatory_matrix_manifest").parent,
        results_root=contract.formal_result_root("confirmatory"),
        families=("dense",),
        scope_amendment=contract.source_path("scope_amendment"),
    )
    if (
        formal_confirmatory.get("complete") is not True
        or formal_confirmatory.get("valid_units") != 126
        or formal_confirmatory.get("expected_units") != 126
    ):
        raise ValueError("Five-stage summary requires the strict 126-unit stage-5 audit")
    for seed, matrix_path in matrix_paths.items():
        configs = _confirmatory_configs(matrix_path)
        for partition, root in (
            (PARTITIONS[0], contract.result_root("confirmatory") / f"seed{seed}"),
            (PARTITIONS[1], contract.formal_result_root("confirmatory") / f"seed{seed}"),
        ):
            rows, source = _collect_partition(
                root,
                configs,
                suite="confirmatory",
                partition=partition,
                repository=contract.repository,
                seed=seed,
            )
            task_rows.extend(rows)
            partition_sources.append(source)

    return {
        "dynamics_audit": dynamics_audit,
        "formal_confirmatory": formal_confirmatory,
        "matrix_manifest_sha256": matrix_manifest_sha256,
        "partition_sources": partition_sources,
        "protocol_path": protocol_path,
        "trajectories": summarize_five_stage_trajectories(task_rows),
    }


def _summary_manifest(
    contract: Any,
    material: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dynamics_audit = material["dynamics_audit"]
    formal_confirmatory = material["formal_confirmatory"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "complete": True,
        "families": ["dense"],
        "coverage": {
            "runs": EXPECTED_RUNS,
            "stages_per_run": EXPECTED_STAGES,
            "trajectory_rows": EXPECTED_TRAJECTORY_ROWS,
            "tasks_per_stage": len(DECONTAMINATED_TASK_NAMES),
            "task_units": EXPECTED_TASK_UNITS,
            "dynamics_units": 728,
            "formal_stage5_units": 182,
        },
        "inference_boundary": {
            "dynamics_stages": list(DYNAMICS_STAGES),
            "formal_inference_stage": FINAL_INFERENCE_STAGE,
            "formal_inference_reads_joined_outputs": False,
            "formal_inference_roots": {
                suite: _relative(contract.formal_result_root(suite), contract.repository)
                for suite in ("hybrid", "confirmatory")
            },
            "interpretation": (
                "The joined CSV and figures are descriptive trajectory artifacts only. Stage 5 "
                "is read from the pre-existing formal roots, while stages 1-4 are read from "
                "disjoint dynamics roots; formal hybrid and confirmatory inference continues to "
                "consume only the original stage-5 roots."
            ),
        },
        "sources": {
            "implementation": _file_identity(Path(__file__).resolve(), contract.repository),
            "contract": _file_identity(contract.path, contract.repository),
            "dynamics_audit": {
                "complete": True,
                "expected_units": dynamics_audit["expected_units"],
                "valid_units": dynamics_audit["valid_units"],
                "contract_sha256": dynamics_audit["contract"]["sha256"],
            },
            "confirmatory_protocol": _file_identity(material["protocol_path"], contract.repository),
            "confirmatory_matrix_manifest_sha256": material["matrix_manifest_sha256"],
            "formal_confirmatory_audit": {
                "complete": True,
                "expected_units": formal_confirmatory["expected_units"],
                "valid_units": formal_confirmatory["valid_units"],
                "protocol_sha256": formal_confirmatory["protocol_sha256"],
                "matrix_manifest_sha256": formal_confirmatory["matrix_manifest_sha256"],
            },
            "partitions": material["partition_sources"],
        },
        "outputs": outputs,
    }


def _expected_output_paths(output_dir: Path, repository: Path) -> dict[str, str]:
    return {
        "trajectory_csv": _relative(output_dir / "five_stage_retrieval_dynamics.csv", repository),
        "figure_svg": _relative(output_dir / "five_stage_retrieval_dynamics.svg", repository),
        "figure_pdf": _relative(output_dir / "five_stage_retrieval_dynamics.pdf", repository),
    }


def _recompute_outputs_without_overwrite(
    trajectory_rows: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    repository: Path,
) -> dict[str, dict[str, Any]]:
    """Render to an isolated temporary directory and return expected artifact hashes."""

    with tempfile.TemporaryDirectory(prefix="dense-retrieval-dynamics-audit-") as temporary:
        temporary_root = Path(temporary).resolve()
        regenerated = write_summary_outputs(
            trajectory_rows,
            output_dir=temporary_root,
            repository=temporary_root,
        )
        expected_paths = _expected_output_paths(output_dir, repository)
        return {
            name: {**record, "path": expected_paths[name]} for name, record in regenerated.items()
        }


def build_dense_retrieval_dynamics_summary(
    contract_path: str | Path = "configs/dense_retrieval_dynamics_extension.json",
    output_dir: str | Path = "reports/dense-retrieval-dynamics",
) -> dict[str, Any]:
    contract = load_dynamics_contract(contract_path)
    material = _collect_summary_material(contract)
    output = Path(output_dir).resolve()
    outputs = write_summary_outputs(
        material["trajectories"],
        output_dir=output,
        repository=contract.repository,
    )
    manifest = _summary_manifest(contract, material, outputs)
    _verify_outputs(outputs, contract.repository)
    _atomic_json(output / "summary_manifest.json", manifest)
    written = json.loads((output / "summary_manifest.json").read_text(encoding="utf-8"))
    if written != manifest:
        raise ValueError("Written Dense retrieval-dynamics manifest differs after serialization")
    return manifest


def audit_dense_retrieval_dynamics_summary(
    contract_path: str | Path = "configs/dense_retrieval_dynamics_extension.json",
    output_dir: str | Path = "reports/dense-retrieval-dynamics",
) -> dict[str, Any]:
    """Strictly audit the existing manifest and artifacts without replacing them."""

    contract = load_dynamics_contract(contract_path)
    output = Path(output_dir).resolve()
    manifest_path = output / "summary_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Dense retrieval-dynamics manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Dense retrieval-dynamics manifest is unreadable") from error
    if not isinstance(manifest, dict):
        raise ValueError("Dense retrieval-dynamics manifest must be a JSON object")

    material = _collect_summary_material(contract)
    reconstructed_outputs = _recompute_outputs_without_overwrite(
        material["trajectories"],
        output_dir=output,
        repository=contract.repository,
    )
    expected_manifest = _summary_manifest(contract, material, reconstructed_outputs)
    if manifest != expected_manifest:
        raise ValueError(
            "Dense retrieval-dynamics manifest differs from the recomputed source-bound summary"
        )
    _verify_outputs(reconstructed_outputs, contract.repository)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "complete": True,
        "read_only": True,
        "manifest": _file_identity(manifest_path, contract.repository),
        "coverage": {
            "dynamics_units": 728,
            "formal_hybrid_stage5_units": 56,
            "formal_confirmatory_stage5_units": 126,
            "task_units": EXPECTED_TASK_UNITS,
            "trajectory_rows": EXPECTED_TRAJECTORY_ROWS,
        },
        "outputs": reconstructed_outputs,
        "formal_inference_reads_joined_outputs": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/dense_retrieval_dynamics_extension.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dense-retrieval-dynamics"),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Recompute and verify the existing manifest and outputs without replacing them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = (
        audit_dense_retrieval_dynamics_summary(args.contract, args.output_dir)
        if args.audit_only
        else build_dense_retrieval_dynamics_summary(args.contract, args.output_dir)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
