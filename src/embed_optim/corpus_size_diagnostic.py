from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .decontamination import DECONTAMINATED_CORPUS_SIZES, DECONTAMINATED_TASK_NAMES

OPTIMIZERS = ("adamw", "muon", "normuon")
CHALLENGERS = ("muon", "normuon")
LABELS = {"muon": "Muon", "normuon": "NorMuon"}
BLOG_MARKERS = (
    "<!-- CORPUS-SIZE-DIAGNOSTIC:BEGIN -->",
    "<!-- CORPUS-SIZE-DIAGNOSTIC:END -->",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def load_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    protocol_path = Path(path).resolve()
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    analysis = payload.get("analysis", {})
    timing = payload.get("timing", {})
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "post_hoc_discovery_corpus_size_diagnostic"
        or timing.get("discovery_beir_visible") is not True
        or timing.get("corpus_size_association_visible") is not True
        or timing.get("candidate_breadth_protocol_already_frozen") is not True
        or timing.get("candidate_breadth_data_or_scores_visible") is not False
        or timing.get("confirmatory_14_task_matrix_complete") is not False
        or analysis.get("family") != "dense"
        or analysis.get("baseline") != "adamw"
        or analysis.get("challengers") != list(CHALLENGERS)
        or analysis.get("stages") != [1, 2, 3, 4, 5]
        or analysis.get("corpus_transform") != "log10_rows"
        or analysis.get("association") != "spearman_rank_correlation"
        or int(analysis.get("permutations", 0)) != 200_000
        or analysis.get("permutation_tail") != "two_sided_add_one"
        or "post hoc" not in str(payload.get("claim_boundary", ""))
    ):
        raise ValueError("Corpus-size diagnostic protocol differs from its post-hoc contract")
    root = protocol_path.parent.parent
    for source in payload.get("sources", {}).values():
        source_path = (root / str(source.get("path", ""))).resolve()
        if (
            not source_path.is_file()
            or _sha256(source_path) != source.get("sha256")
            or len(_read_csv(source_path)) != int(source.get("rows", -1))
        ):
            raise ValueError(f"Corpus-size diagnostic source differs: {source_path}")
    return protocol_path, payload


def _average_ranks(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = ((start + 1) + stop) / 2
        start = stop
    return ranks


def _correlation(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        raise ValueError("Correlation requires aligned non-constant vectors")
    return float(statistics.correlation(x, y))


def _spearman(left: list[float], right: list[float]) -> float:
    return _correlation(_average_ranks(left), _average_ranks(right))


def _permutation_pvalue(
    left: list[float],
    right: list[float],
    *,
    permutations: int,
    seed: int,
    chunk_size: int = 10_000,
) -> tuple[float, float]:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = math.sqrt(
        float(left_centered @ left_centered) * float(right_centered @ right_centered)
    )
    observed = float(left_centered @ right_centered / denominator)
    generator = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < permutations:
        size = min(chunk_size, permutations - completed)
        indices = np.argsort(generator.random((size, len(right_ranks))), axis=1)
        permuted = right_ranks[indices] - right_ranks.mean()
        coefficients = permuted @ left_centered / denominator
        extreme += int(np.count_nonzero(np.abs(coefficients) >= abs(observed) - 1e-15))
        completed += size
    return observed, (extreme + 1) / (permutations + 1)


def _partial_rank_correlation(
    corpus_values: list[float], deltas: list[float], controls: list[bool]
) -> float:
    x = _average_ranks(corpus_values)
    y = _average_ranks(deltas)
    design = np.column_stack((np.ones(len(controls)), np.asarray(controls, dtype=np.float64)))
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return _correlation(x_residual, y_residual)


def _validated_sources(
    protocol_path: Path, protocol: dict[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    root = protocol_path.parent.parent
    sources = protocol["sources"]
    evaluations = _read_csv((root / sources["evaluation_long"]["path"]).resolve())
    summaries = _read_csv((root / sources["optimizer_summary"]["path"]).resolve())
    if (
        len(evaluations) != 840
        or len(summaries) != 3
        or {row.get("optimizer") for row in summaries} != set(OPTIMIZERS)
        or {row.get("model_family") for row in summaries} != {"dense"}
        or {row.get("model_family") for row in evaluations} != {"dense"}
    ):
        raise ValueError("Corpus-size diagnostic requires the complete Dense discovery sources")
    return evaluations, summaries


def build_diagnostic(
    protocol_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    resolved_protocol, protocol = load_protocol(protocol_path)
    evaluations, summaries = _validated_sources(resolved_protocol, protocol)
    best_runs = {str(row["optimizer"]): str(row["best_run_id"]) for row in summaries}
    lookup: dict[tuple[str, int, str], float] = {}
    for row in evaluations:
        optimizer = str(row["optimizer"])
        if str(row["run_id"]) != best_runs.get(optimizer):
            continue
        identity = (optimizer, int(row["stage"]), str(row["task"]))
        if identity in lookup:
            raise ValueError(f"Duplicate selected discovery result: {identity}")
        lookup[identity] = _finite(row["ndcg_at_10"], "nDCG@10")
    expected = {
        (optimizer, stage, task)
        for optimizer in OPTIMIZERS
        for stage in range(1, 6)
        for task in DECONTAMINATED_TASK_NAMES
    }
    if set(lookup) != expected:
        raise ValueError("Selected discovery trajectories do not have complete task coverage")

    ordered_tasks = sorted(
        DECONTAMINATED_TASK_NAMES, key=lambda task: (DECONTAMINATED_CORPUS_SIZES[task], task)
    )
    small = set(ordered_tasks[:7])
    training_sources = set(protocol["analysis"]["training_source_tasks"])
    task_rows: list[dict[str, Any]] = []
    for optimizer in CHALLENGERS:
        for stage in range(1, 6):
            for task in DECONTAMINATED_TASK_NAMES:
                baseline = lookup[("adamw", stage, task)]
                treatment = lookup[(optimizer, stage, task)]
                task_rows.append(
                    {
                        "optimizer": optimizer,
                        "baseline": "adamw",
                        "stage": stage,
                        "fraction": stage / 5,
                        "task": task,
                        "corpus_rows": DECONTAMINATED_CORPUS_SIZES[task],
                        "log10_corpus_rows": math.log10(DECONTAMINATED_CORPUS_SIZES[task]),
                        "corpus_half": "small" if task in small else "large",
                        "training_source_task": task in training_sources,
                        "treatment_ndcg_at_10": treatment,
                        "baseline_ndcg_at_10": baseline,
                        "delta": treatment - baseline,
                    }
                )

    permutations = int(protocol["analysis"]["permutations"])
    base_seed = int(protocol["analysis"]["permutation_seed"])
    association_rows: list[dict[str, Any]] = []
    for optimizer_index, optimizer in enumerate(CHALLENGERS):
        for stage in range(1, 6):
            rows = [
                row for row in task_rows if row["optimizer"] == optimizer and row["stage"] == stage
            ]
            corpus_values = [float(row["log10_corpus_rows"]) for row in rows]
            deltas = [float(row["delta"]) for row in rows]
            seed = base_seed + optimizer_index * 100 + stage
            rho, pvalue = _permutation_pvalue(
                corpus_values, deltas, permutations=permutations, seed=seed
            )
            association_rows.append(
                {
                    "optimizer": optimizer,
                    "baseline": "adamw",
                    "stage": stage,
                    "fraction": stage / 5,
                    "tasks": len(rows),
                    "spearman_rho": rho,
                    "permutation_pvalue_two_sided": pvalue,
                    "permutations": permutations,
                    "permutation_seed": seed,
                }
            )

    optimizer_summaries: dict[str, Any] = {}
    for optimizer in CHALLENGERS:
        rows = [row for row in task_rows if row["optimizer"] == optimizer and row["stage"] == 5]
        corpus_values = [float(row["log10_corpus_rows"]) for row in rows]
        deltas = [float(row["delta"]) for row in rows]
        loo = []
        for held_out in rows:
            kept = [row for row in rows if row["task"] != held_out["task"]]
            loo.append(
                {
                    "held_out_task": held_out["task"],
                    "spearman_rho": _spearman(
                        [float(row["log10_corpus_rows"]) for row in kept],
                        [float(row["delta"]) for row in kept],
                    ),
                }
            )

        def subset_summary(subset: list[dict[str, Any]]) -> dict[str, Any]:
            values = [float(row["delta"]) for row in subset]
            return {
                "tasks": len(subset),
                "positive_tasks": sum(value > 0 for value in values),
                "mean_delta": statistics.mean(values),
                "median_delta": statistics.median(values),
                "spearman_rho": _spearman(
                    [float(row["log10_corpus_rows"]) for row in subset], values
                ),
            }

        without_nq = [row for row in rows if row["task"] != "NQ"]
        seen = [row for row in rows if row["training_source_task"]]
        unseen = [row for row in rows if not row["training_source_task"]]
        small_rows = [row for row in rows if row["corpus_half"] == "small"]
        large_rows = [row for row in rows if row["corpus_half"] == "large"]
        final_association = next(
            row for row in association_rows if row["optimizer"] == optimizer and row["stage"] == 5
        )
        optimizer_summaries[optimizer] = {
            "final_spearman_rho": final_association["spearman_rho"],
            "final_permutation_pvalue_two_sided": final_association["permutation_pvalue_two_sided"],
            "leave_one_task_out_rho_min": min(row["spearman_rho"] for row in loo),
            "leave_one_task_out_rho_max": max(row["spearman_rho"] for row in loo),
            "leave_one_task_out": loo,
            "exclude_nq_spearman_rho": _spearman(
                [float(row["log10_corpus_rows"]) for row in without_nq],
                [float(row["delta"]) for row in without_nq],
            ),
            "partial_rank_correlation_controlling_training_source": _partial_rank_correlation(
                corpus_values,
                deltas,
                [bool(row["training_source_task"]) for row in rows],
            ),
            "training_source_tasks": subset_summary(seen),
            "other_tasks": subset_summary(unseen),
            "small_corpus_half": subset_summary(small_rows),
            "large_corpus_half": subset_summary(large_rows),
        }

    summary = {
        "schema_version": 1,
        "status": protocol["status"],
        "complete": True,
        "tasks": len(DECONTAMINATED_TASK_NAMES),
        "task_stage_rows": len(task_rows),
        "association_rows": len(association_rows),
        "selection": "same-suite-final-BEIR-selected-discovery-runs",
        "optimizer_summaries": optimizer_summaries,
        "claim_boundary": protocol["claim_boundary"],
    }
    return task_rows, association_rows, summary


def _atomic_figure(figure: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_format = path.suffix.removeprefix(".")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{image_format}")
    metadata = (
        {"Date": None, "Creator": "embedding-optimizer-study"}
        if image_format == "svg"
        else {"CreationDate": None, "ModDate": None, "Creator": "embedding-optimizer-study"}
    )
    try:
        figure.savefig(temporary, format=image_format, bbox_inches="tight", metadata=metadata)
        if image_format == "svg":
            svg = temporary.read_text(encoding="utf-8")
            canonical_svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
            _atomic_bytes(temporary, canonical_svg.encode())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _figure(
    task_rows: list[dict[str, Any]], association_rows: list[dict[str, Any]], output_dir: Path
) -> dict[str, dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    # ACL-compatible PDFs must not contain Matplotlib's default Type-3 fonts.
    # Keep text searchable and embedded as TrueType/Type-42 in both PDF and PS
    # backends; the SVG hashsalt below continues to make the web figure stable.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["svg.hashsalt"] = "corpus-size-diagnostic-v1"
    import matplotlib.pyplot as plt

    colors = {"muon": "#F58518", "normuon": "#54A24B"}
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.3))
    for axis, optimizer in zip(axes[:2], CHALLENGERS, strict=True):
        rows = [row for row in task_rows if row["optimizer"] == optimizer and row["stage"] == 5]
        x = np.array([float(row["log10_corpus_rows"]) for row in rows])
        y = np.array([float(row["delta"]) for row in rows])
        axis.scatter(x, y, color=colors[optimizer], s=28, alpha=0.9)
        coefficients = np.polyfit(x, y, 1)
        grid = np.linspace(float(x.min()), float(x.max()), 100)
        axis.plot(grid, np.polyval(coefficients, grid), color=colors[optimizer], linewidth=1.5)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
        nq = next(row for row in rows if row["task"] == "NQ")
        axis.annotate(
            "NQ",
            (float(nq["log10_corpus_rows"]), float(nq["delta"])),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
        final = next(
            row for row in association_rows if row["optimizer"] == optimizer and row["stage"] == 5
        )
        axis.set_title(f"{LABELS[optimizer]} − AdamW\nSpearman ρ={final['spearman_rho']:.3f}")
        axis.set_xlabel("log10 corpus rows")
        axis.set_ylabel("Final nDCG@10 delta")
        axis.grid(alpha=0.18)
    axis = axes[2]
    for optimizer in CHALLENGERS:
        rows = sorted(
            [row for row in association_rows if row["optimizer"] == optimizer],
            key=lambda row: int(row["stage"]),
        )
        axis.plot(
            [100 * float(row["fraction"]) for row in rows],
            [float(row["spearman_rho"]) for row in rows],
            color=colors[optimizer],
            label=LABELS[optimizer],
            marker="o",
            linewidth=1.7,
        )
    axis.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    axis.set_title("Association emerges late")
    axis.set_xlabel("Training completed (%)")
    axis.set_ylabel("Corpus-size Spearman ρ")
    axis.set_xticks([20, 40, 60, 80, 100])
    axis.set_ylim(-0.2, 0.75)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.18)
    figure.suptitle("Exploratory optimizer gains versus full-corpus size", fontsize=12)
    figure.text(
        0.5,
        -0.02,
        "Post-hoc, same-suite-selected discovery runs; association is descriptive, not causal.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout()
    records = {
        suffix: _atomic_figure(figure, output_dir / f"corpus_size_association.{suffix}")
        for suffix in ("svg", "pdf")
    }
    plt.close(figure)
    return records


def _markdown(summary: dict[str, Any]) -> str:
    muon = summary["optimizer_summaries"]["muon"]
    normuon = summary["optimizer_summaries"]["normuon"]
    return "\n".join(
        [
            "### Exploratory signal: gains concentrate on larger corpora",
            "",
            "![Post-hoc optimizer gains versus corpus size](../reports/corpus-size-diagnostic/corpus_size_association.svg)",
            "",
            "Across the 14 discovery tasks, the final Muon-minus-AdamW delta has Spearman "
            f"ρ={muon['final_spearman_rho']:.3f} with log corpus size "
            f"(200,000-permutation p={muon['final_permutation_pvalue_two_sided']:.3f}); "
            "NorMuon has "
            f"ρ={normuon['final_spearman_rho']:.3f} "
            f"(p={normuon['final_permutation_pvalue_two_sided']:.3f}). The association is weak "
            "at 20% of training and appears mainly late in the trajectory.",
            "",
            "The largest seven corpora show positive deltas for both optimizers in 7/7 tasks, "
            "whereas the smallest seven show "
            f"{muon['small_corpus_half']['positive_tasks']}/7 for Muon and "
            f"{normuon['small_corpus_half']['positive_tasks']}/7 for NorMuon. The result is not "
            "driven by NQ: excluding it raises the correlations to "
            f"{muon['exclude_nq_spearman_rho']:.3f} and "
            f"{normuon['exclude_nq_spearman_rho']:.3f}. Every leave-one-task-out correlation "
            f"remains positive (Muon {muon['leave_one_task_out_rho_min']:.3f}–"
            f"{muon['leave_one_task_out_rho_max']:.3f}; NorMuon "
            f"{normuon['leave_one_task_out_rho_min']:.3f}–"
            f"{normuon['leave_one_task_out_rho_max']:.3f}).",
            "",
            "This diagnostic was added after the association was noticed, uses learning rates "
            "selected on the same BEIR suite, and has only 14 heterogeneous task units. Corpus "
            "size can proxy for many task properties, so the result is a descriptive clue—not "
            "evidence that a larger corpus causes Muon to help. Its value is that it makes the "
            "shortlist–corpus account more specific: the independently frozen candidate-breadth "
            "experiment must test candidate coverage directly.",
        ]
    )


def _latex(summary: dict[str, Any]) -> str:
    muon = summary["optimizer_summaries"]["muon"]
    normuon = summary["optimizer_summaries"]["normuon"]
    return "\n".join(
        [
            r"\paragraph{Post-hoc corpus-size diagnostic.}",
            "The heterogeneous discovery gains are larger on tasks with larger evaluation corpora. "
            f"Across 14 tasks, final $\\Delta$ nDCG@10 versus $\\log_{{10}}$ corpus rows has "
            f"Spearman $\\rho={muon['final_spearman_rho']:.3f}$ for Muon "
            f"(200,000-permutation $p={muon['final_permutation_pvalue_two_sided']:.3f}$) and "
            f"$\\rho={normuon['final_spearman_rho']:.3f}$ for NorMuon "
            f"($p={normuon['final_permutation_pvalue_two_sided']:.3f}$). Both optimizers are "
            "positive on all seven larger-corpus tasks, compared with "
            f"{muon['small_corpus_half']['positive_tasks']}/7 and "
            f"{normuon['small_corpus_half']['positive_tasks']}/7 on the smaller half. "
            "The association emerges mainly late and strengthens after excluding NQ. This analysis "
            "was designed after the association was visible, uses same-suite-selected discovery "
            "rates, and cannot distinguish corpus size from correlated task properties; we report "
            "it only as a clue motivating the independently frozen candidate-breadth diagnostic "
            "(Figure~\\ref{fig:corpus-size-diagnostic}).",
            "",
        ]
    )


def _latex_appendix(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            r"\section{Post-hoc Corpus-Size Diagnostic}",
            _latex(summary),
            r"\begin{figure}[t]",
            r"\centering",
            r"\includegraphics[width=\columnwidth]{../reports/corpus-size-diagnostic/corpus_size_association.pdf}",
            r"\caption{Post-hoc association between Muon-family discovery gains and full-corpus size. The right panel shows the association emerging across training stages.}",
            r"\label{fig:corpus-size-diagnostic}",
            r"\end{figure}",
            "",
        ]
    )


def _render_blog(path: Path, content: str, *, audit_only: bool) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    begin, end = BLOG_MARKERS
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("Blog requires exactly one corpus-size marker pair")
    before, rest = text.split(begin)
    _old, after = rest.split(end)
    block = f"{begin}\n\n{content}\n\n{end}"
    rendered = f"{before}{block}{after}"
    if audit_only:
        observed = text[text.index(begin) : text.index(end) + len(end)]
        if observed != block:
            raise ValueError("Blog corpus-size block differs from recomputed content")
    else:
        _atomic_bytes(path, rendered.encode())
    return {
        "path": str(path),
        "bytes": len(block.encode()),
        "sha256": hashlib.sha256(block.encode()).hexdigest(),
    }


def render_diagnostic(
    protocol_path: str | Path = "configs/corpus_size_diagnostic.json",
    output_dir: str | Path = "reports/corpus-size-diagnostic",
    blog: str | Path = "docs/blog.md",
    paper_appendix: str | Path = "paper/generated/corpus-size-diagnostic-appendix.tex",
    *,
    audit_only: bool = False,
) -> dict[str, Any]:
    resolved_protocol, protocol = load_protocol(protocol_path)
    task_rows, association_rows, summary = build_diagnostic(resolved_protocol)
    output = Path(output_dir).resolve()
    blog_path = Path(blog).resolve()
    paper_appendix_path = Path(paper_appendix).resolve()
    task_fields = list(task_rows[0])
    association_fields = list(association_rows[0])
    with tempfile.TemporaryDirectory(prefix="corpus-size-audit-") as directory:
        target = Path(directory) if audit_only else output
        _atomic_bytes(target / "task_stage_deltas.csv", _csv_bytes(task_rows, task_fields))
        _atomic_bytes(
            target / "stage_association.csv", _csv_bytes(association_rows, association_fields)
        )
        _atomic_json(target / "summary.json", summary)
        figures = _figure(task_rows, association_rows, target)
        expected_files = [
            "task_stage_deltas.csv",
            "stage_association.csv",
            "summary.json",
            "corpus_size_association.svg",
            "corpus_size_association.pdf",
        ]
        if audit_only:
            for name in expected_files:
                expected = target / name
                observed = output / name
                if not observed.is_file() or expected.read_bytes() != observed.read_bytes():
                    raise ValueError(f"Corpus-size output differs from recomputation: {observed}")
        output_records = {
            name: {
                "path": str((output / name).relative_to(resolved_protocol.parent.parent)),
                "bytes": (target / name).stat().st_size,
                "sha256": _sha256(target / name),
            }
            for name in expected_files
        }
    blog_record = _render_blog(blog_path, _markdown(summary), audit_only=audit_only)
    appendix_payload = _latex_appendix(summary).encode()
    if audit_only:
        if (
            not paper_appendix_path.is_file()
            or paper_appendix_path.read_bytes() != appendix_payload
        ):
            raise ValueError("Corpus-size appendix block differs from recomputed content")
    else:
        _atomic_bytes(paper_appendix_path, appendix_payload)
    appendix_record = {
        "path": str(paper_appendix_path.relative_to(resolved_protocol.parent.parent)),
        "bytes": len(appendix_payload),
        "sha256": hashlib.sha256(appendix_payload).hexdigest(),
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "protocol": {
            "path": str(resolved_protocol.relative_to(resolved_protocol.parent.parent)),
            "sha256": _sha256(resolved_protocol),
        },
        "outputs": output_records,
        "publication": {
            "blog_block": blog_record,
            "paper_appendix_block": appendix_record,
        },
        "figures": figures,
        "claim_boundary": protocol["claim_boundary"],
    }
    manifest_path = output / "publication_manifest.json"
    if audit_only:
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != manifest:
            raise ValueError("Corpus-size publication manifest differs from recomputation")
    else:
        _atomic_json(manifest_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the post-hoc corpus-size diagnostic")
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/corpus_size_diagnostic.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/corpus-size-diagnostic"))
    parser.add_argument("--blog", type=Path, default=Path("docs/blog.md"))
    parser.add_argument(
        "--paper-appendix",
        type=Path,
        default=Path("paper/generated/corpus-size-diagnostic-appendix.tex"),
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = render_diagnostic(
        args.protocol,
        args.output_dir,
        args.blog,
        args.paper_appendix,
        audit_only=args.audit_only,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
