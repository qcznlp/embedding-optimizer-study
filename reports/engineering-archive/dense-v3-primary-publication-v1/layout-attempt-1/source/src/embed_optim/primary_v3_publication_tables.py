"""Complete v3 primary display tables from freshly checked inputs; no model admission."""

from collections import Counter
from statistics import fmean

from .bridge_numerics import validate_panel
from .corrected_publication import (
    OPTIMIZERS,
    _final_geometry,
    _validate_contrasts,
    _validate_optimizer_stage,
)
from .corrected_retrieval_bridge import assemble_bridge_rows
from .primary_completion import integer
from .primary_contract import require_same
from .primary_v3_outcomes import TABLE_COUNTS, outcome_tables, recipe_views, system_rows
from .primary_v3_publication_bridge import render_bridge
from .primary_v3_validation import select_recipes

SYSTEM_FIELDS = (
    "wall_time_seconds",
    "samples_per_second",
    "steps_per_second",
    "peak_allocated_gib",
    "peak_reserved_gib",
    "maximum_checkpoint_gib",
    "maximum_optimizer_state_gib",
)
SYSTEM_BOUNDARY = (
    "Four-rate arithmetic means. Wall time is the sum of the five segment maximum-rank "
    "times. Payload sizes are the mean of per-run maxima over the five retained checkpoints, "
    "not the size of a selected checkpoint. Timings are descriptive, not randomized speed effects."
)


def selected_rows(primary, selection, rows):
    selected = select_recipes(selection["run_metrics"], primary.inputs["runs"])
    require_same(selection["selected"], selected)
    require_same(
        sorted(rows, key=lambda r: r["run_id"]),
        sorted(selection["run_metrics"], key=lambda r: r["run_id"]),
    )
    indexed = {row["run_id"]: row for row in rows}
    return [
        {**indexed[selected[name]], "selected_by": "validation_contrastive_loss"}
        for name in OPTIMIZERS
    ]


def systems(rows):
    output = []
    for optimizer in OPTIMIZERS:
        group = [row for row in rows if row["optimizer"] == optimizer]
        if len(group) != 4:
            raise ValueError("System summaries require all four declared rates")
        output.append(
            {
                "optimizer": optimizer,
                "runs": 4,
                **{"mean_" + name: fmean(row[name] for row in group) for name in SYSTEM_FIELDS},
                "gpu_names": sorted({row["gpu_name"] for row in group}),
                "world_sizes": sorted({row["world_size"] for row in group}),
            }
        )
    return output


def summarize(primary, outcomes, checkpoints, pairs, bridge, selection, runs):
    """Recompute outcome statistics and bridge joins; geometry admission is upstream.

    This pure helper does not authenticate checkpoints, raw geometry measurements,
    scores or timings. The public author obtains these only after complete source
    admission; reconstructed/synthetic input alone is never primary admission.
    """
    require_same({name: len(rows) for name, rows in outcomes.items()}, TABLE_COUNTS)
    fresh = outcome_tables(primary, outcomes["all_task_scores"], selection)
    fresh["system_metrics"] = system_rows(primary, runs)
    require_same(outcomes, fresh)
    expected = {}
    for row in primary.inputs["runs"]:
        optimizer = primary.expected_identity(row["run_id"])["recipe"]["optimizer"]
        expected[row["run_id"]] = {name: optimizer[name] for name in ("name", "lr")}
    wanted = Counter((run_id, stage) for run_id in expected for stage in range(1, 6))
    actual = Counter()
    for row in checkpoints:
        run_id = row["run_id"]
        stage = integer(row["stage"], minimum=1)
        if run_id not in expected:
            raise ValueError("Undeclared publication geometry run")
        require_same({"name": row["optimizer"], "lr": row["learning_rate"]}, expected[run_id])
        actual[run_id, stage] += 1
    if actual != wanted:
        raise ValueError("Publication geometry requires the complete sixty-state grid")
    panel = assemble_bridge_rows(
        checkpoints, pairs, fresh["run_stage_scores"], recipe_views(primary)
    )
    require_same(bridge["bridge_rows"], validate_panel(panel))
    bridge_display = render_bridge(bridge)
    selected = selected_rows(primary, selection, fresh["validation_run_metrics"])
    selected_ids = {row["optimizer"]: row["run_id"] for row in selected}
    for row in fresh["secondary_summary"]:
        for side in ("treatment", "baseline"):
            if row[side + "_run_id"] != selected_ids[row[side]]:
                raise ValueError("Secondary comparison does not use validation-only selection")
    return {
        "primary": _validate_contrasts(fresh["primary_summary"], secondary=False),
        "secondary": _validate_contrasts(fresh["secondary_summary"], secondary=True),
        "optimizer_stage": _validate_optimizer_stage(fresh["optimizer_stage_scores"]),
        "validation_selected": selected,
        "systems": systems(fresh["system_metrics"]),
        "systems_boundary": SYSTEM_BOUNDARY,
        "final_geometry": _final_geometry(checkpoints),
        "bridge": bridge_display["rows"],
        "bridge_latex": bridge_display["latex"],
        "outcome_table_counts": dict(TABLE_COUNTS),
        "all_geometry_states_retained": 60,
        "all_geometry_pairs_retained": len(pairs),
        "primary_admission_supplied_by_this_helper": False,
        "scientific_completion": False,
    }
