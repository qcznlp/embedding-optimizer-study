"""Small synthetic guard checks; genuine all-run reconstruction is a separate audit."""

import copy
import json

import pytest

from scripts import audit_primary_v3_training_observations as audit
from scripts import observe_primary_v3_training as obs


def history():
    rows = []
    for step in obs.LOG_STEPS:
        t = step - 1
        factor = t / 391 if t < 391 else (3907 - t) / 3516
        rows.append(
            {
                "step": step,
                "epoch": step * 128 / 500000,
                "learning_rate": 3e-4 * factor,
                "loss": 1 / (step + 1),
                "grad_norm": 2.0,
            }
        )
    rows.append({"step": 3907, "train_loss": 0.5})
    return {"global_step": 3907, "max_steps": 3907, "epoch": 1.0, "log_history": rows}


def test_synthetic_full_logging_schedule_and_terminal_distinction():
    logs = obs.inspect_history(history(), 3e-4)
    assert len(logs) == 391 and logs[0]["learning_rate"] == 0
    assert logs[-1]["step"] == 3900 and all(r["step"] != 3907 for r in logs)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "reordered",
        "floating_step",
        "boolean_step",
        "wrong_lr",
        "nan",
        "negative_loss",
        "negative_norm",
        "bad_epoch",
        "extra_field",
        "terminal_loss",
        "wrong_horizon",
        "wrong_end_step",
    ],
)
def test_invalid_logging_rejected(change):
    state = history()
    rows = state["log_history"]
    if change == "missing":
        rows.pop(3)
    elif change == "duplicate":
        rows[3] = copy.deepcopy(rows[2])
    elif change == "reordered":
        rows[3], rows[4] = rows[4], rows[3]
    elif change == "floating_step":
        rows[0]["step"] = 1.0
    elif change == "boolean_step":
        rows[0]["step"] = True
    elif change == "wrong_lr":
        rows[1]["learning_rate"] *= 4
    elif change == "nan":
        rows[1]["grad_norm"] = float("nan")
    elif change == "negative_loss":
        rows[1]["loss"] = -0.1
    elif change == "negative_norm":
        rows[1]["grad_norm"] = -0.1
    elif change == "bad_epoch":
        rows[1]["epoch"] += 0.01
    elif change == "extra_field":
        rows[1]["invented_timestamp"] = 0
    elif change == "terminal_loss":
        rows[-1]["loss"] = 0.2
    elif change == "wrong_horizon":
        state["global_step"] = 3900
    else:
        rows[-1]["step"] = 3900
    with pytest.raises(ValueError):
        obs.inspect_history(state, 3e-4)


@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf"), float("-inf")])
def test_non_numeric_telemetry_rejected(value):
    with pytest.raises(ValueError):
        obs.finite(value)


@pytest.mark.parametrize("value", [0, -1])
def test_positive_time_required(value):
    with pytest.raises(ValueError):
        obs.finite(value, positive=True)


def test_input_hash_and_json_duplicate_key_refusals(tmp_path):
    p = tmp_path / "input.json"
    p.write_text('{"x": 1, "x": 2}')
    with pytest.raises(ValueError, match="Duplicate"):
        obs.read(p)
    p.write_text('{"x": 1}')
    identity = obs.identity(p)
    p.write_text('{"x": 2}')
    with pytest.raises(ValueError, match="binding"):
        obs.read(p, identity)


def test_unrelated_admission_cannot_supply_primary_evidence(tmp_path):
    p = tmp_path / "not-genuine.json"
    p.write_text(json.dumps({"scientific_completion": False}))
    with pytest.raises(ValueError, match="exact existing"):
        obs.load_parent(p)


def test_ordinary_paths_and_output_preservation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="absolute"):
        obs.ordinary("relative")
    with pytest.raises(ValueError, match="absolute"):
        obs.ordinary(tmp_path / ".." / "escape")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="absolute"):
        obs.ordinary(tmp_path / "link" / "payload.json")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="fresh output"):
        obs.produce(tmp_path / "parent.json", tmp_path / "native", tmp_path, copy_inputs=False)


def test_cpu_guard_precedes_any_input_read(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="hidden CUDA"):
        obs.produce(
            tmp_path / "absent.json", tmp_path / "native", tmp_path / "new", copy_inputs=False
        )
    assert not (tmp_path / "new").exists()


def synthetic_file_parent():
    runs = {}
    for op, rates in obs.RATES.items():
        for rate in rates:
            run_id = f"verified-v3-{op}-{rate}"
            runs[run_id] = {
                "metadata": {name: {"bytes": 1, "sha256": "0" * 64} for name in obs.METADATA},
                "checkpoints": [
                    {
                        "step": step,
                        "files": [{"path": "trainer_state.json", "bytes": 1, "sha256": "0" * 64}],
                    }
                    for step in obs.STEPS
                ],
            }
    return {"admitted": {"complete_runs": runs}}


def test_synthetic_metadata_inventory_complete():
    assert len(obs.declared_files(synthetic_file_parent())) == 132


@pytest.mark.parametrize(
    "change", ["missing_stage", "wrong_step", "missing_state", "duplicate_state", "missing_run"]
)
def test_incomplete_native_population_rejected(change):
    parent = synthetic_file_parent()
    runs = parent["admitted"]["complete_runs"]
    run = next(iter(runs.values()))
    if change == "missing_stage":
        run["checkpoints"].pop()
    elif change == "wrong_step":
        run["checkpoints"][0]["step"] += 1
    elif change == "missing_state":
        run["checkpoints"][0]["files"] = []
    elif change == "duplicate_state":
        run["checkpoints"][0]["files"] *= 2
    else:
        runs.pop(next(iter(runs)))
    with pytest.raises(ValueError):
        obs.declared_files(parent)


def test_independent_scalar_reference_and_corruption_refusal():
    rows = [{"loss": x, "grad_norm": 10 - x} for x in range(10)]
    mean, sd, median = audit.scalar_window(rows)
    assert mean == 4.5 and median == 5.5
    assert sd**2 == pytest.approx(55 / 6)
    assert audit.compare_table([{"a": 1}], [{"a": 1}])["scalar_fields"] == 1
    for actual in ([{"a": 2}], [{"a": True}], [{"b": 1}], []):
        with pytest.raises(ValueError):
            audit.compare_table(actual, [{"a": 1}])


def test_independent_reference_requires_original_source_bytes(tmp_path):
    (tmp_path / "primary_contract.py").write_text("raise RuntimeError('never execute')\n")
    with pytest.raises(ValueError, match="source changed"):
        audit.pure_references(tmp_path)
