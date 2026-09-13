import copy
import json
import shutil
from pathlib import Path

import pytest

from scripts import dense_run_identity_proposal as proposal

ROOT = Path(__file__).parents[1]


def recipe():
    # Actual archived diagnostic configuration, not a fabricated primary result.
    source = (
        ROOT
        / "reports/engineering-archive/dense-correction-candidate-v1/candidate-source/configs/dense_correctness_candidate.yaml"
    )
    import yaml

    matrix = yaml.safe_load(source.read_text())
    optimizer = copy.deepcopy(matrix["optimizers"][3])
    run_id = optimizer.pop("id")
    return {
        **matrix["common"],
        **matrix["models"]["dense"],
        "model_family": "dense",
        "optimizer": optimizer,
        "run_id": run_id,
    }


def inputs(tmp_path):
    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "data.arrow").write_bytes(b"synthetic test bytes")
    data = {
        "rows": 2,
        "selected_columns": ["query", "positive"],
        "files": proposal.dataset_files(directory),
    }
    config = recipe()
    numerical = {"numerical_policy": config["numerical_policy"]}
    return config, data, {"world_size": 4, "max_steps": 3}, numerical, {"sha256": "a" * 64}


def test_identical_bytes_relocate_without_changing_recipe_identity(tmp_path):
    values = inputs(tmp_path)
    moved = copy.deepcopy(values)
    destination = tmp_path / "moved"
    shutil.copytree(tmp_path / "dataset", destination)
    moved[0].update(
        dataset_path=str(destination),
        output_root="another-output",
        wandb_entity="another",
        wandb_project="tracking-only",
    )
    moved[1]["files"] = proposal.dataset_files(destination)
    proposal.require_identity(proposal.build_identity(*values), proposal.build_identity(*moved))


def test_same_length_data_change_is_not_hidden_by_row_count(tmp_path):
    values = inputs(tmp_path)
    expected = proposal.build_identity(*values)
    (tmp_path / "dataset/data.arrow").write_bytes(b"different test bytes")
    assert len(b"synthetic test bytes") == len(b"different test bytes")
    values[1]["files"] = proposal.dataset_files(tmp_path / "dataset")
    with pytest.raises(ValueError, match="not a continuation"):
        proposal.require_identity(expected, proposal.build_identity(*values))


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 43),
        ("max_length", 4096),
        ("temperature", 0.04),
        ("max_grad_norm", 0.5),
        ("epochs", 2.0),
        ("run_id", "different-run"),
    ],
)
def test_scientific_recipe_changes_reject(tmp_path, field, value):
    values = inputs(tmp_path)
    expected = proposal.build_identity(*values)
    values[0][field] = value
    with pytest.raises(ValueError):
        proposal.require_identity(expected, proposal.build_identity(*values))


@pytest.mark.parametrize("part", ["runtime", "source", "optimizer", "numerical"])
def test_other_scientific_identity_changes_reject(tmp_path, part):
    values = inputs(tmp_path)
    expected = proposal.build_identity(*values)
    if part == "runtime":
        values[2]["max_steps"] = 6
    elif part == "source":
        values[4]["sha256"] = "b" * 64
    elif part == "optimizer":
        values[0]["optimizer"]["lr"] *= 2
    else:
        values[3]["second_divisor"] = 4
    with pytest.raises(ValueError):
        proposal.require_identity(expected, proposal.build_identity(*values))


def test_json_booleans_are_not_equal_to_numeric_values():
    with pytest.raises(ValueError):
        proposal.require_identity({"value": True}, {"value": 1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_identity_is_rejected(value):
    with pytest.raises(ValueError):
        proposal.canonical({"value": value})


def test_missing_recipe_field_is_not_accepted(tmp_path):
    values = inputs(tmp_path)
    del values[0]["seed"]
    with pytest.raises(ValueError):
        proposal.build_identity(*values)


@pytest.mark.parametrize("index", [2, 4])
def test_unbound_runtime_or_source_is_rejected(tmp_path, index):
    values = list(inputs(tmp_path))
    values[index] = {}
    with pytest.raises(ValueError):
        proposal.build_identity(*values)


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_invalid_dataset_identity_paths_are_rejected(tmp_path, name):
    values = inputs(tmp_path)
    values[1]["files"][0]["path"] = name
    with pytest.raises(ValueError):
        proposal.build_identity(*values)


def test_duplicate_dataset_file_is_rejected(tmp_path):
    values = inputs(tmp_path)
    values[1]["files"] *= 2
    with pytest.raises(ValueError):
        proposal.build_identity(*values)


def test_empty_dataset_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        proposal.dataset_files(tmp_path)


def test_dataset_symlink_is_rejected_without_following_it(tmp_path):
    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "not-data").symlink_to(tmp_path / "missing-target")
    with pytest.raises(ValueError, match="symlinks"):
        proposal.dataset_files(directory)


def test_changed_file_during_hashing_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "data.arrow"
    path.write_bytes(b"before")
    original = proposal.hashlib.file_digest

    def mutate(handle, algorithm):
        value = original(handle, algorithm)
        path.write_bytes(b"changed while hashing")
        return value

    monkeypatch.setattr(proposal.hashlib, "file_digest", mutate)
    with pytest.raises(ValueError, match="changed while hashing"):
        proposal.dataset_files(tmp_path)


def test_identity_is_plain_roundtrippable_json(tmp_path):
    value = proposal.build_identity(*inputs(tmp_path))
    proposal.require_identity(value, json.loads(json.dumps(value)))
