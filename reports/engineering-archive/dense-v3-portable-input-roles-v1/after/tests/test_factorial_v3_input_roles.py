"""Location admission only; no GPU or synthetic scientific acceptance."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from embed_optim import factorial_v3_inputs as inputs

REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE = REPOSITORY / inputs.EVIDENCE


def records():
    return tuple(
        inputs._fixed_audit(EVIDENCE, name) for name in ("data-first.json", "states-first.json")
    )


def locations(tmp_path, evidence=EVIDENCE):
    return inputs.Locations(
        *(tmp_path / name for name in ("repository", "primary_source", "experiment", "data_store")),
        evidence,
    )


def test_every_recorded_input_maps_to_explicit_role(tmp_path):
    data, states = records()
    roots = inputs._recorded_roots(data, states)
    local = locations(tmp_path)
    paths = list(data["inputs"])
    paths += list(inputs._fixed_audit(EVIDENCE, "verification.json")["files"])
    for row in states["states"]:
        paths.append(row["checkpoint"])
        paths.extend(
            row[key]["path"]
            for key in ("original_durability_receipt", "original_remote_audit_receipt")
        )
    for recorded in paths:
        role = next(role for role, root in roots.items() if Path(recorded).is_relative_to(root))
        assert local.locate(recorded) == getattr(local, role) / Path(recorded).relative_to(
            roots[role]
        )


@pytest.mark.parametrize(
    "suffix", ["../escape", "nested/../../escape", "a//b", "a/./b", "a\\b", "a\x00b", ""]
)
def test_noncanonical_and_root_only_paths_refused(tmp_path, suffix):
    roots = inputs._recorded_roots(*records())
    with pytest.raises(ValueError):
        locations(tmp_path).locate(str(roots["data_store"]) + "/" + suffix)


def test_unknown_and_sibling_namespaces_refused(tmp_path):
    roots = inputs._recorded_roots(*records())
    for value in (
        "relative/file",
        "/unbound/file",
        str(roots["data_store"].parent / "not-data/file"),
        str(roots["data_store"]) + "-other/file",
    ):
        with pytest.raises(ValueError):
            locations(tmp_path).locate(value)


@pytest.mark.parametrize("role", ["repository", "primary_source", "experiment", "data_store"])
def test_symlink_role_and_descendant_refused(tmp_path, role):
    roots = inputs._recorded_roots(*records())
    local = locations(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    getattr(local, role).symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        local.locate(str(roots[role] / "nested/file"))
    getattr(local, role).unlink()
    getattr(local, role).mkdir()
    (getattr(local, role) / "nested").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        local.locate(str(roots[role] / "nested/file"))


@pytest.mark.parametrize("name", ["data-first.json", "states-first.json"])
def test_modified_audit_refused_before_using_recorded_namespace(tmp_path, name):
    evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, evidence)
    value = json.loads((evidence / name).read_text())
    value["untrusted"] = True
    (evidence / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="audit differs"):
        locations(tmp_path, evidence).locate("/unbound/file")


def test_role_extraction_is_not_specific_to_the_producer_machine():
    # Synthetic lexical transformation only, never passed to authenticated load_inputs.
    data, states = records()
    roots = inputs._recorded_roots(data, states)
    replacements = {old: Path("/synthetic-recording") / role for role, old in roots.items()}
    replacements[roots["evidence"]] = replacements[roots["repository"]] / inputs.EVIDENCE

    def replace(value):
        for old, new in replacements.items():
            if Path(value).is_relative_to(old):
                return str(new / Path(value).relative_to(old))
        raise AssertionError(value)

    data = copy.deepcopy(data)
    states = copy.deepcopy(states)
    data["inputs"] = {replace(path): binding for path, binding in data["inputs"].items()}
    states["audit_source"]["path"] = replace(states["audit_source"]["path"])
    for row in states["states"]:
        row["checkpoint"] = replace(row["checkpoint"])
    assert inputs._recorded_roots(data, states) == {
        role: replacements[old] for role, old in roots.items()
    }
