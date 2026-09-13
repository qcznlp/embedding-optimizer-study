"""Scoped exact-append/read-only checks, not waiver of the original failed guard."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).with_name("recover_observed_commit.py")
spec = importlib.util.spec_from_file_location("observed_training_recovery", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def values():
    old = b"original rule\n"
    prefix = "corrected-dense-correctness-v3/training-dynamics/explicit-fixture"
    paths = [prefix + "/native/inputs/admission.json", prefix + "/native/training-trajectories.pdf"]
    new = old + "".join(p + " filter=lfs diff=lfs merge=lfs -text\n" for p in paths).encode()
    return old, new, prefix, paths


def test_exact_two_literal_new_paths_and_old_bytes_retained():
    old, new, prefix, paths = values()
    assert m.attributes(old, new, prefix) == paths


@pytest.mark.parametrize("change", ["old_rule", "wildcard", "other_file", "missing", "extra"])
def test_no_unrelated_attribute_change_admitted(change):
    old, new, prefix, _ = values()
    if change == "old_rule":
        new = new.replace(b"original rule", b"changed rule")
    elif change == "wildcard":
        new = new.replace(b"admission.json", b"*.json")
    elif change == "other_file":
        new = new.replace(b"native/inputs/admission.json", b"historical.json")
    elif change == "missing":
        new = new.splitlines(keepends=True)[0]
    else:
        new += b"* filter=lfs diff=lfs merge=lfs -text\n"
    with pytest.raises(ValueError):
        m.attributes(old, new, prefix)
