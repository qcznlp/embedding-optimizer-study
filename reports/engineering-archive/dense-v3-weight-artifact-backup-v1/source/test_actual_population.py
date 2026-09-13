"""Explicit host-only integration check, outside the default portable test suite."""

import importlib.util
from pathlib import Path


def test_actual_selection_retains_all_native_files_and_no_code():
    script = Path(__file__).with_name("backup.py")
    spec = importlib.util.spec_from_file_location("actual_weight_artifact_backup", script)
    backup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backup)
    selected = backup.select_sources()
    assert len(selected) == 297
    counts = {
        label: sum(name.startswith(label + "/") for name in selected) for label in backup.ROOTS
    }
    assert counts == {"approximate": 137, "exact": 137, "update-map": 10, "update-map-readout": 7}
    assert all(source.suffix != ".py" for source, _ in selected.values())
