"""The current research/recovery inputs must survive wheel installation.

These checks concern distribution coverage, not native experiment admission or
permission to execute an unreleased training protocol.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from embed_optim.distribution_audit import _data_files

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = re.compile(r"configs/[A-Za-z0-9_./-]+\.(?:json|yaml|yml|txt|lock)")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def test_current_v3_configuration_reference_closure_is_distributed():
    installed = _data_files((ROOT / "pyproject.toml").read_text(), ROOT)
    pending = [path.relative_to(ROOT).as_posix() for path in (ROOT / "configs").glob("*v3*.json")]
    assert len(pending) >= 14
    pending.append("configs/dense_no_packing_state_operator_claim_wording_amendment.json")
    checked = set()
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        checked.add(name)
        path = ROOT / name
        assert path.is_file(), f"Missing configuration reference: {name}"
        assert name in installed, f"Current configuration omitted from distribution: {name}"
        assert installed[name] == PurePosixPath("share/embedding-optimizer-study") / name
        if path.suffix == ".json":
            pending.extend(
                value
                for value in strings(json.loads(path.read_text()))
                if REFERENCE.fullmatch(value)
            )


def test_actual_recovery_scripts_and_guides_are_distributed():
    installed = _data_files((ROOT / "pyproject.toml").read_text(), ROOT)
    paths = [
        "CURRENT_EXPERIMENT.md",
        "scripts/restore_primary_v3.py",
        "scripts/restore_functional_analysis.py",
        "scripts/restore_factorial_probes.py",
        "docs/paper-results-reproduction.md",
        *(path.relative_to(ROOT).as_posix() for path in (ROOT / "docs").glob("*restoration.md")),
    ]
    for name in paths:
        assert name in installed, f"Current recovery entry omitted from distribution: {name}"
        assert installed[name] == PurePosixPath("share/embedding-optimizer-study") / name
