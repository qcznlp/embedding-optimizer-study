"""Offline candidate for the original v3 job-record layout, not a dispatcher.

No model imports, process inspection, lease acquisition, authority creation or
retry entry point. A future separately authorized recovery must integrate this
component and preserve/reuse the accepted baseline through its own native gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat


STEPS = (782, 1563, 2345, 3126, 3907)
RATES = {
    "adamw": ("1e-6", "3e-6", "1e-5", "3e-5"),
    "muon": ("1e-4", "3e-4", "1e-3", "3e-3"),
    "normuon": ("1e-4", "3e-4", "1e-3", "3e-3"),
}
RUNS = tuple(sorted(f"verified-v3-{optimizer}-{rate}"
                    for optimizer, rates in RATES.items() for rate in rates))
CELLS = ("pretrained",) + tuple(f"{run}/checkpoint-{step}"
                                for run in RUNS for step in STEPS)
JSON_ROLES = ("admission", "started", "exited", "encoded", "verified", "features-verified")
SUFFIXES = tuple(f"{role}.json" for role in JSON_ROLES) + ("log",)
EXPECTED_FILES = frozenset(f"{cell}.{suffix}" for cell in CELLS for suffix in SUFFIXES)


def validate_cells(cells):
    """Require the exact original all-rate, all-stage order, before any writes."""
    if isinstance(cells, (str, bytes)) or tuple(cells) != CELLS:
        raise ValueError("Require the unchanged, ordered 61-state v3 population")
    return CELLS


def ordinary_directory(root):
    root = Path(root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("Use an absolute non-traversing job-record directory")
    for path in (*reversed(root.parents), root):
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise ValueError("Job-record ancestry must contain ordinary directories only")
    return root


def prepare_record_parents(jobs_root, cells):
    """Populate a newly created EMPTY jobs root; never repair an existing attempt.

The original coordinator's exclusive jobs-root creation remains necessary.
Only its twelve missing run subdirectories are created here. All existing
exclusive JSON/log writers and every native record filename stay unchanged.
"""
    validate_cells(cells)
    root = ordinary_directory(jobs_root)
    expected = root.stat()
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        actual = os.fstat(descriptor)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("Job-record root changed before directory creation")
        if os.listdir(descriptor):
            raise ValueError("Require a fresh empty jobs root; preserve existing entries")
        for run in RUNS:
            os.mkdir(run, mode=0o700, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    ordinary_directory(root)
    return tuple(root / run for run in RUNS)


def inventory(jobs_root, cells):
    """Inspect only the declared root and twelve direct run directories."""
    validate_cells(cells)
    root = ordinary_directory(jobs_root)
    files, directories = {}, set()
    for path in root.iterdir():
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode) and path.name in RUNS:
            directories.add(path.name)
            for leaf in path.iterdir():
                relative = f"{path.name}/{leaf.name}"
                if not stat.S_ISREG(leaf.lstat().st_mode) or relative not in EXPECTED_FILES:
                    raise ValueError("Unexpected or non-regular nested job-record entry")
                files[relative] = leaf
        elif stat.S_ISREG(mode) and path.name in EXPECTED_FILES:
            files[path.name] = path
        else:
            raise ValueError("Unexpected or non-regular top-level job-record entry")
    if directories != set(RUNS):
        raise ValueError("Missing declared job-record directories")
    return files


def existing_records(jobs_root, cells, role):
    """Read a complete declared record family, retaining missing/in-flight states.

This is a file-layout component, not a completion or worker-liveness verifier.
Original source/authority/terminal-record and exact-handle checks are still
required by the future observer. Partial JSON raises; it is never counted.
"""
    if role not in JSON_ROLES:
        raise ValueError("Unknown job-record family")
    files = inventory(jobs_root, cells)
    selected = []
    for cell in CELLS:
        name = f"{cell}.{role}.json"
        if name not in files:
            continue
        path = files[name]
        before = path.stat()
        value = json.loads(path.read_bytes())
        after = path.lstat()
        if (not stat.S_ISREG(after.st_mode)
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError("Job record changed during read")
        if not isinstance(value, dict) or value.get("cell") != cell:
            raise ValueError("Job record cell differs from its exact native filename")
        selected.append(path)
    return tuple(sorted(selected))


if __name__ == "__main__":
    raise SystemExit("Offline record-layout candidate only; no recovery execution is authorized")
