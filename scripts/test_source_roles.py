#!/usr/bin/env python3
"""Run the complete test suite in explicit current and historical source roles.

Historical contracts are immutable. Their consumers run in ordinary private
copies with authenticated original dependencies; current-source tests still run
against the current checkout. This is not admission of a changed training worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROLE_FILE = Path("configs/source_test_roles.json")
CLOSED = Path("reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed")
CLOSED_SHA = "746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Not a repository-relative member: {name}")
    if "gpu.py" in path.parts:
        raise ValueError("Protected helper is outside test assembly scope")
    target = root / path
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
        raise ValueError(f"Not a regular contained member: {name}")
    return target


def load_roles(root: Path) -> dict:
    value = json.loads(member(root, str(ROLE_FILE)).read_text())
    if value.get("schema_version") != 1:
        raise ValueError("Unknown source-test role schema")
    roles = value["historical_roles"]
    if set(roles) != {"original-analysis", "original-factorial"}:
        raise ValueError("Unexpected historical source roles")
    all_tests = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py")}
    assigned: set[str] = set()
    for role in roles.values():
        names = role["tests"]
        if not names or len(names) != len(set(names)) or assigned.intersection(names):
            raise ValueError("Duplicate or empty historical test assignment")
        if not set(names).issubset(all_tests):
            raise ValueError("Historical role names an absent test module")
        assigned.update(names)
        if not role["changes"]:
            raise ValueError("Historical role has no source identity")
        for target, binding in role["changes"].items():
            member(root, target)
            if sha(member(root, binding["origin"])) != binding["sha256"]:
                raise ValueError(f"Historical source identity differs: {target}")
    current = sorted(all_tests - assigned)
    if not current:
        raise ValueError("Current-source tests are absent")
    return {"current": {"tests": current, "changes": {}}, **roles}


def inventory(root: Path) -> dict:
    names = set(
        subprocess.check_output(
            ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=root
        )
        .decode()
        .split("\0")
    ) - {""}
    manifest = member(root, str(CLOSED / "manifest.json"))
    if sha(manifest) != CLOSED_SHA:
        raise ValueError("Original closed numerical inventory differs")
    names.update(str(CLOSED / name) for name in json.loads(manifest.read_text())["files"])
    names.add(str(CLOSED / "manifest.json"))
    records = {}
    for name in sorted(names):
        if "gpu.py" in Path(name).parts:
            continue
        # Git also lists deliberately deleted tracked files; do not resurrect them.
        if not (root / name).exists() and not (root / name).is_symlink():
            continue
        path = member(root, name)
        records[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    return records


def assemble(root: Path, output: Path, roles: dict, files: dict) -> dict:
    assemblies = {}
    for name, role in roles.items():
        destination = root if name == "current" else output / name
        if name != "current":
            destination.mkdir(exist_ok=False)
            for label, binding in files.items():
                replacement = role["changes"].get(label)
                source = member(root, replacement["origin"] if replacement else label)
                expected = replacement["sha256"] if replacement else binding["sha256"]
                if sha(source) != expected:
                    raise ValueError(f"Source changed during test assembly: {label}")
                target = destination / label
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if sha(target) != expected:
                    raise ValueError(f"Test assembly copy differs: {label}")
        for label in role["tests"]:
            if label not in files or sha(member(destination, label)) != files[label]["sha256"]:
                raise ValueError(f"Test assertion source differs: {label}")
        assemblies[name] = {**role, "root": str(destination)}
    return assemblies


def read_results(path: Path) -> dict:
    document = ET.parse(path).getroot()
    cases = list(document.iter("testcase"))
    if not cases:
        raise ValueError("No test cases executed")
    counts = {
        kind: sum(case.find(kind) is not None for case in cases)
        for kind in ("failure", "error", "skipped")
    }
    identifiers = [case.attrib.get("classname", "") + "::" + case.attrib["name"] for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate executed test case")
    return {"cases": len(cases), **counts, "identifiers": identifiers}


def execute(name: str, role: dict, output: Path, files: dict) -> dict:
    root = Path(role["root"])
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(root / "src") + os.pathsep + str(root),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    xml = output / f"{name}.xml"
    command = [sys.executable, "-B", "-m", "pytest", "-q", *role["tests"], f"--junitxml={xml}"]
    with (output / f"{name}.log").open("xb") as stream:
        child = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    result = {"role": name, "command": command, "cwd": str(root), "exit_code": child.returncode}
    try:
        result.update(read_results(xml))
        for label in role["tests"]:
            if sha(member(root, label)) != files[label]["sha256"]:
                raise ValueError(f"Test source changed during execution: {label}")
        result["complete"] = child.returncode == 0 and not any(
            result[key] for key in ("failure", "error", "skipped")
        )
    except (ValueError, OSError, ET.ParseError) as error:
        result.update(complete=False, validation_error=str(error))
    (output / f"{name}-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"command", "identifiers"}}
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory outside the checkout; failed attempts are retained",
    )
    args = parser.parse_args()
    root = args.repository.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Require a new output directory outside the checkout")
    roles = load_roles(root)
    files = inventory(root)
    output.mkdir(parents=True, exist_ok=False)
    assemblies = assemble(root, output, roles, files)
    (output / "inputs.json").write_text(
        json.dumps({"files": files, "roles": assemblies}, indent=2, sort_keys=True) + "\n"
    )
    # Separate OS processes avoid shared Python import caches across source versions.
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [
            pool.submit(execute, name, role, output, files) for name, role in assemblies.items()
        ]
        results = [job.result() for job in jobs]
    identifiers = [item for result in results for item in result.get("identifiers", [])]
    summary = {
        "complete": all(result["complete"] for result in results)
        and len(set(identifiers)) == len(identifiers),
        "cases": len(identifiers),
        "test_modules": sum(len(role["tests"]) for role in roles.values()),
        "roles": [
            {key: value for key, value in result.items() if key not in {"command", "identifiers"}}
            for result in results
        ],
        "role_file_sha256": sha(root / ROLE_FILE),
        "fresh_training_worker_admission": False,
        "remote_publication": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
