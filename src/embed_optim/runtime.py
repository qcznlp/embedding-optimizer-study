from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _resolve_spec(path: str | Path, prefix: Path | None = None) -> Path:
    path = Path(path)
    if path.is_file() or path.is_absolute() or path.parent != Path("configs"):
        return path
    prefix = Path(sys.prefix) if prefix is None else prefix
    installed = prefix / "share" / "embedding-optimizer-study" / "configs" / path.name
    return installed if installed.is_file() else path


def load_runtime_spec(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_spec(path)
    try:
        spec = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid formal runtime specification {resolved}: {error}") from error
    if (
        spec.get("schema_version") != 1
        or not isinstance(spec.get("python_major_minor"), str)
        or not isinstance(spec.get("torch_cuda"), str)
        or not isinstance(spec.get("packages"), dict)
        or not spec["packages"]
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in spec["packages"].items()
        )
    ):
        raise RuntimeError(f"Invalid formal runtime specification schema: {resolved}")
    return spec


def runtime_snapshot(packages: list[str]) -> dict[str, Any]:
    import torch

    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_cuda": torch.version.cuda,
        "packages": versions,
    }


def runtime_problems(spec: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    python = str(actual.get("python", ""))
    expected_python = spec["python_major_minor"]
    if not (python == expected_python or python.startswith(f"{expected_python}.")):
        problems.append(f"python is {python or 'missing'}, expected {expected_python}.x")
    if actual.get("torch_cuda") != spec["torch_cuda"]:
        problems.append(
            f"torch CUDA build is {actual.get('torch_cuda')!r}, expected {spec['torch_cuda']!r}"
        )
    actual_packages = actual.get("packages", {})
    for package, expected in spec["packages"].items():
        observed = actual_packages.get(package)
        if observed != expected:
            problems.append(f"{package} is {observed!r}, expected {expected!r}")
    return problems


def verify_runtime_spec(path: str | Path) -> dict[str, Any]:
    spec = load_runtime_spec(path)
    actual = runtime_snapshot(list(spec["packages"]))
    problems = runtime_problems(spec, actual)
    if problems:
        raise RuntimeError("Formal runtime mismatch: " + "; ".join(problems))
    return actual


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify an interpreter against the frozen formal experiment runtime"
    )
    parser.add_argument("--spec", default="configs/formal_runtime.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_runtime_spec(args.spec)
    actual = runtime_snapshot(list(spec["packages"]))
    problems = runtime_problems(spec, actual)
    print(
        json.dumps(
            {"valid": not problems, "problems": problems, "runtime": actual},
            indent=2,
            sort_keys=True,
        )
    )
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
