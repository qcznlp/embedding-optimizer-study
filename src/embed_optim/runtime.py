from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _constraint_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise RuntimeError(
                f"Invalid formal runtime constraint at {path}:{line_number}: {line!r}"
            )
        package, version = (part.strip() for part in line.split("==", maxsplit=1))
        if not package or not version or package in versions:
            raise RuntimeError(
                f"Invalid formal runtime constraint at {path}:{line_number}: {line!r}"
            )
        versions[package] = version
    return versions


def _validate_reconstruction(spec: dict[str, Any], spec_path: Path) -> None:
    reconstruction = spec.get("reconstruction")
    if reconstruction is None:
        return
    if not isinstance(reconstruction, dict) or set(reconstruction) != {
        "platform",
        "torch_backend",
        "constraints",
        "base_lock",
        "flash_lock",
    }:
        raise RuntimeError(f"Invalid formal runtime reconstruction schema: {spec_path}")
    if not all(
        isinstance(reconstruction.get(key), str) and reconstruction[key]
        for key in ("platform", "torch_backend")
    ):
        raise RuntimeError(f"Invalid formal runtime reconstruction schema: {spec_path}")

    resolved_inputs: dict[str, Path] = {}
    for name in ("constraints", "base_lock", "flash_lock"):
        identity = reconstruction.get(name)
        if (
            not isinstance(identity, dict)
            or set(identity) != {"path", "sha256"}
            or not isinstance(identity.get("path"), str)
            or not identity["path"]
            or not isinstance(identity.get("sha256"), str)
            or len(identity["sha256"]) != 64
        ):
            raise RuntimeError(f"Invalid formal runtime reconstruction schema: {spec_path}")
        path = (spec_path.parent / identity["path"]).resolve()
        if not path.is_file():
            raise RuntimeError(f"Formal runtime reconstruction input is missing: {path}")
        observed = _sha256(path)
        if observed != identity["sha256"]:
            raise RuntimeError(
                f"Formal runtime reconstruction input hash mismatch for {path}: "
                f"expected {identity['sha256']}, observed {observed}"
            )
        resolved_inputs[name] = path

    constraints = _constraint_versions(resolved_inputs["constraints"])
    if constraints != spec["packages"]:
        raise RuntimeError(
            "Formal runtime constraints differ from the package-version specification: "
            f"{resolved_inputs['constraints']}"
        )


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
    _validate_reconstruction(spec, resolved)
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
