from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _credential_patterns() -> dict[str, re.Pattern[bytes]]:
    # Split high-confidence prefixes so the scanner's own source is safe to bundle.
    return {
        "Weights & Biases API token": re.compile(b"wandb" + b"_v1_" + rb"[A-Za-z0-9_-]{20,}"),
        "GitHub personal access token": re.compile(b"gh" + rb"[pousr]_[A-Za-z0-9]{20,}"),
        "GitHub fine-grained token": re.compile(b"github" + b"_pat_" + rb"[A-Za-z0-9_]{20,}"),
        "AWS access key": re.compile(b"AK" + rb"IA[0-9A-Z]{16}"),
        "OpenAI project key": re.compile(b"sk" + b"-proj-" + rb"[A-Za-z0-9_-]{20,}"),
        "private key": re.compile(b"-----BEGIN " + rb"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }


def _credential_findings(scope: str, member: str, payload: bytes) -> list[str]:
    return [
        f"{scope} credential pattern: {member}: {label}"
        for label, pattern in _credential_patterns().items()
        if pattern.search(payload)
    ]


def _checkout_path_findings(scope: str, member: str, payload: bytes) -> list[str]:
    producer_path = b"/root" + b"/embedding-optimizer-study"
    return [f"{scope} producer checkout path: {member}"] if producer_path in payload else []


def _section(text: str, name: str) -> str:
    marker = f"[{name}]"
    if marker not in text:
        raise ValueError(f"Missing {marker} in pyproject.toml")
    return text.split(marker, 1)[1].split("\n[", 1)[0]


def _project_identity(text: str) -> tuple[str, str, str]:
    section = _section(text, "project")
    fields = dict(re.findall(r'^(name|version)\s*=\s*"([^"]+)"$', section, re.MULTILINE))
    if set(fields) != {"name", "version"}:
        raise ValueError("Project name/version could not be parsed from pyproject.toml")
    distribution = re.sub(r"[-_.]+", "_", fields["name"])
    return fields["name"], fields["version"], distribution


def _project_scripts(text: str) -> dict[str, str]:
    section = _section(text, "project.scripts")
    scripts = dict(re.findall(r'^([A-Za-z0-9_-]+)\s*=\s*"([^"]+)"$', section, re.MULTILINE))
    if not scripts:
        raise ValueError("No project scripts could be parsed from pyproject.toml")
    return scripts


def _data_files(text: str) -> dict[str, PurePosixPath]:
    section = _section(text, "tool.setuptools.data-files")
    groups = re.findall(r'^"([^"]+)"\s*=\s*\[(.*?)^\]$', section, flags=re.MULTILINE | re.DOTALL)
    files = {
        source: PurePosixPath(destination) / Path(source).name
        for destination, sources in groups
        for source in re.findall(r'^\s+"([^"]+)",?$', sources, flags=re.MULTILINE)
    }
    if not files:
        raise ValueError("No setuptools data files could be parsed from pyproject.toml")
    return files


def _entry_points(archive: zipfile.ZipFile, path: str) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read_string(archive.read(path).decode("utf-8"))
    if "console_scripts" not in parser:
        return {}
    return dict(parser["console_scripts"])


def _artifact_paths(dist_dir: Path, distribution: str, version: str) -> tuple[Path, Path]:
    wheels = sorted(dist_dir.glob(f"{distribution}-{version}-*.whl"))
    sdist = dist_dir / f"{distribution}-{version}.tar.gz"
    if len(wheels) != 1:
        raise ValueError(
            f"Expected one wheel for {distribution} {version} in {dist_dir}, got {wheels}"
        )
    if not sdist.is_file():
        raise FileNotFoundError(sdist)
    return wheels[0], sdist


def audit_distribution(
    repo_root: str | Path = ".", *, dist_dir: str | Path = "dist"
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    project, version, distribution = _project_identity(text)
    scripts = _project_scripts(text)
    data_files = _data_files(text)
    resolved_dist = Path(dist_dir)
    if not resolved_dist.is_absolute():
        resolved_dist = root / resolved_dist
    wheel, sdist = _artifact_paths(resolved_dist.resolve(), distribution, version)

    problems: list[str] = []
    package_sources = sorted((root / "src" / "embed_optim").glob("*.py"))
    wheel_prefix = f"{distribution}-{version}"
    package_members = {path: path.relative_to(root / "src").as_posix() for path in package_sources}
    data_members = {
        (root / source): f"{wheel_prefix}.data/data/{installed.as_posix()}"
        for source, installed in data_files.items()
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        corrupt = archive.testzip()
        if corrupt is not None:
            problems.append(f"wheel CRC failure: {corrupt}")
        for member in archive.infolist():
            if not member.is_dir():
                problems.extend(
                    _credential_findings("wheel", member.filename, archive.read(member))
                )
                problems.extend(
                    _checkout_path_findings("wheel", member.filename, archive.read(member))
                )
        expected_package = set(package_members.values())
        expected_data = set(data_members.values())
        expected_metadata = {
            f"{wheel_prefix}.dist-info/METADATA",
            f"{wheel_prefix}.dist-info/WHEEL",
            f"{wheel_prefix}.dist-info/RECORD",
            f"{wheel_prefix}.dist-info/entry_points.txt",
            f"{wheel_prefix}.dist-info/licenses/LICENSE",
        }
        missing_wheel = sorted((expected_package | expected_data | expected_metadata) - names)
        problems.extend(f"wheel missing: {path}" for path in missing_wheel)
        for source, member in {**package_members, **data_members}.items():
            if member in names and archive.read(member) != source.read_bytes():
                problems.append(
                    f"wheel content mismatch: {member} != {source.relative_to(root).as_posix()}"
                )
        license_member = f"{wheel_prefix}.dist-info/licenses/LICENSE"
        if (
            license_member in names
            and archive.read(license_member) != (root / "LICENSE").read_bytes()
        ):
            problems.append("wheel content mismatch: bundled LICENSE != LICENSE")
        entry_path = f"{wheel_prefix}.dist-info/entry_points.txt"
        observed_scripts = _entry_points(archive, entry_path) if entry_path in names else {}
        if observed_scripts != scripts:
            problems.append("wheel console scripts differ from pyproject.toml")

    sdist_prefix = f"{distribution}-{version}/"
    expected_sdist_sources = {
        "README.md",
        "LICENSE",
        "pyproject.toml",
        *(path.relative_to(root).as_posix() for path in package_sources),
        *data_files,
    }
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
        payloads: dict[str, bytes] = {}
        missing_sdist = sorted(
            source for source in expected_sdist_sources if f"{sdist_prefix}{source}" not in names
        )
        problems.extend(f"sdist missing: {path}" for path in missing_sdist)
        # Reading every regular member validates tar headers and gzip payloads.
        for member in archive.getmembers():
            if member.isfile():
                extracted = archive.extractfile(member)
                if extracted is None:
                    problems.append(f"sdist unreadable: {member.name}")
                    continue
                payload = extracted.read()
                payloads[member.name] = payload
                problems.extend(_credential_findings("sdist", member.name, payload))
                problems.extend(_checkout_path_findings("sdist", member.name, payload))
        for source in sorted(expected_sdist_sources):
            member = f"{sdist_prefix}{source}"
            if member not in names:
                continue
            if payloads.get(member) != (root / source).read_bytes():
                problems.append(f"sdist content mismatch: {source}")

    return {
        "schema_version": 1,
        "complete": not problems,
        "project": project,
        "version": version,
        "declared_console_scripts": len(scripts),
        "declared_data_files": len(data_files),
        "package_modules": len(package_sources),
        "wheel": {
            "path": str(wheel),
            "bytes": wheel.stat().st_size,
            "sha256": _sha256(wheel),
        },
        "sdist": {
            "path": str(sdist),
            "bytes": sdist.stat().st_size,
            "sha256": _sha256(sdist),
        },
        "problems": problems,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the built wheel and sdist against the repository distribution contract"
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = audit_distribution(args.repo_root, dist_dir=args.dist_dir)
    except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(json.dumps({"schema_version": 1, "complete": False, "error": str(error)}))
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
