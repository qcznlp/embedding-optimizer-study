from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

_CONFIG_DOCUMENT_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
_CONFIG_DEPENDENCY_SUFFIXES = frozenset({*_CONFIG_DOCUMENT_SUFFIXES, ".lock", ".txt"})
_EXECUTABLE_CONFIG_REFERENCE_KEYS = frozenset(
    {
        "config",
        "formal_runtime",
        "matrix",
        "path",
        "protocol",
        "scope_amendment",
        "spec",
    }
)
_EXECUTABLE_CONFIG_REFERENCE_SUFFIXES = ("_config", "_matrix", "_protocol", "_spec")


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


def _data_files(text: str, root: Path) -> dict[str, PurePosixPath]:
    section = _section(text, "tool.setuptools.data-files")
    groups = re.findall(r'^"([^"]+)"\s*=\s*\[(.*?)^\]$', section, flags=re.MULTILINE | re.DOTALL)
    files: dict[str, PurePosixPath] = {}
    for destination, sources in groups:
        for pattern in re.findall(r'^\s+"([^"]+)",?$', sources, flags=re.MULTILINE):
            matches = (
                sorted(path for path in root.glob(pattern) if path.is_file())
                if any(character in pattern for character in "*?[")
                else [root / pattern]
            )
            for path in matches:
                source = path.relative_to(root).as_posix()
                files[source] = PurePosixPath(destination) / path.name
    if not files:
        raise ValueError("No setuptools data files could be parsed from pyproject.toml")
    return files


def _dense_dynamics_distribution_problems(
    root: Path, data_files: dict[str, PurePosixPath]
) -> list[str]:
    manifest = root / "reports/dense-retrieval-dynamics/summary_manifest.json"
    if not manifest.is_file():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["Dense retrieval-dynamics distribution manifest is unreadable"]
    if payload.get("complete") is not True:
        return []
    expected = {
        "reports/dense-retrieval-dynamics/summary_manifest.json",
        "reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.csv",
        "reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.svg",
        "reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.pdf",
    }
    return [
        f"completed Dense retrieval-dynamics distribution missing: {source}"
        for source in sorted(expected)
        if source not in data_files or not (root / source).is_file()
    ]


def _candidate_breadth_distribution_problems(
    root: Path, data_files: dict[str, PurePosixPath]
) -> list[str]:
    summary = root / "reports/candidate-breadth/summary.json"
    if not summary.is_file():
        return []
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["Candidate-breadth distribution summary is unreadable"]
    if payload.get("status") != "complete":
        return []
    expected = {
        "reports/candidate-breadth/data-audit.json",
        "reports/candidate-breadth/summary.json",
        "reports/candidate-breadth/calibration_by_width.csv",
        "reports/candidate-breadth/high_dose_contrasts.csv",
        "reports/candidate-breadth/candidate_breadth_calibration.svg",
        "reports/candidate-breadth/candidate_breadth_calibration.pdf",
        "reports/candidate-breadth/publication_manifest.json",
    }
    return [
        f"completed candidate-breadth distribution missing: {source}"
        for source in sorted(expected)
        if source not in data_files or not (root / source).is_file()
    ]


def _wandb_source_receipt_distribution_problems(
    root: Path,
    data_files: dict[str, PurePosixPath],
) -> list[str]:
    """Validate and require packaging of a source-audit receipt when present."""

    source = "reports/wandb/dense_source_provenance_audit.json"
    path = root / source
    if not path.is_file():
        return []
    problems: list[str] = []
    if source not in data_files:
        problems.append(f"W&B source-audit receipt is not declared for distribution: {source}")
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [*problems, "W&B source-audit receipt is unreadable"]
    if not isinstance(envelope, dict):
        return [*problems, "W&B source-audit receipt is not an object"]
    audit = envelope.get("audit")
    expected_hash = envelope.get("audit_sha256")
    if not isinstance(audit, dict) or not isinstance(expected_hash, str):
        return [*problems, "W&B source-audit receipt envelope is malformed"]
    try:
        canonical_audit = json.dumps(
            audit,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        canonical_audit = None
    observed_hash = (
        hashlib.sha256(canonical_audit).hexdigest() if canonical_audit is not None else None
    )
    if (
        envelope.get("schema_version") != 1
        or envelope.get("kind") != "dense-wandb-source-provenance-audit"
        or set(envelope) != {"schema_version", "kind", "audit", "audit_sha256"}
        or expected_hash != observed_hash
    ):
        problems.append("W&B source-audit receipt envelope/hash differs")

    runs = audit.get("runs")
    expected_counts = {
        "discovery": 12,
        "hybrid": 4,
        "confirmatory": 9,
        "short-branch": 9,
    }
    valid_runs = (
        isinstance(runs, list) and len(runs) == 34 and all(isinstance(run, dict) for run in runs)
    )
    phase_values = [run.get("phase") for run in runs] if valid_runs else []
    valid_phases = valid_runs and all(isinstance(phase, str) for phase in phase_values)
    observed_counts = Counter(phase_values) if valid_phases else Counter()
    source_ids = [run.get("source_wandb_run_id") for run in runs] if valid_runs else []
    valid_source_ids = valid_runs and all(
        isinstance(source_id, str) and bool(source_id) for source_id in source_ids
    )
    digest_fields_valid = valid_runs and all(
        isinstance(run.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", run[key]) is not None
        for run in runs
        for key in ("config_sha256", "run_config_sha256", "history_sha256")
    )
    remote_contract_valid = valid_runs and all(
        run.get("group") == "dense"
        and isinstance(run.get("optimizer"), str)
        and bool(run["optimizer"])
        and isinstance(run.get("seed"), int)
        and not isinstance(run["seed"], bool)
        and run.get("required_tags") == sorted({"dense", run["optimizer"], f"seed-{run['seed']}"})
        and run.get("config_keys") == 179
        and isinstance(run.get("git_commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", run["git_commit"]) is not None
        and run.get("git_commit_validation") == "reachable-local-ref"
        for run in runs
    )
    if (
        audit.get("schema_version") != 1
        or audit.get("status") != "passed"
        or audit.get("remote_access") != "read-only"
        or audit.get("entity") != "stevezenguom"
        or audit.get("project") != "embedding-optimizer-study"
        or audit.get("expected_git_remote")
        != "https://github.com/qcznlp/embedding-optimizer-study.git"
        or audit.get("expected_counts") != expected_counts
        or audit.get("verified_runs") != 34
        or observed_counts != Counter(expected_counts)
        or not valid_source_ids
        or len(set(source_ids)) != 34
        or not digest_fields_valid
        or not remote_contract_valid
        or any(run.get("state") != "finished" for run in runs if isinstance(run, dict))
    ):
        problems.append("W&B source-audit receipt does not prove the frozen 34-run contract")
    return problems


def _runtime_config_references(package_sources: list[Path], root: Path) -> set[str]:
    # Runtime defaults are expressed both as ``Path("configs/...")`` and as
    # plain strings accepted by argparse/helper APIs.  Scan quoted literals
    # instead of only the former spelling so a valid console entry point cannot
    # silently depend on a config omitted from the wheel.
    pattern = re.compile(r"""["'](configs/[^"']+)["']""")
    references = {
        match
        for path in package_sources
        for match in pattern.findall(path.read_text(encoding="utf-8"))
    }
    return {source for source in references if (root / source).is_file()}


def _is_repository_only_provenance_key(key: str) -> bool:
    return key == "source_bindings" or key.endswith(("_bindings", "_manifest", "_manifest_path"))


def _is_executable_config_reference_key(key: str) -> bool:
    return key in _EXECUTABLE_CONFIG_REFERENCE_KEYS or key.endswith(
        _EXECUTABLE_CONFIG_REFERENCE_SUFFIXES
    )


def _config_reference(
    value: str,
    source: Path,
    root: Path,
    *,
    allow_repository_root: bool,
    include_missing_relative: bool,
) -> str | None:
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or any(char.isspace() for char in value)
    ):
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.suffix.lower() not in _CONFIG_DEPENDENCY_SUFFIXES:
        return None
    explicit_repo_path = relative.parts[:1] == ("configs",)
    candidate = root / relative if explicit_repo_path else source.parent / relative
    resolved = candidate.resolve()
    allowed_root = root.resolve() if allow_repository_root else (root / "configs").resolve()
    if not resolved.is_relative_to(allowed_root):
        return None
    if not explicit_repo_path and not include_missing_relative and not resolved.is_file():
        return None
    return resolved.relative_to(root).as_posix()


def _document_config_references(
    document: Any,
    *,
    source: Path,
    root: Path,
) -> tuple[set[str], set[str]]:
    executable: set[str] = set()
    provenance: set[str] = set()

    def visit(value: Any, *, key: str | None = None, repository_only: bool = False) -> None:
        if isinstance(value, dict):
            for raw_child_key, child in value.items():
                child_key = str(raw_child_key)
                visit(
                    child,
                    key=child_key,
                    repository_only=(
                        repository_only or _is_repository_only_provenance_key(child_key)
                    ),
                )
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key=key, repository_only=repository_only)
            return
        if not isinstance(value, str) or key is None:
            return
        executable_key = _is_executable_config_reference_key(key)
        reference = _config_reference(
            value,
            source,
            root,
            allow_repository_root=False,
            # Generic ``path`` fields often describe dataset/report artifacts.
            # Missing relative paths are dependencies only for explicit config-loader keys.
            include_missing_relative=executable_key and key != "path",
        )
        if reference is None:
            return
        if repository_only:
            provenance.add(reference)
        elif executable_key:
            executable.add(reference)

    visit(document)
    if source.resolve() == (root / "configs/formal_runtime.json").resolve() and isinstance(
        document, dict
    ):
        reconstruction = document.get("reconstruction")
        if isinstance(reconstruction, dict):
            for name in ("constraints", "base_lock", "flash_lock"):
                identity = reconstruction.get(name)
                if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
                    continue
                reference = _config_reference(
                    identity["path"],
                    source,
                    root,
                    allow_repository_root=True,
                    include_missing_relative=True,
                )
                if reference is not None:
                    executable.add(reference)
    return executable, provenance


def _transitive_config_references(
    declared_sources: set[str], root: Path
) -> tuple[set[str], set[str]]:
    roots = {
        source
        for source in declared_sources
        if source.startswith("configs/")
        and PurePosixPath(source).suffix.lower() in _CONFIG_DOCUMENT_SUFFIXES
    }
    pending = sorted(roots, reverse=True)
    visited: set[str] = set()
    executable: set[str] = set()
    provenance: set[str] = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        path = root / source
        if not path.is_file() or path.suffix.lower() not in _CONFIG_DOCUMENT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        document = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        direct_executable, direct_provenance = _document_config_references(
            document,
            source=path,
            root=root,
        )
        executable.update(direct_executable)
        provenance.update(direct_provenance)
        pending.extend(
            sorted(
                (
                    reference
                    for reference in direct_executable
                    if PurePosixPath(reference).suffix.lower() in _CONFIG_DOCUMENT_SUFFIXES
                    and reference not in visited
                ),
                reverse=True,
            )
        )
    return executable, provenance


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
    data_files = _data_files(text, root)
    resolved_dist = Path(dist_dir)
    if not resolved_dist.is_absolute():
        resolved_dist = root / resolved_dist
    wheel, sdist = _artifact_paths(resolved_dist.resolve(), distribution, version)

    problems: list[str] = []
    problems.extend(_dense_dynamics_distribution_problems(root, data_files))
    problems.extend(_candidate_breadth_distribution_problems(root, data_files))
    problems.extend(_wandb_source_receipt_distribution_problems(root, data_files))
    package_sources = sorted((root / "src" / "embed_optim").glob("*.py"))
    runtime_configs = _runtime_config_references(package_sources, root)
    declared_sources = set(data_files)
    problems.extend(
        f"pyproject data-files missing runtime config: {source}"
        for source in sorted(runtime_configs - declared_sources)
    )
    executable_configs, provenance_configs = _transitive_config_references(
        declared_sources,
        root,
    )
    problems.extend(
        f"pyproject data-files missing executable config dependency: {source}"
        for source in sorted(executable_configs - declared_sources)
    )
    repository_only_provenance = sorted(provenance_configs - declared_sources)
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
        "runtime_config_references": len(runtime_configs),
        "transitive_executable_config_references": len(executable_configs),
        "repository_only_provenance_config_references": repository_only_provenance,
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
