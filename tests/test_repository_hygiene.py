from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _tracked_files() -> list[Path]:
    if not (ROOT / ".git").exists():
        pytest.skip("credential hygiene requires a Git repository checkout")
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / path.decode() for path in completed.stdout.split(b"\0") if path]


def _credential_patterns() -> dict[str, re.Pattern[bytes]]:
    # Keep high-confidence prefixes split so this regression does not match its
    # own source while still rejecting complete credentials in any tracked file.
    return {
        "Weights & Biases API token": re.compile(b"wandb" + b"_v1_" + rb"[A-Za-z0-9_-]{20,}"),
        "GitHub personal access token": re.compile(b"gh" + rb"[pousr]_[A-Za-z0-9]{20,}"),
        "GitHub fine-grained token": re.compile(b"github" + b"_pat_" + rb"[A-Za-z0-9_]{20,}"),
        "AWS access key": re.compile(b"AK" + rb"IA[0-9A-Z]{16}"),
        "OpenAI project key": re.compile(b"sk" + b"-proj-" + rb"[A-Za-z0-9_-]{20,}"),
        "private key": re.compile(b"-----BEGIN " + rb"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }


def test_tracked_files_do_not_contain_high_confidence_credentials() -> None:
    findings: list[str] = []
    patterns = _credential_patterns()
    for path in _tracked_files():
        if not path.is_file():
            continue
        payload = path.read_bytes()
        for label, pattern in patterns.items():
            if pattern.search(payload):
                findings.append(f"{path.relative_to(ROOT)}: {label}")

    assert not findings, "Potential credentials in tracked files:\n" + "\n".join(findings)


def test_complete_git_history_does_not_contain_high_confidence_credentials() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("credential history hygiene requires a Git repository checkout")
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert shallow == "false", "Credential history audit requires a complete clone"

    findings: list[str] = []
    for label, pattern in _credential_patterns().items():
        extended_pattern = pattern.pattern.decode().replace("(?:", "(")
        commits = subprocess.run(
            [
                "git",
                "log",
                "--all",
                "--extended-regexp",
                f"-G{extended_pattern}",
                "--format=%H",
                "--",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if commits:
            findings.append(f"{label}: {len(set(commits))} history commit(s)")

    messages = subprocess.run(
        ["git", "log", "--all", "--format=%B"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    findings.extend(
        f"{label}: commit message"
        for label, pattern in _credential_patterns().items()
        if pattern.search(messages)
    )
    assert not findings, "Potential credentials in Git history:\n" + "\n".join(findings)


def test_distributable_receipts_do_not_embed_checkout_paths() -> None:
    receipts = (
        ROOT / "reports/confirmatory-data/receipt.json",
        ROOT / "reports/short-branch/subset-receipt.json",
    )

    def paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "path" and isinstance(item, str):
                    yield item
                yield from paths(item)
        elif isinstance(value, list):
            for item in value:
                yield from paths(item)

    for receipt in receipts:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        embedded = list(paths(payload))
        assert embedded
        assert not [path for path in embedded if Path(path).is_absolute()]
