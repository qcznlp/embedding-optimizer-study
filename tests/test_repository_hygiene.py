from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _tracked_files() -> list[Path]:
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
