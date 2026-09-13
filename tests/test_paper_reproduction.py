"""Version-isolation/refusal controls; full real execution is a separate integration run."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from embed_optim import paper_reproduction as subject

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed"
PAPER = ROOT / "paper/current"


def test_actual_bundle_and_shared_inputs():
    root, manifest = subject.verify_bundle(BUNDLE)
    assert root == BUNDLE and len(manifest["files"]) == 189
    assert len(subject.shared_inputs(root / "expected-paper", PAPER)) == 8
    for name in ("config.py", "optimizers.py"):
        assert (
            subject.SOURCE_ROLES["current_training"][name]
            != subject.SOURCE_ROLES["original_numerical_consumers"][name]
        )


@pytest.mark.parametrize("kind", ["changed", "extra", "missing", "symlink", "manifest"])
def test_bundle_refuses_mutation(tmp_path, kind):
    copied = tmp_path / "bundle"
    shutil.copytree(BUNDLE, copied)
    entry = copied / "source/original_summary.py"
    if kind == "changed":
        entry.write_bytes(entry.read_bytes() + b"\n")
    elif kind == "extra":
        (copied / "extra.txt").write_text("not declared")
    elif kind == "missing":
        entry.unlink()
    elif kind == "symlink":
        entry.unlink()
        entry.symlink_to(BUNDLE / "source/original_summary.py")
    else:
        path = copied / "manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        subject.verify_bundle(copied)


@pytest.mark.parametrize("name", ["../file", "/file", "a/../file", "./file", ""])
def test_manifest_local_name_refusal(name):
    with pytest.raises(ValueError, match="Nonlocal"):
        subject.local_name(name)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', "[]"])
def test_json_refusal(tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        subject.read(path)


def test_shared_input_difference(tmp_path):
    copied = tmp_path / "paper"
    shutil.copytree(PAPER, copied)
    path = copied / "results.tex"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Numerical/reviewed"):
        subject.shared_inputs(copied, PAPER)


@pytest.fixture
def orchestration(monkeypatch, tmp_path):
    """Synthetic orchestration only; no forged actual numerical completion."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(subject, "verify_bundle", lambda p: (BUNDLE, {}))
    monkeypatch.setattr(subject.current_paper, "read_snapshot", lambda p: (PAPER, b"snapshot", {}))
    monkeypatch.setattr(subject, "shared_inputs", lambda *args: {"synthetic": True})
    return tmp_path / "new-output"


def test_child_failure_preserved_before_document(orchestration, monkeypatch):
    calls = []

    def fail(*args):
        calls.append("numerical")
        raise ValueError("original child failure")

    monkeypatch.setattr(subject, "execute_numerical", fail)
    monkeypatch.setattr(
        subject.current_paper, "build_current_paper", lambda *a: calls.append("document")
    )
    with pytest.raises(ValueError, match="original child"):
        subject.reproduce(BUNDLE, PAPER, orchestration)
    assert calls == ["numerical"]
    record = json.loads((orchestration / "failed.json").read_text())
    assert record["automatic_retry"] is False and record["outputs_preserved"] is True
    assert not (orchestration / "complete.json").exists()


def test_existing_output_is_preserved(orchestration, monkeypatch):
    orchestration.mkdir()
    keep = orchestration / "keep.txt"
    keep.write_text("previous output")
    monkeypatch.setattr(subject, "execute_numerical", lambda *a: pytest.fail("must not execute"))
    with pytest.raises(ValueError, match="Preserve previous"):
        subject.reproduce(BUNDLE, PAPER, orchestration)
    assert keep.read_text() == "previous output"
    assert not (orchestration / "failed.json").exists()


def test_input_output_overlap_refused(orchestration):
    with pytest.raises(ValueError, match="inside input"):
        subject.reproduce(BUNDLE, PAPER, PAPER / "must-not-create")


def test_visible_gpu_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        subject, "verify_bundle", lambda *a: pytest.fail("must fail before input access")
    )
    with pytest.raises(ValueError, match="CPU reproduction"):
        subject.reproduce(BUNDLE, PAPER, tmp_path / "new")


def test_child_version_isolation(tmp_path, monkeypatch):
    observed = {}

    class Child:
        pid = 999999999

        def __init__(self, command, **kwargs):
            observed.update(command=command, **kwargs)

        def wait(self, *, timeout):
            assert timeout == 2400
            return 0

    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setattr(subject.subprocess, "Popen", Child)
    record = subject.execute_numerical(BUNDLE, tmp_path / "numerical", tmp_path / "log")
    assert record["exit_code"] == 0
    assert observed["cwd"] == BUNDLE / "primary"
    assert observed["env"]["PYTHONPATH"] == observed["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert observed["start_new_session"] is True
    assert observed["command"][2] == str(BUNDLE / "replay_complete.py")
    assert "--manifest-sha256" in observed["command"]


def test_failed_child_exit_is_recorded(tmp_path, monkeypatch):
    class Child:
        def __init__(self, *args, **kwargs):
            pass

        def wait(self, *, timeout):
            return 17

    monkeypatch.setattr(subject.subprocess, "Popen", Child)
    with pytest.raises(ValueError, match="Original numerical replay failed"):
        subject.execute_numerical(BUNDLE, tmp_path / "numerical", tmp_path / "log")
    assert json.loads((tmp_path / "numerical-exit.json").read_text())["exit_code"] == 17


def test_timeout_signals_only_created_group(tmp_path, monkeypatch):
    signals = []

    class Child:
        pid = 999999997
        calls = 0

        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"] is True

        def wait(self, *, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("synthetic", timeout)
            return -15

    monkeypatch.setattr(subject.subprocess, "Popen", Child)
    monkeypatch.setattr(subject.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        subject.execute_numerical(BUNDLE, tmp_path / "numerical", tmp_path / "log")
    assert signals == [(999999997, subject.signal.SIGTERM)]


def test_foreign_role_is_not_admitted(tmp_path, monkeypatch):
    # Numerical completion is synthetic here; original admission is not simulated in real execution.
    value = {
        "scope": subject.NUMERICAL_SCOPE,
        "input_manifest_sha256": subject.BUNDLE_SHA256,
        "factorial_counts": subject.FACTORIAL_COUNTS,
        "complete_paper_source_and_extracted_text_exact": True,
        "all_original_scientific_rules_unchanged": True,
        "native_model_admission_is_upstream_provenance": True,
        "primary_completion": {},
        "strict_document": {},
        "pdf": {},
        "loaded_project_modules": {"embed_optim.config": {"role": "../outside.py"}},
    }
    monkeypatch.setattr(
        subject,
        "read",
        lambda p: (
            value
            if p.name == "complete.json"
            else {"failure": None, "producer_reads_refused": [], "network_refused": 0}
        ),
    )
    monkeypatch.setattr(subject, "identity", lambda p: {})
    with pytest.raises(ValueError, match="Nonlocal"):
        subject.validate_numerical(tmp_path, BUNDLE)
