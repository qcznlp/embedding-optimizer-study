import json
from hashlib import sha256
from pathlib import Path

import pytest

from embed_optim.runtime import _resolve_spec, load_runtime_spec, runtime_problems

ROOT = Path(__file__).parents[1]


def _spec():
    return {
        "schema_version": 1,
        "python_major_minor": "3.12",
        "torch_cuda": "12.9",
        "packages": {"torch": "2.9.1+cu129", "pylate": "1.6.0"},
    }


def test_runtime_contract_accepts_exact_formal_environment():
    actual = {
        "python": "3.12.3",
        "torch_cuda": "12.9",
        "packages": {"torch": "2.9.1+cu129", "pylate": "1.6.0"},
    }

    assert runtime_problems(_spec(), actual) == []


def test_runtime_contract_reports_every_mismatch():
    actual = {
        "python": "3.11.9",
        "torch_cuda": "12.8",
        "packages": {"torch": "2.9.0", "pylate": None},
    }

    problems = runtime_problems(_spec(), actual)

    assert len(problems) == 4
    assert any("python" in problem for problem in problems)
    assert any("CUDA" in problem for problem in problems)
    assert any("torch is" in problem for problem in problems)
    assert any("pylate is" in problem for problem in problems)


def test_runtime_spec_rejects_invalid_schema(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"schema_version": 2}))

    with pytest.raises(RuntimeError, match="schema"):
        load_runtime_spec(path)


def test_runtime_spec_resolves_installed_package_data(tmp_path, monkeypatch):
    installed = tmp_path / "share" / "embedding-optimizer-study" / "configs"
    installed.mkdir(parents=True)
    expected = installed / "formal_runtime.json"
    expected.write_text(json.dumps(_spec()))
    monkeypatch.chdir(tmp_path)

    assert _resolve_spec("configs/formal_runtime.json", prefix=tmp_path) == expected
    assert load_runtime_spec(expected) == _spec()


def test_checked_in_formal_reconstruction_is_hash_bound_and_matches_packages():
    spec = load_runtime_spec(ROOT / "configs" / "formal_runtime.json")

    assert spec["reconstruction"]["torch_backend"] == "cu129"
    assert spec["reconstruction"]["platform"] == "x86_64-manylinux_2_28"
    assert set(spec["packages"]) == {
        line.split("==", maxsplit=1)[0]
        for line in (ROOT / "configs" / "formal_runtime_constraints.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    }


def test_runtime_spec_rejects_changed_reconstruction_input(tmp_path):
    constraints = tmp_path / "constraints.txt"
    base_lock = tmp_path / "base.lock"
    flash_lock = tmp_path / "flash.lock"
    constraints.write_text("pylate==1.6.0\ntorch==2.9.1+cu129\n", encoding="utf-8")
    base_lock.write_text("locked base\n", encoding="utf-8")
    flash_lock.write_text("locked flash\n", encoding="utf-8")

    spec = _spec()
    spec["reconstruction"] = {
        "platform": "x86_64-manylinux_2_28",
        "torch_backend": "cu129",
        "constraints": {
            "path": constraints.name,
            "sha256": sha256(constraints.read_bytes()).hexdigest(),
        },
        "base_lock": {
            "path": base_lock.name,
            "sha256": sha256(base_lock.read_bytes()).hexdigest(),
        },
        "flash_lock": {
            "path": flash_lock.name,
            "sha256": sha256(flash_lock.read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert load_runtime_spec(path) == spec

    base_lock.write_text("mutated base\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_runtime_spec(path)
