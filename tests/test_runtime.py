import json

import pytest

from embed_optim.runtime import _resolve_spec, load_runtime_spec, runtime_problems


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
