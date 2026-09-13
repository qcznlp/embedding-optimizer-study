"""Source-version routing preserves all tests and refuses incomplete executions."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts import test_source_roles as runner

ROOT = Path(__file__).resolve().parents[1]


def test_real_roles_partition_every_test_without_overlap():
    roles = runner.load_roles(ROOT)
    assigned = [name for role in roles.values() for name in role["tests"]]
    expected = {path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").rglob("test_*.py")}
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == expected
    assert len(roles["original-analysis"]["tests"]) == 27
    assert len(roles["original-factorial"]["tests"]) == 3
    assert "tests/test_current_primary_source.py" in roles["current"]["tests"]
    assert "tests/test_source_roles.py" in roles["current"]["tests"]
    assert len(roles["original-analysis"]["changes"]) == 6
    assert len(roles["original-factorial"]["changes"]) == 1
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["test"]["steps"]
    formal = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Install the actual hash-locked scientific runtime"
    )
    suite = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Run every test in its explicit source version"
    )
    assert formal < suite
    command = steps[formal]["run"]
    assert "--require-hashes" in command and "--torch-backend cu129" in command
    assert "--overrides requirements-formal.lock" in command
    assert "--dry-run" not in command and "import flash_attn_2_cuda" in command
    assert "embed_optim.runtime --spec configs/formal_runtime.json" in command
    assert steps[formal]["env"]["FLASH_ATTENTION_FORCE_BUILD"] == "TRUE"
    assert "FLASH_ATTENTION_SKIP_CUDA_BUILD" not in steps[formal]["env"]
    assert "uv run --no-sync" in steps[suite]["run"]
    assert all(step.get("continue-on-error", False) is False for step in steps)


@pytest.fixture
def small_repo(tmp_path):
    for name, text in {
        "tests/test_current.py": "current test",
        "tests/test_analysis.py": "analysis test",
        "tests/test_factorial.py": "factorial test",
        "src/module.py": "current code",
        "archive/module.py": "original code",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    change = {
        "src/module.py": {
            "origin": "archive/module.py",
            "sha256": hashlib.sha256(b"original code").hexdigest(),
        }
    }
    roles = {
        "original-analysis": {"changes": deepcopy(change), "tests": ["tests/test_analysis.py"]},
        "original-factorial": {"changes": deepcopy(change), "tests": ["tests/test_factorial.py"]},
    }
    config = tmp_path / runner.ROLE_FILE
    config.parent.mkdir()
    config.write_text(json.dumps({"schema_version": 1, "historical_roles": roles}))
    return tmp_path


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "missing", "wrong_hash", "traversal", "empty", "schema", "unknown_role"],
)
def test_invalid_role_definitions_refused(small_repo, mutation):
    path = small_repo / runner.ROLE_FILE
    payload = json.loads(path.read_text())
    role = payload["historical_roles"]["original-analysis"]
    if mutation == "duplicate":
        role["tests"].append("tests/test_factorial.py")
    elif mutation == "missing":
        role["tests"].append("tests/test_absent.py")
    elif mutation == "wrong_hash":
        role["changes"]["src/module.py"]["sha256"] = "0" * 64
    elif mutation == "traversal":
        role["changes"]["src/module.py"]["origin"] = "../elsewhere.py"
    elif mutation == "empty":
        role["tests"] = []
    elif mutation == "schema":
        payload["schema_version"] = 2
    else:
        payload["historical_roles"]["unexpected"] = deepcopy(role)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        runner.load_roles(small_repo)


@pytest.mark.parametrize("name", ["../outside", "/outside", "ssh/gpu.py", "missing.py"])
def test_member_boundary(tmp_path, name):
    with pytest.raises(ValueError):
        runner.member(tmp_path, name)


def test_source_copy_preserves_test_bytes_and_current_tree(small_repo, tmp_path):
    roles = runner.load_roles(small_repo)
    files = {
        path.relative_to(small_repo).as_posix(): {
            "sha256": runner.sha(path),
            "bytes": path.stat().st_size,
        }
        for path in small_repo.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "role-outputs"
    output.mkdir()
    copies = runner.assemble(small_repo, output, roles, files)
    assert (small_repo / "src/module.py").read_text() == "current code"
    for name in ("original-analysis", "original-factorial"):
        root = Path(copies[name]["root"])
        assert (root / "src/module.py").read_text() == "original code"
        for test in roles[name]["tests"]:
            assert (root / test).read_bytes() == (small_repo / test).read_bytes()


@pytest.mark.parametrize("kind", ["failure", "error", "skipped"])
def test_nonpassing_case_is_not_suppressed(tmp_path, kind):
    path = tmp_path / "result.xml"
    path.write_text(
        f'<testsuites><testsuite><testcase classname="tests.one" name="one"><{kind}/></testcase></testsuite></testsuites>'
    )
    result = runner.read_results(path)
    assert result[kind] == 1
    assert result["cases"] == 1


@pytest.mark.parametrize(
    "body", ["", '<testcase classname="one" name="one"/><testcase classname="one" name="one"/>']
)
def test_empty_or_duplicate_case_inventory_refused(tmp_path, body):
    path = tmp_path / "result.xml"
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    with pytest.raises(ValueError):
        runner.read_results(path)
