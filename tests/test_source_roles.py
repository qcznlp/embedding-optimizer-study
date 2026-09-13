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
    numerical = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Reproduce complete numerical findings and reviewed paper"
    )
    assert formal < numerical < suite
    retained = next(
        step for step in steps if step.get("name") == "Retain actual test and document receipts"
    )
    assert retained["if"] == "always()"
    assert "${{ runner.temp }}/complete-paper/\n" in retained["with"]["path"]
    command = steps[formal]["run"]
    assert "--require-hashes" in command and "--torch-backend cu129" in command
    assert "--overrides requirements-formal.lock" in command
    assert "--dry-run" not in command and "import flash_attn_2_cuda" in command
    assert "embed_optim.runtime --spec configs/formal_runtime.json" in command
    assert workflow["jobs"]["test"]["runs-on"] == "ubuntu-24.04"
    assert "--no-deps --require-hashes -r requirements-ci-flash-wheel.txt" in command
    assert "--no-deps --require-hashes -r requirements-primary-replay.txt" in command
    replay_lock = (ROOT / "requirements-primary-replay.txt").read_text()
    assert "huggingface-hub==1.28.0" in replay_lock
    assert "58a8bacb03072edfc38067065e9dc24bbb34805410fcd36a1632de0b329660bb" in replay_lock
    binary_lock = (ROOT / "requirements-ci-flash-wheel.txt").read_text()
    assert "/resolve/18c1ab7abc636d2542be6831591f31c171c7dcf6/" in binary_lock
    assert (
        "--hash=sha256:9feca56918f1603358f32ce0c7ab06d221e6123b8b1072566880479f3b1aa1ba"
        in binary_lock
    )
    assert "_GLIBCXX_USE_CXX11_ABI" in command and "varlen_bwd" in command
    assert steps[formal]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert "FLASH_ATTENTION_SKIP_CUDA_BUILD" not in steps[formal]["env"]
    assert "uv run --no-sync" in steps[suite]["run"]
    assert all(step.get("continue-on-error", False) is False for step in steps)
    emulator = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Prepare authenticated CPU instruction emulation"
    )
    probe = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Verify original functional decisions under CPU emulation"
    )
    assert formal < emulator < probe < numerical < suite
    assert (
        "94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50" in steps[emulator]["run"]
    )
    assert "sha256sum --check" in steps[emulator]["run"]
    assert "https://downloadmirror.intel.com/924984/" in steps[emulator]["run"]
    for index in (probe, numerical):
        assert "-skx -force_emulate skx --" in steps[index]["run"]
        assert steps[index]["env"]["OPENBLAS_CORETYPE"] == "SkylakeX"
        assert steps[index]["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "if" not in steps[index]
    assert 'r["decisions_exact"] is True' in steps[probe]["run"]
    assert "make -C paper release" in steps[numerical]["run"]
    assert "${{ runner.temp }}/cpu-replay-diagnostics/\n" in retained["with"]["path"]
    assert all(
        forbidden not in step.get("run", "")
        for step in steps
        for forbidden in ("ptrace_scope", "setenforce", "-attach-pid", "-no-follow-child")
    )


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
