from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

from embed_optim.distribution_audit import audit_distribution


def _fixture_project(
    root: Path,
    *,
    include_wheel_data: bool = True,
    include_credential: bool = False,
    undeclared_runtime_config: bool = False,
    runtime_config_as_string_only: bool = False,
) -> None:
    (root / "src/embed_optim").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "dist").mkdir()
    package_source = "def main(): pass\n"
    if undeclared_runtime_config:
        (root / "configs").mkdir()
        (root / "configs/required.json").write_text("{}\n")
        package_source = (
            'DEFAULT_STRING = "configs/required.json"\n\ndef main(): pass\n'
            if runtime_config_as_string_only
            else (
                "from pathlib import Path\n\n"
                'DEFAULT_PATH = Path("configs/required.json")\n\n'
                "def main(): pass\n"
            )
        )
    (root / "src/embed_optim/__init__.py").write_text(package_source)
    (root / "docs/blog.md").write_text("# Result-safe blog\n")
    (root / "README.md").write_text("# Demo\n")
    (root / "LICENSE").write_text("license\n")
    (root / "pyproject.toml").write_text(
        """[project]
name = "demo-project"
version = "1.2.3"

[project.scripts]
demo-command = "embed_optim:main"

[tool.setuptools.data-files]
"share/demo" = [
  "docs/blog.md",
]
"""
    )

    prefix = "demo_project-1.2.3"
    wheel = root / "dist/demo_project-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("embed_optim/__init__.py", package_source)
        if include_wheel_data:
            archive.writestr(f"{prefix}.data/data/share/demo/blog.md", "# Result-safe blog\n")
        archive.writestr(f"{prefix}.dist-info/METADATA", "Name: demo-project\nVersion: 1.2.3\n")
        archive.writestr(f"{prefix}.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        archive.writestr(f"{prefix}.dist-info/RECORD", "")
        archive.writestr(
            f"{prefix}.dist-info/entry_points.txt",
            "[console_scripts]\ndemo-command = embed_optim:main\n",
        )
        archive.writestr(f"{prefix}.dist-info/licenses/LICENSE", "license\n")
        if include_credential:
            archive.writestr("unexpected/leak.txt", "wandb" + "_v1_" + "A" * 24)

    sources = {
        "README.md": "# Demo\n",
        "LICENSE": "license\n",
        "pyproject.toml": (root / "pyproject.toml").read_text(),
        "src/embed_optim/__init__.py": package_source,
        "docs/blog.md": "# Result-safe blog\n",
    }
    if include_credential:
        sources["unexpected/leak.txt"] = "github" + "_pat_" + "B" * 24
    with tarfile.open(root / "dist/demo_project-1.2.3.tar.gz", "w:gz") as archive:
        for relative, content in sources.items():
            payload = content.encode()
            member = tarfile.TarInfo(f"{prefix}/{relative}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_distribution_audit_reads_actual_wheel_and_sdist_contracts(tmp_path: Path):
    _fixture_project(tmp_path)

    result = audit_distribution(tmp_path)

    assert result["complete"] is True
    assert result["project"] == "demo-project"
    assert result["version"] == "1.2.3"
    assert result["declared_console_scripts"] == 1
    assert result["declared_data_files"] == 1
    assert result["runtime_config_references"] == 0
    assert result["package_modules"] == 1
    assert result["problems"] == []
    assert len(result["wheel"]["sha256"]) == 64
    assert len(result["sdist"]["sha256"]) == 64


def test_distribution_audit_rejects_declared_data_missing_from_wheel(tmp_path: Path):
    _fixture_project(tmp_path, include_wheel_data=False)

    result = audit_distribution(tmp_path)

    assert result["complete"] is False
    assert result["problems"] == ["wheel missing: demo_project-1.2.3.data/data/share/demo/blog.md"]


def test_distribution_audit_rejects_stale_archive_content(tmp_path: Path):
    _fixture_project(tmp_path)
    (tmp_path / "docs/blog.md").write_text("# Changed after build\n")

    result = audit_distribution(tmp_path)

    assert result["complete"] is False
    assert result["problems"] == [
        "wheel content mismatch: demo_project-1.2.3.data/data/share/demo/blog.md != docs/blog.md",
        "sdist content mismatch: docs/blog.md",
    ]


def test_distribution_audit_rejects_credentials_without_echoing_them(tmp_path: Path):
    _fixture_project(tmp_path, include_credential=True)

    result = audit_distribution(tmp_path)

    assert result["complete"] is False
    assert result["problems"] == [
        "wheel credential pattern: unexpected/leak.txt: Weights & Biases API token",
        "sdist credential pattern: demo_project-1.2.3/unexpected/leak.txt: "
        "GitHub fine-grained token",
    ]
    assert not any("A" * 24 in problem or "B" * 24 in problem for problem in result["problems"])


def test_distribution_audit_rejects_undeclared_runtime_config(tmp_path: Path):
    _fixture_project(tmp_path, undeclared_runtime_config=True)

    result = audit_distribution(tmp_path)

    assert result["complete"] is False
    assert result["runtime_config_references"] == 1
    assert result["problems"] == [
        "pyproject data-files missing runtime config: configs/required.json"
    ]


def test_distribution_audit_discovers_plain_string_runtime_config(tmp_path: Path):
    _fixture_project(
        tmp_path,
        undeclared_runtime_config=True,
        runtime_config_as_string_only=True,
    )

    result = audit_distribution(tmp_path)

    assert result["runtime_config_references"] == 1
    assert result["problems"] == [
        "pyproject data-files missing runtime config: configs/required.json"
    ]
