from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

from embed_optim.distribution_audit import (
    _candidate_breadth_distribution_problems,
    _transitive_config_references,
    audit_distribution,
)


def _fixture_project(
    root: Path,
    *,
    include_wheel_data: bool = True,
    include_credential: bool = False,
    undeclared_runtime_config: bool = False,
    runtime_config_as_string_only: bool = False,
    config_closure: str | None = None,
) -> None:
    (root / "src/embed_optim").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "dist").mkdir()
    package_source = "def main(): pass\n"
    if undeclared_runtime_config:
        (root / "configs").mkdir(exist_ok=True)
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
    declared_data = {"docs/blog.md": "share/demo/blog.md"}
    config_groups = ""
    if config_closure is not None:
        if config_closure not in {"entry-only", "complete"}:
            raise ValueError(config_closure)
        generated = root / "configs/generated"
        generated.mkdir(parents=True, exist_ok=True)
        (root / "configs/entry.json").write_text(
            """{
  "matrix": "configs/generated/seed.yaml",
  "source_bindings": {
    "matrix_manifest": {"path": "configs/generated/manifest.json"}
  }
}
"""
        )
        (generated / "seed.yaml").write_text("formal_runtime: ../formal_runtime.json\n")
        (generated / "manifest.json").write_text("{}\n")
        (root / "configs/formal_runtime.json").write_text("{}\n")
        declared_data["configs/entry.json"] = "share/demo/configs/entry.json"
        root_config_sources = ['  "configs/entry.json",']
        if config_closure == "complete":
            declared_data["configs/formal_runtime.json"] = "share/demo/configs/formal_runtime.json"
            declared_data["configs/generated/seed.yaml"] = "share/demo/configs/generated/seed.yaml"
            root_config_sources.append('  "configs/formal_runtime.json",')
            config_groups += (
                '\n"share/demo/configs/generated" = [\n  "configs/generated/seed.yaml",\n]\n'
            )
        root_config_group = "\n".join(root_config_sources)
        config_groups = f'\n"share/demo/configs" = [\n{root_config_group}\n]\n{config_groups}'
    pyproject_text = (
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
        + config_groups
    )
    (root / "pyproject.toml").write_text(pyproject_text)

    prefix = "demo_project-1.2.3"
    wheel = root / "dist/demo_project-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("embed_optim/__init__.py", package_source)
        if include_wheel_data:
            for source, destination in declared_data.items():
                archive.writestr(
                    f"{prefix}.data/data/{destination}",
                    (root / source).read_text(),
                )
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
        "pyproject.toml": pyproject_text,
        "src/embed_optim/__init__.py": package_source,
        **{source: (root / source).read_text() for source in declared_data},
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
    assert result["transitive_executable_config_references"] == 0
    assert result["repository_only_provenance_config_references"] == []
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


def test_completed_candidate_breadth_outputs_must_be_packaged(tmp_path: Path) -> None:
    report = tmp_path / "reports" / "candidate-breadth"
    report.mkdir(parents=True)
    expected = {
        "data-audit.json",
        "summary.json",
        "calibration_by_width.csv",
        "high_dose_contrasts.csv",
        "candidate_breadth_calibration.svg",
        "candidate_breadth_calibration.pdf",
        "publication_manifest.json",
    }
    for name in expected:
        (report / name).write_text(
            '{"status":"complete"}\n' if name == "summary.json" else "artifact\n",
            encoding="utf-8",
        )
    declared = {
        f"reports/candidate-breadth/{name}": Path("share/candidate-breadth") / name
        for name in expected
    }
    assert _candidate_breadth_distribution_problems(tmp_path, declared) == []

    declared.pop("reports/candidate-breadth/data-audit.json")
    assert _candidate_breadth_distribution_problems(tmp_path, declared) == [
        "completed candidate-breadth distribution missing: "
        "reports/candidate-breadth/data-audit.json"
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


def test_transitive_config_references_are_controlled_and_recursive(tmp_path: Path):
    generated = tmp_path / "configs/generated"
    generated.mkdir(parents=True)
    (tmp_path / "configs/entry.json").write_text(
        """{
  "matrix": "configs/generated/seed.yaml",
  "source_bindings": {
    "matrix_manifest": "configs/generated/manifest.json"
  }
}
"""
    )
    (generated / "seed.yaml").write_text("formal_runtime: ../missing-runtime.json\n")
    (generated / "manifest.json").write_text("{}\n")

    executable, provenance = _transitive_config_references(
        {"configs/entry.json"},
        tmp_path,
    )

    assert executable == {
        "configs/generated/seed.yaml",
        "configs/missing-runtime.json",
    }
    assert provenance == {"configs/generated/manifest.json"}


def test_transitive_config_references_include_runtime_reconstruction_inputs(tmp_path: Path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "formal_runtime.json").write_text(
        """{
  "reconstruction": {
    "constraints": {"path": "formal_runtime_constraints.txt"},
    "base_lock": {"path": "../requirements-formal.lock"},
    "flash_lock": {"path": "../requirements-formal-flash.txt"}
  }
}
"""
    )

    executable, provenance = _transitive_config_references(
        {"configs/formal_runtime.json"},
        tmp_path,
    )

    assert executable == {
        "configs/formal_runtime_constraints.txt",
        "requirements-formal-flash.txt",
        "requirements-formal.lock",
    }
    assert provenance == set()


def test_runtime_reconstruction_reference_cannot_escape_repository(tmp_path: Path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "formal_runtime.json").write_text(
        """{
  "reconstruction": {
    "base_lock": {"path": "../../outside.lock"}
  }
}
"""
    )

    executable, provenance = _transitive_config_references(
        {"configs/formal_runtime.json"},
        tmp_path,
    )

    assert executable == set()
    assert provenance == set()


def test_distribution_audit_rejects_missing_executable_config_closure(tmp_path: Path):
    _fixture_project(tmp_path, config_closure="entry-only")

    result = audit_distribution(tmp_path)

    assert result["complete"] is False
    assert result["transitive_executable_config_references"] == 2
    assert result["repository_only_provenance_config_references"] == [
        "configs/generated/manifest.json"
    ]
    assert result["problems"] == [
        "pyproject data-files missing executable config dependency: configs/formal_runtime.json",
        "pyproject data-files missing executable config dependency: configs/generated/seed.yaml",
    ]


def test_distribution_audit_accepts_complete_executable_config_closure(tmp_path: Path):
    _fixture_project(tmp_path, config_closure="complete")

    result = audit_distribution(tmp_path)

    assert result["complete"] is True
    assert result["declared_data_files"] == 4
    assert result["transitive_executable_config_references"] == 2
    assert result["repository_only_provenance_config_references"] == [
        "configs/generated/manifest.json"
    ]
    assert result["problems"] == []
