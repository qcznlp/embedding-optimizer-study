from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]
ACTIVE_PUBLICATION_ROOTS = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "pyproject.toml",
    ROOT / "docs",
    ROOT / "paper",
    ROOT / "src",
)


def test_active_publication_surface_is_paper_only() -> None:
    retired_token = "".join(("bl", "og"))
    retired_path = Path("docs") / f"{retired_token}.md"
    assert not (ROOT / retired_path).exists()

    for root in ACTIVE_PUBLICATION_ROOTS:
        paths = (root,) if root.is_file() else (path for path in root.rglob("*") if path.is_file())
        for path in paths:
            relative = path.relative_to(ROOT).as_posix().lower()
            if relative.startswith("paper/vendor/"):
                continue
            assert retired_token not in relative
            try:
                text = path.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                continue
            assert retired_token not in text, relative
