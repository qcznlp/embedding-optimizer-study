from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

MAIN_END_LABEL = "paper-main-end"
DEFAULT_MAX_MAIN_PAGE = 8


def _label_pages(aux_text: str, label: str) -> list[str]:
    pattern = re.compile(
        rf"^\\newlabel\{{{re.escape(label)}\}}\{{\{{[^{{}}]*\}}\{{([^{{}}]+)\}}",
        flags=re.MULTILINE,
    )
    return pattern.findall(aux_text)


def audit_paper_layout(
    paper_dir: str | Path = "paper",
    *,
    max_main_page: int = DEFAULT_MAX_MAIN_PAGE,
    label: str = MAIN_END_LABEL,
) -> dict[str, Any]:
    if isinstance(max_main_page, bool) or max_main_page < 1:
        raise ValueError("Maximum main-text page must be a positive integer")
    paper = Path(paper_dir).resolve()
    aux_path = paper / "build/main.aux"
    try:
        aux_text = aux_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Paper layout audit cannot read {aux_path}") from error
    pages = _label_pages(aux_text, label)
    if len(pages) != 1:
        raise ValueError(
            f"Paper layout audit requires exactly one {label!r} label in {aux_path}; "
            f"found {len(pages)}"
        )
    page_text = pages[0]
    if not page_text.isascii() or not page_text.isdecimal():
        raise ValueError(f"Paper main-text page is not an Arabic integer: {page_text!r}")
    page = int(page_text)
    if page < 1 or page > max_main_page:
        raise ValueError(
            f"Paper main text ends on page {page}, beyond the {max_main_page}-page limit"
        )
    return {
        "complete": True,
        "aux_path": str(aux_path),
        "label": label,
        "main_end_page": page,
        "max_main_page": max_main_page,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed unless the compiled paper main-text endpoint fits its page budget"
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--max-main-page", type=int, default=DEFAULT_MAX_MAIN_PAGE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            audit_paper_layout(args.paper_dir, max_main_page=args.max_main_page),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
