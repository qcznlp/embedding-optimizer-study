from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

MAIN_END_LABEL = "paper-main-end"
DEFAULT_MAX_MAIN_PAGE = 8

# Every manuscript float is classified by where it is allowed to land.  This is
# intentionally independent of the result values: the checked-in pending draft
# and the final renderer must expose the same float-label topology.
MAIN_TEXT_FLOAT_LABELS = (
    "tab:discovery-results",
    "fig:dense-discovery-dynamics",
    "tab:common-state-results",
    "tab:intervention-results",
    "tab:causal-chain-summary",
    "tab:confirmation-results",
)
APPENDIX_FLOAT_LABELS = (
    "tab:claim-firewall",
    "tab:training-systems-results",
    "tab:basis-sensitivity-results",
    "tab:representation-results",
    "tab:tail-identity-results",
    "tab:tail-persistence-results",
    "tab:spectral-factorial-results",
    "tab:spectral-tail-results",
    "tab:causal-temporal-diagnostics",
    "tab:causal-temporal-estimates",
    "tab:causal-temporal-pairs",
    "tab:causal-dose-diagnostics",
    "tab:causal-dose-anchors",
    "tab:causal-forward-rmse",
    "fig:extended-retrieval-dynamics",
    "tab:extended-retrieval-dynamics",
    "tab:denseon-per-task-results",
    "tab:task-delta-stability",
)
FLOAT_LABELS = MAIN_TEXT_FLOAT_LABELS + APPENDIX_FLOAT_LABELS


def _label_pages(aux_text: str, label: str) -> list[str]:
    pattern = re.compile(
        rf"^\\newlabel\{{{re.escape(label)}\}}\{{\{{[^{{}}]*\}}\{{([^{{}}]+)\}}",
        flags=re.MULTILINE,
    )
    return pattern.findall(aux_text)


def _page_for_unique_label(aux_text: str, label: str, aux_path: Path) -> int:
    pages = _label_pages(aux_text, label)
    if len(pages) != 1:
        raise ValueError(
            f"Paper layout audit requires exactly one {label!r} label in {aux_path}; "
            f"found {len(pages)}"
        )
    page_text = pages[0]
    if not page_text.isascii() or not page_text.isdecimal():
        raise ValueError(f"Paper page for {label!r} is not an Arabic integer: {page_text!r}")
    page = int(page_text)
    if page < 1:
        raise ValueError(f"Paper page for {label!r} must be positive: {page}")
    return page


def _observed_float_labels(aux_text: str) -> set[str]:
    return set(re.findall(r"^\\newlabel\{((?:tab|fig):[^{}]+)\}", aux_text, flags=re.MULTILINE))


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
    page = _page_for_unique_label(aux_text, label, aux_path)
    if page < 1 or page > max_main_page:
        raise ValueError(
            f"Paper main text ends on page {page}, beyond the {max_main_page}-page limit"
        )

    expected_float_labels = set(FLOAT_LABELS)
    observed_float_labels = _observed_float_labels(aux_text)
    if observed_float_labels != expected_float_labels:
        missing = sorted(expected_float_labels - observed_float_labels)
        unexpected = sorted(observed_float_labels - expected_float_labels)
        raise ValueError(
            "Paper float-label topology differs from the frozen main/appendix contract: "
            f"missing={missing}, unexpected={unexpected}"
        )

    main_float_pages = {
        float_label: _page_for_unique_label(aux_text, float_label, aux_path)
        for float_label in MAIN_TEXT_FLOAT_LABELS
    }
    appendix_float_pages = {
        float_label: _page_for_unique_label(aux_text, float_label, aux_path)
        for float_label in APPENDIX_FLOAT_LABELS
    }
    escaped_main = {
        float_label: float_page
        for float_label, float_page in main_float_pages.items()
        if float_page > page
    }
    if escaped_main:
        raise ValueError(
            "Main-text floats land after the audited main-text endpoint "
            f"on page {page}: {escaped_main}"
        )
    premature_appendix = {
        float_label: float_page
        for float_label, float_page in appendix_float_pages.items()
        if float_page <= page
    }
    if premature_appendix:
        raise ValueError(
            "Appendix floats do not land after the audited main-text endpoint: "
            f"{premature_appendix}"
        )
    return {
        "complete": True,
        "aux_path": str(aux_path),
        "label": label,
        "main_end_page": page,
        "max_main_page": max_main_page,
        "main_float_pages": main_float_pages,
        "appendix_float_pages": appendix_float_pages,
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
