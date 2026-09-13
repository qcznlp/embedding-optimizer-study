"""Complete active-v3 float topology, including the required exact-sensitivity appendix."""

import argparse
import json
from pathlib import Path

from . import paper_layout as original

MAIN_TEXT_FLOAT_LABELS = original.MAIN_TEXT_FLOAT_LABELS
APPENDIX_FLOAT_LABELS = (*original.APPENDIX_FLOAT_LABELS, "tab:exact-geometry-sensitivity")
FLOAT_LABELS = MAIN_TEXT_FLOAT_LABELS + APPENDIX_FLOAT_LABELS


def audit_paper_layout(paper_dir):
    """Inspect every expected label without removing the new float from the old inventory."""
    paper = Path(paper_dir).absolute()
    if not paper.is_dir() or any(path.is_symlink() for path in (paper, *paper.parents)):
        raise ValueError("Require an ordinary active-v3 paper directory")
    aux = paper / "build/main.aux"
    if not aux.is_file() or aux.is_symlink() or aux.parent.is_symlink():
        raise ValueError("Require an ordinary compiled active-v3 auxiliary file")
    text = aux.read_text(encoding="utf-8")
    page = original._page_for_unique_label(text, original.MAIN_END_LABEL, aux)
    if page > original.DEFAULT_MAX_MAIN_PAGE:
        raise ValueError("Active-v3 main text exceeds the unchanged eight-page limit")
    observed = original._observed_float_labels(text)
    if observed != set(FLOAT_LABELS):
        raise ValueError(
            "Active-v3 float topology differs: "
            f"missing={sorted(set(FLOAT_LABELS) - observed)}, "
            f"unexpected={sorted(observed - set(FLOAT_LABELS))}"
        )
    main = {
        name: original._page_for_unique_label(text, name, aux) for name in MAIN_TEXT_FLOAT_LABELS
    }
    appendix = {
        name: original._page_for_unique_label(text, name, aux) for name in APPENDIX_FLOAT_LABELS
    }
    if any(value > page for value in main.values()):
        raise ValueError("An active-v3 main float lands after the main-text endpoint")
    if any(value <= page for value in appendix.values()):
        raise ValueError("An active-v3 appendix float lands in the main text")
    return {
        "scope": "dense_primary_v3_complete_float_topology",
        "complete": True,
        "aux_path": str(aux),
        "label": original.MAIN_END_LABEL,
        "main_end_page": page,
        "max_main_page": original.DEFAULT_MAX_MAIN_PAGE,
        "main_float_pages": main,
        "appendix_float_pages": appendix,
        "all_original_float_requirements_retained": True,
        "scientific_completion": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit_paper_layout(args.paper_dir), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
