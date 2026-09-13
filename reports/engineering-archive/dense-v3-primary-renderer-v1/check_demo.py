"""Read-only layout/source/test verification with a separately written receipt."""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, require_same, verify_file
from embed_optim.primary_v3_validation_io import write_new


def run(root, work, output):
    if sys.flags.optimize or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or output.exists():
        raise ValueError("Require CPU-only assertions and a new bounded check receipt")
    archive = root / "reports/engineering-archive/dense-v3-primary-renderer-v1"
    record = read_json(work / "rendered.json")
    assert record["scientific_completion"] is False and record["manuscript_installed"] is False
    for row in record["sources"]:
        verify_file(row["path"], row)
    suite = list(ET.parse(archive / "focused-first.xml").iter("testsuite"))
    assert len(suite) == 1
    require_same(
        {k: int(suite[0].attrib[k]) for k in ("tests", "failures", "errors", "skipped")},
        {"tests": 14, "failures": 0, "errors": 0, "skipped": 0},
    )
    checked = []
    for case in record["cases"]:
        name = case["kind"]
        verify_file(work / (name + ".tex"), case["tex"])
        verify_file(work / (name + ".json"), case["render"])
        log = (work / (name + ".log")).read_text()
        assert "Output written on " + name + ".pdf" in log
        assert "Overfull" not in log and not any(line.startswith("!") for line in log.splitlines())
        pdf = work / (name + ".pdf")
        fonts = subprocess.run(
            ["pdffonts", str(pdf)], check=True, text=True, capture_output=True
        ).stdout
        info = subprocess.run(
            ["pdfinfo", str(pdf)], check=True, text=True, capture_output=True
        ).stdout
        assert "Type 3" not in fonts and "Type 1" in fonts
        pages = next(line for line in info.splitlines() if line.startswith("Pages:"))
        assert int(pages.split(":", 1)[1]) == 2
        checked.append(
            {
                "kind": name,
                "pages": 2,
                "type3_fonts": False,
                "overfull_boxes": False,
                "font_report": fonts,
                "pdf": {"path": str(pdf), **file_identity(pdf)},
            }
        )
    require_same([r["kind"] for r in checked], ["signal", "constant", "train_only", "near_null"])
    write_new(
        output,
        {
            "scope": "synthetic_original_bridge_rendering_component_check",
            "passed": True,
            "focused_tests": 14,
            "samples": checked,
            "candidate_integrated_into_package": False,
            "complete_primary_publication_verified": False,
            "complete_paper_layout_verified": False,
            "scientific_completion": False,
            "manuscript_installed": False,
            "artifacts": [
                {"path": str(p), **file_identity(p)} for p in sorted(work.rglob("*")) if p.is_file()
            ],
            "checker": {"path": str(Path(__file__).absolute()), **file_identity(__file__)},
        },
    )
    print(
        {"passed": True, "samples": 4, "complete_paper_verified": False, **file_identity(output)},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "workdir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.repository.resolve(), args.workdir.resolve(), args.output.absolute())
