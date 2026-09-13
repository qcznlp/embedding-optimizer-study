"""Generate isolated synthetic layout samples, never manuscript includes."""

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path

from embed_optim.primary_contract import file_identity, read_json, verify_file
from embed_optim.primary_v3_validation_io import write_new


def run(root, work):
    if (
        sys.flags.optimize
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or not work.is_dir()
        or any(work.iterdir())
    ):
        raise ValueError("Require CPU-only assertions and a new empty temporary directory")
    archive = root / "reports/engineering-archive/dense-v3-primary-renderer-v1"
    probe_path = archive / "probe-result.json"
    assert (
        file_identity(probe_path)["sha256"]
        == "82e19b038d89e9bf7d7c4f55d40d822d059051cb61ebc65a22b4044f8a9f835c"
    )
    probe = read_json(probe_path)
    for row in probe["sources"]:
        verify_file(row["path"], row)
    candidate = archive / "candidate/primary_v3_publication_render.py"
    sources = [
        {"path": str(p), **file_identity(p)}
        for p in (Path(__file__).absolute(), candidate, root / "paper/acl.sty", probe_path)
    ]
    renderer = runpy.run_path(str(candidate))["render_bridge"]
    shutil.copyfile(root / "paper/acl.sty", work / "acl.sty")
    records = []
    for case in probe["cases"]:
        kind = case["kind"]
        rendered = renderer(case["complete_kernel_tables"])
        prefix = (
            "\\documentclass[11pt]{article}\n\\usepackage[review]{acl}\n"
            "\\usepackage{times,latexsym}\n\\usepackage[T1]{fontenc}\n"
            "\\usepackage{booktabs}\n"
        )
        body = (
            "\\begin{document}\n\\section*{Synthetic boundary layout sample}\n"
            "This is an artificial " + kind.replace("_", " ") + " case, not a model result.\n"
            "\\CorrectedGeometryBridgeFinding\n\\CorrectedGeometryBridgeTable\n"
            "\\end{document}\n"
        )
        with (work / (kind + ".tex")).open("x") as stream:
            stream.write(prefix + rendered["latex"] + body)
        write_new(work / (kind + ".json"), rendered)
        records.append(
            {
                "kind": kind,
                "tex": file_identity(work / (kind + ".tex")),
                "render": file_identity(work / (kind + ".json")),
            }
        )
    for row in sources:
        verify_file(row["path"], row)
    write_new(
        work / "rendered.json",
        {
            "scope": "synthetic_primary_bridge_layout_samples",
            "sources": sources,
            "cases": records,
            "compile_verified": False,
            "scientific_completion": False,
            "manuscript_installed": False,
        },
    )
    print(
        {
            "samples_generated": 4,
            "scientific_completion": False,
            **file_identity(work / "rendered.json"),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    run(args.repository.resolve(), args.workdir.resolve())
