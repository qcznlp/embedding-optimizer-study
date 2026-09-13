"""Assemble the already-audited primary sources without deployment or execution.

Copies exact frozen bytes into a new ordinary directory, with a complete static
local import closure. Old source locks and all failed evidence stay unchanged.
The original primary protocol remains DRAFT here: no status-only authorization.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path


def identity(path):
    if path.is_symlink() or not path.is_file() or path.name == "gpu.py":
        raise ValueError("Require an ordinary, in-scope source file")
    with path.open("rb") as stream:
        return {
            "bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
        }


def prepare(repository, candidate, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or output.exists():
        raise ValueError("CPU-only source preparation requires a new output directory")
    protocol_path = repository / "configs/dense_primary_v3_protocol.json"
    if (
        identity(protocol_path)["sha256"]
        != "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
    ):
        raise ValueError("The declared primary protocol changed")
    protocol = json.loads(protocol_path.read_text())
    records = {}

    def add(name, origin, expected=None):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("Unsafe source path")
        source = origin / relative
        observed = identity(source)
        if expected is not None and observed != {k: expected[k] for k in ("bytes", "sha256")}:
            raise ValueError(f"Frozen source differs: {name}")
        if name in records:
            if records[name]["identity"] != observed:
                raise ValueError(f"Conflicting source roles: {name}")
            return
        records[name] = {"origin": str(source), "identity": observed}

    for key, root in (("training_sources", candidate), ("consumer_sources", repository)):
        for name, expected in protocol[key].items():
            add(name, root, expected)
    for key in ("data_amendment", "input_bindings", "natural_acceptance", "primary_parent"):
        row = protocol[key]
        add(row["path"], repository, row)
    add("configs/dense_primary_v3_protocol.json", repository)
    for name in (
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "configs/dense_no_packing_retrain.yaml",
        "tests/test_losses.py",
        "tests/test_optimizers.py",
        "tests/test_collators.py",
        "tests/test_runtime.py",
    ):
        add(name, candidate)

    # Include the real import dependencies, not only the nine numerical-identity
    # members. Bound v3 consumer files take precedence, never a foreign install.
    parsed = set()
    while True:
        pending = [
            name
            for name in records
            if name.startswith("src/embed_optim/") and name.endswith(".py") and name not in parsed
        ]
        if not pending:
            break
        for name in pending:
            parsed.add(name)
            tree = ast.parse(Path(records[name]["origin"]).read_text())
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    modules = (
                        [node.module.split(".")[0]] if node.module else [a.name for a in node.names]
                    )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("embed_optim.")
                ):
                    modules = [node.module.split(".")[1]]
                elif isinstance(node, ast.Import):
                    modules = [
                        a.name.split(".")[1]
                        for a in node.names
                        if a.name.startswith("embed_optim.")
                    ]
                for module in modules:
                    dependency = f"src/embed_optim/{module}.py"
                    if dependency in records:
                        continue
                    origin = repository if dependency in protocol["consumer_sources"] else candidate
                    if (origin / dependency).is_file():
                        add(dependency, origin)
                    elif node.module is not None:
                        raise ValueError(f"Missing actual local import dependency: {dependency}")
    output.mkdir(parents=True)
    for name, row in sorted(records.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["origin"], target)
        if identity(target) != row["identity"] or identity(Path(row["origin"])) != row["identity"]:
            raise ValueError("Source changed during exact assembly")
    receipt = {
        "scope": "primary-v3-exact-source-assembly",
        "formal_execution": False,
        "committed": False,
        "protocol": identity(protocol_path),
        "files": records,
        "excluded_work": ["new factorial Trainer components", "paper rendering expansion"],
        "boundary": "One real source root assembled from audited bytes; not a released contract, whole-repository green claim, or scientific result.",
    }
    with (output / "source-assembly.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "assembled": str(output),
                "files": len(records),
                "receipt": identity(output / "source-assembly.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.repository.resolve(), args.candidate.resolve(), args.output.resolve())
