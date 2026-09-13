"""Version-isolated complete numerical reconstruction and reviewed-paper build.

Historical source contracts are immutable. Numerical consumers execute in their
authenticated original closure, in a fresh process; the reviewed document uses
the installed current component. Their eight shared inputs must agree exactly.
This command does not train, download, install a manuscript or publish a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import current_paper

BUNDLE_SHA256 = "746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7"
ENTRY_SHA256 = "1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566"
NUMERICAL_SCOPE = "actual-combined-dense-paper-numerical-portability-v1"
SCOPE = "dense-v3-version-isolated-paper-reproduction-v1"
SHARED_INPUTS = (
    "generated/optimizer-primary.tex",
    "generated/dimension-utilization.tex",
    "generated/recipe-sensitivity.tex",
    "generated/state-operator-factorial.tex",
    "results.tex",
    "figures/weight-to-retrieval-map.pdf",
    "figures/full-rate-retrieval-trajectories.pdf",
    "figures/optimizer-weight-dimension-map.pdf",
)
FACTORIAL_COUNTS = {
    "beir_seed_task_scores": 168,
    "factorial_cell_summary": 4,
    "estimand_seed_task_contrasts": 126,
    "estimand_summary": 3,
    "probe_checkpoint_metrics": 60,
    "probe_task_metrics": 840,
}
SOURCE_ROLES = {
    "current_training": {
        "config.py": "25d85021a0d78918374cb995016e80563c5ede9b8e87499809e336323105436b",
        "optimizers.py": "7d7d4f200c410cb1b24e1773ae72811582f43a0c794a5d1b2af269c123e69b1e",
    },
    "original_numerical_consumers": {
        "config.py": "8f9d0ab7251b4051c06a5b9a30bea5d8fa47b51513eae78bc205c953facab91a",
        "optimizers.py": "12158eee448b4a7cc4ae5d4dd9e2b3492f52e452b0cb7ed4b2a96e996e71d3a6",
    },
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ordinary(value: str | Path, *, directory: bool = False) -> Path:
    path = Path(value)
    need(path.is_absolute() and ".." not in path.parts, "Explicit ordinary absolute path required")
    need(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink path refused")
    need(path.is_dir() if directory else path.is_file(), "Required local input is missing")
    if not directory:
        need(stat.S_ISREG(path.stat().st_mode), "Regular file required")
    return path


def identity(path: Path) -> dict[str, Any]:
    ordinary(path)
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    need(
        all(
            getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "Input changed during authentication",
    )
    return {"bytes": after.st_size, "sha256": digest}


def read(path: Path) -> dict[str, Any]:
    before = identity(path)

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            need(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_: str) -> None:
        raise ValueError("Nonfinite JSON value")

    value = json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)
    need(identity(path) == before and isinstance(value, dict), "Invalid or changed JSON object")
    return value


def local_name(name: str) -> str:
    path = Path(name)
    need(
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == name,
        "Nonlocal manifest member",
    )
    return name


def verify_bundle(bundle: str | Path) -> tuple[Path, dict[str, Any]]:
    root = ordinary(bundle, directory=True)
    manifest_path = root / "manifest.json"
    need(identity(manifest_path)["sha256"] == BUNDLE_SHA256, "Not the accepted numerical closure")
    manifest = read(manifest_path)
    need(
        manifest["scope"] == NUMERICAL_SCOPE and len(manifest["files"]) == 189,
        "Wrong closure scope",
    )
    for name, expected in manifest["files"].items():
        actual = identity(root / local_name(name))
        need(actual == {key: expected[key] for key in actual}, "Changed numerical member: " + name)
    files = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "Symlink in numerical closure")
        if path.is_dir():
            continue
        ordinary(path)
        files.add(path.relative_to(root).as_posix())
    need(files == set(manifest["files"]) | {"manifest.json"}, "Extra or missing numerical member")
    need(identity(root / "replay_complete.py")["sha256"] == ENTRY_SHA256, "Changed original entry")
    for name, digest in SOURCE_ROLES["original_numerical_consumers"].items():
        need(
            identity(root / "primary/src/embed_optim" / name)["sha256"] == digest,
            "Wrong numerical role",
        )
    for name, digest in SOURCE_ROLES["current_training"].items():
        need(
            identity(Path(__file__).absolute().parent / name)["sha256"] == digest,
            "Wrong training role",
        )
    return root, manifest


def shared_inputs(numerical_paper: Path, reviewed_paper: Path) -> dict[str, Any]:
    result = {}
    for name in SHARED_INPUTS:
        expected = identity(reviewed_paper / name)
        need(
            identity(numerical_paper / name) == expected,
            "Numerical/reviewed input differs: " + name,
        )
        result[name] = expected
    return result


def write(path: Path, value: dict[str, Any]) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execute_numerical(bundle: Path, output: Path, log_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        str(bundle / "replay_complete.py"),
        "replay",
        "--bundle",
        str(bundle),
        "--manifest-sha256",
        BUNDLE_SHA256,
        "--output",
        str(output),
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    with log_path.open("xb") as log:
        child = subprocess.Popen(
            command,
            cwd=bundle / "primary",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=2400)
        except BaseException:
            # Only this call's newly created, CPU-only process group is owned.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            raise
    receipt = {"command": command, "cwd": str(bundle / "primary"), "exit_code": code}
    write(log_path.parent / "numerical-exit.json", receipt)
    need(code == 0, "Original numerical replay failed; outputs retained, no retry")
    return receipt


def validate_numerical(output: Path, bundle: Path) -> dict[str, Any]:
    value = read(output / "complete.json")
    need(
        value["scope"] == NUMERICAL_SCOPE and value["input_manifest_sha256"] == BUNDLE_SHA256,
        "Wrong numerical completion",
    )
    need(value["factorial_counts"] == FACTORIAL_COUNTS, "Incomplete factorial population")
    for key in (
        "complete_paper_source_and_extracted_text_exact",
        "all_original_scientific_rules_unchanged",
        "native_model_admission_is_upstream_provenance",
    ):
        need(value[key] is True, "Incomplete numerical gate: " + key)
    for relative, key in (
        ("primary/reconstructed/complete.json", "primary_completion"),
        ("document/document.json", "strict_document"),
        ("document/paper/build/main.pdf", "pdf"),
    ):
        need(identity(output / relative) == value[key], "Changed numerical result: " + key)
    for path in (output / "io-boundary.json", output / "primary/io-boundary.json"):
        io = read(path)
        need(
            io["failure"] is None
            and io["producer_reads_refused"] == []
            and io["network_refused"] == 0,
            "Numerical I/O boundary failed",
        )
    for module, binding in value["loaded_project_modules"].items():
        role = local_name(binding["role"])
        need(role.startswith("primary/src/embed_optim/"), "Foreign analytical module: " + module)
        need(
            identity(bundle / role) == {key: binding[key] for key in ("bytes", "sha256")},
            "Wrong analytical module identity: " + module,
        )
    for name in ("config", "optimizers"):
        binding = value["loaded_project_modules"]["embed_optim." + name]
        need(
            binding["sha256"] == SOURCE_ROLES["original_numerical_consumers"][name + ".py"],
            "Historical consumer was replaced",
        )
    return value


def reproduce(bundle: str | Path, paper_dir: str | Path, output: str | Path) -> dict[str, Any]:
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "Set CUDA_VISIBLE_DEVICES='' for CPU reproduction",
    )
    bundle, _ = verify_bundle(bundle)
    paper, snapshot_raw, _ = current_paper.read_snapshot(paper_dir)
    shared_inputs(bundle / "expected-paper", paper)
    root = Path(output)
    need(root.is_absolute() and ".." not in root.parts, "New absolute output required")
    ordinary(root.parent, directory=True)
    need(
        not root.exists() and not root.is_symlink(),
        "Preserve previous output; choose a new directory",
    )
    need(
        not root.is_relative_to(bundle) and not root.is_relative_to(paper),
        "Output inside input refused",
    )
    root.mkdir()
    try:
        numerical_exit = execute_numerical(bundle, root / "numerical", root / "numerical.log")
        native = validate_numerical(root / "numerical", bundle)
        joined = shared_inputs(root / "numerical/document/paper", paper)
        document = current_paper.build_current_paper(paper, root / "reviewed")
        need(
            shared_inputs(root / "reviewed/paper", paper) == joined,
            "Compiled numerical input changed",
        )
        verify_bundle(bundle)
        need(current_paper.read_snapshot(paper)[1] == snapshot_raw, "Reviewed source changed")
        result = {
            "scope": SCOPE,
            "complete": True,
            "numerical_and_reviewed_document_complete": True,
            "input_manifest_sha256": BUNDLE_SHA256,
            "numerical_entry_sha256": ENTRY_SHA256,
            "document_snapshot_sha256": current_paper.SNAPSHOT_SHA256,
            "source_roles": SOURCE_ROLES,
            "original_contracts_modified": False,
            "numerical_exit": numerical_exit,
            "numerical_completion": identity(root / "numerical/complete.json"),
            "numerical_loaded_modules": native["loaded_project_modules"],
            "shared_inputs": joined,
            "reviewed_document": document,
            "pdf": identity(root / "reviewed/paper/build/main.pdf"),
            "source": identity(Path(__file__).absolute()),
            "fresh_training_or_encoding": False,
            "physical_second_host": False,
            "gpu_resume_verified_here": False,
            "repository_publication_complete": False,
        }
        write(root / "complete.json", result)
        return result
    except BaseException as error:
        write(
            root / "failed.json",
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "outputs_preserved": True,
                "automatic_retry": False,
            },
        )
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerical-bundle", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = reproduce(args.numerical_bundle, args.paper_dir, args.output)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "pdf": result["reviewed_document"]["pdf"],
                "repository_publication_complete": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
