"""Fresh CLI refusal controls for real missing primary data and an unreleased draft."""

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import file_identity
from embed_optim.primary_v3_validation_io import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repository",
        "training-root",
        "experiment-root",
        "reference",
        "probe-root",
        "protocol",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.repository.resolve(), args.output_root.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not output.is_dir() or any(output.iterdir()):
        raise ValueError("Require CPU-only visibility and an empty refusal-control directory")
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=f"{root / 'src'}:{root}",
        PYTHONDONTWRITEBYTECODE="1",
    )
    records = []
    for module, action in (
        ("primary_v3_dimension_exports", "plan"),
        ("primary_v3_dimension_exports", "inspect"),
        ("primary_v3_dimension_exports", "export"),
        ("primary_v3_dimensions", "compute"),
        ("primary_v3_dimensions", "inspect"),
    ):
        target = output / f"forbidden-{module}-{action}"
        command = [sys.executable, "-B", "-m", f"embed_optim.{module}", action]
        for name in (
            "repository",
            "training_root",
            "experiment_root",
            "reference",
            "probe_root",
            "protocol",
        ):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command.extend(["--output", str(target)])
        if module == "primary_v3_dimensions":
            command.extend(
                [
                    "--vectors",
                    str(output / "absent-primary-vectors"),
                    "--expected-vector-manifest-sha256",
                    "0" * 64,
                ]
            )
        elif action == "inspect":
            command.extend(["--expected-manifest-sha256", "0" * 64])
        child = subprocess.run(
            command, cwd=root, env=environment, capture_output=True, text=True, timeout=45
        )
        expected = "not execution authorized" if action == "export" else "ordinary retained run"
        if child.returncode != 1 or expected not in child.stderr or target.exists():
            raise ValueError(f"CLI did not refuse before output: {module}:{action}: {child.stderr}")
        records.append(
            {
                "module": module,
                "action": action,
                "command": command,
                "exit_code": child.returncode,
                "stdout": child.stdout,
                "stderr": child.stderr,
                "output_absent": True,
                "expected_boundary": expected,
            }
        )
    names = (
        "scripts/audit_dense_v3_dimension_cli.py",
        "src/embed_optim/primary_v3_dimension_exports.py",
        "src/embed_optim/primary_v3_dimensions.py",
        "src/embed_optim/primary_v3_dimension_contract.py",
    )
    result = {
        "scope": "engineering_real_primary_dimension_cli_refusals",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "sources": [{"path": name, **file_identity(root / name)} for name in names],
        "protocol": {"path": str(args.protocol), **file_identity(args.protocol)},
        "records": records,
        "scientific_completion": False,
    }
    write_new(output / "result.json", result)
    print({"passed": True, "cases": len(records), **file_identity(output / "result.json")})


if __name__ == "__main__":
    main()
