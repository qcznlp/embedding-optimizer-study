"""Fresh CPU readback, changed-contract refusals and semantic checks on altered score copies."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from embed_optim.primary_contract import canonical, file_identity, read_json, verify_file
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_validation import ValidationContract
from embed_optim.primary_v3_validation_io import inspect_saved, read_record_file, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    root, work = args.repository.resolve(), args.workdir.resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not work.is_dir() or any(work.iterdir()):
        raise ValueError("Require a fresh CPU-only admission namespace")
    primary = PrimaryV3Contract.load(
        root / "configs/dense_primary_v3_protocol.json", root, args.training_root
    )
    protocol = root / "configs/dense_primary_v3_validation_protocol.json"
    if (
        file_identity(protocol)["sha256"]
        != "96a8aa2fc684d29ec67649c2e637c0cd4c7dcdd0f31f2e354b7d0955a18fb1c3"
    ):
        raise ValueError("Actual validation proposal differs")
    validated = ValidationContract.load(protocol, primary)
    producer_path = Path("/tmp/dense-v3-validation.Gf6KJ2/result.json")
    if (
        file_identity(producer_path)["sha256"]
        != "51936005e30ddd2519fba66e5d6847abaaee25cb0ad3d6eb0d4e2e6294b88200"
    ):
        raise ValueError("Actual score producer differs")
    producer = read_json(producer_path)
    for row in producer["artifacts"]:
        verify_file(row["path"], row)
    plan = read_json(producer["plan"]["path"])
    before = [
        *plan["source_bindings"],
        *plan["input_bindings"],
        *producer["artifacts"],
        {"path": str(producer_path), **file_identity(producer_path)},
    ]
    identities = plan["selection"]["row_identities"]
    positives = [
        inspect_saved(r["output"], r["plan"], identities) for r in producer["worker"]["records"]
    ]
    records = []

    def refused(name, action, artifacts):
        try:
            action()
        except (ValueError, KeyError) as error:
            records.append(
                {"case": name, "rejected": True, "error": str(error), "artifacts": artifacts}
            )
        else:
            raise AssertionError(f"Invalid validation case accepted: {name}")

    changes = [
        ("missing_source_closure", ("sources",), {}),
        ("old_primary", ("primary", "sha256"), "0" * 64),
        ("wrong_primary_path", ("primary", "path"), "configs/validation_probe.json"),
        ("short_data", ("dataset", "rows"), 4095),
        ("missing_files", ("dataset", "files"), []),
        ("other_temperature", ("settings", "temperature"), 0.2),
        ("in_batch_negatives", ("settings", "in_batch_negatives"), True),
        ("margin_tie", ("settings", "positive_margin_is_a_tie_breaker"), True),
        ("beir_selection", ("settings", "beir_is_a_selection_input"), True),
        ("relaxed_replay", ("settings", "reference_replay", "atol"), 1),
        (
            "rewrite_fp32_results",
            ("settings", "reference_replay", "changes_recorded_metrics"),
            True,
        ),
        ("scientific_completion", ("scientific_completion",), True),
    ]
    for name, keys, value in changes:
        modified = copy.deepcopy(validated.payload)
        node = modified
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
        path = work / f"{name}.json"
        write_new(path, modified)
        refused(
            name,
            lambda p=path: ValidationContract.load(p, primary),
            [{"path": str(path), **file_identity(path)}],
        )

    source = producer["worker"]["records"][0]
    for name in (
        "changed_loss",
        "changed_query_id",
        "reversed_rows",
        "duplicate_row",
        "dropped_row",
        "changed_summary",
    ):
        target = work / name
        shutil.copytree(source["output"], target)
        rows = read_record_file(target / "sample_scores.jsonl")
        if name == "changed_loss":
            rows[0]["metrics"]["contrastive_loss"] += 0.1
        elif name == "changed_query_id":
            rows[0]["row"]["query_id"] += 1
        elif name == "reversed_rows":
            rows.reverse()
        elif name == "duplicate_row":
            rows[-1] = copy.deepcopy(rows[0])
        elif name == "dropped_row":
            rows.pop()
        else:
            summary = read_json(target / "summary.json")
            summary["groups"][0]["contrastive_loss"] += 0.1
            (target / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
        if name != "changed_summary":
            (target / "sample_scores.jsonl").write_text(
                "".join(canonical(r).decode() + "\n" for r in rows)
            )
        # Deliberately refresh only the copied file checksums: semantic readers must
        # still reject these known altered diagnostic copies. Originals stay intact.
        manifest = read_json(target / "manifest.json")
        for key, file_name in (
            ("sample_scores", "sample_scores.jsonl"),
            ("summary", "summary.json"),
        ):
            manifest["outputs"][key] = {"path": file_name, **file_identity(target / file_name)}
        (target / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        artifacts = [{"path": str(p), **file_identity(p)} for p in sorted(target.iterdir())]
        refused(name, lambda t=target: inspect_saved(t, source["plan"], identities), artifacts)

    missing = work / "must-remain-absent"
    command = [
        sys.executable,
        "-B",
        "-m",
        "embed_optim.primary_v3_validation_io",
        "select",
        "--protocol",
        str(protocol),
        "--primary-protocol",
        str(primary.path),
        "--repository",
        str(root),
        "--training-root",
        str(args.training_root.resolve()),
        "--experiment-root",
        "/root/embedding-optimizer-study",
        "--data-root",
        "/tmp/dense-partition-candidate.kHXoGW/validation",
        "--output-root",
        str(missing),
    ]
    result = subprocess.run(
        command,
        cwd=root,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{root / 'src'}:{root}",
        },
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if (
        result.returncode == 0
        or "Require an ordinary retained run directory" not in result.stderr
        or missing.exists()
    ):
        raise ValueError("Actual incomplete primary selection did not fail before output")
    for row in before:
        verify_file(row["path"], row)
    value = {
        "scope": "engineering_dense_v3_validation_admission",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "fresh_cpu_diagnostic_readbacks": positives,
        "all_changed_cases_rejected": True,
        "changed_cases": records,
        "actual_missing_primary_selection": {
            "command": command,
            "returncode": result.returncode,
            "stderr": result.stderr,
        },
        "unchanged_inputs_and_producer_artifacts": before,
        "source": {"path": str(Path(__file__).resolve()), **file_identity(Path(__file__))},
        "model_updates": 0,
        "gpu_workers": 0,
        "scientific_completion": False,
        "primary_recipe_selection_produced": False,
        "boundary": "Three actual saved diagnostic score bundles are reread in a fresh CPU process. Twelve altered contracts and six checksum-consistent but semantically invalid score copies are refused. No formal recipe is selected; the actual CLI rejects missing v3 primary runs. All changed copies are explicitly invalid diagnostic fixtures, never source results.",
    }
    write_new(work / "result.json", value)
    print(
        {
            "fresh_cpu_readbacks": len(positives),
            "changed_cases_rejected": len(records),
            "missing_primary_selection_rejected": True,
            **file_identity(work / "result.json"),
        }
    )


if __name__ == "__main__":
    main()
