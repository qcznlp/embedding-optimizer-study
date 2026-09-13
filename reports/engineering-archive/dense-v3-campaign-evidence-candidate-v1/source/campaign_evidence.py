"""Read immutable v3 training evidence from local roles, without model execution.

Isolated preparation only. This does not implement PrimaryV3Contract.complete_run,
grant source/publication admission, repeat tensor checks, or read producer paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

PROTOCOL = "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b"
CONTINUATION = "c1602e493b68fe52ce487389a5275452bd4c003706af3d66c61b7c6f0cd9d1cf"
OBSERVER = "57287dad0574787e3b868a187acc41db1af97b37172dc75698c9b0942b96d2da"
ANCHORS = {
    "admission.json": (19515868, "be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067"),
    "input-view-audit.json": (4463, "13cb27173e48b8262aae77cba6ed5ebb03d9839f10fec409af06beac570af473"),
    "source-assembly.json": (15970, "e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8"),
    "evaluation-authorization.json": (6588, "2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c"),
}
TERMINAL = {
    "verified-v3-normuon-1e-4": {
        "pid": 1959407, "start_ticks": 295778135,
        "started": "98ff6ebf287b6bb177b4949a0a2cf58f4c80dc990c0528719b2cef1b4ecf6ca4",
        "terminated": "1d9dda131c2e1e150bfc98f45786f034c2acd57a5c7755f1cb63d9673dc113c2",
    },
    "verified-v3-normuon-3e-3": {
        "pid": 1959573, "start_ticks": 295779725,
        "started": "8eec4b13a049ce62b8cc66ea520ab93e6fc906e570dfbd550277dd7b1e77ea4a",
        "terminated": "758e999d2fc495247a4a02bd17b4d81d208c9e85ba60f22443ed90015d58d246",
    },
}
RATES = {"adamw": ("1e-6", "3e-6", "1e-5", "3e-5"),
         "muon": ("1e-4", "3e-4", "1e-3", "3e-3"),
         "normuon": ("1e-4", "3e-4", "1e-3", "3e-3")}
RUNS = tuple(f"verified-v3-{op}-{rate}" for op, rates in RATES.items() for rate in rates)
STEPS = [782, 1563, 2345, 3126, 3907]
METADATA = {"accepted_timing.json", "checkpoint_schedule.json", "completed.json",
            "dense_run_contract.json", "run_config.json", "trainer_state_final.json"}
SEAL = "dense_checkpoint_seal.json"
INFERENCE = {"model.safetensors", "config.json", "config_sentence_transformers.json",
             "modules.json", "sentence_bert_config.json", "tokenizer.json",
             "tokenizer_config.json", "1_Pooling/config.json"}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def same(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def relative(name):
    require(isinstance(name, str) and bool(name) and "\\" not in name and "\0" not in name,
            "Invalid local role name")
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name and ".." not in p.parts,
            "Noncanonical local role name")
    return name


def ordinary_root(root):
    root = Path(root)
    require(root.is_absolute() and ".." not in root.parts, "Use an explicit absolute local root")
    require(all(not p.is_symlink() for p in (root, *root.parents)), "Symlinked root refused")
    require(root.is_dir(), "Missing local evidence root")
    return root


def fingerprint(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def inventory(root):
    """Include directories as well as files, so empty extras are not invisible."""
    root = ordinary_root(root)
    result = {"": fingerprint(root.lstat())}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in [*dirs, *files]:
            p = Path(directory) / name
            st = p.lstat()
            require(stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode),
                    "Symlink or special evidence entry refused")
            result[p.relative_to(root).as_posix()] = fingerprint(st)
    return result


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read_local(root, name, sha, size=None):
    """Open every role component relative to pinned directory FDs; never provenance paths."""
    root = ordinary_root(root)
    parts = PurePosixPath(relative(name)).parts
    parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            first = os.fstat(stream.fileno())
            require(stat.S_ISREG(first.st_mode), "Evidence must be an ordinary file")
            raw = stream.read()
            same(fingerprint(os.fstat(stream.fileno())), fingerprint(first), "File changed during read")
        same(fingerprint(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)),
             fingerprint(first), "File replaced during read")
    finally:
        os.close(parent)
    require(hashlib.sha256(raw).hexdigest() == sha and (size is None or len(raw) == size),
            f"Evidence byte binding differs: {name}")
    value = json.loads(raw, object_pairs_hook=unique)
    canonical(value)
    return value


def file_binding(value):
    require(type(value.get("bytes")) is int and value["bytes"] >= 0
            and isinstance(value.get("sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", value["sha256"]) is not None,
            "Malformed original file identity")
    return {k: value[k] for k in ("bytes", "sha256")}


def _validate_anchors(anchors):
    """Semantic checks supplement, never replace, the public reader's fixed byte anchors."""
    parent, view, assembly, auth = (anchors[n] for n in ANCHORS)
    same(parent["primary_protocol_sha256"], PROTOCOL, "Wrong admission protocol")
    require(parent["scientific_completion"] is False and parent["committed_source_release"] is False,
            "Admission cannot acquire release/scientific authority")
    same(digest(parent["admitted"]), parent["admitted_sha256"], "Admitted population digest differs")
    complete = parent["actual_completion"]
    require(complete["original_single_selection_guard_passed"] is False
            and complete["committed_source_release"] is False and complete["scientific_completion"] is False,
            "Historical completion flags changed")
    same(complete["primary_protocol_sha256"], PROTOCOL, "Wrong completion protocol")
    same([r["run_id"] for r in complete["rows"]], list(RUNS), "Wrong ordered twelve-run population")
    same(sorted(parent["admitted"]["complete_runs"]), sorted(RUNS), "Wrong admitted runs")
    same(view["scope"], "actual_full_dataset_idempotent_selection_audit", "Wrong view audit scope")
    same(view["source_sha256"], CONTINUATION, "Wrong view audit source")
    require(view["scientific_completion"] is False, "View audit is not scientific completion")
    semantic = view["audit"]["semantic_view"]
    same(semantic["fingerprints"], ["0c6bd82f699a563c", "5a2cdf9a1ae149bc", "0c29f5d460d1b4d7"],
         "Do not rewrite distinct data-history fingerprints")
    require(semantic["rows"] == 500000 and semantic["indices_none"] is True
            and semantic["features_and_formats_equal"] is True
            and semantic["entire_arrow_tables_equal_including_metadata"] is True,
            "Missing full original view-equivalence proof")
    require(assembly["committed"] is False and assembly["formal_execution"] is False,
            "Source assembly flags changed")
    same(assembly["protocol"]["sha256"], PROTOCOL, "Wrong assembled protocol")
    require(len(assembly["files"]) == 56, "Incomplete recorded assembly")
    same(auth["protocol_sha256"], PROTOCOL, "Wrong original evaluation protocol")
    require(auth["committed_source_release"] is False and auth["native_draft_guard_modified"] is False,
            "Historical evaluation authority changed")
    same(sorted(auth["original_completed_runs"]), sorted(set(RUNS) - set(TERMINAL)),
         "Wrong ten exit-observed runs")
    return parent


def _validate_run(row, anchors, records, native):
    run, proof, expected, binding = (row[k] for k in ("run_id", "proof", "expected", "binding"))
    parent, view, assembly, auth = (anchors[n] for n in ANCHORS)
    same(proof, parent["admitted"]["complete_runs"][run], "Copied proof differs")
    same(records["proof"], proof, "Original proof bytes do not match admission")
    same(proof["run_id"], run, "Wrong proof run")
    same(proof["protocol_sha256"], PROTOCOL, "Wrong proof protocol")
    same(proof["run_identity_sha256"], digest(expected), "Wrong full recipe identity")
    for key, wanted in {"whole_run_artifacts_verified": True, "explicit_two_selection_view_audit_passed": True,
                        "original_single_selection_guard_passed": False, "committed_source_release": False,
                        "scientific_completion": False}.items():
        same(proof[key], wanted, "Historical proof flag changed")
    same(proof["steps"], STEPS, "Incomplete full horizon")
    same(proof["dataset_fingerprint"], view["audit"]["semantic_view"]["fingerprints"][-1],
         "Wrong actual data-history fingerprint")
    same(proof["input_view_audit_sha256"], ANCHORS["input-view-audit.json"][1], "Wrong view audit binding")
    same(expected["data"]["selected_columns"], view["audit"]["semantic_view"]["columns"], "Wrong data columns")
    same(expected["data"]["rows"], 500000, "Wrong training rows")
    for f in expected["source"]["local_files"]:
        same(file_binding(f), assembly["files"]["src/embed_optim/" + relative(f["path"])]["identity"],
             "Recipe source differs from recorded assembly")
    same(records["started"]["run_id"], run, "Wrong original start")
    if run not in TERMINAL:
        same(proof["scope"], "dense_primary_v3_explicit_view_history_completion", "Wrong exit-observed proof type")
        same(proof["source_sha256"], CONTINUATION, "Wrong original proof source")
        same(records["exited"]["run_id"], run, "Wrong original exit")
        same(records["exited"]["exit_code"], 0, "Original worker did not exit zero")
        require(binding["exit_code_observed"] is True and type(binding["exit_code"]) is int
                and binding["exit_code"] == 0, "Observed exit metadata changed")
    else:
        wanted = TERMINAL[run]
        same(proof["scope"], "dense_primary_v3_orphan_worker_artifact_completion", "Wrong termination proof type")
        same(proof["observer_source_sha256"], OBSERVER, "Wrong original observer")
        same(proof["original_view_reader_sha256"], CONTINUATION, "Wrong original view reader")
        require(proof["original_supervisor_exit_receipt_present"] is False and proof["cuda_initialized"] is False,
                "Fabricated original-supervisor evidence")
        require(binding["exit_code_observed"] is False and binding["exit_code"] is None,
                "Do not infer an unobserved OS exit code")
        observed = records["terminated"]
        same(observed, proof["process_observation"], "Original termination witness differs")
        for key, value in {"run_id": run, "pid": wanted["pid"], "start_ticks": wanted["start_ticks"],
                           "process_termination_observed_via_pidfd": True,
                           "os_exit_code_observed": False, "training_exit_code": None}.items():
            same(observed[key], value, "Wrong original terminal observation")
        same(records["started"]["torchrun_pid"], wanted["pid"], "Wrong original worker PID")
        same(observed["initial_live_witness"]["started_receipt_sha256"], wanted["started"], "Wrong initial start witness")
    require(set(proof["metadata"]) == METADATA, "Missing original metadata")
    same(native["dense_run_contract.json"], expected, "Recorded run metadata differs")
    same(native["checkpoint_schedule.json"], {"max_steps": 3907, "fractions": [.2, .4, .6, .8, 1.0], "steps": STEPS},
         "Wrong original checkpoint schedule")
    for key in ("run_identity_sha256", "system_metrics", "accepted_timing"):
        same(native["completed.json"][key], proof[key], "Completion metadata differs")
    require(native["trainer_state_final.json"]["global_step"] == 3907
            and native["trainer_state_final.json"]["max_steps"] == 3907
            and native["trainer_state_final.json"]["epoch"] == 1.0, "Incomplete recorded trainer horizon")
    same([p["step"] for p in proof["checkpoints"]], STEPS, "Missing checkpoint population")
    same([p["scheduler_step"] for p in proof["deep_checkpoint_checks"]], STEPS, "Missing original deep checks")
    require(all(p["parameter_states"] == 134 for p in proof["deep_checkpoint_checks"]), "Incomplete original tensor-state proof")
    required = INFERENCE | {"dense_run_contract.json", "dense_numerical_contract.json", "optimizer.pt", "scheduler.pt",
                            "trainer_state.json", "training_args.bin", *(f"rng_state_{i}.pth" for i in range(4))}
    for checkpoint in proof["checkpoints"]:
        step = checkpoint["step"]
        same(checkpoint["run_identity_sha256"], digest(expected), "Wrong checkpoint recipe")
        names = [relative(f["path"]) for f in checkpoint["files"]]
        require(names[-1] == SEAL and names[:-1] == sorted(set(names[:-1]))
                and SEAL not in names[:-1] and required.issubset(names), "Incomplete sealed inventory")
        for f in checkpoint["files"]:
            file_binding(f)
        same(file_binding(checkpoint["files"][-1]), checkpoint["checkpoint_seal"], "Seal inventory differs")
        state = native[f"checkpoint-{step}/trainer_state.json"]
        same(state["global_step"], step, "Wrong saved trainer step")
        for key, wanted in {
            "checkpoint_bytes": sum(f["bytes"] for f in checkpoint["files"]),
            "optimizer_state_bytes": next(f["bytes"] for f in checkpoint["files"] if f["path"] == "optimizer.pt")
        }.items():
            same(proof["system_metrics"][key][f"checkpoint-{step}"], wanted, "Recorded inventory byte total differs")


def _roles(anchors):
    """Return only canonical local-role names; stored absolute paths are never opened."""
    parent = _validate_anchors(anchors)
    auth = anchors["evaluation-authorization.json"]
    roles = {}
    for row in parent["actual_completion"]["rows"]:
        run, proof, binding = row["run_id"], row["proof"], row["binding"]
        roles[f"receipts/{run}.proof.json"] = file_binding(binding)
        if run in TERMINAL:
            roles[f"receipts/{run}.started.json"] = {"sha256": TERMINAL[run]["started"]}
            roles[f"receipts/{run}.terminated.json"] = {"sha256": TERMINAL[run]["terminated"]}
        else:
            original = auth["original_completed_runs"][run]
            same(binding["sha256"], original["view-verified"], "Wrong original proof hash")
            for kind in ("started", "exited"):
                roles[f"receipts/{run}.{kind}.json"] = {"sha256": original[kind]}
        for name, identity in proof["metadata"].items():
            roles[f"native/{run}/{relative(name)}"] = file_binding(identity)
        for checkpoint in proof["checkpoints"]:
            files = [f for f in checkpoint["files"] if f["path"] == "trainer_state.json"]
            require(len(files) == 1, "Missing or duplicate saved trainer state")
            roles[f"native/{run}/checkpoint-{checkpoint['step']}/trainer_state.json"] = file_binding(files[0])
    require(len(roles) == 168, "Incomplete local evidence roles")
    return roles


def read_complete_training_population(root):
    """Authenticate prior completion proofs. Never claim fresh model/data verification."""
    before = inventory(root)
    anchors = {name: read_local(root, f"anchors/{name}", sha, size) for name, (size, sha) in ANCHORS.items()}
    roles = _roles(anchors)
    wanted = set(roles) | {f"anchors/{name}" for name in ANCHORS}
    directories = {""} | {p.as_posix() for name in wanted for p in PurePosixPath(name).parents if p.as_posix() != "."}
    same(sorted(before), sorted(wanted | directories), "Missing or extra evidence path")
    payloads = {name: read_local(root, name, b["sha256"], b.get("bytes")) for name, b in roles.items()}
    parent = anchors["admission.json"]
    for row in parent["actual_completion"]["rows"]:
        run = row["run_id"]
        records = {kind: payloads[f"receipts/{run}.{kind}.json"] for kind in
                   ("proof", "started", "terminated" if run in TERMINAL else "exited")}
        native = {name.removeprefix(f"native/{run}/"): value for name, value in payloads.items()
                  if name.startswith(f"native/{run}/")}
        _validate_run(row, anchors, records, native)
    same(inventory(root), before, "Evidence tree changed during read")
    return {
        "scope": "primary-v3-local-role-training-evidence-v1",
        "original_evidence_authenticated": True, "runs": 12, "checkpoints": 60,
        "files_authenticated": len(wanted), "native_metadata_files": 132,
        "original_exit_zero_runs": [run for run in RUNS if run not in TERMINAL],
        "original_exit_unobserved_runs": [run for run in RUNS if run in TERMINAL],
        "primary_protocol_sha256": PROTOCOL,
        "original_completion_rows": parent["actual_completion"]["rows"],
        "scientific_completion": False, "committed_source_release": False,
        "fresh_model_or_data_validation": False, "formal_primary_contract_admission": False,
        "original_single_selection_guard_passed": False,
        "original_absolute_paths_opened": False,
        "boundary": "Previously accepted evidence replay only. No training/evaluation, tensor checks, functional recovery, scientific admission or publication. Original flags, fingerprints and unknown exit codes remain unchanged.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = read_complete_training_population(args.root)
    # Full original rows remain available to an explicit future versioned consumer.
    print(json.dumps({k: v for k, v in result.items() if k != "original_completion_rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
