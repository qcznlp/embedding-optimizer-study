"""One exact, owner-directed BEIR resource handoff; not a general process killer.

Default is read-only. --pause suspends the three verified CPU dispatch processes,
archives their state, then terminates only their verified leaf evaluation workers.
No process-group signals, process enumeration, environment reads or source edits.
The original dispatch chain remains stopped with its lease and in-memory progress.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/root/embedding-optimizer-study")
SCRIPT = str(ROOT / "scripts/eval/dense_no_packing_parallel.py")
CHAIN = (
    (196647, 246327790, "embed_optim.corrected_completion_pipeline"),
    (870313, 257721545, "embed_optim.corrected_beir_evaluation"),
    (870864, 257733056, SCRIPT),
)
SOURCE_HASHES = {
    "src/embed_optim/corrected_completion_pipeline.py": "7f3e6f440d4b79b2a7aea3d5336b9458fbeb8cb61c5537fdda13fea72d591972",
    "src/embed_optim/corrected_beir_evaluation.py": "01c2f871b55b1629898c060f8024a6226d186aa9d270f7c764e5267461036b91",
    "scripts/eval/dense_no_packing_parallel.py": "89b650c1d10a41eb2999083604a1b1d0c996c45cd02c7dee0028e918f3d4752a",
    "scripts/eval/dense_parallel.py": "2b476585c17f636fbce604e5ff9ff051c69077d49340ead63916afab426a8c72",
    "scripts/eval/dense_sequential.py": "f92a3c9fbfcfed7b7f593dbbf5bdd85571f185087a1170fc3999fcd370b21906",
    "src/embed_optim/evaluation_utils.py": "529436e73759051b5c3000579fe9d66ce12c4444132eca4e7a26827054683e53",
}
LEDGER = ROOT / "logs/dense-no-packing-finalization/pipeline-ledger.json"
LEDGER_SHA = "95c69483f60badddd3c13e175951bcb52682523589c62bf898a57eb786d4e4ef"
MODEL_ROOT = ROOT / "outputs/dense-no-packing-v1/dense"
RESULT_ROOT = ROOT / "results/dense-no-packing-beir"
TASKS = {"ClimateFEVER", "FEVER"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_stat(raw):
    fields = raw[raw.rfind(")") + 2 :].split()
    return {"state": fields[0], "ppid": int(fields[1]), "start_ticks": int(fields[19])}


def stat(pid):
    return parse_stat((Path("/proc") / str(pid) / "stat").read_text())


def option(argv, name):
    if argv.count(name) != 1 or argv.index(name) + 1 >= len(argv):
        raise ValueError(f"Missing or duplicate approved argument {name}")
    return argv[argv.index(name) + 1]


def validate_worker_argv(argv):
    if SCRIPT not in argv or "--worker" not in argv:
        raise ValueError("Not the exact project worker")
    task = option(argv, "--worker")
    model = Path(option(argv, "--models"))
    result = Path(option(argv, "--results_folder"))
    if task not in TASKS or not model.is_relative_to(MODEL_ROOT):
        raise ValueError("Worker model/task outside the verified handoff")
    if len(model.relative_to(MODEL_ROOT).parts) != 2 or not model.name.startswith("checkpoint-"):
        raise ValueError("Unexpected checkpoint topology")
    if not result.is_relative_to(RESULT_ROOT / "dense") or ".." in result.parts:
        raise ValueError("Worker result root outside the project")
    if ".." in model.parts or not all(
        x in argv for x in ("--bf16", "--fa2", "--local", "--decontaminated")
    ):
        raise ValueError("Worker configuration mismatch")
    return {"task": task, "model": str(model), "results": str(result)}


def inspect(pid, start, expected, parent=None, worker=False):
    before = stat(pid)
    if before["start_ticks"] != start or (parent is not None and before["ppid"] != parent):
        raise ValueError(f"Process identity mismatch for exact handle {pid}")
    path = Path("/proc") / str(pid)
    argv = [x.decode() for x in (path / "cmdline").read_bytes().split(b"\0") if x]
    if expected not in argv:
        raise ValueError(f"Exact project entrypoint mismatch for {pid}")
    children = [int(x) for x in (path / "task" / str(pid) / "children").read_text().split()]
    record = {"pid": pid, **before, "entrypoint": expected, "children": children}
    if worker:
        record.update(validate_worker_argv(argv))
        if children:
            raise ValueError("Refusing a non-leaf worker")
        status = dict(
            line.split(":", 1) for line in (path / "status").read_text().splitlines() if ":" in line
        )
        term_bit = 1 << (signal.SIGTERM - 1)
        if any(int(status[k].strip(), 16) & term_bit for k in ("SigCgt", "SigIgn", "SigBlk")):
            raise ValueError("SIGTERM is not default and unblocked; no signal allowed")
        record["sigterm_default_unblocked"] = True
    if stat(pid)["start_ticks"] != start:
        raise ValueError("Process identity raced during verification")
    return record


def verify_sources():
    for name, expected in SOURCE_HASHES.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Bound source changed: {name}")
    if digest(LEDGER) != LEDGER_SHA:
        raise ValueError("The active ledger advanced; re-plan instead of guessing")
    ledger = json.loads(LEDGER.read_text())
    if ledger.get("complete") or ledger.get("active_step") != "decontaminated-beir":
        raise ValueError("Main is not in the reviewed evaluation state")
    if len(ledger["steps"]) != 5 or sum(x.get("complete") is True for x in ledger["steps"]) != 4:
        raise ValueError("Completed-step prefix changed")


def inspect_tree():
    verify_sources()
    chain = []
    for i, (pid, start, entrypoint) in enumerate(CHAIN):
        item = inspect(pid, start, entrypoint, CHAIN[i - 1][0] if i else 1)
        if i < 2 and item["children"] != [CHAIN[i + 1][0]]:
            raise ValueError("Unexpected direct controller child")
        chain.append(item)
    workers = []
    if len(chain[-1]["children"]) != 8:
        raise ValueError("Eight-worker topology changed; re-plan read-only")
    for pid in chain[-1]["children"]:
        first = stat(pid)
        if first["ppid"] != CHAIN[-1][0]:
            raise ValueError("Worker no longer belongs to the verified scheduler")
        workers.append(inspect(pid, first["start_ticks"], SCRIPT, CHAIN[-1][0], worker=True))
    return {"chain": chain, "workers": workers}


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def wait_state(item, accepted, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            state = stat(item["pid"])
        except FileNotFoundError:
            if "absent" in accepted:
                return "absent"
            raise
        if state["start_ticks"] != item["start_ticks"]:
            raise ValueError("Exact process no longer exists; never use its reused PID")
        if state["state"] in accepted:
            return state["state"]
        time.sleep(0.1)
    raise TimeoutError(f"Exact handle {item['pid']} did not reach {sorted(accepted)}")


def pause(output):
    if output.exists() or not output.is_absolute() or not output.parent.is_dir():
        raise ValueError("Use a new absolute output directory with an existing parent")
    plan = inspect_tree()
    output.mkdir()
    plan.update(
        observed_at_utc=datetime.now(UTC).isoformat(),
        scientific_completion=False,
        authority="Owner: 请你持续推进目标，优先保证任务的正确性，科学性; announced project-only GPU handoff.",
        source_hashes=SOURCE_HASHES,
        ledger_sha256=LEDGER_SHA,
    )
    write_new(output / "before.json", plan)
    fds = {}
    events = []
    try:
        for item in plan["chain"] + plan["workers"]:
            fds[item["pid"]] = os.pidfd_open(item["pid"])
            if stat(item["pid"])["start_ticks"] != item["start_ticks"]:
                raise ValueError("PID changed while opening race-safe handle")
        for item in plan["chain"] + plan["workers"]:
            signal.pidfd_send_signal(fds[item["pid"]], signal.SIGSTOP)
            events.append({"pid": item["pid"], "signal": "SIGSTOP"})
            wait_state(item, {"T"})
        # Everything that can dispatch/write results is now stopped. Revalidate ownership.
        frozen = inspect_tree()
        if [[x["pid"], x["start_ticks"]] for x in frozen["workers"]] != [
            [x["pid"], x["start_ticks"]] for x in plan["workers"]
        ]:
            raise ValueError("Worker set changed before scheduler suspension")
        files = [
            LEDGER,
            ROOT / "logs/dense-no-packing-finalization/controller.lease",
            ROOT / "logs/dense-no-packing-finalization/05-decontaminated-beir.attempt-1.log",
        ]
        files += sorted(RESULT_ROOT.rglob("*.json"))
        files += [ROOT / x for x in SOURCE_HASHES]
        inventory = []
        for source in files:
            if source.is_symlink() or not source.is_file():
                raise ValueError("Unexpected archive input")
            raw = source.read_bytes()
            relative = source.relative_to(ROOT)
            target = output / "preserved" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(raw)
            inventory.append(
                {
                    "path": str(relative),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        write_new(output / "preservation.json", inventory)
        for item in plan["workers"]:
            inspect(item["pid"], item["start_ticks"], SCRIPT, CHAIN[-1][0], worker=True)
            signal.pidfd_send_signal(fds[item["pid"]], signal.SIGTERM)
            signal.pidfd_send_signal(fds[item["pid"]], signal.SIGCONT)
            events.extend(
                [
                    {"pid": item["pid"], "signal": "SIGTERM"},
                    {"pid": item["pid"], "signal": "SIGCONT"},
                ]
            )
        terminal = [
            {"pid": x["pid"], "state": wait_state(x, {"Z", "absent"}, 20)} for x in plan["workers"]
        ]
        for item in plan["chain"]:
            wait_state(item, {"T"})
        for item in inventory:
            if digest(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("Preserved source/evidence changed during handoff")
        with (ROOT / "logs/dense-no-packing-finalization/controller.lease").open("r") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease_held = True
            else:
                raise ValueError("Original controller lease was unexpectedly released")
        result = {
            "observed_at_utc": datetime.now(UTC).isoformat(),
            "status": "paused_for_correctness_audit",
            "scientific_completion": False,
            "controller_chain_retained_stopped": True,
            "original_lease_held": lease_held,
            "worker_terminal_states": terminal,
            "preserved_file_count": len(inventory),
            "all_preserved_bytes_unchanged": True,
            "events": events,
            "production_source_modified": False,
            "checkpoint_modified": False,
            "process_group_signals_used": False,
        }
        write_new(output / "result.json", result)
        return result
    except BaseException as error:
        write_new(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "events": events,
                "warning": "No automatic continuation on failure; inspect exact handles before proceeding.",
            },
        )
        raise
    finally:
        for fd in fds.values():
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pause:
        if args.output is None:
            parser.error("--pause requires --output")
        result = pause(args.output)
    else:
        result = inspect_tree()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
