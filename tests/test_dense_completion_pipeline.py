from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.dense_completion_pipeline import (
    PipelineStep,
    ProcessIdentity,
    _completion_input_binding,
    _exclusive_controller_lease,
    _matching_completed_prefix,
    _repository_contract_sources,
    _step_contract,
    _validate_training_inputs,
    _wait_for_training,
    parse_args,
    pipeline_steps,
    run_pipeline,
)

SCOPE = {
    "path": "configs/dense_scope_amendment.json",
    "sha256": "a" * 64,
    "status": "user_directed_post_hoc_scope_amendment",
    "amended_at_utc": "2026-08-31T00:00:00Z",
    "claim_boundary": "Dense only",
}


def _args():
    return Namespace(
        workdir=Path.cwd(),
        scope_amendment=Path("configs/dense_scope_amendment.json"),
        python="/usr/bin/python3",
        gpus="0,1,2,3,4,5,6,7",
        gpus_b="4,5,6,7",
        worker_retries=2,
        include_validation=False,
    )


def test_dense_pipeline_never_schedules_late_family():
    steps = pipeline_steps(_args())
    commands = [token for step in steps for token in step.command]

    assert "late" not in commands
    assert [step.name for step in steps[:7]] == [
        "hybrid-training-audit",
        "confirmatory-training-audit-seed-314159",
        "confirmatory-training-audit-seed-271828",
        "confirmatory-training-audit-seed-161803",
        "short-branch-training-audit-seed-314159",
        "short-branch-training-audit-seed-271828",
        "short-branch-training-audit-seed-161803",
    ]
    protocol_scoped_modules = {
        "embed_optim.temporal_short_branch",
        "embed_optim.dose_band_analysis",
    }
    contract_scoped_modules = {
        "embed_optim.dense_retrieval_dynamics_evaluation",
        "embed_optim.dense_retrieval_dynamics_summary",
    }
    assert all(
        (
            step.command[2] in protocol_scoped_modules
            and "configs/causal_chain_analysis.json" in step.command
        )
        or (
            step.command[2] in contract_scoped_modules
            and "configs/dense_retrieval_dynamics_extension.json" in step.command
        )
        or any(token.endswith("/configs/dense_scope_amendment.json") for token in step.command)
        for step in steps[7:]
    )


def test_extended_retrieval_dynamics_builds_and_audits_all_five_stages():
    steps = pipeline_steps(_args())
    by_name = {step.name: step for step in steps}
    names = [step.name for step in steps]

    expected = (
        "hybrid-adamw-dynamics-evaluation",
        "confirmatory-dynamics-evaluation",
        "dense-retrieval-dynamics-audit",
        "dense-retrieval-dynamics-summary-build",
        "dense-retrieval-dynamics-summary-audit",
    )
    assert all(name in by_name for name in expected)
    assert names.index(expected[0]) > names.index("hybrid-adamw-summary")
    assert names.index(expected[1]) > names.index("confirmatory-summary")
    assert [names.index(name) for name in expected] == sorted(
        names.index(name) for name in expected
    )
    assert names.index("dense-retrieval-dynamics-summary-audit") == (
        names.index("dense-retrieval-dynamics-summary-build") + 1
    )
    assert "--audit-only" in by_name["dense-retrieval-dynamics-audit"].command
    assert "--audit-only" in by_name["dense-retrieval-dynamics-summary-audit"].command
    for name in expected[:2]:
        command = by_name[name].command
        assert command[command.index("--gpus-a") + 1] == "0,1,2,3,4,5,6,7"
        assert command[command.index("--gpus-b") + 1] == "4,5,6,7"


def test_new_mechanism_analyses_build_then_audit_in_dependency_order():
    steps = pipeline_steps(_args())
    names = [step.name for step in steps]

    temporal = names.index("temporal-short-branch-analysis")
    predictors = names.index("temporal-short-branch-predictors")
    dose = names.index("dose-band-analysis")
    assert names[predictors - 1] == "short-branch-summary"
    assert names[predictors + 1] == "temporal-short-branch-predictors-audit"
    assert names[temporal - 1] == "tail-stability-summary"
    assert names[temporal + 1] == "temporal-short-branch-audit"
    assert names[dose - 1] == "spectral-transplant-summary"
    assert names[dose + 1] == "dose-band-audit"
    assert steps[temporal].command[2] == "embed_optim.temporal_short_branch"
    assert steps[temporal + 1].command[-1] == "--audit"
    assert (
        steps[temporal + 1]
        .command[steps[temporal + 1].command.index("--scope-amendment") + 1]
        .endswith("/configs/dense_scope_amendment.json")
    )
    assert steps[dose].command[2] == "embed_optim.dose_band_analysis"
    assert steps[dose + 1].command[-1] == "--audit"
    assert steps[predictors].command[2] == "embed_optim.temporal_short_branch_predictors"
    assert steps[predictors + 1].command[-1] == "--audit"
    for step in (steps[predictors], steps[predictors + 1]):
        assert step.command[step.command.index("--families") + 1] == "dense"
        assert step.command[step.command.index("--analysis-protocol") + 1] == (
            "configs/causal_chain_analysis.json"
        )
    assert steps[predictors].command[-2:] == ("--device", "cuda:0")
    assert "--device" not in steps[predictors + 1].command
    for step in (
        steps[temporal],
        steps[temporal + 1],
        steps[dose],
        steps[dose + 1],
    ):
        assert "--families" not in step.command
        assert step.command[step.command.index("--protocol") + 1] == (
            "configs/causal_chain_analysis.json"
        )


def test_temporal_predictor_uses_a_declared_pipeline_gpu():
    args = _args()
    args.gpus = "8,9,10,11,12,13,14,15"

    step = next(
        step for step in pipeline_steps(args) if step.name == "temporal-short-branch-predictors"
    )

    assert step.command[-2:] == ("--device", "cuda:8")


def test_mechanism_analysis_integration_contract_matches_pipeline():
    contract = json.loads(
        Path("configs/mechanism_analysis_integration.json").read_text(encoding="utf-8")
    )
    steps = {step.name: step for step in pipeline_steps(_args())}

    assert contract["families"] == ["dense"]
    assert contract["protocol"] == "configs/causal_chain_analysis.json"
    for analysis in contract["analyses"]:
        build = steps[f"{analysis['name']}-analysis"]
        audit = steps[f"{analysis['name']}-audit"]
        assert build.command[2] == analysis["module"]
        assert audit.command[2] == analysis["module"]
        assert audit.command[-1] == analysis["audit_flag"]
        prerequisite = analysis.get("prerequisite")
        if prerequisite:
            build = steps[prerequisite["name"]]
            audit = steps[f"{prerequisite['name']}-audit"]
            assert build.command[2] == prerequisite["module"]
            assert audit.command[-1] == prerequisite["audit_flag"]


def test_dense_pipeline_resume_prefix_requires_same_command():
    steps = [PipelineStep("one", ("python", "one")), PipelineStep("two", ("python", "two"))]
    previous = {
        "steps": [
            {"name": "one", "command": ["python", "one"], "complete": True},
            {"name": "two", "command": ["python", "changed"], "complete": True},
        ]
    }

    assert _matching_completed_prefix(previous, steps) == 1


def test_dense_pipeline_cli_rejects_partial_gpu_set():
    with pytest.raises(SystemExit):
        parse_args(["--gpus", "0,1,2,3"])
    with pytest.raises(SystemExit):
        parse_args(["--training-ledgers", "same.json", "same.json"])
    with pytest.raises(SystemExit):
        parse_args(["--wait-pids", "12", "12"])


def test_wait_for_training_tracks_process_identity(monkeypatch):
    args = _args()
    args.wait_pids = [123]
    args.wait_command_fragment = "embed_optim.family_training_queue"
    args.poll_seconds = 0.001
    command = f"{sys.executable} -m embed_optim.family_training_queue --pool a"
    identities = iter(
        [
            ProcessIdentity(123, 456, command),
            ProcessIdentity(123, 456, command),
            None,
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline._read_process_identity",
        lambda _pid: next(identities),
    )
    monkeypatch.setattr("embed_optim.dense_completion_pipeline.time.sleep", sleeps.append)

    _wait_for_training(args)

    assert sleeps == [args.poll_seconds]


def test_wait_for_training_rejects_wrong_or_reused_pid(monkeypatch):
    args = _args()
    args.wait_pids = [123]
    args.wait_command_fragment = "embed_optim.family_training_queue"
    args.poll_seconds = 0.001
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline._read_process_identity",
        lambda _pid: ProcessIdentity(123, 456, "unrelated-command"),
    )
    with pytest.raises(RuntimeError, match="not the requested Dense training queue"):
        _wait_for_training(args)

    command = f"{sys.executable} -m embed_optim.family_training_queue --pool a"
    identities = iter([ProcessIdentity(123, 456, command), ProcessIdentity(123, 789, command)])
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline._read_process_identity",
        lambda _pid: next(identities),
    )
    with pytest.raises(RuntimeError, match="was reused"):
        _wait_for_training(args)


def test_completion_controller_lease_rejects_a_second_instance(monkeypatch, tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    args = Namespace(
        **{
            **vars(_args()),
            "workdir": tmp_path,
            "scope_amendment": tmp_path / "scope.json",
            "log_dir": Path("logs/completion"),
        }
    )
    step = PipelineStep("one", (sys.executable, "-c", "print('one')"))
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline.resolve_scope",
        lambda *_args, **_kwargs: (("dense",), SCOPE),
    )
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline.pipeline_steps", lambda _args: [step]
    )
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline._wait_for_training",
        lambda _args: pytest.fail("the second controller must fail before waiting"),
    )
    lease_path = tmp_path / args.log_dir / "controller.lease"
    with _exclusive_controller_lease(
        lease_path,
        controller="completion",
        workdir=tmp_path,
        step_contract_sha256="0" * 64,
    ):
        with pytest.raises(RuntimeError, match="controller lease is already held"):
            run_pipeline(args)


def test_controller_child_inherits_lease_after_parent_copy_closes(tmp_path: Path):
    lease_path = tmp_path / "controller.lease"
    child = None
    try:
        with _exclusive_controller_lease(
            lease_path,
            controller="completion",
            workdir=tmp_path,
            step_contract_sha256="0" * 64,
        ) as lease_fd:
            child = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
                stdin=subprocess.PIPE,
                pass_fds=(lease_fd,),
            )

        assert child.poll() is None
        with pytest.raises(RuntimeError, match="controller lease is already held"):
            with _exclusive_controller_lease(
                lease_path,
                controller="completion",
                workdir=tmp_path,
                step_contract_sha256="1" * 64,
            ):
                pass
    finally:
        if child is not None and child.poll() is None:
            assert child.stdin is not None
            child.stdin.close()
            child.wait(timeout=5)

    with _exclusive_controller_lease(
        lease_path,
        controller="completion",
        workdir=tmp_path,
        step_contract_sha256="2" * 64,
    ):
        pass


def _write_training_inputs(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    scope_path = config_dir / "dense_scope_amendment.json"
    scope_path.write_text("{}\n", encoding="utf-8")
    scope = {**SCOPE, "sha256": hashlib.sha256(scope_path.read_bytes()).hexdigest()}
    plan = config_dir / "dense_training_queue.json"
    plan.write_text('{"schema_version":1}\n', encoding="utf-8")
    plan_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()
    pools = {}
    for pool in ("a", "b"):
        jobs = []
        for index in range(1, 10):
            matrix = config_dir / f"{pool}-{index}.yaml"
            output = tmp_path / "outputs" / f"{pool}-{index}"
            jobs.append(
                SimpleNamespace(
                    identity=f"confirmatory/seed-{index}/dense/{pool}-{index}",
                    matrix=matrix,
                    config=SimpleNamespace(output_dir=output),
                )
            )
        pools[pool] = jobs
    plan_payload = {
        "scope_amendment": {"path": "configs/dense_scope_amendment.json"},
    }
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline.load_queue_plan",
        lambda _path: (plan.resolve(), plan_payload, pools),
    )
    ledgers = []
    for pool in ("a", "b"):
        path = tmp_path / "logs" / f"queue-{pool}.json"
        path.parent.mkdir(exist_ok=True)
        payload = {
            "schema_version": 1,
            "complete": True,
            "plan": {"path": str(plan.resolve()), "sha256": plan_sha256},
            "pool": pool,
            "family": "dense",
            "jobs": [
                {
                    "index": index,
                    "identity": job.identity,
                    "matrix": str(job.matrix),
                    "output_dir": str(job.config.output_dir),
                    "complete": True,
                }
                for index, job in enumerate(pools[pool], start=1)
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        ledgers.append(path)
    return scope, plan, ledgers


def _run_args(tmp_path: Path, plan: Path, ledgers: list[Path], *, resume: bool = False):
    return Namespace(
        workdir=tmp_path,
        scope_amendment=tmp_path / "configs" / "dense_scope_amendment.json",
        training_plan=plan,
        training_ledgers=ledgers,
        log_dir=Path("logs/completion"),
        python=sys.executable,
        gpus="0,1,2,3,4,5,6,7",
        gpus_b="4,5,6,7",
        wait_pids=[],
        wait_command_fragment="embed_optim.family_training_queue",
        poll_seconds=0.001,
        worker_retries=0,
        step_retries=0,
        retry_delay=0.0,
        resume=resume,
        include_validation=False,
    )


def test_completion_rejects_source_mutation_while_waiting(monkeypatch, tmp_path: Path):
    scope, plan, ledgers = _write_training_inputs(tmp_path, monkeypatch)
    source = tmp_path / "src" / "embed_optim" / "marker.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    args = _run_args(tmp_path, plan, ledgers)
    step = PipelineStep("one", (sys.executable, "-c", "print('one')"))
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline.resolve_scope",
        lambda *_args, **_kwargs: (("dense",), scope),
    )
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline.pipeline_steps", lambda _args: [step]
    )
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline._wait_for_training",
        lambda _args: source.write_text("VALUE = 2\n", encoding="utf-8"),
    )
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline._validate_training_inputs",
        lambda **_kwargs: pytest.fail("mutated source must fail before ledger validation"),
    )

    with pytest.raises(RuntimeError, match="step contract changed"):
        run_pipeline(args)

    assert not (tmp_path / args.log_dir / "pipeline-ledger.json").exists()


def test_training_input_gate_requires_unique_complete_a_and_b_ledgers(monkeypatch, tmp_path: Path):
    scope, plan, ledgers = _write_training_inputs(tmp_path, monkeypatch)

    observed = _validate_training_inputs(
        workdir=tmp_path,
        scope=scope,
        training_plan=plan,
        training_ledgers=ledgers,
    )

    assert [record["pool"] for record in observed["training_ledgers"]] == ["a", "b"]
    assert observed["training_plan"]["sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match="exactly two unique"):
        _validate_training_inputs(
            workdir=tmp_path,
            scope=scope,
            training_plan=plan,
            training_ledgers=[ledgers[0], ledgers[0]],
        )

    payload = json.loads(ledgers[1].read_text(encoding="utf-8"))
    payload["pool"] = "a"
    ledgers[1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not finish cleanly"):
        _validate_training_inputs(
            workdir=tmp_path,
            scope=scope,
            training_plan=plan,
            training_ledgers=ledgers,
        )


def test_training_input_gate_rehashes_current_plan_and_scope(monkeypatch, tmp_path: Path):
    scope, plan, ledgers = _write_training_inputs(tmp_path, monkeypatch)
    scope_path = tmp_path / scope["path"]

    plan.write_text('{"schema_version":1,"changed":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not finish cleanly"):
        _validate_training_inputs(
            workdir=tmp_path,
            scope=scope,
            training_plan=plan,
            training_ledgers=ledgers,
        )

    plan_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()
    for ledger in ledgers:
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        payload["plan"]["sha256"] = plan_sha256
        ledger.write_text(json.dumps(payload), encoding="utf-8")
    scope_path.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="different scope amendment"):
        _validate_training_inputs(
            workdir=tmp_path,
            scope=scope,
            training_plan=plan,
            training_ledgers=ledgers,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(complete=False),
        lambda payload: payload.update(family="late"),
        lambda payload: payload["plan"].update(sha256="0" * 64),
        lambda payload: payload["jobs"][0].update(complete=False),
        lambda payload: payload.update(failed_job="some-job"),
    ],
)
def test_training_input_gate_rejects_incomplete_or_mismatched_ledger(
    monkeypatch, tmp_path: Path, mutation
):
    scope, plan, ledgers = _write_training_inputs(tmp_path, monkeypatch)
    payload = json.loads(ledgers[0].read_text(encoding="utf-8"))
    mutation(payload)
    ledgers[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="did not finish cleanly"):
        _validate_training_inputs(
            workdir=tmp_path,
            scope=scope,
            training_plan=plan,
            training_ledgers=ledgers,
        )


def test_resume_rechecks_complete_ledger_and_reruns_after_training_provenance_change(
    monkeypatch, tmp_path: Path
):
    scope, plan, ledgers = _write_training_inputs(tmp_path, monkeypatch)
    args = _run_args(tmp_path, plan, ledgers)
    step = PipelineStep("one", (sys.executable, "-c", "print('one')"))
    validations = 0
    original_validate = _validate_training_inputs

    def counted_validate(**kwargs):
        nonlocal validations
        validations += 1
        return original_validate(**kwargs)

    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline.resolve_scope",
        lambda *_args, **_kwargs: (("dense",), scope),
    )
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline.pipeline_steps", lambda _args: [step]
    )
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline._validate_training_inputs", counted_validate
    )

    assert run_pipeline(args) == 0
    ledger_path = tmp_path / args.log_dir / "pipeline-ledger.json"
    first = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert first["steps"][0]["input_binding"] == first["input_binding"]

    args.resume = True
    before_complete_resume = validations
    Path(first["steps"][0]["attempts"][0]["log"]["path"]).unlink()
    assert run_pipeline(args) == 0
    assert validations > before_complete_resume
    unchanged = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert unchanged["steps"][0]["finished_at"] != first["steps"][0]["finished_at"]

    pool_a = json.loads(ledgers[0].read_text(encoding="utf-8"))
    pool_a["finished_at"] = "changed-but-still-complete"
    ledgers[0].write_text(json.dumps(pool_a), encoding="utf-8")
    assert run_pipeline(args) == 0
    changed = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert changed["training_ledgers"] != first["training_ledgers"]
    assert changed["steps"][0]["input_binding"] == changed["input_binding"]
    assert changed["steps"][0]["finished_at"] != first["steps"][0]["finished_at"]


def test_resume_does_not_grandfather_legacy_unbound_step(monkeypatch, tmp_path: Path):
    scope, plan, ledgers = _write_training_inputs(tmp_path, monkeypatch)
    args = _run_args(tmp_path, plan, ledgers)
    step = PipelineStep("one", (sys.executable, "-c", "print('one')"))
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline.resolve_scope",
        lambda *_args, **_kwargs: (("dense",), scope),
    )
    monkeypatch.setattr(
        "embed_optim.dense_completion_pipeline.pipeline_steps", lambda _args: [step]
    )
    assert run_pipeline(args) == 0
    ledger_path = tmp_path / args.log_dir / "pipeline-ledger.json"
    legacy = json.loads(ledger_path.read_text(encoding="utf-8"))
    original_finished_at = legacy["steps"][0]["finished_at"]
    legacy["steps"][0].pop("input_binding")
    legacy.pop("input_binding")
    ledger_path.write_text(json.dumps(legacy), encoding="utf-8")

    args.resume = True
    assert run_pipeline(args) == 0
    upgraded = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert upgraded["steps"][0]["finished_at"] != original_finished_at
    assert upgraded["steps"][0]["input_binding"] == upgraded["input_binding"]


def test_completed_prefix_binds_scope_plan_ledgers_and_step_contract():
    steps = [PipelineStep("one", ("python", "one"))]
    training_inputs = {
        "training_plan": {"path": "/plan", "bytes": 1, "sha256": "1" * 64},
        "training_ledgers": [
            {"pool": "a", "path": "/a", "bytes": 1, "sha256": "2" * 64},
            {"pool": "b", "path": "/b", "bytes": 1, "sha256": "3" * 64},
        ],
    }
    binding = _completion_input_binding(
        scope=SCOPE,
        training_inputs=training_inputs,
        step_contract=_step_contract(steps),
        pipeline_arguments={"frozen": True},
    )
    previous = {
        "steps": [
            {
                "name": "one",
                "command": ["python", "one"],
                "input_binding": binding,
                "complete": True,
            }
        ]
    }

    assert _matching_completed_prefix(previous, steps, binding) == 1
    changed = {**binding, "step_contract_sha256": "0" * 64}
    assert _matching_completed_prefix(previous, steps, changed) == 0


def test_repository_contract_changes_with_downstream_module_source(tmp_path: Path):
    module = tmp_path / "src" / "embed_optim" / "downstream.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    steps = [PipelineStep("downstream", ("python", "-m", "embed_optim.downstream"))]
    before = _step_contract(
        steps,
        implementation_paths=_repository_contract_sources(tmp_path),
    )
    module.write_text("VALUE = 2\n", encoding="utf-8")
    after = _step_contract(
        steps,
        implementation_paths=_repository_contract_sources(tmp_path),
    )

    assert before["sha256"] != after["sha256"]


def test_repository_contract_binds_formal_reconstruction_locks(tmp_path: Path):
    module = tmp_path / "src" / "embed_optim" / "downstream.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    lock = tmp_path / "requirements-formal.lock"
    flash = tmp_path / "requirements-formal-flash.txt"
    constraints = tmp_path / "configs" / "formal_runtime_constraints.txt"
    constraints.parent.mkdir(parents=True)
    lock.write_text("base v1\n", encoding="utf-8")
    flash.write_text("flash v1\n", encoding="utf-8")
    constraints.write_text("torch==2.9.1+cu129\n", encoding="utf-8")
    steps = [PipelineStep("downstream", ("python", "-m", "embed_optim.downstream"))]
    before = _step_contract(steps, implementation_paths=_repository_contract_sources(tmp_path))

    lock.write_text("base v2\n", encoding="utf-8")
    after = _step_contract(steps, implementation_paths=_repository_contract_sources(tmp_path))

    assert before["sha256"] != after["sha256"]


def test_repository_contract_excludes_generated_paper_outputs_and_fetched_vendor(tmp_path: Path):
    module = tmp_path / "src" / "embed_optim" / "downstream.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    generated = tmp_path / "paper" / "generated" / "confirmation.tex"
    generated.parent.mkdir(parents=True)
    generated.write_text("old generated result\n", encoding="utf-8")
    results = tmp_path / "paper" / "results.tex"
    results.write_text("old result include\n", encoding="utf-8")

    before = _repository_contract_sources(tmp_path)
    generated.write_text("new generated result\n", encoding="utf-8")
    results.write_text("new result include\n", encoding="utf-8")
    vendor_style = tmp_path / "paper" / "vendor" / "acl.sty"
    vendor_bibliography = tmp_path / "paper" / "vendor" / "acl_natbib.bst"
    vendor_style.parent.mkdir(parents=True)
    vendor_style.write_text("fetched style\n", encoding="utf-8")
    vendor_bibliography.write_text("fetched bibliography style\n", encoding="utf-8")
    after = _repository_contract_sources(tmp_path)

    assert before == after
    assert generated.resolve() not in before
    assert results.resolve() not in before
    assert vendor_style.resolve() not in after
    assert vendor_bibliography.resolve() not in after
