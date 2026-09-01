from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from embed_optim.candidate_breadth_release import (
    RELEASE_STEP_NAMES,
    UPSTREAM_FINALIZATION_STEP_NAMES,
    PipelineStep,
    _canonical_hash,
    _declared_data_output,
    _read_finalization_ledger,
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


def _record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_upstream(repository: Path) -> Path:
    completion = repository / "logs" / "completion.json"
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_text('{"complete":true}\n', encoding="utf-8")
    source = repository / "historical-finalizer.py"
    source.write_text("FROZEN = True\n", encoding="utf-8")
    contract_steps = [
        {"index": index, "name": name, "command": ["python", name]}
        for index, name in enumerate(UPSTREAM_FINALIZATION_STEP_NAMES, start=1)
    ]
    contract_body = {
        "steps": contract_steps,
        "implementation_sources": [_record(source)],
    }
    contract = {
        "schema_version": 1,
        **contract_body,
        "sha256": _canonical_hash(contract_body),
    }
    completion_source = _record(completion)
    binding = {
        "scope_amendment": SCOPE,
        "completion_ledger": completion_source,
        "step_contract_sha256": contract["sha256"],
    }
    records = []
    for step in contract_steps:
        log = repository / "logs" / f"{step['index']:02d}-{step['name']}.log"
        log.write_text(f"complete {step['name']}\n", encoding="utf-8")
        records.append(
            {
                **step,
                "input_binding": binding,
                "attempts": [
                    {
                        "attempt": 1,
                        "started_at": "2026-08-31T00:00:00Z",
                        "finished_at": "2026-08-31T00:00:01Z",
                        "return_code": 0,
                        "log": _record(log),
                    }
                ],
                "complete": True,
                "finished_at": "2026-08-31T00:00:01Z",
            }
        )
    ledger = repository / "logs" / "finalization.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "started_at": "2026-08-31T00:00:00Z",
                "finished_at": "2026-08-31T00:01:00Z",
                "families": ["dense"],
                "scope_amendment": SCOPE,
                "completion_ledger": completion_source,
                "step_contract": contract,
                "input_binding": binding,
                "steps": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ledger


def _args(repository: Path) -> Namespace:
    return Namespace(
        workdir=repository,
        scope_amendment=Path("configs/dense_scope_amendment.json"),
        upstream_finalization_ledger=Path("logs/finalization.json"),
        protocol=Path("configs/candidate_breadth_probe.json"),
        data_output=Path("data/candidate-breadth"),
        summary_dir=Path("reports/candidate-breadth"),
        blog=Path("docs/blog.md"),
        paper=Path("paper/generated/candidate-breadth.tex"),
        publication_manifest=Path("reports/candidate-breadth/publication_manifest.json"),
        log_dir=Path("logs/candidate-release"),
        python="/usr/bin/python3",
        gpus="0,1,2,3,4,5,6,7",
        worker_retries=2,
        step_retries=0,
        retry_delay=0.0,
        resume=False,
    )


def test_release_steps_bind_candidate_outputs_and_dense_release_gates(tmp_path: Path) -> None:
    steps = pipeline_steps(_args(tmp_path))

    assert tuple(step.name for step in steps) == RELEASE_STEP_NAMES
    assert steps[0].command[-1] == "--resume"
    assert "--audit-only" in steps[1].command
    source_receipt = steps[1].command[steps[1].command.index("--receipt") + 1]
    assert source_receipt.endswith("/reports/candidate-breadth/data-audit.json")
    assert steps[2].command[steps[2].command.index("--source-audit-receipt") + 1] == source_receipt
    assert steps[2].command[steps[2].command.index("--gpus") + 1] == "0,1,2,3,4,5,6,7"
    assert steps[2].command[steps[2].command.index("--retries") + 1] == "2"
    assert steps[11].command == steps[10].command + ("--audit-only",)
    assert steps[17].command == steps[11].command
    assert steps[16].command == (
        "make",
        "-C",
        "paper",
        "release",
        "PYTHON=/usr/bin/python3",
    )
    assert steps[-2].command == ("uv", "build")
    assert steps[-1].command[2] == "embed_optim.distribution_audit"
    for step in (*steps[5:10], steps[12], steps[18]):
        assert step.command[step.command.index("--families") + 1] == "dense"
        assert step.command[step.command.index("--scope-amendment") + 1].endswith(
            "/configs/dense_scope_amendment.json"
        )


def test_upstream_finalization_ledger_hashes_all_logs_and_completion(tmp_path: Path) -> None:
    ledger = _write_upstream(tmp_path)

    payload, source = _read_finalization_ledger(
        ledger,
        expected_scope=SCOPE,
        repository=tmp_path,
    )

    assert payload["complete"] is True
    assert len(payload["steps"]) == 18
    assert source == _record(ledger)


def test_upstream_finalization_treats_implementation_sources_as_historical(
    tmp_path: Path,
) -> None:
    ledger = _write_upstream(tmp_path)
    historical_source = tmp_path / "historical-finalizer.py"
    frozen_sha256 = json.loads(ledger.read_text(encoding="utf-8"))["step_contract"][
        "implementation_sources"
    ][0]["sha256"]

    # The post-hoc publication checkout is expected to differ from the earlier
    # formal finalizer checkout.  Preserve and authenticate the recorded
    # contract, but do not reinterpret it as a hash of the current worktree.
    historical_source.write_text("FROZEN = False\n", encoding="utf-8")
    payload, _ = _read_finalization_ledger(
        ledger,
        expected_scope=SCOPE,
        repository=tmp_path,
    )

    assert payload["step_contract"]["implementation_sources"][0]["sha256"] == frozen_sha256
    assert _record(historical_source)["sha256"] != frozen_sha256


def test_upstream_finalization_accepts_a_logged_execution_error_before_success(
    tmp_path: Path,
) -> None:
    ledger = _write_upstream(tmp_path)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    record = payload["steps"][0]
    failed_log = tmp_path / "logs" / "01-start-failure.log"
    failed_log.write_text("pipeline execution error: unavailable\n", encoding="utf-8")
    record["attempts"].insert(
        0,
        {
            "attempt": 1,
            "started_at": "2026-08-31T00:00:00Z",
            "finished_at": "2026-08-31T00:00:01Z",
            "return_code": None,
            "execution_error": "OSError: unavailable",
            "log": _record(failed_log),
        },
    )
    record["attempts"][1]["attempt"] = 2
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    observed, _ = _read_finalization_ledger(
        ledger,
        expected_scope=SCOPE,
        repository=tmp_path,
    )

    assert observed["steps"][0]["attempts"][-1]["return_code"] == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload.update(complete=False), "not complete"),
        (lambda payload: payload.update(failed_step="tests"), "not complete"),
        (
            lambda payload: payload["steps"][0].update(name="different"),
            "step 1 is invalid",
        ),
        (
            lambda payload: payload["step_contract"].update(sha256="0" * 64),
            "contract hash differs",
        ),
        (
            lambda payload: payload["steps"][-1]["attempts"][-1].update(return_code=1),
            "step 18 is invalid",
        ),
        (lambda payload: payload.update(input_binding={}), "input binding differs"),
    ],
)
def test_upstream_finalization_ledger_rejects_structural_mutations(
    tmp_path: Path, mutation, match: str
) -> None:
    ledger = _write_upstream(tmp_path)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    mutation(payload)
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=match):
        _read_finalization_ledger(
            ledger,
            expected_scope=SCOPE,
            repository=tmp_path,
        )


def test_upstream_finalization_rejects_changed_log_or_completion(tmp_path: Path) -> None:
    ledger = _write_upstream(tmp_path)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    log = Path(payload["steps"][4]["attempts"][0]["log"]["path"])
    log.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="step 5 log source changed"):
        _read_finalization_ledger(
            ledger,
            expected_scope=SCOPE,
            repository=tmp_path,
        )

    ledger = _write_upstream(tmp_path)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    completion = Path(payload["completion_ledger"]["path"])
    completion.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completion ledger source changed"):
        _read_finalization_ledger(
            ledger,
            expected_scope=SCOPE,
            repository=tmp_path,
        )


def test_release_pipeline_writes_content_bound_ledger(monkeypatch, tmp_path: Path) -> None:
    protocol = tmp_path / "configs" / "candidate_breadth_probe.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(
        json.dumps({"evaluation": {"data_output": "data/candidate-breadth"}}) + "\n",
        encoding="utf-8",
    )
    marker = tmp_path / "src" / "marker.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("VALUE = 1\n", encoding="utf-8")
    upstream = tmp_path / "logs" / "finalization.json"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("{}\n", encoding="utf-8")
    upstream_source = _record(upstream)
    steps = [
        PipelineStep("one", (sys.executable, "-c", "print('one')")),
        PipelineStep("two", (sys.executable, "-c", "print('two')")),
    ]
    args = _args(tmp_path)
    monkeypatch.setattr(
        "embed_optim.candidate_breadth_release.resolve_scope",
        lambda *_args, **_kwargs: (("dense",), SCOPE),
    )
    monkeypatch.setattr("embed_optim.candidate_breadth_release.pipeline_steps", lambda _args: steps)
    monkeypatch.setattr(
        "embed_optim.candidate_breadth_release._repository_contract_sources",
        lambda _repository: (marker,),
    )
    monkeypatch.setattr(
        "embed_optim.candidate_breadth_release._read_finalization_ledger",
        lambda *_args, **_kwargs: ({"complete": True}, upstream_source),
    )

    assert run_pipeline(args) == 0

    ledger = json.loads(
        (tmp_path / args.log_dir / "pipeline-ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["complete"] is True
    assert ledger["upstream_finalization_ledger"] == upstream_source
    assert ledger["candidate_protocol"] == _record(protocol)
    assert [step["name"] for step in ledger["steps"]] == ["one", "two"]
    assert all(step["attempts"][-1]["return_code"] == 0 for step in ledger["steps"])


def test_cli_rejects_invalid_gpu_and_retry_values() -> None:
    args = parse_args([])
    assert args.gpus == "0,1,2,3,4,5,6,7"
    assert args.upstream_finalization_ledger == Path(
        "logs/dense-finalization-pipeline/pipeline-ledger.json"
    )
    with pytest.raises(SystemExit):
        parse_args(["--gpus", "0,0"])
    with pytest.raises(SystemExit):
        parse_args(["--gpus", "cpu"])
    with pytest.raises(SystemExit):
        parse_args(["--worker-retries", "-1"])
    with pytest.raises(SystemExit):
        parse_args(["--step-retries", "-1"])


def test_protocol_declares_the_only_accepted_candidate_data_output(tmp_path: Path) -> None:
    protocol = tmp_path / "configs" / "candidate_breadth_probe.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(
        json.dumps({"evaluation": {"data_output": "data/frozen-candidates"}}),
        encoding="utf-8",
    )

    assert _declared_data_output(protocol) == (tmp_path / "data/frozen-candidates").resolve()

    protocol.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Cannot resolve"):
        _declared_data_output(protocol)


def test_completion_gates_require_the_posthoc_release_after_canonical_finalization() -> None:
    gates = (Path(__file__).parents[1] / "docs/completion-gates.md").read_text(encoding="utf-8")

    assert "12 discovery final checkpoints × 6 nested widths = 72 run-width cells" in gates
    assert "all 21 release steps" in gates
    assert "logs/candidate-breadth-release/pipeline-ledger.json" in gates
    assert gates.index("embed-optim-dense-finalize") < gates.index(
        "embed-optim-candidate-breadth-release"
    )


def test_publication_keeps_candidate_breadth_claim_boundaries_visible() -> None:
    root = Path(__file__).parents[1]
    paper = (root / "paper/main.tex").read_text(encoding="utf-8")
    blog = (root / "docs/blog.md").read_text(encoding="utf-8")

    for document in (paper, blog):
        normalized = " ".join(document.split()).lower()
        assert "designed after" in normalized
        assert "unjudged relevant" in normalized
        assert "formal mediation" in normalized
    assert "Muon-family high-dose ordering reverses with candidate breadth" in paper
    assert "width-2,048 endpoint reversals" in paper
    assert "directly supports missing-candidate coverage as the mechanism" not in paper
    assert r"\CandidateBreadthDiscussion" in paper
    assert r"\CandidateBreadthConclusion" in paper
    assert r"\CandidateBreadthFigure" in paper
    assert "does not establish that contribution causally" in " ".join(blog.split())
