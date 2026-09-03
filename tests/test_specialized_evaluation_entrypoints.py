from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim import confirmatory_evaluation, hybrid_evaluation


def test_hybrid_entrypoint_deep_audits_before_locked_evaluator(monkeypatch):
    args = Namespace(matrix="configs/hybrid_adamw.yaml")
    configs = [SimpleNamespace(run_id="hybrid")]
    audit = {"complete": True, "errors": [], "verified_runs": 8}
    calls = []
    monkeypatch.setattr(hybrid_evaluation.evaluate_matrix, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        hybrid_evaluation.evaluate_matrix, "_selected_configs", lambda selected: configs
    )
    monkeypatch.setattr(
        hybrid_evaluation,
        "audit_hybrid_training",
        lambda selected: calls.append(("audit", selected)) or audit,
    )
    monkeypatch.setattr(
        hybrid_evaluation,
        "run_evaluation_after_specialized_audit",
        lambda selected_args, selected_audit, *, label: (
            calls.append(("evaluate", selected_args, selected_audit, label)) or 0
        ),
    )

    hybrid_evaluation.main([])

    assert calls == [
        ("audit", configs),
        ("evaluate", args, audit, "hybrid AdamW control"),
    ]


def test_confirmatory_training_audit_binds_seed_view_to_six_runs(monkeypatch):
    seed = 314159
    configs = [SimpleNamespace(seed=seed) for _ in range(6)]
    protocol = {
        "confirmatory_data": {"seeds": [seed]},
        "training": {"runs_per_seed": 6},
    }
    dataset = {"rows": 500_000, "training_view_fingerprint": "seed-view"}
    expected = {"complete": True, "verified_runs": 6, "errors": []}
    monkeypatch.setattr(
        confirmatory_evaluation,
        "load_confirmatory_protocol",
        lambda _path: ("protocol.json", protocol),
    )
    monkeypatch.setattr(
        confirmatory_evaluation,
        "audit_confirmatory_view",
        lambda _path, selected_seed: dataset if selected_seed == seed else None,
    )
    calls = []
    monkeypatch.setattr(
        confirmatory_evaluation,
        "audit_derived_training_artifacts",
        lambda selected, receipt, *, deep: calls.append((selected, receipt, deep)) or expected,
    )

    result = confirmatory_evaluation.audit_confirmatory_training("protocol.json", seed, configs)

    assert result is expected
    assert calls == [(configs, dataset, True)]


def test_confirmatory_entrypoint_uses_specialized_preflight_for_each_seed(tmp_path, monkeypatch):
    seed = 314159
    protocol_path = tmp_path / "protocol.json"
    generated = tmp_path / "matrices"
    args = Namespace(
        protocol=protocol_path,
        experiment_matrix=tmp_path / "experiment.yaml",
        validation_spec=tmp_path / "validation.json",
        matrix_dir=generated,
        results_root=tmp_path / "results",
        log_dir=tmp_path / "logs",
        gpus_a="0,1,2,3",
        gpus_b="4,5,6,7",
        late_port_a=29710,
        late_port=29720,
        worker_python="/usr/bin/python3",
        families=["dense"],
        scope_amendment=Path("configs/dense_scope_amendment.json"),
        audit_only=False,
        receipt=tmp_path / "receipt.json",
    )
    protocol = {
        "confirmatory_data": {"seeds": [seed]},
        "training": {"matrix_output_dir": str(generated)},
    }
    configs = [SimpleNamespace(seed=seed, model_family="dense") for _ in range(3)]
    training_audit = {"complete": True, "errors": [], "verified_runs": 3}
    receipt = {"complete": True, "valid_units": 42}
    monkeypatch.setattr(confirmatory_evaluation, "parse_args", lambda _argv: args)
    monkeypatch.setattr(
        confirmatory_evaluation,
        "load_confirmatory_protocol",
        lambda _path: (protocol_path, protocol),
    )
    monkeypatch.setattr(confirmatory_evaluation, "audit_confirmatory_matrices", lambda *a, **k: {})
    monkeypatch.setattr(confirmatory_evaluation, "load_matrix", lambda _path: configs)
    monkeypatch.setattr(
        confirmatory_evaluation,
        "audit_confirmatory_training",
        lambda _path, selected_seed, selected_configs, families: training_audit,
    )
    calls = []
    monkeypatch.setattr(
        confirmatory_evaluation,
        "run_evaluation_after_specialized_audit",
        lambda worker_args, audit, *, label: calls.append((worker_args, audit, label)) or 0,
    )
    monkeypatch.setattr(
        confirmatory_evaluation,
        "audit_confirmatory_evaluations",
        lambda *a, **k: receipt,
    )
    writes = []
    monkeypatch.setattr(
        confirmatory_evaluation,
        "_atomic_json",
        lambda path, payload: writes.append((path, payload)),
    )

    confirmatory_evaluation.main([])

    assert len(calls) == 1
    assert calls[0][1:] == (training_audit, f"confirmatory seed {seed}")
    assert calls[0][0].matrix == str((generated / f"seed{seed}.yaml").resolve())
    assert calls[0][0].scope_amendment == args.scope_amendment
    assert writes == [(args.receipt, receipt)]


def test_confirmatory_evaluation_defaults_dense_with_explicit_two_family_opt_in():
    assert confirmatory_evaluation.parse_args([]).families == ["dense"]
    assert len(confirmatory_evaluation.parse_args([]).tasks) == 14
    assert confirmatory_evaluation.parse_args(["--families", "dense", "late"]).families == [
        "dense",
        "late",
    ]


@pytest.mark.parametrize(
    "tasks",
    [
        ["SciFact", "SciFact"],
        ["not-a-frozen-task"],
        [],
    ],
)
def test_confirmatory_task_selection_rejects_invalid_values(tasks):
    with pytest.raises(ValueError):
        confirmatory_evaluation.normalize_confirmatory_tasks(tasks)


def test_confirmatory_explicit_task_subset_is_preserved():
    selected = ["SciFact", "NFCorpus"]
    assert confirmatory_evaluation.parse_args(["--tasks", *selected]).tasks == selected


def test_confirmatory_subset_audit_ignores_valid_cached_superset(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    generated = tmp_path / "generated"
    generated.mkdir()
    matrix_path = generated / "seed314159.yaml"
    matrix_path.write_text("runs: []\n")
    configs = [SimpleNamespace(model_family="dense", run_id=f"run-{index}") for index in range(3)]
    selected = ("SciFact", "NFCorpus")
    rows = []
    for config in configs:
        for task in (*selected, "Touche2020"):
            result = tmp_path / f"{config.run_id}-{task}.json"
            result.write_text("{}")
            rows.append(
                {
                    "model_family": "dense",
                    "run_id": config.run_id,
                    "task": task,
                    "stage": 5,
                    "result_path": str(result),
                }
            )

    monkeypatch.setattr(
        confirmatory_evaluation,
        "load_confirmatory_protocol",
        lambda _path: (
            protocol_path,
            {
                "confirmatory_data": {"seeds": [314159]},
                "training": {"matrix_output_dir": str(generated)},
            },
        ),
    )
    monkeypatch.setattr(
        confirmatory_evaluation,
        "audit_confirmatory_matrices",
        lambda *args, **kwargs: {"manifest_sha256": "matrix-manifest"},
    )
    monkeypatch.setattr(
        confirmatory_evaluation,
        "resolve_scope",
        lambda families, amendment: (("dense",), {"active_scope": "test"}),
    )
    monkeypatch.setattr(confirmatory_evaluation, "load_matrix", lambda _path: configs)
    monkeypatch.setattr(
        confirmatory_evaluation, "collect_evaluations", lambda _root, _configs: rows
    )
    provenance_calls = []
    monkeypatch.setattr(
        confirmatory_evaluation,
        "checkpoint_paths",
        lambda config, stages: [Path(f"/{config.run_id}/checkpoint-5")],
    )
    monkeypatch.setattr(
        confirmatory_evaluation,
        "audit_evaluation_artifacts",
        lambda root, checkpoints, selected_rows: (
            provenance_calls.append((root, checkpoints, selected_rows))
            or {
                "input_manifest": {"path": "inputs.json", "sha256": "a" * 64},
                "result_files": {"root": str(root), "files": len(selected_rows)},
            }
        ),
    )

    audit = confirmatory_evaluation.audit_confirmatory_evaluations(
        protocol_path,
        matrix_dir=generated,
        results_root=tmp_path / "results",
        families=("dense",),
        tasks=selected,
    )

    assert audit["complete"] is True
    assert audit["tasks"] == list(selected)
    assert audit["valid_units"] == audit["expected_units"] == 6
    assert len(audit["result_sources"]) == 6
    assert len(provenance_calls) == 1
    assert len(provenance_calls[0][1]) == 3
    assert len(provenance_calls[0][2]) == 9
    assert audit["per_seed"]["314159"]["formal_artifacts"]["input_manifest"]["sha256"] == "a" * 64


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinal"])
def test_confirmatory_subset_audit_rejects_incomplete_or_ambiguous_rows(
    tmp_path, monkeypatch, mutation
):
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "seed314159.yaml").write_text("runs: []\n")
    configs = [SimpleNamespace(model_family="dense", run_id=f"run-{index}") for index in range(3)]
    rows = []
    for config in configs:
        for task in ("SciFact", "NFCorpus"):
            result = tmp_path / f"{config.run_id}-{task}.json"
            result.write_text("{}")
            rows.append(
                {
                    "model_family": "dense",
                    "run_id": config.run_id,
                    "task": task,
                    "stage": 5,
                    "result_path": str(result),
                }
            )
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(dict(rows[-1]))
    else:
        rows[-1]["stage"] = 4

    monkeypatch.setattr(
        confirmatory_evaluation,
        "load_confirmatory_protocol",
        lambda _path: (
            protocol_path,
            {
                "confirmatory_data": {"seeds": [314159]},
                "training": {"matrix_output_dir": str(generated)},
            },
        ),
    )
    monkeypatch.setattr(
        confirmatory_evaluation,
        "audit_confirmatory_matrices",
        lambda *args, **kwargs: {"manifest_sha256": "matrix-manifest"},
    )
    monkeypatch.setattr(
        confirmatory_evaluation,
        "resolve_scope",
        lambda families, amendment: (("dense",), {"active_scope": "test"}),
    )
    monkeypatch.setattr(confirmatory_evaluation, "load_matrix", lambda _path: configs)
    monkeypatch.setattr(
        confirmatory_evaluation, "collect_evaluations", lambda _root, _configs: rows
    )
    monkeypatch.setattr(
        confirmatory_evaluation,
        "checkpoint_paths",
        lambda config, stages: [Path(f"/{config.run_id}/checkpoint-5")],
    )
    monkeypatch.setattr(
        confirmatory_evaluation,
        "audit_evaluation_artifacts",
        lambda root, checkpoints, selected_rows: {
            "input_manifest": {"path": "inputs.json", "sha256": "a" * 64},
            "result_files": {"root": str(root), "files": len(selected_rows)},
        },
    )

    with pytest.raises(ValueError, match="coverage"):
        confirmatory_evaluation.audit_confirmatory_evaluations(
            protocol_path,
            matrix_dir=generated,
            results_root=tmp_path / "results",
            families=("dense",),
            tasks=("SciFact", "NFCorpus"),
        )
