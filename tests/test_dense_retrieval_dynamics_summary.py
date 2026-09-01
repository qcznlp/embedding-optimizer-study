from __future__ import annotations

import hashlib

import pytest

from embed_optim import dense_retrieval_dynamics_summary as summary
from embed_optim.decontamination import DECONTAMINATED_TASK_NAMES


def _task_rows():
    rows = []
    runs = [
        ("hybrid", 42, "hybrid_adamw", f"hybrid-{index}", (index + 1) * 1e-6) for index in range(4)
    ]
    runs.extend(
        ("confirmatory", seed, optimizer, f"{optimizer}-selected", learning_rate)
        for seed in (314159, 271828, 161803)
        for optimizer, learning_rate in (
            ("adamw", 3e-5),
            ("muon", 1e-3),
            ("normuon", 3e-3),
        )
    )
    for run_index, (suite, seed, optimizer, run_id, learning_rate) in enumerate(runs):
        for stage, step in enumerate((782, 1563, 2345, 3126, 3907), start=1):
            partition = "dynamics-stage1-4" if stage < 5 else "formal-stage5"
            for task_index, task in enumerate(DECONTAMINATED_TASK_NAMES):
                score = 0.30 + run_index * 0.005 + stage * 0.01 + task_index * 0.001
                digest = hashlib.sha256(
                    f"{suite}/{seed}/{run_id}/{stage}/{task}".encode()
                ).hexdigest()
                rows.append(
                    {
                        "suite": suite,
                        "training_seed": seed,
                        "partition": partition,
                        "model_family": "dense",
                        "optimizer": optimizer,
                        "learning_rate": learning_rate,
                        "aux_learning_rate": 3e-6,
                        "run_id": run_id,
                        "stage": stage,
                        "fraction": stage / 5,
                        "checkpoint_step": step,
                        "task": task,
                        "ndcg_at_10": score,
                        "result_path": f"results/{suite}/{seed}/{run_id}/{stage}/{task}.json",
                        "result_sha256": digest,
                    }
                )
    return rows


def test_five_stage_summary_has_exactly_65_rows_and_mean_median():
    rows = summary.summarize_five_stage_trajectories(_task_rows())

    assert len(rows) == 65
    assert len({(row["suite"], row["training_seed"], row["run_id"]) for row in rows}) == 13
    assert all(row["tasks_completed"] == 14 for row in rows)
    assert all(row["joined_summary_used_for_formal_inference"] is False for row in rows)
    first = rows[0]
    scores = [0.30 + 0.01 + index * 0.001 for index in range(14)]
    assert first["mean_ndcg_at_10"] == pytest.approx(sum(scores) / 14)
    assert first["median_ndcg_at_10"] == pytest.approx((scores[6] + scores[7]) / 2)
    assert sum(row["formal_source_stage5"] for row in rows) == 13


def test_summary_rejects_stage_five_in_the_dynamics_partition():
    rows = _task_rows()
    final = next(row for row in rows if row["stage"] == 5)
    final["partition"] = "dynamics-stage1-4"

    with pytest.raises(ValueError, match="inference boundary"):
        summary.summarize_five_stage_trajectories(rows)


def test_summary_outputs_pdf_svg_csv_and_hashes_every_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    matplotlib = pytest.importorskip("matplotlib")
    monkeypatch.setitem(matplotlib.rcParams, "pdf.fonttype", 3)
    monkeypatch.setitem(matplotlib.rcParams, "ps.fonttype", 3)
    trajectories = summary.summarize_five_stage_trajectories(_task_rows())
    output = tmp_path / "reports/dense-retrieval-dynamics"

    records = summary.write_summary_outputs(
        trajectories,
        output_dir=output,
        repository=tmp_path,
    )

    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42
    assert set(records) == {"trajectory_csv", "figure_svg", "figure_pdf"}
    assert records["trajectory_csv"]["rows"] == 65
    for record in records.values():
        path = tmp_path / record["path"]
        assert path.is_file()
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (tmp_path / records["figure_svg"]["path"]).read_text().startswith("<?xml")
    assert (tmp_path / records["figure_pdf"]["path"]).read_bytes().startswith(b"%PDF")


def test_output_hash_gate_rejects_tampering(tmp_path):
    trajectories = summary.summarize_five_stage_trajectories(_task_rows())
    records = summary.write_summary_outputs(
        trajectories,
        output_dir=tmp_path / "reports/dense-retrieval-dynamics",
        repository=tmp_path,
    )
    table = tmp_path / records["trajectory_csv"]["path"]
    table.write_text(table.read_text() + "tampered\n")

    with pytest.raises(ValueError, match="output differs"):
        summary._verify_outputs(records, tmp_path)


def test_audit_only_recomputes_without_rewriting_and_rejects_tampering(tmp_path, monkeypatch):
    repository = tmp_path.resolve()
    contract_path = repository / "configs/contract.json"
    protocol_path = repository / "configs/protocol.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text('{"contract": true}\n')
    protocol_path.write_text('{"protocol": true}\n')

    class Contract:
        def __init__(self):
            self.path = contract_path
            self.repository = repository

        def formal_result_root(self, suite):
            return self.repository / f"results/formal-{suite}"

    contract = Contract()
    material = {
        "dynamics_audit": {
            "complete": True,
            "expected_units": 728,
            "valid_units": 728,
            "formal_inference_uses_dynamics_rows": False,
            "contract": {"sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest()},
        },
        "formal_confirmatory": {
            "complete": True,
            "expected_units": 126,
            "valid_units": 126,
            "protocol_sha256": "1" * 64,
            "matrix_manifest_sha256": "2" * 64,
        },
        "matrix_manifest_sha256": "2" * 64,
        "partition_sources": [],
        "protocol_path": protocol_path,
        "trajectories": summary.summarize_five_stage_trajectories(_task_rows()),
    }
    monkeypatch.setattr(summary, "load_dynamics_contract", lambda _: contract)
    monkeypatch.setattr(summary, "_collect_summary_material", lambda _: material)

    output = repository / "reports/dense-retrieval-dynamics"
    summary.build_dense_retrieval_dynamics_summary(contract_path, output)
    protected = [
        output / "summary_manifest.json",
        output / "five_stage_retrieval_dynamics.csv",
        output / "five_stage_retrieval_dynamics.svg",
        output / "five_stage_retrieval_dynamics.pdf",
    ]
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in protected}

    receipt = summary.audit_dense_retrieval_dynamics_summary(contract_path, output)

    assert receipt["complete"] is True
    assert receipt["read_only"] is True
    assert receipt["coverage"] == {
        "dynamics_units": 728,
        "formal_hybrid_stage5_units": 56,
        "formal_confirmatory_stage5_units": 126,
        "task_units": 910,
        "trajectory_rows": 65,
    }
    assert before == {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in protected}

    table = output / "five_stage_retrieval_dynamics.csv"
    table.write_text(table.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="output differs"):
        summary.audit_dense_retrieval_dynamics_summary(contract_path, output)


def test_cli_exposes_read_only_audit_mode():
    args = summary.parse_args(["--audit-only"])

    assert args.audit_only is True
