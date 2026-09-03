import json
from types import SimpleNamespace

from embed_optim.corrected_progress import _log_progress, build_progress, main


def test_corrected_progress_is_artifact_only_and_counts_runs(monkeypatch, tmp_path):
    configs = [
        SimpleNamespace(
            run_id="padded-adamw",
            optimizer=SimpleNamespace(name="adamw", lr=3e-6),
            output_dir=tmp_path / "outputs/padded-adamw",
        ),
        SimpleNamespace(
            run_id="padded-muon",
            optimizer=SimpleNamespace(name="muon", lr=3e-4),
            output_dir=tmp_path / "outputs/padded-muon",
        ),
        SimpleNamespace(
            run_id="padded-normuon",
            optimizer=SimpleNamespace(name="normuon", lr=3e-4),
            output_dir=tmp_path / "outputs/padded-normuon",
        ),
    ]
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "dense-padded-adamw.log").write_text("134/134\n10/3907\r20/3907\n", encoding="utf-8")
    (log_dir / "dense-padded-muon.log").write_text(
        "3/3907\nTraceback (most recent call last)\n", encoding="utf-8"
    )
    configs[0].output_dir.mkdir(parents=True)
    (configs[0].output_dir / "checkpoint_schedule.json").write_text(
        json.dumps({"steps": [782, 1563, 2345, 3126, 3907]}), encoding="utf-8"
    )
    monkeypatch.setattr("embed_optim.corrected_progress.load_matrix", lambda _: configs)
    monkeypatch.setattr(
        "embed_optim.corrected_progress._run_is_complete",
        lambda config: config.run_id == "padded-adamw",
    )
    monkeypatch.setattr(
        "embed_optim.corrected_progress._checkpoint_is_resumable",
        lambda path: path.name == "checkpoint-782",
    )

    report = build_progress("matrix.yaml", log_dir)

    assert report["observation"] == "artifact_only_no_process_inspection"
    assert report["complete_runs"] == 1
    assert report["started_incomplete_runs"] == 1
    assert report["pending_runs"] == 1
    assert report["resumable_checkpoints"] == 1
    assert report["error_markers"]["traceback"] == 1
    assert report["error_markers"]["nccl_error"] == 0
    assert report["control_plane_warning_markers"]["tcpstore_heartbeat_disconnect"] == 0
    assert report["runs"][0]["latest_log_step"] == 20
    assert report["runs"][0]["declared_total_steps"] == 3907
    assert report["planned_runs"] == 3
    assert report["planned_checkpoints"] == 15
    assert report["planned_beir_task_units"] == 210
    assert report["canonical_handoff"] == "PROJECT_STATUS.md"
    assert report["study_status"] == "active"
    assert report["active_phase"] == "corrected_dense_no_packing_training"


def test_progress_separates_tcpstore_heartbeat_warning_from_fatal_nccl_error(tmp_path):
    log = tmp_path / "train.log"
    log.write_bytes(
        b"100/3907\n"
        b'[rank0]:[W] ProcessGroupNCCL.cpp Failed to check the "should dump" flag '
        b"on TCPStore, with error: Broken pipe\n"
    )

    step, total, errors, warnings = _log_progress(log)

    assert (step, total) == (100, 3907)
    assert errors["nccl_error"] == 0
    assert warnings["tcpstore_heartbeat_disconnect"] == 1


def test_progress_counts_fatal_nccl_data_plane_marker(tmp_path):
    log = tmp_path / "train.log"
    log.write_bytes(
        b"200/3907\ntorch.distributed.DistBackendError: NCCL error in: ProcessGroupNCCL.cpp\n"
    )

    _, _, errors, warnings = _log_progress(log)

    assert errors["nccl_error"] == 1
    assert warnings["tcpstore_heartbeat_disconnect"] == 0


def test_corrected_progress_cli_writes_same_atomic_snapshot(monkeypatch, tmp_path, capsys):
    report = {
        "schema_version": 1,
        "scope": "corrected_dense_no_packing",
        "complete_runs": 2,
    }
    monkeypatch.setattr(
        "embed_optim.corrected_progress.build_progress", lambda matrix, log_dir: report.copy()
    )
    output = tmp_path / "nested" / "progress.json"

    main(["--matrix", "matrix.yaml", "--log-dir", "logs", "--output", str(output)])

    written = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert written["complete_runs"] == 2
    assert written["observed_at_utc"].endswith("+00:00")
    assert not output.with_name(".progress.json.tmp").exists()
