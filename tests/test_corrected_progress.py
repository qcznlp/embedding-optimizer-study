import json
from types import SimpleNamespace

from embed_optim.corrected_progress import build_progress


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
    assert report["runs"][0]["latest_log_step"] == 20
    assert report["runs"][0]["declared_total_steps"] == 3907
