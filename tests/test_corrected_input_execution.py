import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embed_optim.config import load_matrix
from embed_optim.corrected_beir_evaluation import (
    _model_folder,
    _selected_configs,
    _source_manifest,
)
from embed_optim.corrected_input_execution import (
    PADDED_DENSE_RECEIPT,
    require_corrected_training_receipt,
    require_independently_padded_dense,
)
from embed_optim.corrected_validation_matrix import build_jobs


def test_corrected_dense_execution_is_forced_and_verified():
    first = SimpleNamespace(can_flatten_inputs=True)
    model = SimpleNamespace(_first_module=lambda: first)

    assert require_independently_padded_dense(model) == PADDED_DENSE_RECEIPT
    assert first.can_flatten_inputs is False
    require_corrected_training_receipt(
        {"model_family": "dense", "input_execution": dict(PADDED_DENSE_RECEIPT)}
    )


def test_corrected_dense_execution_rejects_missing_controls_or_receipts():
    model = SimpleNamespace(_first_module=lambda: SimpleNamespace())
    with pytest.raises(RuntimeError, match="does not expose can_flatten_inputs"):
        require_independently_padded_dense(model)
    with pytest.raises(RuntimeError, match="lacks the corrected"):
        require_corrected_training_receipt({"model_family": "dense"})


def test_corrected_validation_matrix_accepts_only_the_twelve_padded_runs(monkeypatch, tmp_path):
    matrix = Path(__file__).parents[1] / "configs" / "dense_no_packing_retrain.yaml"
    configs = load_matrix(matrix)
    monkeypatch.setattr(
        "embed_optim.corrected_validation_matrix._final_checkpoint",
        lambda config: tmp_path / config.run_id / "checkpoint-3907",
    )

    jobs = build_jobs(configs, tmp_path / "results")

    assert len(jobs) == 12
    assert len({job.label for job in jobs}) == 12
    assert all(job.output_dir.parent == (tmp_path / "results").resolve() for job in jobs)
    invalid = [
        SimpleNamespace(**{**config.__dict__, "dense_can_flatten_inputs": True})
        for config in configs
    ]
    with pytest.raises(ValueError, match="12 padded Dense"):
        build_jobs(invalid, tmp_path / "invalid")


def test_corrected_preflight_receipt_is_bound_to_padded_mode():
    report = json.loads(
        (Path(__file__).parents[1] / "reports/dense-no-packing/preflight-selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["input_execution"] == PADDED_DENSE_RECEIPT


def test_corrected_beir_selection_requires_terminal_padding_receipts(monkeypatch, tmp_path):
    matrix = Path(__file__).parents[1] / "configs" / "dense_no_packing_retrain.yaml"
    configs = [
        dataclasses.replace(config, output_root=str(tmp_path)) for config in load_matrix(matrix)
    ]
    for config in configs:
        config.output_dir.mkdir(parents=True)
        (config.output_dir / "completed.json").write_text(
            json.dumps(
                {
                    "model_family": "dense",
                    "input_execution": dict(PADDED_DENSE_RECEIPT),
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr("embed_optim.corrected_beir_evaluation.load_matrix", lambda _: configs)

    selected_matrix, selected = _selected_configs(matrix, [configs[3].run_id])

    assert selected_matrix == matrix.resolve()
    assert [config.run_id for config in selected] == [configs[3].run_id]
    assert _model_folder(Path("/tmp/padded-run/checkpoint-782")) == "padded-run__checkpoint-782"


def test_corrected_beir_source_manifest_covers_wrapper_and_legacy_workers():
    root = Path(__file__).parents[1]
    manifest = _source_manifest(root)

    assert "scripts/eval/dense_no_packing_parallel.py" in manifest
    assert "scripts/eval/dense_parallel.py" in manifest
    assert "scripts/eval/dense_sequential.py" in manifest
    assert "src/embed_optim/corrected_input_execution.py" in manifest
    assert all(
        identity["bytes"] > 0 and len(identity["sha256"]) == 64 for identity in manifest.values()
    )
