"""Complete native-shape synthetic geometry reconstruction, not primary findings."""

import csv
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from embed_optim import primary_v3_geometry_reconstruction as replay
from embed_optim.primary_contract import read_json
from embed_optim.primary_v3_contract import PrimaryV3Contract
from embed_optim.primary_v3_dimension_inference import FunctionalInferenceContract
from embed_optim.primary_v3_geometry import TABLE_COUNTS
from embed_optim.primary_v3_geometry_primitives import metric, stage_records
from embed_optim.primary_v3_outcomes import csv_bytes
from embed_optim.primary_v3_reconstruction_authoring import build
from embed_optim.primary_v3_reconstruction_inputs import ReconstructionInput
from embed_optim.primary_v3_reconstruction_sources import INFERENCE_PROTOCOL
from scripts.geometry_reconstruction_controls import CASES, mutate, stage_change
from scripts.geometry_reconstruction_fixture import make, synthetic_metric
from scripts.geometry_reconstruction_oracle import audit as independent_oracle

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "reports/engineering-archive/dense-full-identity-v1/candidate-source"


@pytest.fixture(autouse=True)
def isolate_old_archival_test_aliases(monkeypatch):
    for side in ("before", "after"):
        name = "embed_optim._main_resume_" + side
        module = sys.modules.get(name)
        if module is not None:
            assert (
                Path(module.__file__)
                == ROOT
                / f"reports/engineering-archive/main-resume-v1/{side}/corrected_completion_pipeline.py"
            )
            monkeypatch.delitem(sys.modules, name)


@pytest.fixture(scope="module")
def contract():
    primary = PrimaryV3Contract.load(
        ROOT / "configs/dense_primary_v3_protocol.json", ROOT, CANDIDATE
    )
    return FunctionalInferenceContract.load(ROOT / INFERENCE_PROTOCOL, primary)


@pytest.fixture(scope="module")
def archive(contract, tmp_path_factory):
    root = tmp_path_factory.mktemp("explicit-synthetic-geometry-reconstruction")
    return make(root / "fixture", contract)


def test_complete_native_geometry_reconstruction(archive, tmp_path):
    result = replay.reconstruct(ReconstructionInput.load(*archive), tmp_path / "recomputed")
    assert result["original_geometry_aggregate_numerics_verified"] is True
    assert result["table_counts"] == TABLE_COUNTS
    assert result["checkpoints_reconstructed"] == 60
    assert result["raw_matrix_records_authenticated"] == 5280
    assert result["spectral_health_records_authenticated"] == 10560
    assert result["run_pairs_recomputed"] == 660
    assert result["upstream_admission_simulated"] is True
    assert len(result["outputs"]) == 8
    for key in (
        "raw_weight_metrics_recomputed",
        "spectral_health_remeasured",
        "basis_orthogonality_remeasured",
        "checkpoint_tensors_revalidated",
        "model_encoding_repeated",
        "retrieval_repeated",
        "original_outcomes_recomputed",
        "raw_vector_features_recomputed",
        "bridge_and_functional_inference_recomputed",
        "primary_scientific_admission",
        "manuscript_installed",
        "scientific_completion",
    ):
        assert result[key] is False
    for name in ("summary.json", *(name + ".csv" for name in TABLE_COUNTS)):
        assert (tmp_path / "recomputed/geometry" / name).read_bytes() == (
            archive[0] / "geometry" / name
        ).read_bytes()
    addresses = read_json(tmp_path / "recomputed/local_addresses.json")
    assert len(addresses) == len({row["local"] for row in addresses}) == 137


def test_independent_set_and_rational_oracle(archive):
    result = independent_oracle(archive[0])
    assert result["passed"] is True
    assert result["native_coordinate_bases_checked"] == 14480
    assert result["exact_set_overlap_pairs"] == 660
    assert result["undefined_pairs"] == 132
    assert result["real_weight_spectra_verified"] is False


@pytest.mark.parametrize("case", CASES)
def test_fully_rehashed_semantic_controls(archive, tmp_path, case):
    root = tmp_path / ("rehashed-" + case)
    shutil.copytree(archive[0], root)
    anchor = mutate(root, case)
    inputs = ReconstructionInput.load(root, anchor)
    with pytest.raises((ValueError, KeyError)) as error:
        replay.reconstruct(inputs, tmp_path / "refused")
    assert "SHA-256 mismatch" not in str(error.value)
    assert not (tmp_path / "refused/reconstruction.json").exists()
    if case in {"checkpoint_row", "geometry_table", "geometry_summary", "basis_values"}:
        assert (tmp_path / "refused/fresh_checkpoint_rows.json").is_file()


@pytest.mark.parametrize(
    "case",
    [
        "stage_anchor",
        "record_missing",
        "record_duplicate",
        "record_shape",
        "entry_count",
        "zero_presence",
        "metric_schema",
        "sketch_rank",
        "health_rank",
        "health_signal",
        "health_gap",
        "health_missing",
    ],
)
def test_raw_stage_schema_refusals(contract, archive, case):
    admitted = read_json(archive[0] / "vectors/admission.json")
    run = sorted((archive[0] / "geometry/runs").iterdir())[0]
    checked = admitted["complete_runs"][run.name]
    expected = contract.primary.expected_identity(run.name)
    raw = read_json(run / f"checkpoint-{checked['steps'][1]}.json")
    stage_change(raw, case)
    with pytest.raises((ValueError, KeyError)):
        stage_records(
            raw,
            expected,
            checked,
            admitted["reference"],
            contract.bridge.geometry.payload["settings"],
            2,
        )


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "1.0"])
def test_raw_metric_rejects_nonfinite_negative_and_untyped_scalars(contract, value):
    row = synthetic_metric([768, 768], 1.0)
    row["frobenius_norm"] = value
    with pytest.raises(ValueError):
        metric(row, [768, 768], contract.bridge.geometry.payload["settings"])


def test_raw_metric_refuses_zero_spectral_inconsistency(contract):
    row = synthetic_metric([768, 768], 0.0, displacement=True)
    row["spectral_norm"] = 1.0
    with pytest.raises(ValueError):
        metric(row, [768, 768], contract.bridge.geometry.payload["settings"], displacement=True)


@pytest.mark.parametrize(
    "field,key",
    [
        (field, key)
        for field in ("row_norms", "column_norms")
        for key in ("mean", "cv", "gini", "max_to_median")
    ]
    + [(field, None) for field in ("top_1pct_row_energy", "top_10pct_row_energy")],
)
def test_exact_zero_matrix_cannot_retain_nonzero_norm_statistics(contract, field, key):
    row = synthetic_metric([768, 768], 0.0, displacement=True)
    if key is None:
        row[field] = 0.25
    else:
        row[field][key] = 0.25
    with pytest.raises(ValueError):
        metric(row, [768, 768], contract.bridge.geometry.payload["settings"], displacement=True)


def test_stage_thread_count_is_frozen(contract, archive):
    from embed_optim.primary_v3_geometry_primitives import inspect_run_primitives

    admitted = read_json(archive[0] / "vectors/admission.json")
    run = sorted((archive[0] / "geometry/runs").iterdir())[0]
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with pytest.raises(ValueError, match="thread count"):
            inspect_run_primitives(
                run,
                contract.primary.expected_identity(run.name),
                admitted["complete_runs"][run.name],
                admitted["reference"],
                contract.bridge.geometry,
            )
    finally:
        torch.set_num_threads(previous)


def test_actual_authoring_still_refuses_missing_primary(contract, tmp_path):
    args = SimpleNamespace(
        experiment_root=tmp_path / "missing-primary", output=tmp_path / "forbidden"
    )
    with pytest.raises(ValueError, match="ordinary retained run"):
        build(contract, args)
    assert not args.output.exists()


def test_controls_require_explicit_synthetic_namespace(archive):
    with pytest.raises(ValueError, match="rehashed synthetic geometry"):
        mutate(archive[0], "geometry_table")


def test_output_cannot_overwrite_or_enter_archive(archive, tmp_path):
    inputs = ReconstructionInput.load(*archive)
    for path in (archive[0], archive[0] / "nested", tmp_path):
        with pytest.raises(ValueError, match="new geometry reconstruction"):
            replay.reconstruct(inputs, path)


def test_foreign_imported_package_remains_rejected(archive, monkeypatch):
    foreign = ModuleType("embed_optim._untrusted_geometry")
    foreign.__file__ = str(ROOT / "tests/test_primary_v3_geometry_reconstruction.py")
    monkeypatch.setitem(sys.modules, "embed_optim._untrusted_geometry", foreign)
    with pytest.raises(ValueError, match="source is absent"):
        ReconstructionInput.load(*archive)


@pytest.mark.parametrize("case", ["basis", "entry_denominator", "pair_overlap", "group_mean"])
def test_independent_oracle_refuses_inconsistent_fixture(archive, tmp_path, case):
    root = tmp_path / ("oracle-control-" + case)
    shutil.copytree(archive[0], root)
    if case == "basis":
        from safetensors.torch import load_file, save_file

        path = sorted((root / "geometry/runs").glob("*/checkpoint-1563-bases.safetensors"))[0]
        values = load_file(path)
        key = next(iter(values))
        values[key] = values[key] * 0.5
        save_file(values, path)
    else:
        filename, key = {
            "entry_denominator": ("checkpoint_geometry", "hidden_parameters"),
            "pair_overlap": ("run_pair_subspace_overlap", "mean_subspace_overlap"),
            "group_mean": (
                "optimizer_pair_subspace_summary",
                "mean_subspace_overlap_across_rate_pairs",
            ),
        }[case]
        path = root / "geometry" / (filename + ".csv")
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        row = next(r for r in rows if r[key] != "")
        row[key] = int(row[key]) + 1 if case == "entry_denominator" else float(row[key]) + 0.125
        path.write_bytes(csv_bytes(rows))
    with pytest.raises(AssertionError):
        independent_oracle(root)
