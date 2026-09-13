import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import validate_dense_correction_candidate as audit

ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / audit.ARCHIVE


def payload(path):
    return json.loads((ARCHIVE / path).read_text())


def test_declared_matrix_changes_preserve_the_scientific_grid():
    old = yaml.safe_load((ROOT / "configs/dense_no_packing_retrain.yaml").read_text())
    new = yaml.safe_load(
        (ARCHIVE / "candidate-source/configs/dense_correctness_candidate.yaml").read_text()
    )
    audit.check_matrix(old, new)


@pytest.mark.parametrize("field", ["seed", "max_length", "global_batch_size", "dataset_path"])
def test_undeclared_scientific_change_is_rejected(field):
    old = yaml.safe_load((ROOT / "configs/dense_no_packing_retrain.yaml").read_text())
    new = yaml.safe_load(
        (ARCHIVE / "candidate-source/configs/dense_correctness_candidate.yaml").read_text()
    )
    new["common"][field] = "undeclared-change"
    with pytest.raises(AssertionError):
        audit.check_matrix(old, new)


@pytest.mark.parametrize("label,device", [("cpu", "cpu"), ("gpu", "cuda")])
def test_reference_audit_checks_actual_weight_and_state_records(label, device):
    audit.check_reference(payload(f"reference/optimizer-{label}-v2.json"), device)


def test_reference_summary_cannot_hide_a_failed_update():
    value = payload("reference/optimizer-cpu-v2.json")
    value["records"][-1]["steps"][-1]["candidate_weight_and_state_bitwise_equal_to_official"] = (
        False
    )
    with pytest.raises(AssertionError):
        audit.check_reference(value, "cpu")


@pytest.mark.parametrize("label,fixture", [("short", "short"), ("max-context", "max_context")])
def test_actual_gradient_fixtures_preserve_the_full_and_tail_checks(label, fixture):
    assert (
        audit.check_gradients(payload(f"{label}/result.json"), fixture)["raw_and_clipped_checks"]
        == 9
    )


def test_tail_cannot_be_relabelled_as_a_full_accumulation_group():
    value = payload("short/result.json")
    value["records"][0]["records"][-1]["trainer_divisors"] = [[4] * 4] * 4
    with pytest.raises(AssertionError):
        audit.check_gradients(value, "short")


def test_engineering_pass_cannot_be_promoted_to_scientific_completion():
    value = payload("short/result.json")
    value["scientific_completion"] = True
    with pytest.raises(AssertionError):
        audit.check_gradients(value, "short")


def test_source_contract_failures_are_reported_not_waived():
    result = audit.check_candidate_tests(ARCHIVE / "tests/candidate-base-full-v2.xml")
    assert result["full_suite_passed"] is False
    assert len(result["unwaived_source_contract_failures"]) == 11


def test_sealed_initial_candidate_sources_match_the_earlier_receipt():
    for item in payload("prepared-source.json")["candidate_sources"]:
        audit.compare_binding(item["identity"], ARCHIVE / "v1-source" / item["relative_path"], ROOT)


def test_optimized_python_cannot_create_a_false_acceptance_receipt(tmp_path):
    output = tmp_path / "not-written.json"
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "scripts.validate_dense_correction_candidate",
            "--repository",
            str(ROOT),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Audit assertions must not be disabled" in result.stderr
    assert not output.exists()
