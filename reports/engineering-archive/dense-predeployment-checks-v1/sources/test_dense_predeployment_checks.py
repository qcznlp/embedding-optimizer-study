import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import validate_dense_predeployment_checks as audit

ROOT = Path(__file__).parents[1]


def payload():
    return json.loads((ROOT / audit.ARCHIVE / "admission/result.json").read_text())


def test_complete_admission_record_recomputes_all_proposed_comparisons():
    assert audit.check_admission(payload())["prototype_exact_comparisons"] == 42


def test_admission_failure_is_not_promoted_to_full_acceptance():
    value = payload()
    value["full_recipe_resume_contract_passed"] = True
    with pytest.raises(AssertionError):
        audit.check_admission(value)


def test_summary_cannot_hide_an_unchecked_field():
    value = payload()
    value["records"][0]["actual_preload_gate"]["accepted_preload_gate"] = False
    with pytest.raises(AssertionError):
        audit.check_admission(value)


def test_prototype_inputs_are_recomputed_not_just_trusted_flags():
    value = payload()
    case = next(x for x in value["prototype_cases"] if x["case"] == "seed")
    case["requested_identity"] = value["expected_identities"][case["run_id"]]
    with pytest.raises(AssertionError):
        audit.check_admission(value)


def test_missing_comparison_is_not_accepted():
    value = payload()
    value["prototype_cases"].pop()
    with pytest.raises(AssertionError):
        audit.check_admission(value)


def test_relocation_must_remain_a_positive_control():
    value = payload()
    case = next(x for x in value["prototype_cases"] if x["case"] == "identical_relocation")
    case["should_equal"] = False
    with pytest.raises(AssertionError):
        audit.check_admission(value)


def test_optimized_python_cannot_issue_an_acceptance_receipt(tmp_path):
    output = tmp_path / "not-created.json"
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "scripts.validate_dense_predeployment_checks",
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
