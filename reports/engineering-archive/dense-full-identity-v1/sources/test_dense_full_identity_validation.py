import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_dense_full_identity import check_admission, check_tests

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports/engineering-archive/dense-full-identity-v1"


def payload():
    return json.loads((ARCHIVE / "admission/result.json").read_text())


def test_real_completed_admission_receipt_covers_all_algorithms_and_cases():
    assert check_admission(payload()) == {
        "positive_controls": 6,
        "changed_identities_rejected": 36,
        "changed_real_payloads_rejected": 3,
    }


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "false_positive", "false_negative", "updates", "pickle"]
)
def test_bad_admission_claim_is_not_accepted(mutation):
    value = copy.deepcopy(payload())
    if mutation == "missing":
        value["records"].pop()
    elif mutation == "duplicate":
        value["records"][-1] = value["records"][0]
    elif mutation == "false_positive":
        value["records"][-1]["observed"]["accepted"] = True
    elif mutation == "false_negative":
        value["records"][0]["observed"]["accepted"] = False
    elif mutation == "updates":
        value["model_updates_executed"] = 1
    elif mutation == "pickle":
        value["model_or_pickle_loaded"] = True
    with pytest.raises(AssertionError):
        check_admission(value)


def test_candidate_failure_receipt_is_not_converted_into_green_suite():
    result = check_tests(ARCHIVE / "tests/full-first.xml")
    assert result["full_suite_passed"] is False and len(result["unwaived_source_failures"]) == 11


def test_optimized_python_refuses_before_any_receipt_write(tmp_path):
    destination = tmp_path / "must-not-exist.json"
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-B",
            "-m",
            "scripts.validate_dense_full_identity",
            "--repository",
            str(ROOT),
            "--output",
            str(destination),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0 and "Audit assertions must not be disabled" in result.stderr
    assert not destination.exists()
