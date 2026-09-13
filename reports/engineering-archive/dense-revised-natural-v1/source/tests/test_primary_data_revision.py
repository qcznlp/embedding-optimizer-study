import json
import os
import subprocess
import sys
from collections import Counter

import pytest
from datasets import Dataset

from embed_optim.data import SPLITS
from embed_optim.primary_data_revision import (
    QUOTAS,
    SCOPE,
    STATUS,
    TEXT_COLUMNS,
    load_amendment,
    select_natural_coverage,
)
from scripts.audit_dense_revised_natural import run_owned


def fixture_rows(count=45):
    return [
        {
            "sample_id": source_index * count + i,
            "source": source,
            "query_id": i,
            **{column: f"{source}-{i}-{column}" for column in TEXT_COLUMNS},
            "length": 100,
            "extra_metadata": i,
        }
        for source_index, source in enumerate(SPLITS)
        for i in range(count)
    ]


MANDATORY = [0, 1, 2, 45, 135, 180]


def test_seven_source_coverage_is_deterministic_and_retains_every_field():
    original = fixture_rows()
    dataset = Dataset.from_list(original)
    selected, receipt = select_natural_coverage(dataset, MANDATORY)
    repeated, again = select_natural_coverage(dataset, MANDATORY)
    assert len(selected) == 288 and receipt == again and list(selected) == list(repeated)
    assert set(MANDATORY).issubset(receipt["selected_positions"])
    assert Counter(selected["source"]) == QUOTAS
    assert list(selected) == [original[i] for i in receipt["selected_positions"]]
    assert receipt["expected_global_query_groups"] == [128, 128, 32]


@pytest.mark.parametrize(
    "targets", [MANDATORY[:5], [*MANDATORY, 181], [*MANDATORY[:5], -1], [*MANDATORY[:5], True]]
)
def test_rejects_wrong_replacement_scope(targets):
    with pytest.raises(ValueError, match="exactly six"):
        select_natural_coverage(Dataset.from_list(fixture_rows()), targets)


def test_requires_all_seven_sources():
    with pytest.raises(ValueError, match="all seven"):
        select_natural_coverage(Dataset.from_list(fixture_rows()[:-45]), MANDATORY)


@pytest.mark.parametrize("column", TEXT_COLUMNS)
@pytest.mark.parametrize("invalid", [None, ""])
def test_invalid_mandatory_group_cannot_be_skipped(column, invalid):
    rows = fixture_rows()
    rows[0][column] = invalid
    with pytest.raises(ValueError, match="invalid original text"):
        select_natural_coverage(Dataset.from_list(rows), MANDATORY)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "old"),
        ("status", "reviewed_primary_execution_lock"),
        ("execution_authorized", True),
        ("schema_version", True),
    ],
)
def test_data_amendment_cannot_become_primary_authorization(tmp_path, field, value):
    payload = {
        "scope": SCOPE,
        "status": STATUS,
        "schema_version": 1,
        "execution_authorized": False,
        "source_bindings": {},
        field: value,
    }
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_amendment(path, tmp_path)


def test_owned_cpu_launcher_records_identity_and_waits_for_terminal(tmp_path):
    code = run_owned(
        [sys.executable, "-c", "print('owned fixture')"],
        tmp_path,
        {**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        tmp_path,
    )
    assert code == 0
    receipt = json.loads((tmp_path / "launcher.json").read_text())
    assert receipt["pid"] == receipt["process_group"] and receipt["start_ticks"] > 0
    assert (tmp_path / "launcher.log").read_text().strip() == "owned fixture"


def test_timeout_requests_owned_launcher_shutdown_and_waits(tmp_path, monkeypatch):
    original = subprocess.Popen
    live = []

    class TimeoutOnce:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)
            self.first = True
            live.append(self.inner)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def wait(self, timeout):
            if self.first:
                self.first = False
                raise subprocess.TimeoutExpired("owned fixture", timeout)
            return self.inner.wait(timeout=timeout)

    monkeypatch.setattr(subprocess, "Popen", TimeoutOnce)
    with pytest.raises(RuntimeError, match="1800-second"):
        run_owned(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            {**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            tmp_path,
        )
    assert len(live) == 1 and live[0].poll() is not None
