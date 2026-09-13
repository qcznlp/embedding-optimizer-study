import pytest

from embed_optim.collators import TEXT_COLUMNS
from scripts.audit_dense_gpu_max_context import LONG_ROW_COUNT, REPEATS, SHORT_ROWS, stress_rows


def test_stress_rows_keep_exact_identities_and_short_tail():
    actual, short = stress_rows(288), SHORT_ROWS(288)
    assert [x["example_id"] for x in actual] == list(range(288))
    assert actual[LONG_ROW_COUNT:] == short[LONG_ROW_COUNT:]
    for index in range(LONG_ROW_COUNT):
        for name in TEXT_COLUMNS:
            assert actual[index][name] == (short[index][name] + " ") * REPEATS
        assert actual[index]["length"] == sum(len(actual[index][c]) for c in TEXT_COLUMNS)


def test_stress_rows_are_deterministic():
    assert stress_rows(288) == stress_rows(288)


@pytest.mark.parametrize("count", [0, 32, 128, 256, 289])
def test_stress_cannot_silently_change_epoch_topology(count):
    with pytest.raises(ValueError):
        stress_rows(count)
