import pytest
from datasets import Dataset

from scripts.prepare_dense_data_protection import overlaps, query_digest


@pytest.mark.parametrize("variant", [" A Test ", "a\t test", "A\nTEST", "Ａ TEST"])
def test_declared_normalized_equality(variant):
    assert query_digest(variant) == query_digest("a test")


@pytest.mark.parametrize("invalid", [None, "", " \n\t"])
def test_rejects_empty_or_missing_query(invalid):
    with pytest.raises(ValueError):
        query_digest(invalid)


def test_does_not_claim_near_duplicate_equivalence():
    assert query_digest("a test") != query_digest("a test?")


def test_cross_namespace_text_overlap_not_just_integer_id():
    data = Dataset.from_list(
        [
            {"sample_id": 0, "source": "training_source", "query_id": 10, "query": " A Test "},
            {"sample_id": 1, "source": "training_source", "query_id": 20, "query": "No match"},
        ]
    )
    protected = {query_digest("a test"): [{"task": "test_fixture", "query_id": "999"}]}
    result = overlaps(data, protected)
    assert len(result) == 1
    assert result[0]["sample_id"] == 0 and result[0]["query_id"] == 10
    assert result[0]["held_out_matches"][0]["query_id"] == "999"
