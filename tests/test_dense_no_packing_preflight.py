from datasets import Dataset

from embed_optim.dense_no_packing_preflight import _ids_sha256, _selected_indices


def test_longest_row_preflight_selection_is_deterministic():
    dataset = Dataset.from_dict(
        {
            "sample_id": [30, 20, 10, 40],
            "length": [100, 200, 200, 50],
        }
    )

    indices = _selected_indices(dataset, 3)

    assert indices == [2, 1, 0]
    assert _ids_sha256([10, 20, 30]) == _ids_sha256([10, 20, 30])
    assert _ids_sha256([10, 20, 30]) != _ids_sha256([20, 10, 30])
