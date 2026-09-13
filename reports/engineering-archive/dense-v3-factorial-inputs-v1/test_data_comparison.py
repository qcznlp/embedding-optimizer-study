"""Negative controls for exact full-group comparison, not experiment results."""

import copy

import pytest
from datasets import Dataset

from audit_data import compare_all_fields


def rows():
    result = []
    for index in range(3):
        row = {'sample_id': index, 'source': 'test', 'query_id': index,
               'positive_id': index + 10, 'query': f'q{index}', 'positive': f'p{index}',
               'length': index + 1}
        for negative in range(7):
            row[f'negative_{negative}'] = f'n{index}-{negative}'
            row[f'negative_{negative}_id'] = index * 7 + negative + 100
        result.append(row)
    return result


def test_exact_selection_including_nonascending_calibration_order():
    values = rows()
    parent = Dataset.from_list(values)
    selected = Dataset.from_list([values[2], values[0]])
    result = compare_all_fields(parent, selected, [2, 0])
    assert result['rows'] == 2 and len(result['columns']) == 21
    assert result['differing_rows'] == []


@pytest.mark.parametrize('column', list(rows()[0]))
def test_every_text_identity_and_length_field_is_compared(column):
    values = rows()
    modified = copy.deepcopy(values)
    original = modified[1][column]
    modified[1][column] = original + '-changed' if isinstance(original, str) else original + 1000
    result = compare_all_fields(Dataset.from_list(values), Dataset.from_list(modified), [0, 1, 2])
    assert len(result['differing_rows']) == 1
    assert result['differing_rows'][0]['child_row'] == 1
    assert result['differing_rows'][0]['fields'] == [column]


def test_omitted_column_is_not_silently_ignored():
    parent = Dataset.from_list(rows())
    with pytest.raises(ValueError, match='column'):
        compare_all_fields(parent, parent.remove_columns('length'), [0, 1, 2])


def test_wrong_row_count_is_not_silently_ignored():
    parent = Dataset.from_list(rows())
    with pytest.raises(ValueError, match='cardinality'):
        compare_all_fields(parent, parent.select([0, 1]), [0, 1, 2])
