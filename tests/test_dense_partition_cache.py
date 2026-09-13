from collections import OrderedDict
from pathlib import Path

import pyarrow as pa

from scripts.materialize_dense_partition_revision import PinnedSource


def test_bounded_cache_hits_and_evictions_keep_score_values_identical():
    class FixtureParquet:
        def __init__(self):
            self.calls = []

        def read_row_group(self, row_group, columns):
            self.calls.append(row_group)
            return pa.Table.from_pylist(
                [{"query_id": row_group, "document_ids": [1, 2], "scores": [1.0, 0.2]}]
            )

    source = PinnedSource.__new__(PinnedSource)
    path = Path("/fixture.parquet")
    reader = FixtureParquet()
    source.parquets, source.cache = {path: reader}, OrderedDict()
    expected = source.score_record(path, 0, 0)
    assert source.score_record(path, 0, 0) == expected and reader.calls == [0]
    for row_group in range(1, 65):
        source.score_record(path, row_group, 0)
    assert len(source.cache) == 64 and (path, 0) not in source.cache
    assert source.score_record(path, 0, 0) == expected and len(source.cache) == 64
    assert reader.calls[-1] == 0
