# Zero-statistic consistency omission

Owned session 3807 is terminal/exit 1. All ten explicit zero-matrix controls fail
because the initial archive reader accepted nonzero row/column distribution fields
or top-row energy alongside an exactly zero Frobenius norm. This is an actual new
reader omission, not a failed scientific tolerance or a training-kernel failure.

`zero-initial.xml` and `source/` preserve the exact tests and initial primitive.
The added check requires zero statistics for zero matrices; it changes no stored
measurement, threshold, spectrum or aggregation kernel. The separate real-record
retry still matches all nine diagnostic checkpoint rows. Later results do not
overwrite this first failure or certify the first smoke's source as the fixed reader.
