"""Small component checks; not retrieval experiments or whole-grid acceptance."""

import unittest
from fractions import Fraction

from analyze import decompose, query_counts


class QuerySupportTests(unittest.TestCase):
    def test_query_rows_are_not_qrel_covered_counts(self):
        result = query_counts(["a", "b", "unused"], ["a", "a", "b"], [1, 0, 0], ["a", "b"])
        self.assertEqual(result, {"query_file_rows": 3, "qrel_rows": 3, "qrel_covered_queries": 2,
                                  "queries_with_positive_qrel": 1, "queries_without_positive_qrel": 1})

    def test_missing_query_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            query_counts(["a"], ["b"], [1], ["b"])

    def test_duplicate_query_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate query-file"):
            query_counts(["a", "a"], ["a"], [1], ["a"])

    def test_changed_membership_rejected(self):
        with self.assertRaisesRegex(ValueError, "membership"):
            query_counts(["a", "b"], ["a"], [1], ["b"])

    def test_duplicate_recorded_query_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate recorded"):
            query_counts(["a"], ["a"], [1], ["a", "a"])


class ContributionTests(unittest.TestCase):
    def test_exact_conservation_with_negative_contributions(self):
        values = [Fraction(2701, 1000), Fraction(-864, 1000)] + [Fraction(1, 1000)] * 12
        mean, contributions, signs = decompose(values)
        self.assertEqual(sum(contributions), mean)
        self.assertEqual(signs, {"positive_tasks": 13, "negative_tasks": 1, "tied_tasks": 0})

    def test_zero_net_preserves_cancellation(self):
        mean, contributions, signs = decompose([Fraction(1), Fraction(-1)] + [Fraction(0)] * 12)
        self.assertEqual(mean, 0)
        self.assertEqual(contributions[:2], [Fraction(1, 14), Fraction(-1, 14)])
        self.assertEqual(signs["tied_tasks"], 12)

    def test_missing_task_rejected(self):
        with self.assertRaisesRegex(ValueError, "fourteen"):
            decompose([Fraction(0)] * 13)


if __name__ == "__main__":
    unittest.main()
