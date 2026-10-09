import unittest

import pandas as pd

from data_utils import normalize_bug_labels


class NormalizeBugLabelsTests(unittest.TestCase):
    def test_numeric_bug_counts_become_binary_without_dropping_defects(self):
        labels = pd.Series([0, 1, 2, 12])

        result = normalize_bug_labels(labels)

        self.assertEqual(result.astype(int).tolist(), [0, 1, 1, 1])

    def test_boolean_text_labels_are_supported(self):
        labels = pd.Series(["true", "false", "yes", "no"])

        result = normalize_bug_labels(labels)

        self.assertEqual(result.astype(int).tolist(), [1, 0, 1, 0])

    def test_unrecognized_labels_remain_missing(self):
        labels = pd.Series(["0", "unknown", None])

        result = normalize_bug_labels(labels)

        self.assertEqual(result.iloc[0], 0)
        self.assertTrue(result.iloc[1:].isna().all())


if __name__ == "__main__":
    unittest.main()
