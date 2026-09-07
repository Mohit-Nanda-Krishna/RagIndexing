"""Small unit tests for benchmark result handling without loading the embedding model."""

import unittest

import numpy as np

from benchmark import recall_at_k, valid_ids


class BenchmarkUtilityTests(unittest.TestCase):
    def test_valid_ids_removes_faiss_missing_markers(self) -> None:
        self.assertEqual(valid_ids(np.array([4, -1, 2, 99]), 10), [4, 2])

    def test_recall_uses_the_flat_result_count_as_its_denominator(self) -> None:
        self.assertEqual(recall_at_k([1, 2, 8], [1, 2, 3, 4, 5]), 0.4)
        self.assertEqual(recall_at_k([1, 2], []), None)


if __name__ == "__main__":
    unittest.main()
