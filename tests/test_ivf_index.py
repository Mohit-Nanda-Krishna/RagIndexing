"""Tests for the IVF implementation using deterministic, lightweight vectors."""

import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from ivf_index import IVFIndex


class IVFIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        # Two well-separated groups make list assignment and nprobe effects clear.
        left = np.column_stack((np.arange(40, dtype=np.float32) * 0.1, np.zeros(40)))
        right = np.column_stack(
            (6.0 + np.arange(40, dtype=np.float32) * 0.1, np.zeros(40))
        )
        self.vectors = np.vstack((left, right)).astype(np.float32)

    def test_creation_and_build_creates_requested_lists(self) -> None:
        index = IVFIndex(nlist=2, nprobe=1, dimension=2)
        self.assertFalse(index.is_trained)

        index.build(self.vectors)

        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, len(self.vectors))
        self.assertEqual(len(index.list_sizes), 2)
        self.assertEqual(index.list_sizes.sum(), len(self.vectors))
        self.assertTrue(np.all(index.list_sizes > 0))

    def test_add_keeps_implicit_faiss_row_ids(self) -> None:
        index = IVFIndex(nlist=2, nprobe=2)
        index.build(self.vectors)
        added = np.array([[20.0, 0.0]], dtype=np.float32)
        index.add(added)

        distances, ids = index.search(added, k=1)

        self.assertEqual(index.ntotal, len(self.vectors) + 1)
        self.assertEqual(ids[0, 0], len(self.vectors))
        self.assertAlmostEqual(float(distances[0, 0]), 0.0)

    def test_search_returns_l2_results_in_distance_order(self) -> None:
        index = IVFIndex(nlist=2, nprobe=2)
        index.build(self.vectors)
        query = np.array([[0.05, 0.0]], dtype=np.float32)

        distances, ids = index.search(query, k=3)
        flat = faiss.IndexFlatL2(2)
        flat.add(self.vectors)
        expected_distances, expected_ids = flat.search(query, k=3)

        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(distances, expected_distances)
        self.assertTrue(np.all(np.diff(distances[0]) >= 0))

    def test_more_probes_improves_or_maintains_recall_against_flat(self) -> None:
        index = IVFIndex(nlist=2, nprobe=1)
        index.build(self.vectors)
        query = np.array([[4.9, 0.0]], dtype=np.float32)

        flat = faiss.IndexFlatL2(2)
        flat.add(self.vectors)
        _, exact_ids = flat.search(query, k=2)

        _, one_probe_ids = index.search(query, k=2)
        index.nprobe = 2
        _, all_probe_ids = index.search(query, k=2)

        exact_set = set(exact_ids[0])
        recall_one_probe = len(exact_set.intersection(one_probe_ids[0])) / 2
        recall_all_probe = len(exact_set.intersection(all_probe_ids[0])) / 2
        self.assertLess(recall_one_probe, 1.0)
        self.assertGreaterEqual(recall_all_probe, recall_one_probe)
        self.assertEqual(recall_all_probe, 1.0)

    def test_invalid_and_empty_inputs_raise_clear_errors(self) -> None:
        with self.assertRaises(ValueError):
            IVFIndex(nlist=0)
        with self.assertRaises(ValueError):
            IVFIndex(nlist=2, nprobe=3)

        index = IVFIndex(nlist=2)
        with self.assertRaises(RuntimeError):
            index.search(np.array([[0.0, 0.0]], dtype=np.float32), k=1)
        with self.assertRaises(ValueError):
            index.build(np.empty((0, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            index.build(np.ones((1, 2), dtype=np.float32))

        index.build(self.vectors)
        with self.assertRaises(ValueError):
            index.search(np.ones((1, 3), dtype=np.float32), k=1)
        with self.assertRaises(ValueError):
            index.search(np.ones(2, dtype=np.float32), k=1)
        with self.assertRaises(ValueError):
            index.search(np.ones((1, 2), dtype=np.float32), k=0)

    def test_save_and_load_preserve_search_behavior(self) -> None:
        index = IVFIndex(nlist=2, nprobe=2)
        index.build(self.vectors)
        query = np.array([[2.1, 0.0]], dtype=np.float32)
        expected = index.search(query, k=3)

        with tempfile.TemporaryDirectory() as temporary_directory:
            saved_path = Path(temporary_directory) / "ivf.index"
            index.save(saved_path)
            loaded = IVFIndex.load(saved_path)
            actual = loaded.search(query, k=3)

        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
