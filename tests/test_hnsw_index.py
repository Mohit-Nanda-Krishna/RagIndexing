"""Tests for the HNSW implementation's Flat-compatible FAISS API."""

import unittest

import faiss
import numpy as np

from hnsw_index import HNSWIndex


class HNSWIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vectors = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [4.0, 4.0]], dtype=np.float32
        )

    def test_creation_exposes_configurable_hnsw_parameters(self) -> None:
        index = HNSWIndex(2, M=8, efConstruction=48, efSearch=32)

        self.assertEqual(index.dimension, 2)
        self.assertEqual(index.M, 8)
        self.assertEqual(index.efConstruction, 48)
        self.assertEqual(index.efSearch, 32)
        self.assertEqual(index.ntotal, 0)

    def test_empty_index_returns_faiss_missing_result_markers(self) -> None:
        index = HNSWIndex(2)
        distances, ids = index.search(np.array([[0.0, 0.0]], dtype=np.float32), k=2)

        np.testing.assert_array_equal(ids, np.array([[-1, -1]]))
        np.testing.assert_array_equal(
            distances,
            np.full((1, 2), np.finfo(np.float32).max, dtype=np.float32),
        )

    def test_add_build_and_search_return_insertion_position_ids(self) -> None:
        index = HNSWIndex(2, M=8, efConstruction=64, efSearch=64)
        index.add(self.vectors[:2])
        index.add(self.vectors[2:])

        distances, ids = index.search(np.array([[0.1, 0.0]], dtype=np.float32), k=3)

        self.assertEqual(index.ntotal, 4)
        np.testing.assert_array_equal(ids[0], np.array([0, 1, 2]))
        np.testing.assert_allclose(distances[0], np.array([0.01, 0.81, 4.01]))

        index.build(self.vectors[:1])
        self.assertEqual(index.ntotal, 1)

    def test_validates_vector_dimension_and_top_k(self) -> None:
        index = HNSWIndex(2)
        with self.assertRaisesRegex(ValueError, "dimension 3"):
            index.add(np.zeros((1, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "positive"):
            index.search(np.zeros((1, 2), dtype=np.float32), k=0)

    def test_high_ef_search_matches_flat_on_small_dataset(self) -> None:
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(80, 5)).astype(np.float32)
        queries = rng.normal(size=(6, 5)).astype(np.float32)
        hnsw = HNSWIndex(5, M=16, efConstruction=128, efSearch=128)
        hnsw.build(vectors)
        flat = faiss.IndexFlatL2(5)
        flat.add(vectors)

        hnsw_distances, hnsw_ids = hnsw.search(queries, k=10)
        flat_distances, flat_ids = flat.search(queries, k=10)

        np.testing.assert_array_equal(hnsw_ids, flat_ids)
        np.testing.assert_allclose(hnsw_distances, flat_distances)


if __name__ == "__main__":
    unittest.main()
