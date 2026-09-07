"""Deterministic tests for standalone FAISS Product Quantization."""

import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from pq_index import PQIndex


class PQIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        # 624 is FAISS's recommended minimum for 16-centroid codebooks.
        self.vectors = rng.normal(size=(640, 4)).astype(np.float32)

    def test_construction_exposes_compression_configuration(self) -> None:
        index = PQIndex(4, m=2, nbits=3)
        self.assertEqual(index.codebook_shape, (2, 8, 2))
        self.assertEqual(index.code_size, 1)
        self.assertEqual(index.ntotal, 0)
        self.assertFalse(index.is_trained)

    def test_build_trains_codebooks_and_stores_compressed_codes(self) -> None:
        index = PQIndex(4, m=2, nbits=3)
        index.build(self.vectors)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.ntotal, len(self.vectors))
        self.assertEqual(index.compressed_size_bytes, len(self.vectors) * index.code_size)
        centroids = faiss.vector_to_array(index.index.pq.centroids)
        self.assertEqual(centroids.size, 2 * 8 * 2)

    def test_add_preserves_implicit_ids_and_search_returns_ordered_results(self) -> None:
        index = PQIndex(4, m=2, nbits=3)
        index.build(self.vectors)
        added = np.array([[30.0, -20.0, 10.0, 5.0]], dtype=np.float32)
        index.add(added)
        distances, ids = index.search(added, k=index.ntotal)
        self.assertIn(len(self.vectors), ids[0])
        self.assertSetEqual(set(ids[0]), set(range(index.ntotal)))
        # PQ compares against reconstructed code vectors, so even a vector
        # queried against itself can have a non-zero approximate distance.
        self.assertGreaterEqual(float(distances[0, 0]), 0.0)
        self.assertTrue(np.all(np.diff(distances[0]) >= 0))

    def test_pq_is_approximate_but_returns_flat_comparable_ids(self) -> None:
        index = PQIndex(4, m=2, nbits=3)
        index.build(self.vectors)
        queries = self.vectors[:8]
        _, pq_ids = index.search(queries, k=5)
        flat = faiss.IndexFlatL2(4)
        flat.add(self.vectors)
        _, flat_ids = flat.search(queries, k=5)
        recalls = [len(set(a).intersection(b)) / 5 for a, b in zip(pq_ids, flat_ids)]
        self.assertTrue(all(0.0 <= recall <= 1.0 for recall in recalls))
        self.assertGreater(np.mean(recalls), 0.0)

    def test_invalid_parameters_dimensions_and_untrained_operations_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            PQIndex(5, m=2)
        with self.assertRaisesRegex(ValueError, "between"):
            PQIndex(4, m=2, nbits=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            PQIndex(4, m=0)

        index = PQIndex(4, m=2, nbits=3)
        with self.assertRaises(RuntimeError):
            index.add(self.vectors)
        with self.assertRaises(RuntimeError):
            index.search(self.vectors[:1], 1)
        with self.assertRaisesRegex(ValueError, "at least 8"):
            index.train(self.vectors[:7])
        with self.assertRaisesRegex(ValueError, "dimension 3"):
            index.build(np.ones((16, 3), dtype=np.float32))

    def test_parameter_configurations_and_save_load(self) -> None:
        index = PQIndex(4, m=1, nbits=4)
        index.build(self.vectors)
        query = self.vectors[:2]
        expected = index.search(query, 4)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pq.index"
            index.save(path)
            loaded = PQIndex.load(path)
            actual = loaded.search(query, 4)
        self.assertEqual((loaded.m, loaded.nbits), (1, 4))
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
