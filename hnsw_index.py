"""FAISS HNSW index using the same L2 search contract as the Flat index.

The project currently uses FAISS indexes directly: ``add(vectors)`` followed by
``search(query_vectors, k)`` returning ``(distances, indices)``.  This wrapper
keeps that contract while making HNSW experiment parameters explicit.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class HNSWIndex:
    """An L2 HNSW index backed by :class:`faiss.IndexHNSWFlat`.

    Vector IDs are FAISS implicit insertion positions, matching the existing
    ``IndexFlatL2`` implementation. FAISS returns squared L2 distances and
    ``-1`` IDs for result slots that cannot be filled (for example, an empty
    index or ``k`` larger than the number of indexed vectors); corresponding
    distances are FAISS's maximum ``float32`` sentinel.

    Args:
        dimension: Number of components in each embedding.
        M: Maximum graph connectivity parameter.
        efConstruction: Candidate-list size used while building the graph.
        efSearch: Candidate-list size used while querying the graph.
    """

    def __init__(
        self,
        dimension: int,
        M: int = 32,
        efConstruction: int = 200,
        efSearch: int = 64,
    ) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(M, int) or M <= 0:
            raise ValueError("M must be a positive integer")
        if not isinstance(efConstruction, int) or efConstruction <= 0:
            raise ValueError("efConstruction must be a positive integer")
        if not isinstance(efSearch, int) or efSearch <= 0:
            raise ValueError("efSearch must be a positive integer")

        self.dimension = dimension
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = efConstruction
        self.index.hnsw.efSearch = efSearch

    @property
    def ntotal(self) -> int:
        """Number of vectors currently stored in the index."""
        return self.index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        """Add a batch of embeddings in insertion order."""
        self.index.add(self._validate_vectors(vectors, "vectors"))

    def build(self, vectors: np.ndarray) -> None:
        """Replace the current graph with one built from ``vectors``."""
        self.index.reset()
        self.add(vectors)

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return FAISS ``(squared_l2_distances, insertion_position_ids)``."""
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        queries = self._validate_vectors(query_vectors, "query_vectors")
        return self.index.search(queries, k)

    def save(self, path: str | Path) -> None:
        """Persist the underlying FAISS index in the project's native format."""
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "HNSWIndex":
        """Load an HNSW index previously persisted with :meth:`save`."""
        loaded_index = faiss.read_index(str(path))
        if not hasattr(loaded_index, "hnsw"):
            raise ValueError(f"{path} is not a FAISS HNSW index")

        instance = cls.__new__(cls)
        instance.dimension = loaded_index.d
        instance.M = loaded_index.hnsw.nb_neighbors(1)
        instance.efConstruction = loaded_index.hnsw.efConstruction
        instance.efSearch = loaded_index.hnsw.efSearch
        instance.index = loaded_index
        return instance

    def _validate_vectors(self, vectors: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(vectors)
        if array.ndim != 2:
            raise ValueError(f"{name} must have shape (n, {self.dimension})")
        if array.shape[1] != self.dimension:
            raise ValueError(
                f"{name} has dimension {array.shape[1]}; expected {self.dimension}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values")
        return np.ascontiguousarray(array, dtype=np.float32)
