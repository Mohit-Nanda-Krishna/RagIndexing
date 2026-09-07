"""Standalone FAISS Product Quantization index with the project's L2 contract."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import faiss
import numpy as np


SearchResults = Tuple[np.ndarray, np.ndarray]


class PQIndex:
    """An L2 ``faiss.IndexPQ`` with configurable compression parameters.

    ``m`` splits each vector into that many equally sized subvectors.  Each
    subvector is encoded with a codebook containing ``2 ** nbits`` centroids.
    Consequently, ``dimension`` must be divisible by ``m`` and each indexed
    vector occupies ``m * nbits / 8`` bytes (rounded up by FAISS).

    IDs are FAISS implicit insertion IDs, matching the project's Flat, IVF,
    and HNSW wrappers. Search returns FAISS's ``(squared_l2_distances, ids)``.
    """

    def __init__(self, dimension: int, m: int = 48, nbits: int = 4) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(m, int) or isinstance(m, bool) or m < 1:
            raise ValueError("m must be a positive integer")
        if dimension % m != 0:
            raise ValueError("dimension must be divisible by m")
        if not isinstance(nbits, int) or isinstance(nbits, bool) or not 1 <= nbits <= 16:
            raise ValueError("nbits must be an integer between 1 and 16")

        self.dimension = dimension
        self.m = m
        self.nbits = nbits
        self.index = self._new_index()

    def _new_index(self) -> faiss.IndexPQ:
        return faiss.IndexPQ(self.dimension, self.m, self.nbits, faiss.METRIC_L2)

    @property
    def is_trained(self) -> bool:
        """Whether the PQ codebooks have been trained."""
        return bool(self.index.is_trained)

    @property
    def ntotal(self) -> int:
        """Number of compressed vectors stored in the index."""
        return self.index.ntotal

    @property
    def code_size(self) -> int:
        """Bytes used for one compressed vector code."""
        return self.index.code_size

    @property
    def compressed_size_bytes(self) -> int:
        """Size of stored PQ codes, excluding FAISS/codebook metadata."""
        return self.ntotal * self.code_size

    @property
    def codebook_shape(self) -> tuple[int, int, int]:
        """Return ``(m, 2**nbits, dimension/m)`` for the learned codebooks."""
        return (self.m, 1 << self.nbits, self.dimension // self.m)

    def train(self, vectors: np.ndarray) -> None:
        """Train the per-subvector codebooks from representative embeddings."""
        prepared = self._validate_vectors(vectors, "vectors")
        minimum_training_vectors = 1 << self.nbits
        if len(prepared) < minimum_training_vectors:
            raise ValueError(
                f"PQ training needs at least {minimum_training_vectors} vectors for nbits={self.nbits}"
            )
        self.index.train(prepared)

    def add(self, vectors: np.ndarray) -> None:
        """Encode and add vectors after the codebooks have been trained."""
        if not self.is_trained:
            raise RuntimeError("train or build must be called before add")
        self.index.add(self._validate_vectors(vectors, "vectors"))

    def build(self, vectors: np.ndarray) -> None:
        """Train fresh codebooks and encode all supplied corpus vectors."""
        prepared = self._validate_vectors(vectors, "vectors")
        self.index = self._new_index()
        self.train(prepared)
        self.index.add(prepared)

    def search(self, query_vectors: np.ndarray, k: int) -> SearchResults:
        """Return approximate squared-L2 ``(distances, implicit_ids)``."""
        if not self.is_trained:
            raise RuntimeError("train or build must be called before search")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        return self.index.search(self._validate_vectors(query_vectors, "query_vectors"), k)

    def save(self, path: Union[str, Path]) -> None:
        """Persist compressed codes and trained codebooks in FAISS format."""
        if not self.is_trained:
            raise RuntimeError("train or build must be called before save")
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PQIndex":
        """Load a standalone FAISS ``IndexPQ`` persisted by :meth:`save`."""
        loaded = faiss.read_index(str(path))
        if not isinstance(loaded, faiss.IndexPQ):
            raise ValueError("the provided file does not contain a FAISS IndexPQ")

        instance = cls.__new__(cls)
        instance.dimension = loaded.d
        instance.m = loaded.pq.M
        instance.nbits = loaded.pq.nbits
        instance.index = loaded
        return instance

    def _validate_vectors(self, vectors: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(vectors)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
        if array.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one vector")
        if array.shape[1] != self.dimension:
            raise ValueError(
                f"{name} has dimension {array.shape[1]}, expected {self.dimension}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must not contain NaN or infinite values")
        return np.ascontiguousarray(array, dtype=np.float32)
