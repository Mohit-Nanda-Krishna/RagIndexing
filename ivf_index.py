"""FAISS IVF-Flat index compatible with this project's Flat L2 baseline.

The existing baseline uses ``faiss.IndexFlatL2`` and returns FAISS's raw
``(distances, indices)`` tuple.  ``IVFIndex`` intentionally keeps those
vector, distance, ID, and search-result conventions while exposing IVF's
``nlist`` and ``nprobe`` experiment parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import faiss
import numpy as np


ArrayLike = np.ndarray
SearchResults = Tuple[np.ndarray, np.ndarray]


class IVFIndex:
    """An L2 FAISS ``IndexIVFFlat`` index with configurable probing.

    Parameters
    ----------
    nlist:
        Number of coarse clusters (inverted lists) to train.
    nprobe:
        Number of nearest inverted lists to search per query.  It must not
        exceed ``nlist``.  Larger values trade latency for recall; setting it
        equal to ``nlist`` searches every list and is exact for IVF-Flat.
    dimension:
        Optional expected embedding dimension.  If omitted, it is inferred
        from the first call to :meth:`build`.

    Notes
    -----
    IDs are FAISS's implicit row IDs, matching the existing Flat baseline:
    the first vector added has ID 0, the next has ID 1, and so on.
    """

    def __init__(
        self, nlist: int = 100, nprobe: int = 1, dimension: Optional[int] = None
    ) -> None:
        if not isinstance(nlist, int) or isinstance(nlist, bool) or nlist < 1:
            raise ValueError("nlist must be a positive integer")
        if (
            not isinstance(nprobe, int)
            or isinstance(nprobe, bool)
            or nprobe < 1
            or nprobe > nlist
        ):
            raise ValueError("nprobe must be a positive integer no greater than nlist")
        if (
            dimension is not None
            and (not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1)
        ):
            raise ValueError("dimension must be a positive integer when provided")

        self.nlist = nlist
        self.dimension = dimension
        self._nprobe = nprobe
        self._index: Optional[faiss.IndexIVFFlat] = None
        # FAISS's IVF index references its quantizer; retain the Python object
        # as well so it cannot be garbage-collected while the IVF index is used.
        self._quantizer: Optional[faiss.Index] = None

    @property
    def nprobe(self) -> int:
        """Number of inverted lists searched for each query."""
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            or value > self.nlist
        ):
            raise ValueError("nprobe must be a positive integer no greater than nlist")
        self._nprobe = value
        if self._index is not None:
            self._index.nprobe = value

    @property
    def is_trained(self) -> bool:
        """Whether coarse centroids have been trained."""
        return self._index is not None and self._index.is_trained

    @property
    def ntotal(self) -> int:
        """Number of vectors currently stored in the index."""
        return 0 if self._index is None else self._index.ntotal

    @property
    def list_sizes(self) -> np.ndarray:
        """Number of vectors assigned to each inverted list."""
        self._require_index()
        return np.asarray(
            [self._index.invlists.list_size(list_id) for list_id in range(self.nlist)],
            dtype=np.int64,
        )

    def build(self, vectors: ArrayLike) -> None:
        """Train the coarse quantizer and add all corpus vectors.

        ``vectors`` must be a non-empty 2D numeric array with at least one
        row per requested inverted list.  IVF needs that minimum to train a
        distinct centroid for each list.
        """
        prepared = self._prepare_vectors(vectors, "vectors")
        if len(prepared) < self.nlist:
            raise ValueError("nlist cannot exceed the number of training vectors")

        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        index.train(prepared)
        index.add(prepared)
        index.nprobe = self.nprobe
        self._quantizer = quantizer
        self._index = index

    def add(self, vectors: ArrayLike) -> None:
        """Add vectors after :meth:`build` has trained the coarse quantizer."""
        self._require_index()
        prepared = self._prepare_vectors(vectors, "vectors")
        self._index.add(prepared)

    def search(self, query_vectors: ArrayLike, k: int) -> SearchResults:
        """Return FAISS L2 ``(distances, indices)`` for a batch of queries."""
        self._require_index()
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        prepared = self._prepare_vectors(query_vectors, "query_vectors")
        return self._index.search(prepared, k)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the trained index using FAISS's native index format."""
        self._require_index()
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Union[str, Path], nprobe: Optional[int] = None) -> "IVFIndex":
        """Load an index saved with :meth:`save`.

        ``nprobe`` may override the saved probing configuration for an
        experiment; otherwise the saved value is retained.
        """
        index = faiss.read_index(str(path))
        if not isinstance(index, faiss.IndexIVF):
            raise ValueError("the provided file does not contain a FAISS IVF index")

        selected_nprobe = index.nprobe if nprobe is None else nprobe
        instance = cls(nlist=index.nlist, nprobe=selected_nprobe, dimension=index.d)
        # ``read_index`` already returns the concrete IVF object.  Keeping
        # that object directly avoids creating another SWIG ownership wrapper
        # around its quantizer.
        instance._index = index
        instance._index.nprobe = instance.nprobe
        return instance

    def _prepare_vectors(self, vectors: ArrayLike, name: str) -> np.ndarray:
        array = np.asarray(vectors)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
        if array.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one vector")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain numeric values")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must not contain NaN or infinite values")

        if self.dimension is None:
            self.dimension = array.shape[1]
        if array.shape[1] != self.dimension:
            raise ValueError(
                f"{name} has dimension {array.shape[1]}, expected {self.dimension}"
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    def _require_index(self) -> None:
        if self._index is None:
            raise RuntimeError("build must be called before this operation")
