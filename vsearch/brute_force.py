"""Exact (brute-force) nearest-neighbour search.

This is the ground-truth baseline: it compares the query against every
stored vector, so its answers are always correct. The HNSW index is
measured against it for both accuracy (recall) and speed.
"""

from __future__ import annotations

import numpy as np

METRICS = ("cosine", "l2")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # leave all-zero vectors unchanged
    return vectors / norms


class BruteForceIndex:
    """Stores vectors in one matrix and scans all of them on every query."""

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        self.dim = dim
        self.metric = metric
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return self._vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        """Add one vector of shape (dim,) or many of shape (n, dim)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected vectors of dimension {self.dim}, got shape {vectors.shape}"
            )
        if self.metric == "cosine":
            vectors = _normalize(vectors)
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) of the k nearest vectors, closest first.

        Distances are (1 - cosine similarity) for "cosine" and squared
        Euclidean distance for "l2". Smaller always means closer.
        """
        if len(self) == 0:
            raise ValueError("cannot search an empty index")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(
                f"expected a query of shape ({self.dim},), got {query.shape}"
            )
        k = min(k, len(self))

        if self.metric == "cosine":
            query = _normalize(query[None, :])[0]
            distances = 1.0 - self._vectors @ query
        else:
            diff = self._vectors - query
            distances = np.einsum("ij,ij->i", diff, diff)

        # argpartition finds the k smallest in O(n) without sorting everything;
        # then only those k are sorted.
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest = nearest[np.argsort(distances[nearest])]
        return nearest, distances[nearest]