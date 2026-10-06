"""HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.

A from-scratch implementation of the algorithm described by Malkov and
Yashunin (2016). Vectors are nodes in a stack of graphs: sparse upper
layers for long jumps, a dense bottom layer for precise search.
"""

from __future__ import annotations

import heapq
import math
import random

import numpy as np

from vsearch.brute_force import METRICS, _normalize


class HNSWIndex:
    """Approximate nearest-neighbour index with the same API as BruteForceIndex."""

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if M < 2:
            raise ValueError("M must be at least 2")
        self.dim = dim
        self.metric = metric
        self.M = M  # max links per node on layers above 0
        self.M0 = 2 * M  # layer 0 is denser, so it gets double
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._level_mult = 1.0 / math.log(M)
        self._rng = random.Random(seed)

        self._vectors = np.empty((1024, dim), dtype=np.float32)
        self._count = 0
        # _links[node][layer] is the list of neighbour ids of `node` on `layer`.
        self._links: list[list[list[int]]] = []
        self._entry_point = -1
        self._max_layer = -1

    def __len__(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _distances(self, query: np.ndarray, ids: list[int]) -> np.ndarray:
        """Distance from `query` to each node in `ids`, in one NumPy call."""
        vecs = self._vectors[ids]
        if self.metric == "cosine":
            return 1.0 - vecs @ query
        diff = vecs - query
        return np.einsum("ij,ij->i", diff, diff)

    def _pairwise(self, ids: list[int]) -> np.ndarray:
        """Matrix of distances between every pair of nodes in `ids`."""
        vecs = self._vectors[ids]
        if self.metric == "cosine":
            return 1.0 - vecs @ vecs.T
        sq = np.einsum("ij,ij->i", vecs, vecs)
        return sq[:, None] + sq[None, :] - 2.0 * (vecs @ vecs.T)

    # ------------------------------------------------------------------
    # Core graph routines
    # ------------------------------------------------------------------

    def _search_layer(
        self,
        query: np.ndarray,
        entry: list[tuple[float, int]],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Best-first search on one layer.

        Starts from `entry` and returns up to `ef` (distance, id) pairs,
        closest first. `candidates` is a min-heap of nodes still to expand;
        `results` is a max-heap (distances negated) of the best `ef` so far.
        """
        visited = {node for _, node in entry}
        candidates = list(entry)
        heapq.heapify(candidates)
        results = [(-dist, node) for dist, node in entry]
        heapq.heapify(results)

        while candidates:
            dist, node = heapq.heappop(candidates)
            # Stop when the closest unexpanded node is worse than our worst result.
            if len(results) >= ef and dist > -results[0][0]:
                break
            neighbours = [n for n in self._links[node][layer] if n not in visited]
            if not neighbours:
                continue
            visited.update(neighbours)
            dists = self._distances(query, neighbours).tolist()
            for d, n in zip(dists, neighbours):
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted((-neg_dist, node) for neg_dist, node in results)

    def _select_neighbours(
        self, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """Pick up to `m` links from `candidates` (sorted closest first).

        Heuristic from the paper: keep a candidate only if it is closer to
        the base node than to every neighbour already kept. This spreads
        links in different directions instead of bunching them in one
        cluster, which keeps the graph navigable.
        """
        ids = [node for _, node in candidates]
        pair = self._pairwise(ids).tolist()
        kept: list[int] = []  # positions into `candidates`
        for i, (dist, _) in enumerate(candidates):
            if len(kept) == m:
                break
            row = pair[i]
            if all(row[j] > dist for j in kept):
                kept.append(i)
        return [ids[i] for i in kept]

    def _random_level(self) -> int:
        """Top layer for a new node; each higher layer is exponentially rarer."""
        return int(-math.log(1.0 - self._rng.random()) * self._level_mult)

    def _insert(self, vector: np.ndarray) -> None:
        node = self._count
        if node == self._vectors.shape[0]:  # out of room: double the storage
            grown = np.empty((2 * node, self.dim), dtype=np.float32)
            grown[:node] = self._vectors
            self._vectors = grown
        self._vectors[node] = vector
        self._count += 1

        level = self._random_level()
        self._links.append([[] for _ in range(level + 1)])

        if self._entry_point == -1:  # very first node
            self._entry_point, self._max_layer = node, level
            return

        ep = self._entry_point
        entry = [(float(self._distances(vector, [ep])[0]), ep)]

        # Step 1: greedy descent through the layers above this node's top layer.
        for layer in range(self._max_layer, level, -1):
            entry = self._search_layer(vector, entry, 1, layer)

        # Step 2: on every layer the node lives on, find neighbours and link.
        for layer in range(min(level, self._max_layer), -1, -1):
            candidates = self._search_layer(
                vector, entry, self.ef_construction, layer
            )
            neighbours = self._select_neighbours(candidates, self.M)
            self._links[node][layer] = neighbours

            max_links = self.M0 if layer == 0 else self.M
            for nb in neighbours:
                nb_links = self._links[nb][layer]
                nb_links.append(node)
                if len(nb_links) > max_links:  # too many links: re-pick the best
                    dists = self._distances(self._vectors[nb], nb_links).tolist()
                    ranked = sorted(zip(dists, nb_links))
                    self._links[nb][layer] = self._select_neighbours(
                        ranked, max_links
                    )
            entry = candidates

        if level > self._max_layer:  # new node becomes the top entry point
            self._entry_point, self._max_layer = node, level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        for vector in vectors:
            self._insert(vector)

    def search(
        self, query: np.ndarray, k: int = 10, ef_search: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) of roughly the k nearest vectors, closest first."""
        if self._count == 0:
            raise ValueError("cannot search an empty index")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(
                f"expected a query of shape ({self.dim},), got {query.shape}"
            )
        if self.metric == "cosine":
            query = _normalize(query[None, :])[0]
        ef = max(ef_search or self.ef_search, k)

        ep = self._entry_point
        entry = [(float(self._distances(query, [ep])[0]), ep)]
        for layer in range(self._max_layer, 0, -1):
            entry = self._search_layer(query, entry, 1, layer)
        results = self._search_layer(query, entry, ef, 0)[:k]

        ids = np.array([node for _, node in results], dtype=np.int64)
        dists = np.array([dist for dist, _ in results], dtype=np.float32)
        return ids, dists