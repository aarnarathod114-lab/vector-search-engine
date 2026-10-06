"""HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.

A from-scratch implementation of the algorithm described by Malkov and
Yashunin (2016). Vectors are nodes in a stack of graphs: sparse upper
layers for long jumps, a dense bottom layer for precise search.

The file has two halves. The functions marked @njit are the hot loops;
Numba compiles them to machine code the first time they run. They only
touch NumPy arrays and numbers, which is what lets Numba compile them.
The HNSWIndex class below them owns those arrays and is ordinary Python.

How the graph is stored:
  links0[node]          neighbour ids of `node` on layer 0 (up to 2*M)
  degree0[node]         how many slots of that row are in use
  upper_links[row]      neighbour ids on a layer above 0 (up to M)
  upper_degree[row]     how many slots of that row are in use
  upper_offset[node]    row of `node` on layer 1; layer L is row + L - 1
  visited[node]         equals the current search's tag if already seen
"""

from __future__ import annotations

import heapq
import math
import random

import numpy as np
from numba import njit

from vsearch.brute_force import METRICS, _normalize

_INITIAL_CAPACITY = 1024


# ----------------------------------------------------------------------
# Compiled kernels
# ----------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _distance(vectors, query, node, use_l2):
    """Distance from `query` to the stored vector `node`."""
    row = vectors[node]
    total = np.float32(0.0)
    if use_l2:
        for i in range(query.shape[0]):
            diff = row[i] - query[i]
            total += diff * diff
        return total
    for i in range(query.shape[0]):
        total += row[i] * query[i]
    return np.float32(1.0) - total


@njit(cache=True)
def _search_layer(query, entry_ids, entry_dists, ef, layer, graph, tag, use_l2):
    """Best-first search on one layer.

    Starts from the entry nodes and returns (ids, distances) of up to `ef`
    nodes, closest first. `candidates` is a min-heap of nodes still to
    expand; `results` is a max-heap (distances negated) of the best so far.
    """
    vectors, links0, degree0, upper_links, upper_degree, upper_offset, visited = graph
    if layer == 0:
        links, degree = links0, degree0
    else:
        links, degree = upper_links, upper_degree

    candidates = [(entry_dists[0], entry_ids[0])]
    results = [(-entry_dists[0], entry_ids[0])]
    visited[entry_ids[0]] = tag
    for i in range(1, entry_ids.shape[0]):
        visited[entry_ids[i]] = tag
        heapq.heappush(candidates, (entry_dists[i], entry_ids[i]))
        heapq.heappush(results, (-entry_dists[i], entry_ids[i]))
    while len(results) > ef:
        heapq.heappop(results)

    while len(candidates) > 0:
        dist, node = heapq.heappop(candidates)
        # Stop when the closest unexpanded node is worse than our worst result.
        if len(results) >= ef and dist > -results[0][0]:
            break
        row = node
        if layer > 0:
            row = upper_offset[node] + layer - 1
        for j in range(degree[row]):
            nb = links[row, j]
            if visited[nb] == tag:
                continue
            visited[nb] = tag
            d = _distance(vectors, query, nb, use_l2)
            if len(results) < ef or d < -results[0][0]:
                heapq.heappush(candidates, (d, nb))
                heapq.heappush(results, (-d, nb))
                if len(results) > ef:
                    heapq.heappop(results)

    # Popping a max-heap gives the farthest first, so fill from the back.
    n = len(results)
    ids = np.empty(n, dtype=np.int64)
    dists = np.empty(n, dtype=np.float32)
    for i in range(n - 1, -1, -1):
        neg_dist, node = heapq.heappop(results)
        ids[i] = node
        dists[i] = -neg_dist
    return ids, dists


@njit(cache=True)
def _select_neighbours(cand_ids, cand_dists, m, vectors, use_l2):
    """Pick up to `m` links from candidates that are sorted closest first.

    Heuristic from the paper: keep a candidate only if it is closer to
    the base node than to every neighbour already kept. This spreads
    links in different directions instead of bunching them in one
    cluster, which keeps the graph navigable.
    """
    kept = np.empty(m, dtype=np.int64)
    n_kept = 0
    for i in range(cand_ids.shape[0]):
        if n_kept == m:
            break
        candidate = cand_ids[i]
        keep = True
        for j in range(n_kept):
            if _distance(vectors, vectors[candidate], kept[j], use_l2) <= cand_dists[i]:
                keep = False
                break
        if keep:
            kept[n_kept] = candidate
            n_kept += 1
    return kept[:n_kept]


@njit(cache=True)
def _add_link(node, new, layer, graph, use_l2):
    """Link `node` to `new`; if `node` is already full, re-pick its links."""
    vectors, links0, degree0, upper_links, upper_degree, upper_offset, visited = graph
    if layer == 0:
        links, degree, row = links0, degree0, node
    else:
        links, degree, row = upper_links, upper_degree, upper_offset[node] + layer - 1

    limit = links.shape[1]
    count = degree[row]
    if count < limit:
        links[row, count] = new
        degree[row] = count + 1
        return

    pool = np.empty(count + 1, dtype=np.int64)
    pool_dists = np.empty(count + 1, dtype=np.float32)
    for i in range(count):
        pool[i] = links[row, i]
    pool[count] = new
    for i in range(count + 1):
        pool_dists[i] = _distance(vectors, vectors[node], pool[i], use_l2)
    order = np.argsort(pool_dists)
    chosen = _select_neighbours(pool[order], pool_dists[order], limit, vectors, use_l2)
    for i in range(chosen.shape[0]):
        links[row, i] = chosen[i]
    degree[row] = chosen.shape[0]


@njit(cache=True)
def _link_node(node, level, entry_point, max_layer, ef_construction, m, graph, tag, use_l2):
    """Link the already-stored vector `node` into every layer up to `level`."""
    vectors, links0, degree0, upper_links, upper_degree, upper_offset, visited = graph
    query = vectors[node]
    ids = np.empty(1, dtype=np.int64)
    dists = np.empty(1, dtype=np.float32)
    ids[0] = entry_point
    dists[0] = _distance(vectors, query, entry_point, use_l2)

    # Step 1: greedy descent through the layers above this node's top layer.
    for layer in range(max_layer, level, -1):
        tag += 1
        ids, dists = _search_layer(query, ids, dists, 1, layer, graph, tag, use_l2)

    # Step 2: on every layer the node lives on, find neighbours and link.
    for layer in range(min(level, max_layer), -1, -1):
        tag += 1
        ids, dists = _search_layer(
            query, ids, dists, ef_construction, layer, graph, tag, use_l2
        )
        chosen = _select_neighbours(ids, dists, m, vectors, use_l2)
        if layer == 0:
            links, degree, row = links0, degree0, node
        else:
            links, degree, row = upper_links, upper_degree, upper_offset[node] + layer - 1
        for i in range(chosen.shape[0]):
            links[row, i] = chosen[i]
        degree[row] = chosen.shape[0]
        for i in range(chosen.shape[0]):
            _add_link(chosen[i], node, layer, graph, use_l2)


@njit(cache=True)
def _knn_search(query, k, ef, entry_point, max_layer, graph, tag, use_l2):
    """Ride the upper layers down, then search layer 0 with `ef` candidates."""
    vectors = graph[0]
    ids = np.empty(1, dtype=np.int64)
    dists = np.empty(1, dtype=np.float32)
    ids[0] = entry_point
    dists[0] = _distance(vectors, query, entry_point, use_l2)
    for layer in range(max_layer, 0, -1):
        tag += 1
        ids, dists = _search_layer(query, ids, dists, 1, layer, graph, tag, use_l2)
    tag += 1
    ids, dists = _search_layer(query, ids, dists, ef, 0, graph, tag, use_l2)
    return ids[:k], dists[:k]


# ----------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------


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
        self._use_l2 = metric == "l2"
        self._level_mult = 1.0 / math.log(M)
        self._rng = random.Random(seed)

        cap = _INITIAL_CAPACITY
        self._count = 0
        self._vectors = np.zeros((cap, dim), dtype=np.float32)
        self._links0 = np.zeros((cap, self.M0), dtype=np.int64)
        self._degree0 = np.zeros(cap, dtype=np.int64)
        self._levels = np.zeros(cap, dtype=np.int64)
        self._upper_offset = np.full(cap, -1, dtype=np.int64)
        self._visited = np.zeros(cap, dtype=np.int64)
        self._upper_rows = 0  # rows of _upper_links in use
        self._upper_links = np.zeros((cap, M), dtype=np.int64)
        self._upper_degree = np.zeros(cap, dtype=np.int64)
        # Each layer search uses a fresh tag, so `visited` never needs clearing.
        self._tag = 0
        self._entry_point = -1
        self._max_layer = -1

    def __len__(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Graph storage
    # ------------------------------------------------------------------

    def _graph(self):
        """The arrays the compiled kernels work on, bundled as one argument."""
        return (
            self._vectors,
            self._links0,
            self._degree0,
            self._upper_links,
            self._upper_degree,
            self._upper_offset,
            self._visited,
        )

    def level(self, node: int) -> int:
        """Highest layer that `node` appears on."""
        return int(self._levels[node])

    def neighbours(self, node: int, layer: int) -> list[int]:
        """Ids of the nodes that `node` links to on `layer`."""
        if layer == 0:
            return self._links0[node, : self._degree0[node]].tolist()
        row = self._upper_offset[node] + layer - 1
        return self._upper_links[row, : self._upper_degree[row]].tolist()

    @staticmethod
    def _doubled(array: np.ndarray) -> np.ndarray:
        """A copy of `array` with twice as many rows; new rows are zero."""
        return np.concatenate([array, np.zeros_like(array)])

    def _random_level(self) -> int:
        """Top layer for a new node; each higher layer is exponentially rarer."""
        return int(-math.log(1.0 - self._rng.random()) * self._level_mult)

    def _insert(self, vector: np.ndarray) -> None:
        node = self._count
        if node == self._vectors.shape[0]:  # out of room for nodes
            self._vectors = self._doubled(self._vectors)
            self._links0 = self._doubled(self._links0)
            self._degree0 = self._doubled(self._degree0)
            self._levels = self._doubled(self._levels)
            self._upper_offset = self._doubled(self._upper_offset)
            self._visited = self._doubled(self._visited)

        level = self._random_level()
        while self._upper_rows + level > self._upper_links.shape[0]:
            self._upper_links = self._doubled(self._upper_links)
            self._upper_degree = self._doubled(self._upper_degree)

        self._vectors[node] = vector
        self._levels[node] = level
        self._upper_offset[node] = self._upper_rows  # one row per upper layer
        self._upper_rows += level
        self._count += 1

        if self._entry_point == -1:  # very first node
            self._entry_point, self._max_layer = node, level
            return

        _link_node(
            node,
            level,
            self._entry_point,
            self._max_layer,
            self.ef_construction,
            self.M,
            self._graph(),
            self._tag,
            self._use_l2,
        )
        self._tag += self._max_layer + 2  # the kernel used at most this many tags

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
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(
                f"expected a query of shape ({self.dim},), got {query.shape}"
            )
        if self.metric == "cosine":
            query = _normalize(query[None, :])[0]
        ef = max(ef_search or self.ef_search, k)

        ids, dists = _knn_search(
            query,
            k,
            ef,
            self._entry_point,
            self._max_layer,
            self._graph(),
            self._tag,
            self._use_l2,
        )
        self._tag += self._max_layer + 2
        return ids, dists