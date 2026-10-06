"""Tests for the exact index and the HNSW index.

The exact index is checked against a slow, obviously-correct NumPy
calculation. The HNSW index is then checked against the exact index.
"""

import numpy as np
import pytest

from vsearch.brute_force import BruteForceIndex
from vsearch.hnsw import HNSWIndex

DIM = 32
INDEX_TYPES = [BruteForceIndex, HNSWIndex]
METRICS = ["cosine", "l2"]


def clustered(n: int, seed: int = 0) -> np.ndarray:
    """n vectors grouped around 20 centres, like real embeddings.

    The centres are always the same; `seed` only changes which points are
    drawn around them, so data and queries come from the same clusters.
    """
    centres = np.random.default_rng(0).standard_normal((20, DIM)).astype(np.float32)
    rng = np.random.default_rng(seed + 1)
    labels = rng.integers(0, 20, n)
    noise = 0.35 * rng.standard_normal((n, DIM)).astype(np.float32)
    return centres[labels] + noise


def reference_distances(data: np.ndarray, query: np.ndarray, metric: str) -> np.ndarray:
    """Distances computed the slow, simple way, to check the index against."""
    data = data.astype(np.float64)
    query = query.astype(np.float64)
    if metric == "l2":
        return ((data - query) ** 2).sum(axis=1)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    query = query / np.linalg.norm(query)
    return 1.0 - data @ query


# ----------------------------------------------------------------------
# Behaviour both indexes must share
# ----------------------------------------------------------------------


@pytest.mark.parametrize("index_type", INDEX_TYPES)
@pytest.mark.parametrize("metric", METRICS)
def test_stored_vector_is_its_own_nearest_neighbour(index_type, metric):
    data = clustered(500)
    index = index_type(dim=DIM, metric=metric)
    index.add(data)
    for i in (0, 123, 499):
        ids, dists = index.search(data[i], k=1)
        assert ids[0] == i
        assert abs(dists[0]) < 1e-4


@pytest.mark.parametrize("index_type", INDEX_TYPES)
@pytest.mark.parametrize("metric", METRICS)
def test_results_are_sorted_closest_first(index_type, metric):
    data = clustered(500)
    index = index_type(dim=DIM, metric=metric)
    index.add(data)
    ids, dists = index.search(clustered(1, seed=1)[0], k=10)
    assert len(ids) == len(dists) == 10
    assert len(set(ids.tolist())) == 10  # no id returned twice
    assert np.all(np.diff(dists) >= 0)


@pytest.mark.parametrize("index_type", INDEX_TYPES)
def test_k_larger_than_index_returns_everything(index_type):
    data = clustered(5)
    index = index_type(dim=DIM)
    index.add(data)
    ids, _ = index.search(data[0], k=50)
    assert sorted(ids.tolist()) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("index_type", INDEX_TYPES)
def test_add_accepts_single_vectors_and_batches(index_type):
    data = clustered(10)
    index = index_type(dim=DIM)
    index.add(data[0])  # one vector, shape (DIM,)
    index.add(data[1:])  # a batch, shape (9, DIM)
    assert len(index) == 10
    ids, _ = index.search(data[7], k=1)
    assert ids[0] == 7


@pytest.mark.parametrize("index_type", INDEX_TYPES)
def test_invalid_input_is_rejected(index_type):
    with pytest.raises(ValueError):
        index_type(dim=DIM, metric="manhattan")

    index = index_type(dim=DIM)
    with pytest.raises(ValueError):
        index.search(np.zeros(DIM, dtype=np.float32))  # nothing stored yet
    with pytest.raises(ValueError):
        index.add(np.zeros((3, DIM + 1), dtype=np.float32))  # wrong dimension

    index.add(clustered(10))
    with pytest.raises(ValueError):
        index.search(np.zeros(DIM + 1, dtype=np.float32))  # wrong dimension


# ----------------------------------------------------------------------
# Exact index: must agree with the reference calculation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("metric", METRICS)
def test_brute_force_matches_reference(metric):
    data = clustered(1000)
    index = BruteForceIndex(dim=DIM, metric=metric)
    index.add(data)
    for query in clustered(20, seed=1):
        expected = reference_distances(data, query, metric)
        ids, dists = index.search(query, k=10)
        best = np.sort(expected)[:10]
        assert np.allclose(expected[ids], best, atol=1e-5)  # the right vectors
        assert np.allclose(dists, best, atol=1e-4)  # with the right distances


# ----------------------------------------------------------------------
# HNSW index: accuracy and graph structure
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_hnsw():
    """One 2,000-vector HNSW index and its matching exact index."""
    data = clustered(2000)
    exact = BruteForceIndex(dim=DIM)
    exact.add(data)
    hnsw = HNSWIndex(dim=DIM)
    hnsw.add(data)
    return exact, hnsw


def recall_at_10(exact, hnsw, ef_search: int) -> float:
    queries = clustered(100, seed=1)
    hits = 0
    for query in queries:
        truth = set(exact.search(query, k=10)[0].tolist())
        found = set(hnsw.search(query, k=10, ef_search=ef_search)[0].tolist())
        hits += len(truth & found)
    return hits / (len(queries) * 10)


def test_hnsw_recall_is_high(built_hnsw):
    exact, hnsw = built_hnsw
    assert recall_at_10(exact, hnsw, ef_search=50) >= 0.95


def test_hnsw_recall_does_not_drop_with_more_search_effort(built_hnsw):
    exact, hnsw = built_hnsw
    assert recall_at_10(exact, hnsw, 100) >= recall_at_10(exact, hnsw, 10)


def test_hnsw_links_respect_limits(built_hnsw):
    _, hnsw = built_hnsw
    for node in range(len(hnsw)):
        for layer in range(hnsw.level(node) + 1):
            links = hnsw.neighbours(node, layer)
            limit = hnsw.M0 if layer == 0 else hnsw.M
            assert len(links) <= limit
            assert node not in links  # no link to itself
            assert len(links) == len(set(links))  # no repeated link
            # every neighbour must exist on this layer too
            assert all(hnsw.level(nb) >= layer for nb in links)


def test_hnsw_every_node_is_reachable_on_layer_0(built_hnsw):
    _, hnsw = built_hnsw
    seen = {hnsw._entry_point}
    frontier = [hnsw._entry_point]
    while frontier:
        node = frontier.pop()
        for nb in hnsw.neighbours(node, 0):
            if nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    assert len(seen) == len(hnsw)


def test_hnsw_same_seed_builds_same_graph():
    data = clustered(300)
    first, second = HNSWIndex(dim=DIM, seed=7), HNSWIndex(dim=DIM, seed=7)
    first.add(data)
    second.add(data)
    for node in range(len(first)):
        assert first.level(node) == second.level(node)
        for layer in range(first.level(node) + 1):
            assert first.neighbours(node, layer) == second.neighbours(node, layer)