"""Compare this project's HNSW index with FAISS's HNSW index.

FAISS is Meta's C++ similarity-search library and the usual reference
point. Both indexes get the same data, the same graph settings and one
CPU thread, and answer one query at a time, so the numbers are comparable.
"""

import time

import faiss
import numpy as np

from vsearch.brute_force import BruteForceIndex
from vsearch.hnsw import HNSWIndex

N, DIM, K, QUERIES, CLUSTERS = 100_000, 128, 10, 500, 100
M, EF_CONSTRUCTION = 16, 100

faiss.omp_set_num_threads(1)

rng = np.random.default_rng(42)
centres = rng.standard_normal((CLUSTERS, DIM)).astype(np.float32)
labels = rng.integers(0, CLUSTERS, N + QUERIES)
points = centres[labels] + 0.35 * rng.standard_normal((N + QUERIES, DIM)).astype(
    np.float32
)
# Scale every vector to length 1. On unit vectors, FAISS's default L2
# distance ranks neighbours exactly as cosine distance does.
points = np.ascontiguousarray(points / np.linalg.norm(points, axis=1, keepdims=True))
data, queries = points[:N], points[N:]

exact = BruteForceIndex(dim=DIM, metric="cosine")
exact.add(data)
truth = [set(exact.search(q, k=K)[0].tolist()) for q in queries]

# Build a tiny index first so Numba's one-off compile is not timed.
warm = HNSWIndex(dim=DIM)
warm.add(data[:100])
warm.search(queries[0], k=K)

ours = HNSWIndex(dim=DIM, metric="cosine", M=M, ef_construction=EF_CONSTRUCTION)
start = time.perf_counter()
ours.add(data)
ours_build = time.perf_counter() - start

theirs = faiss.IndexHNSWFlat(DIM, M)
theirs.hnsw.efConstruction = EF_CONSTRUCTION
start = time.perf_counter()
theirs.add(data)
faiss_build = time.perf_counter() - start


def recall(found: list[set[int]]) -> float:
    return sum(len(f & t) for f, t in zip(found, truth)) / (QUERIES * K)


print(f"vectors: {N:,}   dim: {DIM}   k: {K}   M: {M}   threads: 1")
print(f"build time   ours: {ours_build:.1f} s   faiss: {faiss_build:.1f} s")
print()
print("ef_search   ours recall   ours ms   faiss recall   faiss ms")
for ef in (10, 20, 50, 100, 200):
    start = time.perf_counter()
    found = [set(ours.search(q, k=K, ef_search=ef)[0].tolist()) for q in queries]
    ours_ms = (time.perf_counter() - start) / QUERIES * 1000
    ours_recall = recall(found)

    theirs.hnsw.efSearch = ef
    start = time.perf_counter()
    found = [set(theirs.search(q[None, :], K)[1][0].tolist()) for q in queries]
    faiss_ms = (time.perf_counter() - start) / QUERIES * 1000

    print(
        f"{ef:>9}   {ours_recall:>11.3f}   {ours_ms:>7.3f}"
        f"   {recall(found):>12.3f}   {faiss_ms:>8.3f}"
    )