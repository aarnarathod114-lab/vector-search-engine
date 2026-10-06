"""Compare HNSW against exact search: recall, query speed and build time."""

import time

import numpy as np

from vsearch.brute_force import BruteForceIndex
from vsearch.hnsw import HNSWIndex

N, DIM, K, QUERIES, CLUSTERS = 10_000, 128, 10, 200, 100

# Clustered data: 100 random centres with noise around each one. Real
# embeddings look like this (similar items sit close together), whereas
# purely random vectors have no structure for any index to exploit.
rng = np.random.default_rng(42)
centres = rng.standard_normal((CLUSTERS, DIM)).astype(np.float32)
labels = rng.integers(0, CLUSTERS, N + QUERIES)
points = centres[labels] + 0.35 * rng.standard_normal((N + QUERIES, DIM)).astype(
    np.float32
)
data, queries = points[:N], points[N:]

exact = BruteForceIndex(dim=DIM, metric="cosine")
exact.add(data)

start = time.perf_counter()
truth = [set(exact.search(q, k=K)[0].tolist()) for q in queries]
exact_ms = (time.perf_counter() - start) / QUERIES * 1000

hnsw = HNSWIndex(dim=DIM, metric="cosine")
start = time.perf_counter()
hnsw.add(data)
build_s = time.perf_counter() - start

print(f"vectors: {N:,}   dim: {DIM}   k: {K}")
print(f"HNSW build time : {build_s:.1f} s")
print(f"exact search    : {exact_ms:.2f} ms/query")
print()
print("ef_search   recall@10   ms/query")
for ef in (10, 20, 50, 100):
    start = time.perf_counter()
    found = [set(hnsw.search(q, k=K, ef_search=ef)[0].tolist()) for q in queries]
    ms = (time.perf_counter() - start) / QUERIES * 1000
    recall = sum(len(f & t) for f, t in zip(found, truth)) / (QUERIES * K)
    print(f"{ef:>9}   {recall:>9.3f}   {ms:>8.2f}")