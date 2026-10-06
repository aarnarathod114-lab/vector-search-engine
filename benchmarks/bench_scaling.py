"""Measure how exact search and HNSW scale as the index grows.

One index of each kind is grown in steps. At each size, the same queries
are run against both, so the table shows where HNSW overtakes exact search.
"""

import time

import numpy as np

from vsearch.brute_force import BruteForceIndex
from vsearch.hnsw import HNSWIndex

SIZES = (10_000, 25_000, 50_000, 100_000)
DIM, K, QUERIES, CLUSTERS, EF_SEARCH = 128, 10, 200, 100, 50

rng = np.random.default_rng(42)
centres = rng.standard_normal((CLUSTERS, DIM)).astype(np.float32)
labels = rng.integers(0, CLUSTERS, SIZES[-1] + QUERIES)
points = centres[labels] + 0.35 * rng.standard_normal(
    (SIZES[-1] + QUERIES, DIM)
).astype(np.float32)
data, queries = points[: SIZES[-1]], points[SIZES[-1] :]

exact = BruteForceIndex(dim=DIM, metric="cosine")
hnsw = HNSWIndex(dim=DIM, metric="cosine")

print(f"dim: {DIM}   k: {K}   ef_search: {EF_SEARCH}   queries: {QUERIES}")
print()
print(" vectors   build (s)   exact ms   hnsw ms   speed-up   recall@10")

done, build_s = 0, 0.0
for size in SIZES:
    exact.add(data[done:size])
    start = time.perf_counter()
    hnsw.add(data[done:size])
    build_s += time.perf_counter() - start
    done = size

    start = time.perf_counter()
    truth = [set(exact.search(q, k=K)[0].tolist()) for q in queries]
    exact_ms = (time.perf_counter() - start) / QUERIES * 1000

    start = time.perf_counter()
    found = [
        set(hnsw.search(q, k=K, ef_search=EF_SEARCH)[0].tolist()) for q in queries
    ]
    hnsw_ms = (time.perf_counter() - start) / QUERIES * 1000

    recall = sum(len(f & t) for f, t in zip(found, truth)) / (QUERIES * K)
    print(
        f"{size:>8,}   {build_s:>9.1f}   {exact_ms:>8.2f}   {hnsw_ms:>7.2f}"
        f"   {exact_ms / hnsw_ms:>7.1f}x   {recall:>9.3f}",
        flush=True,
    )