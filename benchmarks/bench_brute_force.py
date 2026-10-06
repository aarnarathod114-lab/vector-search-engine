"""Time exact search on random vectors to get a baseline."""

import time

import numpy as np

from vsearch.brute_force import BruteForceIndex

N, DIM, K, QUERIES = 100_000, 128, 10, 100

rng = np.random.default_rng(42)
data = rng.standard_normal((N, DIM), dtype=np.float32)
queries = rng.standard_normal((QUERIES, DIM), dtype=np.float32)

index = BruteForceIndex(dim=DIM, metric="cosine")
index.add(data)

# Sanity check: searching for a stored vector must return that vector first.
ids, dists = index.search(data[123], k=K)
print(f"self-search     : id {ids[0]} at distance {dists[0]:.6f} (expect id 123, ~0)")

start = time.perf_counter()
for q in queries:
    index.search(q, k=K)
elapsed = time.perf_counter() - start

print(f"vectors indexed : {len(index):,}")
print(f"avg query time  : {elapsed / QUERIES * 1000:.2f} ms")
print(f"queries / second: {QUERIES / elapsed:.0f}")