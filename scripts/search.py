"""Search the Wikipedia index from the terminal.

Run:  python -m scripts.search                          (keeps asking)
      python -m scripts.search "why is the sky blue"    (one query)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from vsearch.hnsw import HNSWIndex

MODEL_NAME = "all-MiniLM-L6-v2"  # must be the model the index was built with


def show_results(query, model, index, docs, k):
    start = time.perf_counter()
    vector = model.encode([query], normalize_embeddings=True)[0]
    embed_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    ids, dists = index.search(vector, k=k)
    search_ms = (time.perf_counter() - start) * 1000

    print()
    for rank, (doc_id, dist) in enumerate(zip(ids.tolist(), dists.tolist()), 1):
        doc = docs[doc_id]
        print(f"{rank}. {doc['title']}   (similarity {1 - dist:.2f})")
        print(f"   {doc['text'][:150]}...")
        print(f"   {doc['url']}")
    print(
        f"\n[embedding the query: {embed_ms:.0f} ms"
        f"   index search: {search_ms:.2f} ms over {len(index):,} articles]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="?", help="leave out to be asked repeatedly")
    parser.add_argument("-k", type=int, default=5, help="results to show")
    parser.add_argument("--data", type=Path, default=Path("data"), help="index folder")
    args = parser.parse_args()

    # Article titles contain characters some Windows consoles cannot print.
    sys.stdout.reconfigure(errors="replace")

    print("Loading the model and index...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    index = HNSWIndex.load(args.data / "wiki.npz")
    with open(args.data / "wiki_docs.json", encoding="utf-8") as f:
        docs = json.load(f)
    # One throwaway search so the first real one is not slowed by start-up work.
    index.search(model.encode(["warm up"], normalize_embeddings=True)[0], k=1)

    if args.query:
        show_results(args.query, model, index, docs, args.k)
        return
    while True:
        query = input("\nSearch (press Enter on an empty line to quit): ").strip()
        if not query:
            break
        show_results(query, model, index, docs, args.k)


if __name__ == "__main__":
    main()