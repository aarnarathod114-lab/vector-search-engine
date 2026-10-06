"""Everything the API needs to answer a search: the model, the index, the articles."""

import json
import threading
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from vsearch.hnsw import HNSWIndex

MODEL_NAME = "all-MiniLM-L6-v2"  # must be the model the index was built with


class SearchService:
    """Loads everything once, then turns a text query into ranked articles."""

    def __init__(self, data_dir: Path) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = HNSWIndex.load(data_dir / "wiki.npz")
        with open(data_dir / "wiki_docs.json", encoding="utf-8") as f:
            self.docs = json.load(f)
        if len(self.docs) != len(self.index):
            raise ValueError(
                f"index holds {len(self.index)} vectors but there are "
                f"{len(self.docs)} articles; rebuild them together"
            )
        # The web server answers requests on several threads, but neither the
        # tokenizer nor the index's `visited` array is safe to share between
        # two searches at once, so searches take turns.
        self._lock = threading.Lock()
        self.search("warm up", k=1)  # pay one-off start-up costs now, not on a user

    def search(self, query: str, k: int = 5, ef_search: int | None = None) -> dict:
        with self._lock:
            start = time.perf_counter()
            vector = self.model.encode([query], normalize_embeddings=True)[0]
            embed_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            ids, dists = self.index.search(vector, k=k, ef_search=ef_search)
            search_ms = (time.perf_counter() - start) * 1000

        results = []
        for rank, (doc_id, dist) in enumerate(zip(ids.tolist(), dists.tolist()), 1):
            doc = self.docs[doc_id]
            results.append(
                {
                    "rank": rank,
                    "title": doc["title"],
                    "url": doc["url"],
                    "text": doc["text"],
                    "score": round(1.0 - dist, 4),  # cosine similarity
                }
            )
        return {
            "query": query,
            "results": results,
            "embed_ms": round(embed_ms, 2),
            "search_ms": round(search_ms, 3),
            "total_articles": len(self.index),
        }