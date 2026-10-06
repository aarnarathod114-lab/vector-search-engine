"""Build a semantic search index over Simple English Wikipedia.

Each article becomes one vector: its title plus the opening of its text,
embedded with a sentence-embedding model. The vectors go into the HNSW
index, which is saved next to a file holding the articles themselves.

Run:  python -m scripts.build_index --limit 20000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from vsearch.hnsw import HNSWIndex

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_CHARS = 200  # skip stub articles shorter than this
MAX_CHARS = 500  # how much of each article is embedded and shown
CHUNK = 2000  # articles embedded between progress updates


def opening(text: str) -> str:
    """The first MAX_CHARS characters of `text`, cut at a word boundary."""
    text = " ".join(text.split())  # collapse newlines and repeated spaces
    if len(text) <= MAX_CHARS:
        return text
    return text[:MAX_CHARS].rsplit(" ", 1)[0] + "..."


def collect_articles(limit: int) -> list[dict]:
    dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    articles = []
    for row in dataset:
        if len(row["text"]) < MIN_CHARS:
            continue
        articles.append(
            {"title": row["title"], "url": row["url"], "text": opening(row["text"])}
        )
        if len(articles) == limit:
            break
    return articles


def embed(articles: list[dict], cache_dir: Path) -> np.ndarray:
    """Embed every article, CHUNK at a time.

    Each finished chunk is saved to disk, so an interrupted run resumes
    where it stopped and a later run with a bigger --limit reuses the work.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = None
    chunks = []
    embedded_now, start = 0, time.perf_counter()
    for begin in range(0, len(articles), CHUNK):
        batch = articles[begin : begin + CHUNK]
        path = cache_dir / f"chunk_{begin // CHUNK:05d}.npy"
        if path.exists() and len(np.load(path)) == len(batch):
            chunks.append(np.load(path))
            continue
        if model is None:
            model = SentenceTransformer(MODEL_NAME)
        texts = [f"{a['title']}. {a['text']}" for a in batch]
        vectors = model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        np.save(path, vectors)
        chunks.append(vectors)

        embedded_now += len(batch)
        rate = embedded_now / (time.perf_counter() - start)
        done = begin + len(batch)
        remaining = (len(articles) - done) / rate / 60
        print(
            f"  embedded {done:,} / {len(articles):,}"
            f"   {rate:.0f} articles/s   about {remaining:.0f} min left",
            flush=True,
        )
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=20_000, help="articles to index")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output folder")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading Simple English Wikipedia...", flush=True)
    articles = collect_articles(args.limit)
    print(f"Collected {len(articles):,} articles", flush=True)

    print("Embedding...", flush=True)
    vectors = embed(articles, args.out / "embeddings")

    print("Building the HNSW index...", flush=True)
    index = HNSWIndex(dim=vectors.shape[1], metric="cosine")
    start = time.perf_counter()
    index.add(vectors)
    build_s = time.perf_counter() - start

    index.save(args.out / "wiki.npz")
    with open(args.out / "wiki_docs.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False)

    size_mb = (args.out / "wiki.npz").stat().st_size / 1e6
    print(f"Indexed {len(index):,} articles of dimension {index.dim} in {build_s:.1f} s")
    print(f"Saved {args.out / 'wiki.npz'} ({size_mb:.0f} MB) and wiki_docs.json")


if __name__ == "__main__":
    main()