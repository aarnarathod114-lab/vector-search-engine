"""Streamlit demo: semantic search over Simple English Wikipedia.

This is the entry point for the hosted demo. It reuses the same
SearchService as the FastAPI app. If the index is not already in data/,
it is downloaded once from this repository's GitHub release.

Run locally:  streamlit run demo/streamlit_app.py
"""

import re
import sys
import tempfile
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # lets Python find the `vsearch` and `app` folders

from app.service import MODEL_NAME, SearchService  # noqa: E402

REPO = "https://github.com/aarnarathod114-lab/vector-search-engine"
RELEASE = REPO + "/releases/download/index-v1/"
FILES = ("wiki.npz", "wiki_docs.json")
EXAMPLES = (
    "why is the sky blue",
    "animals that live in the ocean",
    "how do plants make food",
    "who invented the telephone",
)


def find_or_download_data() -> Path:
    """Folder holding the index: data/ if it is there, else a downloaded copy."""
    local = ROOT / "data"
    if all((local / name).exists() for name in FILES):
        return local
    cache = Path(tempfile.gettempdir()) / "vector-search-engine"
    cache.mkdir(exist_ok=True)
    for name in FILES:
        target = cache / name
        if not target.exists():
            # Download under a temporary name so that an interrupted
            # download is never mistaken for a complete file.
            partial = cache / (name + ".part")
            urllib.request.urlretrieve(RELEASE + name, partial)
            partial.replace(target)
    return cache


@st.cache_resource(show_spinner="Loading the index and the language model...")
def load_service() -> SearchService:
    """Built once and shared by every visitor."""
    return SearchService(find_or_download_data())


def plain(text: str) -> str:
    """Escape characters Streamlit would otherwise treat as formatting."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$<])", r"\\\1", text)


def use_example(text: str) -> None:
    st.session_state.query = text


st.set_page_config(page_title="Search Wikipedia by meaning", layout="centered")
service = load_service()
index = service.index

st.title("Search Wikipedia by meaning")
st.write(
    f"Ask in your own words. The closest matches come from {len(index):,} "
    "Simple English Wikipedia articles, found by a search index written from scratch."
)

st.text_input(
    "Your question", key="query", max_chars=200, placeholder="how do plants make food"
)
left, right = st.columns(2)
for position, example in enumerate(EXAMPLES):
    column = left if position % 2 == 0 else right
    column.button(example, on_click=use_example, args=(example,))

query = st.session_state.query.strip()
if query:
    found = service.search(query, k=5)

    st.subheader(f"{len(found['results'])} closest articles")
    for item in found["results"]:
        url = item["url"].replace("(", "%28").replace(")", "%29")
        st.markdown(f"**{item['rank']}. [{plain(item['title'])}]({url})**")
        st.markdown(plain(item["text"]))
        score = min(max(item["score"], 0.0), 1.0)
        st.progress(score, text=f"{item['score']:.2f} similarity")

    st.divider()
    st.subheader("Where the time went")
    total = found["embed_ms"] + found["search_ms"]
    model_column, index_column = st.columns(2)
    model_column.metric("Language model", f"{found['embed_ms']:.1f} ms")
    index_column.metric("Index search", f"{found['search_ms']:.2f} ms")
    st.caption(
        f"The index took {found['search_ms'] / total * 100:.0f}% of the time. "
        "It does not compare your question with every article: it walks a graph "
        "of neighbours and checks only a small fraction of them."
    )

st.divider()
st.caption(
    f"Index: HNSW with M = {index.M} and ef_construction = {index.ef_construction}, "
    f"over {index.dim}-dimensional vectors compared by {index.metric} distance. "
    f"Language model: {MODEL_NAME}. Source code: {REPO}"
)