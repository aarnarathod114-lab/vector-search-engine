"""HTTP API for semantic search over the Wikipedia index.

Run:   uvicorn app.main:app --port 8000
Docs:  http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.service import MODEL_NAME, SearchService

DATA_DIR = Path(os.environ.get("VSEARCH_DATA", "data"))
STATIC_DIR = Path(__file__).parent / "static"


class Result(BaseModel):
    rank: int
    title: str
    url: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[Result]
    embed_ms: float
    search_ms: float
    total_articles: int


class Stats(BaseModel):
    articles: int
    dimension: int
    metric: str
    M: int
    ef_construction: int
    ef_search: int
    embedding_model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and the index once, before the first request arrives.
    app.state.service = SearchService(DATA_DIR)
    yield


app = FastAPI(
    title="Vector Search Engine",
    description=(
        "Semantic search over Simple English Wikipedia, "
        "served by an HNSW index written from scratch."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/search", response_model=SearchResponse)
def search(
    request: Request,
    q: Annotated[
        str, Query(min_length=1, max_length=200, description="What to search for")
    ],
    k: Annotated[int, Query(ge=1, le=20, description="Number of results")] = 5,
    ef_search: Annotated[
        int | None,
        Query(ge=1, le=500, description="Search effort: higher is more accurate"),
    ] = None,
):
    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="q must contain some text")
    return request.app.state.service.search(query, k=k, ef_search=ef_search)


@app.get("/stats", response_model=Stats)
def stats(request: Request):
    index = request.app.state.service.index
    return {
        "articles": len(index),
        "dimension": index.dim,
        "metric": index.metric,
        "M": index.M,
        "ef_construction": index.ef_construction,
        "ef_search": index.ef_search,
        "embedding_model": MODEL_NAME,
    }