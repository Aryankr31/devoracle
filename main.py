"""
DevOracle API — FastAPI application.

Endpoints:
  POST /ingest     — trigger GitHub ingestion for a repo
  POST /query      — ask a question about the ingested codebase
  GET  /status     — vector store stats
  GET  /health     — health check
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rich.logging import RichHandler

from config.settings import settings
from ingestion.github_loader import GitHubLoader
from ingestion.chunker import CodeChunker
from retrieval.vector_store import VectorStore
from retrieval.rag_engine import RAGEngine

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("devoracle")

# ── Global singletons ─────────────────────────────────────────────────────────
vector_store: Optional[VectorStore] = None
rag_engine: Optional[RAGEngine] = None
ingestion_status = {"running": False, "last_run": None, "files_ingested": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    global vector_store, rag_engine
    logger.info("🚀 DevOracle starting up...")
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store)
    stats = vector_store.collection_stats()
    logger.info(f"Vector store loaded: {stats}")
    yield
    logger.info("DevOracle shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DevOracle",
    description="Autonomous Institutional Knowledge Engine for engineering teams",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    repo: str                           # e.g. "vercel/next.js"
    branch: Optional[str] = None       # defaults to repo's default branch
    max_files: Optional[int] = 300


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class IngestResponse(BaseModel):
    message: str
    repo: str
    status: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    query: str
    model_used: str


# ── Background ingestion ───────────────────────────────────────────────────────
def _run_ingestion(repo: str, branch: Optional[str], max_files: int):
    """Runs in background thread — streams GitHub → chunks → embeds."""
    global ingestion_status
    ingestion_status["running"] = True

    try:
        loader = GitHubLoader(repo_name=repo)
        chunker = CodeChunker()

        logger.info(f"Starting ingestion for {repo}")
        all_docs = []

        for repo_file in loader.stream_files(branch=branch, max_files=max_files):
            chunks = chunker.chunk_file(repo_file)
            all_docs.extend(chunks)

            # Embed in batches of 200 to avoid memory pressure
            if len(all_docs) >= 200:
                added = vector_store.add_documents(all_docs)
                ingestion_status["files_ingested"] += added
                all_docs = []

        # Flush remaining
        if all_docs:
            vector_store.add_documents(all_docs)

        ingestion_status["last_run"] = repo
        logger.info(f"✅ Ingestion complete for {repo}")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
    finally:
        ingestion_status["running"] = False


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/status")
def status():
    stats = vector_store.collection_stats() if vector_store else {}
    return {
        "vector_store": stats,
        "ingestion": ingestion_status,
        "model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    if ingestion_status["running"]:
        raise HTTPException(status_code=409, detail="Ingestion already running.")

    background_tasks.add_task(
        _run_ingestion,
        repo=req.repo,
        branch=req.branch,
        max_files=req.max_files or settings.max_files_per_ingest,
    )

    return IngestResponse(
        message=f"Ingestion started for {req.repo}",
        repo=req.repo,
        status="running",
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized.")

    stats = vector_store.collection_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No data ingested yet. POST to /ingest first.",
        )

    response = rag_engine.query(req.question, top_k=req.top_k)
    return QueryResponse(**response.to_dict())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=settings.api_port, reload=True)
