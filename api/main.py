"""
DevOracle API — FastAPI application.

Endpoints:
  GET  /health              — health check
  GET  /status              — vector store + ingestion stats
  POST /ingest              — trigger GitHub ingestion (background)
  GET  /ingest/progress     — poll ingestion progress
  POST /query               — ask a question, get answer + sources
  POST /query/stream        — streaming version (SSE)
  DELETE /collection        — wipe vector store and start fresh
"""

import os
import sys
import warnings
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

# ── Suppress noisy lib warnings before any imports ───────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from config.settings import settings
from ingestion.github_loader import GitHubLoader
from ingestion.chunker import CodeChunker
from retrieval.vector_store import VectorStore
from retrieval.rag_engine import RAGEngine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger("devoracle")

# ── Global state ──────────────────────────────────────────────────────────────
vector_store: Optional[VectorStore] = None
rag_engine: Optional[RAGEngine] = None

ingestion_state = {
    "running": False,
    "repo": None,
    "files_processed": 0,
    "chunks_embedded": 0,
    "error": None,
    "last_completed": None,
}


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, rag_engine
    logger.info("🚀 DevOracle API starting...")
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store)
    stats = vector_store.collection_stats()
    logger.info(
        f"✅ Ready — {stats['total_chunks']} chunks indexed "
        f"across {stats['unique_files']} files"
    )
    yield
    logger.info("DevOracle shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DevOracle",
    description=(
        "Autonomous Institutional Knowledge Engine.\n\n"
        "Ingest any GitHub repo, then ask questions about the codebase "
        "with exact file-level source attribution."
    ),
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    repo: str = Field(..., example="pallets/flask", description="GitHub repo — owner/repo format")
    branch: Optional[str] = Field(None, description="Branch to ingest (defaults to repo default)")
    max_files: Optional[int] = Field(300, description="Max files to ingest per run")

    model_config = {
        "json_schema_extra": {"example": {"repo": "pallets/flask", "max_files": 100}}
    }


class QueryRequest(BaseModel):
    question: str = Field(..., example="How does Flask handle request routing?")
    top_k: Optional[int] = Field(None, description="Chunks to retrieve (default 8)")

    model_config = {
        "json_schema_extra": {"example": {"question": "How does Flask handle request routing?"}}
    }


class IngestResponse(BaseModel):
    message: str
    repo: str
    status: str


class SourceReference(BaseModel):
    path: str
    url: str
    repo: str
    extension: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    query: str
    model_used: str


class IngestionProgress(BaseModel):
    running: bool
    repo: Optional[str] = None
    files_processed: int
    chunks_embedded: int
    error: Optional[str] = None
    last_completed: Optional[str] = None


class StatusResponse(BaseModel):
    api_version: str
    vector_store: dict
    ingestion: IngestionProgress
    llm_model: str
    embedding_model: str


# ── Background ingestion ──────────────────────────────────────────────────────
def _run_ingestion(repo: str, branch: Optional[str], max_files: int):
    """Background thread: GitHub → chunk → embed → ChromaDB."""
    global ingestion_state

    ingestion_state.update({
        "running": True,
        "repo": repo,
        "files_processed": 0,
        "chunks_embedded": 0,
        "error": None,
    })

    try:
        loader = GitHubLoader(repo_name=repo)
        chunker = CodeChunker()
        buffer = []

        logger.info(f"📥 Ingesting: {repo}")

        for repo_file in loader.stream_files(branch=branch, max_files=max_files):
            chunks = chunker.chunk_file(repo_file)
            buffer.extend(chunks)
            ingestion_state["files_processed"] += 1

            if len(buffer) >= 200:
                added = vector_store.add_documents(buffer)
                ingestion_state["chunks_embedded"] += added
                buffer = []
                logger.info(
                    f"  ↳ {ingestion_state['files_processed']} files | "
                    f"{ingestion_state['chunks_embedded']} chunks"
                )

        if buffer:
            added = vector_store.add_documents(buffer)
            ingestion_state["chunks_embedded"] += added

        ingestion_state["last_completed"] = repo
        logger.info(
            f"✅ Done: {ingestion_state['files_processed']} files, "
            f"{ingestion_state['chunks_embedded']} chunks"
        )

        # Reinitialize RAG engine so it points to the refreshed collection
        global rag_engine
        rag_engine = RAGEngine(vector_store=vector_store)
        logger.info("🔄 RAG engine reinitialized with fresh collection")

    except Exception as e:
        ingestion_state["error"] = str(e)
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
    finally:
        ingestion_state["running"] = False


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Liveness check."""
    return {"status": "ok", "version": "0.2.0"}


@app.get("/status", response_model=StatusResponse, tags=["System"])
def status():
    """Full system status — vector store, ingestion state, active models."""
    stats = vector_store.collection_stats() if vector_store else {}
    return StatusResponse(
        api_version="0.2.0",
        vector_store=stats,
        ingestion=IngestionProgress(**ingestion_state),
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Kick off GitHub repo ingestion in the background.
    Poll GET /ingest/progress to track it.
    """
    if ingestion_state["running"]:
        raise HTTPException(
            status_code=409,
            detail=f"Ingestion already running for '{ingestion_state['repo']}'. "
                   "Poll /ingest/progress to check status.",
        )

    background_tasks.add_task(
        _run_ingestion,
        repo=req.repo,
        branch=req.branch,
        max_files=req.max_files or settings.max_files_per_ingest,
    )

    return IngestResponse(
        message=f"Ingestion started for '{req.repo}'. Poll /ingest/progress for updates.",
        repo=req.repo,
        status="running",
    )


@app.get("/ingest/progress", response_model=IngestionProgress, tags=["Ingestion"])
def ingest_progress():
    """Poll ingestion progress — files processed, chunks embedded, errors."""
    return IngestionProgress(**ingestion_state)


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(req: QueryRequest):
    """
    Ask any natural language question about the ingested codebase.
    Returns answer + exact file-level source attribution.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized.")

    stats = vector_store.collection_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No data ingested yet. POST to /ingest first.",
        )

    try:
        response = rag_engine.query(req.question, top_k=req.top_k)
        return QueryResponse(**response.to_dict())
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/stream", tags=["Query"])
async def query_stream(req: QueryRequest):
    """
    Streaming version of /query via Server-Sent Events.
    Answer tokens stream in real time — built for the frontend.
    """
    if not vector_store or vector_store.collection_stats()["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No data ingested yet.")

    async def token_generator() -> AsyncGenerator[str, None]:
        docs = vector_store.similarity_search(
            req.question, k=req.top_k or settings.retrieval_top_k
        )
        if not docs:
            yield "data: No relevant code found for this query.\n\n"
            return

        context = "\n\n".join([
            f"--- {d.metadata.get('context_header', d.metadata.get('path', ''))} ---\n{d.page_content}"
            for d in docs
        ])

        from langchain.prompts import ChatPromptTemplate
        from retrieval.rag_engine import SYSTEM_PROMPT, HUMAN_PROMPT

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        messages = prompt.format_messages(context=context, question=req.question)

        async for chunk in rag_engine.llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                # Escape newlines for SSE format
                content = chunk.content.replace("\n", "\\n")
                yield f"data: {content}\n\n"
                await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/collection", tags=["System"])
def delete_collection():
    """
    Wipe the entire vector store.
    Use when switching repos or re-ingesting from scratch.
    """
    if ingestion_state["running"]:
        raise HTTPException(status_code=409, detail="Cannot delete while ingestion is running.")

    try:
        vector_store._chroma_client.delete_collection(vector_store.collection_name)
        vector_store._seen_shas.clear()
        vector_store._init_store()
        ingestion_state.update({
            "repo": None, "files_processed": 0,
            "chunks_embedded": 0, "last_completed": None, "error": None,
        })
        return {"message": "✅ Collection wiped. Ready for fresh ingestion."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=True,
        log_level="warning",
    )
# ── Onboarding Agent ──────────────────────────────────────────────────────────
class OnboardRequest(BaseModel):
    repo: str = Field(..., example="pallets/flask")
    role: str = Field(..., example="backend")

class OnboardResponse(BaseModel):
    repo: str
    role: str
    learning_plan: str
    files_analyzed: list
    total_docs_retrieved: int
    status: str

@app.post("/onboard", response_model=OnboardResponse, tags=["Agents"])
def onboard(req: OnboardRequest):
    """
    Onboarding Agent — generates a personalized learning path
    for a new developer joining the team.
    """
    if not vector_store or vector_store.collection_stats()["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No data ingested yet. POST to /ingest first."
        )
    try:
        from agents.onboarding_agent import OnboardingAgent
        agent = OnboardingAgent(vector_store=vector_store)
        result = agent.run(repo=req.repo, role=req.role)
        return OnboardResponse(
            repo=req.repo,
            role=req.role,
            learning_plan=result["learning_plan"],
            files_analyzed=result["files_analyzed"],
            total_docs_retrieved=result["total_docs_retrieved"],
            status=result["status"],
        )
    except Exception as e:
        logger.error(f"Onboarding agent failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))