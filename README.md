# DevOracle 🔮
### Autonomous Institutional Knowledge Engine

> Ask any question about any codebase. Get answers with exact file references.

---

## Architecture (Phase 1)

```
GitHub Repo
    ↓
GitHubLoader        — streams files, filters by extension, respects limits
    ↓
CodeChunker         — language-aware splitting (preserves functions/classes)
    ↓
OpenAI Embeddings   — text-embedding-3-small
    ↓
ChromaDB            — persistent vector store with SHA-based deduplication
    ↓
RAG Engine          — retrieves top-k chunks, feeds to LLM with citations
    ↓
FastAPI / CLI       — query interface
```

---

## Day 1 Setup (do this exactly)

### 1. Clone & enter project
```bash
git clone https://github.com/YOUR_USERNAME/devoracle.git
cd devoracle
```

### 2. Create virtual environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment
```bash
cp .env.example .env
# Edit .env and add your keys:
#   OPENAI_API_KEY=sk-...
#   GITHUB_TOKEN=ghp_...
```

**GitHub Token:** go to github.com → Settings → Developer settings → Personal access tokens → Tokens (classic) → New token → select `repo` scope (read only)

### 5. Test the pipeline (CLI - no server needed)

**Ingest a small public repo first:**
```bash
python cli.py ingest --repo pallets/flask --max-files 50
```

**Then ask questions:**
```bash
python cli.py query "How does Flask handle request routing?"
python cli.py query "Where is the application context initialized?"
python cli.py query "What does the g object do and where is it defined?"
```

**Check what's in the store:**
```bash
python cli.py status
```

### 6. Start the API server
```bash
uvicorn api.main:app --reload --port 8000
```

**API docs:** http://localhost:8000/docs

**Ingest via API:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo": "pallets/flask", "max_files": 100}'
```

**Query via API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Flask handle errors?"}'
```

---

## Project Structure

```
devoracle/
├── .env.example
├── .env                    ← your keys (gitignored)
├── requirements.txt
├── cli.py                  ← CLI for local testing
├── chroma_db/              ← auto-created, persistent vector store
│
├── config/
│   └── settings.py         ← all config via pydantic-settings
│
├── ingestion/
│   ├── github_loader.py    ← GitHub API → RepoFile stream
│   └── chunker.py          ← language-aware code chunking
│
├── retrieval/
│   ├── vector_store.py     ← ChromaDB wrapper with deduplication
│   └── rag_engine.py       ← RAG chain + source attribution
│
└── api/
    └── main.py             ← FastAPI endpoints
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | required |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) | — |
| `GITHUB_TOKEN` | GitHub personal access token | required |
| `LLM_MODEL` | LLM to use for answers | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `CHROMA_PERSIST_DIR` | Where ChromaDB stores data | `./chroma_db` |

---

## Swapping LLMs

In `.env`:
```bash
# Use Claude instead of GPT
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...
```

The engine auto-detects based on model name prefix.

---

## Cost Estimate

| Operation | Cost |
|---|---|
| Embed 100 files (~500 chunks) | ~$0.001 (text-embedding-3-small) |
| Query (gpt-4o-mini) | ~$0.001 per query |
| Query (claude-3-5-sonnet) | ~$0.003 per query |

Phase 1 should cost you < $1 total for full development.

---

## What's Next (Phase 2)

- [ ] LangGraph multi-agent orchestration
- [ ] Onboarding Agent — personalized learning paths for new devs
- [ ] Neo4j knowledge graph (code ↔ people ↔ decisions)
- [ ] MCP connectors for Jira + GitHub Actions
- [ ] Slack bot interface
- [ ] PR Sentinel GitHub App

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o-mini / Claude 3.5 Sonnet |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (persistent) |
| RAG Framework | LangChain |
| GitHub Ingestion | PyGithub |
| API | FastAPI + Uvicorn |
| CLI | Rich + argparse |
