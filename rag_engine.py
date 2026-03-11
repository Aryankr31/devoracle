"""
RAG Engine — the core reasoning layer.
Takes a query, retrieves relevant code chunks, and synthesizes an answer
with full source attribution.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from retrieval.vector_store import VectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are DevOracle, an expert AI assistant with deep knowledge of this specific codebase.

Your job is to answer questions about the code with precision, referencing exact file paths and functions.

RULES:
- Always cite the specific file path(s) where relevant code lives
- If you're unsure, say so — never hallucinate code that isn't in the context
- Explain *why* code exists, not just what it does
- If the context doesn't contain enough info to fully answer, say what you DO know and what's missing
- Format code snippets with proper markdown code blocks

CONTEXT FROM CODEBASE:
{context}
"""

HUMAN_PROMPT = """{question}"""


@dataclass
class RAGResponse:
    """Structured response from the RAG engine."""
    answer: str
    sources: List[dict] = field(default_factory=list)
    query: str = ""
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "model_used": self.model_used,
        }


def _format_docs(docs: List[Document]) -> str:
    """Format retrieved docs into a context string for the prompt."""
    sections = []
    for doc in docs:
        header = doc.metadata.get("context_header", doc.metadata.get("path", "Unknown"))
        sections.append(f"--- {header} ---\n{doc.page_content}")
    return "\n\n".join(sections)


def _extract_sources(docs: List[Document]) -> List[dict]:
    """Extract unique source references from retrieved documents."""
    seen_paths = set()
    sources = []
    for doc in docs:
        path = doc.metadata.get("path", "")
        if path not in seen_paths:
            seen_paths.add(path)
            sources.append({
                "path": path,
                "url": doc.metadata.get("url", ""),
                "repo": doc.metadata.get("repo", ""),
                "extension": doc.metadata.get("extension", ""),
            })
    return sources


class RAGEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        model: str = None,
        streaming: bool = False,
    ):
        self.vector_store = vector_store
        self.model_name = model or settings.llm_model
        self.streaming = streaming
        self.llm = self._init_llm()
        self.chain = self._build_chain()

    def _init_llm(self):
        """Initialize the right LLM based on model name."""
        if "claude" in self.model_name:
            return ChatAnthropic(
                model=self.model_name,
                anthropic_api_key=settings.anthropic_api_key,
                streaming=self.streaming,
            )
        else:
            return ChatOpenAI(
                model=self.model_name,
                openai_api_key=settings.openai_api_key,
                streaming=self.streaming,
                temperature=0,  # deterministic for code Q&A
            )

    def _build_chain(self):
        """Build the LangChain RAG chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])

        retriever = self.vector_store.as_retriever()

        chain = (
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Ask a question about the codebase.
        Returns answer + source attributions.
        """
        # Retrieve relevant docs separately for source attribution
        docs = self.vector_store.similarity_search(
            question,
            k=top_k or settings.retrieval_top_k,
        )

        if not docs:
            return RAGResponse(
                answer="I couldn't find relevant code for this query. Try ingesting the repository first.",
                query=question,
                model_used=self.model_name,
            )

        # Run the chain
        answer = self.chain.invoke(question)
        sources = _extract_sources(docs)

        logger.info(f"Query answered using {len(docs)} chunks from {len(sources)} files")

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            model_used=self.model_name,
        )

    async def aquery(self, question: str) -> RAGResponse:
        """Async version for FastAPI."""
        docs = self.vector_store.similarity_search(question)
        answer = await self.chain.ainvoke(question)
        return RAGResponse(
            answer=answer,
            sources=_extract_sources(docs),
            query=question,
            model_used=self.model_name,
        )
