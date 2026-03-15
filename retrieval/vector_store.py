"""
Embedder — generates embeddings and manages the ChromaDB vector store.
Handles deduplication via SHA so re-ingestion is safe and incremental.
"""

import logging
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper around ChromaDB + LangChain.
    Supports incremental ingestion (skips already-embedded SHAs).
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_dir: str = None,
    ):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = persist_dir or settings.chroma_persist_dir

        # Local HuggingFace embeddings (FREE, runs on CPU)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._store: Optional[Chroma] = None
        self._seen_shas: set = set()

        self._init_store()

    def _init_store(self):
        """Initialize or load existing ChromaDB collection."""
        self._store = Chroma(
            client=self._chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

        try:
            existing = self._store.get(include=["metadatas"])
            for meta in existing["metadatas"]:
                if sha := meta.get("sha"):
                    self._seen_shas.add(sha)
            logger.info(f"Loaded store with {len(self._seen_shas)} existing unique files")
        except Exception as e:
            logger.warning(f"Could not load existing store metadata: {e}")

    def _filter_new_docs(self, documents: List[Document]) -> List[Document]:
        """Skip documents whose file SHA we've already embedded."""
        new_docs = []
        skipped = 0
        seen_in_batch = set()

        for doc in documents:
            sha = doc.metadata.get("sha", "")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            key = f"{sha}:{chunk_idx}"

            if sha in self._seen_shas and key not in seen_in_batch:
                skipped += 1
                continue

            seen_in_batch.add(key)
            new_docs.append(doc)

        if skipped:
            logger.info(f"Skipped {skipped} already-embedded chunks (incremental mode)")
        return new_docs

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> int:

        new_docs = self._filter_new_docs(documents)

        if not new_docs:
            logger.info("No new documents to embed.")
            return 0

        total_added = 0
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]

            try:
                self._store.add_documents(batch)
                total_added += len(batch)

                for doc in batch:
                    if sha := doc.metadata.get("sha"):
                        self._seen_shas.add(sha)

                logger.info(
                    f"Embedded batch {i // batch_size + 1} "
                    f"({total_added}/{len(new_docs)} docs)"
                )

            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                raise

        logger.info(f"✅ Added {total_added} new chunks to vector store")
        return total_added

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[dict] = None,
    ) -> List[Document]:

        k = k or settings.retrieval_top_k
        return self._store.similarity_search(query, k=k, filter=filter)

    def as_retriever(self, **kwargs):

        return self._store.as_retriever(
            search_kwargs={"k": settings.retrieval_top_k, **kwargs}
        )

    def collection_stats(self) -> dict:

        count = self._store._collection.count()
        return {
            "total_chunks": count,
            "unique_files": len(self._seen_shas),
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
        }