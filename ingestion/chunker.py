"""
Chunker — splits code files into semantically meaningful chunks.
Uses language-aware splitting so functions/classes stay intact.
"""

import logging
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain.schema import Document

from ingestion.github_loader import RepoFile
from config.settings import settings

logger = logging.getLogger(__name__)

# Map file extensions to LangChain Language enum
EXTENSION_TO_LANGUAGE = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".jsx":   Language.JS,
    ".ts":    Language.TS,
    ".tsx":   Language.TS,
    ".go":    Language.GO,
    ".rs":    Language.RUST,
    ".java":  Language.JAVA,
    ".cpp":   Language.CPP,
    ".c":     Language.C,
    ".sol":   Language.SOL,
}


@dataclass
class ChunkedDocument:
    """A chunk ready to be embedded and stored."""
    content: str
    metadata: dict
    chunk_index: int
    total_chunks: int


class CodeChunker:
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitters: dict = {}  # cache splitters per language

    def _get_splitter(self, extension: str) -> RecursiveCharacterTextSplitter:
        """Get (or create) a language-aware splitter for this extension."""
        if extension in self._splitters:
            return self._splitters[extension]

        language = EXTENSION_TO_LANGUAGE.get(extension)

        if language:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            # Fallback: generic splitter for markdown, yaml, txt, etc.
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )

        self._splitters[extension] = splitter
        return splitter

    def chunk_file(self, repo_file: RepoFile) -> List[Document]:
        """
        Split a RepoFile into LangChain Documents.
        Each Document carries full metadata for traceability.
        """
        if not repo_file.content.strip():
            return []

        splitter = self._get_splitter(repo_file.extension)

        try:
            chunks = splitter.split_text(repo_file.content)
        except Exception as e:
            logger.warning(f"Chunking failed for {repo_file.path}: {e}")
            return []

        total = len(chunks)
        documents = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            doc = Document(
                page_content=chunk,
                metadata={
                    **repo_file.to_metadata(),
                    "chunk_index": i,
                    "total_chunks": total,
                    # Prepend file path so LLM always knows context origin
                    "context_header": f"File: {repo_file.path} (chunk {i+1}/{total})",
                },
            )
            documents.append(doc)

        logger.debug(f"{repo_file.path} → {total} chunks")
        return documents

    def chunk_files(self, repo_files: List[RepoFile]) -> List[Document]:
        """Batch chunk a list of RepoFiles."""
        all_docs = []
        for rf in repo_files:
            docs = self.chunk_file(rf)
            all_docs.extend(docs)
        logger.info(f"Chunked {len(repo_files)} files → {len(all_docs)} documents")
        return all_docs
