"""Persistent Chroma-backed vector store and the end-to-end ingest step."""
import uuid
from pathlib import Path
from typing import List, Optional

import chromadb
import numpy as np
from langchain_core.documents import Document

from .config import PDF_DIR, VECTOR_COLLECTION_NAME, VECTOR_STORE_DIR
from .embeddings import EmbeddingManager
from .ingestion import load_pdf, load_pdfs, split_documents


class VectorStore:
    def __init__(
        self,
        collection_name: str = VECTOR_COLLECTION_NAME,
        persist_directory: Path = VECTOR_STORE_DIR,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # Chroma defaults to raw L2 distance; similarity_score below assumes
            # cosine distance (1 - distance), so the space must be pinned here.
            metadata={"description": "PDF document embeddings for RAG", "hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids, metadatas, texts, vectors = [], [], [], []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            texts.append(doc.page_content)
            vectors.append(embedding.tolist())

        self.collection.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)

    def count(self) -> int:
        return self.collection.count()


def ingest_documents(
    documents: List[Document],
    embedder: Optional[EmbeddingManager] = None,
    store: Optional[VectorStore] = None,
) -> int:
    """Chunk, embed, and store already-loaded documents. Returns chunk count.

    Accepts an existing embedder/store so callers holding a live vector store
    (e.g. a running app) reuse it instead of opening a second Chroma client
    against the same persist directory.
    """
    if not documents:
        return 0

    embedder = embedder or EmbeddingManager()
    store = store or VectorStore()

    chunks = split_documents(documents)
    vectors = embedder.generate_embeddings([c.page_content for c in chunks])
    store.add_documents(chunks, vectors)
    return len(chunks)


def ingest_pdf_file(
    pdf_path: Path,
    embedder: Optional[EmbeddingManager] = None,
    store: Optional[VectorStore] = None,
) -> int:
    """Chunk, embed, and store a single PDF. Returns chunk count."""
    return ingest_documents(load_pdf(pdf_path), embedder=embedder, store=store)


def ingest_directory(pdf_directory: Path = PDF_DIR) -> int:
    """Load, chunk, embed, and store every PDF under `pdf_directory`. Returns chunk count."""
    documents = load_pdfs(pdf_directory)
    if not documents:
        print(f"No PDFs found in {pdf_directory}")
        return 0

    embedder = EmbeddingManager()
    store = VectorStore()
    count = ingest_documents(documents, embedder=embedder, store=store)
    print(
        f"Ingested {count} chunks from {len(documents)} pages "
        f"into '{store.collection.name}' ({store.count()} total)."
    )
    return count


if __name__ == "__main__":
    ingest_directory()
