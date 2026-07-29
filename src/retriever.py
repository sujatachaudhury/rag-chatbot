"""Query embedding + similarity search against the persisted Chroma collection."""
from typing import Any, Dict, List

from .config import RETRIEVAL_SCORE_THRESHOLD, RETRIEVAL_TOP_K
from .embeddings import EmbeddingManager
from .vectorstore import VectorStore


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        hits = []
        for doc_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity_score = 1 - distance
            if similarity_score >= score_threshold:
                hits.append(
                    {
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                    }
                )
        return hits
