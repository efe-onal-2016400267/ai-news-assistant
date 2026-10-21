from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from ai_news_assistant.rag.base import Hit, SearchFilter
from ai_news_assistant.rag.config import BGE_QUERY_PREFIX, EMBEDDING_MODEL


class VectorRetriever:
    """The class that will be used to query ChromaDB vector store."""

    def __init__(self, vector_store: Path, config_id: str) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(vector_store)
        # Get the relevant collection
        self.collection = self.client.get_collection(name=config_id)

    def chroma_where(self, search_filter: SearchFilter | None) -> dict[str, Any] | None:
        """Create chromadb where clause, return as dict."""
        if search_filter is None or (search_filter.sources is None and search_filter.since is None):
            return None

        conditions: list[dict[str, Any]] = []
        if search_filter.sources is not None:
            conditions.append({"source": {"$in": list(search_filter.sources)}})
        if search_filter.since is not None:
            conditions.append({"published_at": {"$gte": search_filter.since.timestamp()}})
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        """Query one Chroma collection (one chunking config) with BGE embeddings."""

        where = self.chroma_where(search_filter=search_filter)

        # Check if k is valid
        k = min(k, self.collection.count())
        if k <= 0:
            return []

        # Add the BGE query prefix into the query
        query = BGE_QUERY_PREFIX + query

        # Create the query embedding with the embedding model
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Query the collection
        if where is not None:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(), n_results=k, where=where
            )
        else:
            results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=k)

        metadatas = results["metadatas"]
        ids = results["ids"]
        documents = results["documents"]
        distances = results["distances"]

        assert metadatas is not None and documents is not None and distances is not None

        return [
            Hit(
                chunk_id=chunk_id,
                article_id=str(metadata["article_id"]),
                score=1 - distance,
                text=text,
            )
            for chunk_id, metadata, distance, text in zip(
                ids[0], metadatas[0], distances[0], documents[0], strict=True
            )
        ]
