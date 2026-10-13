from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ai_news_assistant.rag.base import Hit
from ai_news_assistant.rag.config import BGE_QUERY_PREFIX, EMBEDDING_MODEL


class VectorRetriever:
    """The class that will be used to query ChromaDB vector store."""

    def __init__(self, vector_store: Path, config_id: str) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(vector_store)
        # Get the relevant collection
        self.collection = self.client.get_collection(name=config_id)

    def search(self, query: str, k: int) -> list[Hit]:
        """Query one Chroma collection (one chunking config) with BGE embeddings."""

        # Check if k is valid
        k = min(k, self.collection.count())
        if k <= 0:
            return []

        # Add the BGE query prefix into the query
        query = BGE_QUERY_PREFIX + query

        # Create the query embedding with the embedding model
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Query the collection
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
