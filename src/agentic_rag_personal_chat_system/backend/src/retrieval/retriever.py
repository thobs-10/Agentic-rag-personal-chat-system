"""
Retrieval utilities for integrating with the Qdrant vector database.
"""

from typing import Any, Dict, List

from loguru import logger
from sentence_transformers import SentenceTransformer

from src.agentic_rag_personal_chat_system.ingestion.src.component.qdrant_db_client import (
    QdrantDBClient,
)


class Retriever:
    """Class for retrieving relevant documents from Qdrant."""

    def __init__(self, collection_name: str, top_k: int = 5, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever.

        Args:
            collection_name: Name of the vector DB collection to query
            top_k: Number of relevant documents to retrieve
            embedding_model_name: Name of the sentence transformer model
        """
        self.collection_name = collection_name
        self.top_k = top_k

        # Initialize clients
        try:
            self.qdrant_client = QdrantDBClient()
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"Initialized retriever for collection '{collection_name}' with model '{embedding_model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise RuntimeError(f"Retriever initialization failed: {e}") from e

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for the query.

        Args:
            query: The query text

        Returns:
            List of floats representing the query embedding

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            embedding = self.embedding_model.encode(query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise RuntimeError(f"Failed to generate embedding for query: {query[:50]}...") from e

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The query text

        Returns:
            List of relevant documents with metadata
        """
        if not query.strip():
            logger.warning("Empty query provided to retriever")
            return []

        try:
            # Generate embedding for the query
            query_embedding = self._get_query_embedding(query)

            # Search the vector DB
            search_results = self.qdrant_client.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.top_k
            )

            # Format results
            formatted_results = self._format_search_results(search_results)

            logger.info(f"Retrieved {len(formatted_results)} documents from '{self.collection_name}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving from vector DB: {e}")
            return []  # Return empty results on error

    def _format_search_results(self, search_results) -> List[Dict[str, Any]]:
        """Format search results into a consistent structure."""
        formatted_results = []

        for result in search_results:
            payload = result.payload or {}

            formatted_results.append({
                "text": payload.get("text", ""),
                "source": payload.get("source", "unknown"),
                "relevance": float(result.score) if result.score is not None else 0.0,
                "metadata": {k: v for k, v in payload.items() if k not in ["text", "source"]},
            })

        return formatted_results
