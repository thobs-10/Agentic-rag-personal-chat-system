"""
Retrieval utilities for integrating with the Qdrant vector database.
"""

from typing import Dict, List, Any, Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.agentic_rag_personal_chat_system.ingestion.src.component.qdrant_db_client import (
    QdrantDBClient,
)


class Retriever:
    """Class for retrieving relevant documents from Qdrant."""

    def __init__(self, collection_name: str, top_k: int = 5):
        """
        Initialize the retriever.

        Args:
            collection_name: Name of the vector DB collection to query
            top_k: Number of relevant documents to retrieve
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.qdrant_client = QdrantDBClient()

        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Initialized embedding model for retrieval")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for the query.

        Args:
            query: The query text

        Returns:
            List of floats representing the query embedding
        """
        try:
            embedding = self.embedding_model.encode(query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The query text

        Returns:
            List of relevant documents with metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.get_query_embedding(query)

            # Search the vector DB
            search_results = self.qdrant_client.qdrant_client.search(
                collection_name=self.collection_name, query_vector=query_embedding, limit=self.top_k
            )

            # Format results
            formatted_results = []
            for result in search_results:
                payload = result.payload or {}

                # Extract text and source from payload
                text = payload.get("text", "")
                source = payload.get("source", "unknown")

                formatted_results.append(
                    {
                        "text": text,
                        "source": source,
                        "relevance": float(result.score) if result.score is not None else 0.0,
                        "metadata": {
                            k: v for k, v in payload.items() if k not in ["text", "source"]
                        },
                    }
                )

            logger.info(f"Retrieved {len(formatted_results)} documents from {self.collection_name}")
            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving from vector DB: {e}")
            # Return empty results on error
            return []
