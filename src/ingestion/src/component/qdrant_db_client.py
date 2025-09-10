from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient, models

from src.ingestion.src.config import QdrantDBConfig


class QdrantDBClient:
    def __init__(self) -> None:
        try:
            self.qdrant_client = QdrantClient(
                url=QdrantDBConfig().url,
                port=QdrantDBConfig().port,
            )
            logger.info("Qdrant client created successfully")
        except Exception as e:
            logger.exception(f"Failed to connect to Qdrant: {e}")
            raise

    def create_db_collection(
        self,
        collection_name: str,
        vector_size: int = 384,  # Default size for all-MiniLM-L6-v2
    ) -> None:
        """Creates a new collection in Qdrant.

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimensionality of the vectors (default: 384 for all-MiniLM-L6-v2)

        Raises:
            ValueError: If collection_name is empty
        """
        if not collection_name:
            raise ValueError("Collection name cannot be empty")

        try:
            # Delete if exists and create new
            try:
                self.qdrant_client.delete_collection(collection_name=collection_name)
            except Exception:
                pass  # Ignore if collection doesn't exist

            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
            logger.info(f"Successfully created collection: {collection_name}")
        except Exception as e:
            logger.exception(f"Failed to create collection {collection_name}: {e}")
            raise

    def insert_embbedings(
        self,
        collection_name: str,
        embeddings: np.ndarray[Any, Any],
        points: List[PointStruct],
        wait: bool = True,
    ) -> None:
        """Inserts embeddings into the specified Qdrant collection.

        Args:
            collection_name: Name of target collection
            embeddings: Numpy array of embeddings (unused in current implementation)
            points: Pre-formed list of PointStruct objects containing vectors and metadata
            wait: If True, waits for operation completion before returning

        Note:
            The 'embeddings' parameter is currently not used in the method implementation.
            The method name contains a typo ('embbedings' instead of 'embeddings').
        """
        # points = self._get_vector_points(embeddings)
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=wait,
        )
        logger.info("Embeddings are successfully inserted into the vectorDB.")

    def get_collection(self, collection_name: str):
        """Retrieves information about a Qdrant collection.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            The collection information if exists, None otherwise
        """
        try:
            return self.qdrant_client.get_collection(collection_name)
        except Exception:
            return None
