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
        vector_size: Any,
    ) -> None:
        """Creates or recreates a collection in Qdrant with specified parameters.

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimensionality of the vectors to be stored

        Raises:
            ValueError: If collection_name is empty or None
        """
        if not collection_name:
            raise ValueError("Collection name is not present")
        # vector_size = QdrantDBConfig().vector_size
        vector_distance = QdrantDBConfig().vector_distance
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "vector_params": VectorParams(
                    size=vector_size,
                    distance=vector_distance,
                ),
            },
        )
        logger.info(f"Collection: {collection_name} is created.")

    def _get_vector_points(self, embeddings: np.ndarray):
        """Converts numpy embeddings to Qdrant PointStruct format.

        Args:
            embeddings: Numpy array of embeddings to convert

        Returns:
            List of PointStruct objects ready for Qdrant insertion
        """
        return [
            models.PointStruct(id=int(idx), vector=embedding.tolist())
            for idx, embedding in enumerate(embeddings)
        ]

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
