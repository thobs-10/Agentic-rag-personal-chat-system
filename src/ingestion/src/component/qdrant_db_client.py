from typing import Any, List

import numpy as np
from config import QdrantDBConfig
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams


class QdrantDBClient:
    def __init__(self) -> None:
        self.qdrant_client = QdrantClient(
            url=QdrantDBConfig().url,
            port=QdrantDBConfig().port,
        )
        logger.info("Client is created successfully.")

    def create_db_collection(
        self,
        collection_name: str,
        vector_size: Any,
    ) -> None:
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
        """
        Converts a numpy array of embeddings into a list of Qdrant PointStructs.
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
        # points = self._get_vector_points(embeddings)
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=wait,
        )
        logger.info("Embeddings are successfully inserted into the vectorDB.")

    def get_collection(self, collection_name: str):
        try:
            return self.qdrant_client.get_collection(collection_name)
        except Exception:
            return None
