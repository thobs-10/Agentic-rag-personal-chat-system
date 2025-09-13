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

    def insert_embeddings(
        self,
        collection_name: str,
        embeddings: List[np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        wait: bool = True,
    ) -> None:
        """Inserts embeddings into the specified Qdrant collection.

        Args:
            collection_name: Name of target collection
            embeddings: List of embeddings to insert
            metadata: Optional list of metadata dictionaries for each embedding
            wait: If True, waits for operation completion before returning
        """
        if not collection_name:
            raise ValueError("Collection name cannot be empty")

        try:
            # Convert embeddings to points
            points = []
            for idx, embedding in enumerate(embeddings):
                if not isinstance(embedding, (np.ndarray, list)):
                    raise ValueError(f"Invalid embedding type at index {idx}: {type(embedding)}")

                # Convert numpy array to list if needed
                vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

                point_metadata = metadata[idx] if metadata and idx < len(metadata) else {}

                # Create point with metadata
                points.append(
                    models.Record(
                        id=str(uuid4()),  # Use UUID for unique IDs
                        vector=vector,
                        payload=point_metadata,
                    )
                )

            # Upload in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.qdrant_client.upload_records(
                    collection_name=collection_name, records=batch, wait=wait
                )
                logger.debug(f"Uploaded batch of {len(batch)} points")

            logger.info(f"Successfully inserted {len(points)} embeddings into {collection_name}")
        except Exception as e:
            logger.exception(f"Failed to insert embeddings into collection {collection_name}: {e}")
            raise

    def get_collection(self, collection_name: str) -> models.CollectionInfo | None:
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
