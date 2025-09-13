"""Base model interface for embeddings."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            Dimension size
        """
        pass
