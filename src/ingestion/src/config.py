import os
from dataclasses import dataclass
from pydantic import BaseModel
from qdrant_client.models import Distance
from typing import Literal


@dataclass
class QdrantDBConfig(BaseModel):
    vector_size: int = 284
    vector_distance: Literal[Distance.COSINE, Distance.DOT, Distance.EUCLID, Distance.MANHATTAN] = (
        Distance.COSINE
    )
    port: int = 8080
    url: str = "localhost"
    batch_size = 32


@dataclass
class TextSplitter(BaseModel):
    chunk_size: int = 500  # Optimal size for your embedding model and LLM context window
    chunk_overlap: int = 100


@dataclass
class EncodingSentenceModel(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
