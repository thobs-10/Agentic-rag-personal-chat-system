from typing import Literal

from pydantic import BaseModel


class QdrantDBConfig(BaseModel):
    vector_size: int = 284
    vector_distance: Literal["Cosine", "Dot", "Euclid", "Manhattan"] = "Cosine"
    port: int = 6333
    url: str = "localhost"
    batch_size: int = 32


class TextSplitter(BaseModel):
    chunk_size: int = 500  # Optimal size for your embedding model and LLM context window
    chunk_overlap: int = 100


class EncodingSentenceModel(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
