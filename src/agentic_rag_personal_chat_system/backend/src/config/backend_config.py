"""
Configuration settings for the backend services.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = True
    allowed_origins: List[str] = ["http://localhost:3000"]


class VectorDBConfig(BaseModel):
    """Vector database configuration."""

    url: str = "localhost"
    port: int = 6333
    personal_collection: str = "personal_documents"
    technical_collection: str = "technical_documents"
    prefer_grpc: bool = False
    timeout: int = 10


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = "ollama"  # or "openai", "huggingface", etc.
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30


class AgentConfig(BaseModel):
    """Agent configuration."""

    router_threshold: float = 0.6  # Threshold for routing between technical/personal
    max_retries: int = 2
    max_iterations: int = 5
    memory_enabled: bool = True
    retrieval_top_k: int = 5


class BackendConfig(BaseModel):
    """Main backend configuration."""

    api: APIConfig = APIConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()


# Create a singleton instance
config = BackendConfig()
