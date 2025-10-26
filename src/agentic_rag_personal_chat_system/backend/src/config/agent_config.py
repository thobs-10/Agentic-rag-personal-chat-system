from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Agent configuration."""

    router_threshold: float = 0.6  # Threshold for routing between technical/personal
    max_retries: int = 2
    max_iterations: int = 5
    memory_enabled: bool = True
    retrieval_top_k: int = 5
    temperature: float = 0.0
    model_name: str = "deepseek-ai/DeepSeek-V3.1-Terminus"


class PersonalAgentConfig(AgentConfig):
    """Personal agent specific configuration."""

    collection_name: str = "lease_documents"


class TechnicalAgentConfig(AgentConfig):
    """Technical agent specific configuration."""

    collection_name: str = "technical_documents"
