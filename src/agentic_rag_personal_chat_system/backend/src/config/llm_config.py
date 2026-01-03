from langchain.llms import LLM
from pydantic import BaseModel, Field
from typing import Optional


class HuggingFaceInferenceLLMConfig(BaseModel):
    """LLM wrapper for Hugging Face's Inference API."""

    provider: str = "novita"
    model_name: str = "deepseek-ai/DeepSeek-V3.1-Terminus"
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 4096


class OllamainferenceLLMConfig(BaseModel):
    """LLM wrapper for Ollama's Inference API."""

    model_name: str = "llama3.2:3b"
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 4096
    base_url: str = "http://localhost:11434"
