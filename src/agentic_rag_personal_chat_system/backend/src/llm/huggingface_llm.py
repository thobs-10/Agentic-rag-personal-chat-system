"""
LLM integration using Hugging Face's/Ollama Inference API.
"""

import os
import requests
from typing import Any, List, Optional

from huggingface_hub import InferenceClient
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.llms.base import LLM
from loguru import logger

from src.agentic_rag_personal_chat_system.backend.src.config.llm_config import (
    HuggingFaceInferenceLLMConfig,
    OllamainferenceLLMConfig,
)


class OllamaLLMInference(LLM):
    def __init__(
        self,
        config: Optional[OllamainferenceLLMConfig] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if config is None:
            config = OllamainferenceLLMConfig(
                model_name="llama3.2:3b",
                temperature=temperature if temperature is not None else 0.1,
            )
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_tokens = config.max_tokens
        self.base_url = config.base_url

    @property
    def _llm_type(self) -> str:
        return "ollama-inference"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                },
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            generated_text = data.get("response", "").strip()
            return generated_text
        except Exception as e:
            logger.error(f"Error calling Ollama Inference API: {e}")
            raise

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self._call(prompt, stop, None, **kwargs)


def get_ollama_llm(model_name: str = "llama3.2:3b", temperature: float = 0.1) -> OllamaLLMInference:
    """
    Get an instance of the Ollama LLM with the specified model and temperature.

    Args:
        model_name: The name of the model to use
        temperature: The temperature to use for generation

    Returns:
        An instance of OllamaLLMInference
    """
    return OllamaLLMInference(
        model_name=model_name,
        temperature=temperature,
    )


class HuggingFaceInferenceLLM(LLM):
    """LLM wrapper for Hugging Face's Inference API."""

    def __init__(
        self,
        config: Optional[HuggingFaceInferenceLLMConfig] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the HuggingFace LLM.

        Args:
            config: Configuration object (preferred)
            model_name: Model name (alternative to config)
            temperature: Temperature setting (alternative to config)
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)

        # Use config if provided, otherwise create from individual parameters
        if config is None:
            config = HuggingFaceInferenceLLMConfig(
                model_name=model_name or "deepseek-ai/DeepSeek-V3.1-Terminus",
                temperature=temperature if temperature is not None else 0.0,
            )

        self.provider = config.provider
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_tokens = config.max_tokens

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "huggingface-inference"

    @property
    def _api_key(self) -> str:
        """Get the API key from environment variables."""
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            raise ValueError(
                "HF_TOKEN environment variable not set. "
                "Please set it to your Hugging Face API token."
            )
        return api_key

    def _create_client(self) -> InferenceClient:
        """Create and return an InferenceClient."""
        return InferenceClient(
            api_key=self._api_key,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Hugging Face Inference API."""
        try:
            # Create client
            client = self._create_client()

            # Prepare the messages
            messages = [{"role": "user", "content": prompt}]

            # Call the API
            logger.debug(f"Calling {self.model_name} with prompt: {prompt[:100]}...")
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                **kwargs,
            )

            # Extract and return the generated text
            response = completion.choices[0].message.content
            logger.debug(f"Received response: {response[:100]}...")
            return response

        except Exception as e:
            logger.error(f"Error calling Hugging Face Inference API: {e}")
            raise

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async version of _call for better LangGraph compatibility."""
        # For now, we'll use the sync version but wrap it
        # In production, you might want to use aiohttp or similar for true async
        return self._call(prompt, stop, None, **kwargs)


def get_llm(
    model_name: str = "llama3.2:3b",
    temperature: float = 0.1,
    provider: str = "ollama",
) -> LLM:
    """
    Get an instance of the LLM with the specified model, temperature, and provider.

    Args:
        model_name: The name of the model to use
        temperature: The temperature to use for generation
        provider: The LLM provider ('ollama' or 'huggingface')

    Returns:
        An instance of the appropriate LLM class
    """
    if provider.lower() == "ollama":
        return get_ollama_llm(model_name=model_name, temperature=temperature)
    elif provider.lower() == "huggingface":
        return HuggingFaceInferenceLLM(
            model_name=model_name,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. Supported: 'ollama', 'huggingface'"
        )
