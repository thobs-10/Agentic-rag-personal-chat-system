"""
LLM integration using Hugging Face's Inference API.
"""

import os
from typing import Any, List, Optional

from huggingface_hub import InferenceClient
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.llms.base import LLM
from loguru import logger

from src.agentic_rag_personal_chat_system.backend.src.config.llm_config import (
    HuggingFaceInferenceLLMConfig,
)


class HuggingFaceInferenceLLM(LLM):
    """LLM wrapper for Hugging Face's Inference API."""

    def __init__(
        self,
        config: Optional[HuggingFaceInferenceLLMConfig] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
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
                temperature=temperature if temperature is not None else 0.0
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
    model_name: str = "deepseek-ai/DeepSeek-V3.1-Terminus", temperature: float = 0.0
) -> HuggingFaceInferenceLLM:
    """
    Get an instance of the LLM with the specified model and temperature.

    Args:
        model_name: The name of the model to use
        temperature: The temperature to use for generation

    Returns:
        An instance of HuggingFaceInferenceLLM
    """
    return HuggingFaceInferenceLLM(
        model_name=model_name,
        temperature=temperature,
    )
