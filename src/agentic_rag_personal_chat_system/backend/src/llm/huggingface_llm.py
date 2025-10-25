"""
LLM integration using Hugging Face's Inference API.
"""

import os
from typing import Dict, List, Any, Optional

from huggingface_hub import InferenceClient
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from src.agentic_rag_personal_chat_system.backend.src.config.llm_config import (
    HuggingFaceInferenceLLMConfig,
)


class HuggingFaceInferenceLLM(LLM):
    """LLM wrapper for Hugging Face's Inference API."""

    def __init__(
        self,
        config: HuggingFaceInferenceLLMConfig = HuggingFaceInferenceLLMConfig(),
    ):
        self.provider = config.provider
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_tokens = config.max_tokens

    # provider: str = "novita"
    # model_name: str = "deepseek-ai/DeepSeek-V3.1-Terminus"
    # temperature: float = 0.0
    # top_p: float = 0.95
    # max_tokens: int = 4096

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
            provider=self.provider,
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

    def format_messages_for_llm(
        self, system_message: str, human_message: str, chat_history: List = None
    ) -> List[Dict[str, str]]:
        """Format messages for the Hugging Face chat API."""
        formatted_messages = []

        # Add system message
        if system_message:
            formatted_messages.append({"role": "system", "content": system_message})

        # Add chat history if provided
        if chat_history:
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    formatted_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    formatted_messages.append({"role": "assistant", "content": message.content})
                elif isinstance(message, SystemMessage):
                    formatted_messages.append({"role": "system", "content": message.content})

        # Add the current human message
        formatted_messages.append({"role": "user", "content": human_message})

        return formatted_messages


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
