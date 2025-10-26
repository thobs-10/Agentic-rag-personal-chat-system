"""
Technical assistant agent implementation.
"""

from typing import Any, Dict

from loguru import logger

from src.agentic_rag_personal_chat_system.backend.src.agents.base_agent import (
    BaseAgent,
    AgentResponse,
)


class TechnicalAssistant(BaseAgent):
    """Technical assistant specializing in programming, coding, and technical topics."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the technical assistant.

        Args:
            model_name: Name of the LLM model to use
        """
        super().__init__(
            collection_name="technical_collection",
            model_name=model_name,
            temperature=0.0,  # Lower temperature for more precise technical responses
            top_k=5,
        )

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the technical assistant."""
        return """You are a highly knowledgeable Technical Assistant specializing in programming, software development, 
        data science, and technical topics. Your primary goal is to provide accurate, clear, and helpful technical information.

        Guidelines:
        - Provide code examples when appropriate, with proper formatting and comments
        - Explain technical concepts in a clear, structured way
        - If relevant documentation exists in the provided context, reference it
        - When multiple approaches exist, outline the pros and cons of each
        - Be precise about language/framework versions when they matter
        - Admit when you don't have enough information and suggest what additional details would help

        Always format code with proper syntax highlighting and indentation.
        When explaining errors, focus on root causes and solutions.
        """


# Function to be used by the router agent
async def run_technical_assistant(query: str) -> Dict[str, Any]:
    """
    Run the technical assistant on a query.
    This function is used as a tool by the router agent.

    Args:
        query: The user's query

    Returns:
        Response from the technical assistant
    """
    logger.info(f"Technical assistant processing query: {query}")

    # Initialize the assistant
    assistant = TechnicalAssistant()

    # Process the query
    response = await assistant.process_query(query)

    # Convert TypedDict to regular dict for tool compatibility
    return dict(response)
