"""
Personal assistant agent implementation.
"""

from typing import Any, Dict

from loguru import logger

from src.agentic_rag_personal_chat_system.backend.src.agents.base_agent import BaseAgent


class PersonalAssistant(BaseAgent):
    """Personal assistant specializing in general knowledge and conversational topics."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the personal assistant.

        Args:
            model_name: Name of the LLM model to use
        """
        super().__init__(
            collection_name="personal_collection",
            model_name=model_name,
            temperature=0.3,  # Slightly higher temperature for more natural responses
            provider="ollama",
            top_k=3,
        )

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the personal assistant."""
        return """You are a helpful Personal Assistant focused on providing accurate information
        and engaging conversation on general topics. You have access to personal documents 
        and general knowledge.

        Guidelines:
        - Respond in a conversational, friendly manner
        - Provide helpful information from the given context when available
        - For questions about personal information, only use what's provided in the context
        - Be concise but comprehensive in your answers
        - If you don't have enough information, politely say so
        - Maintain a helpful, supportive tone throughout the conversation

        Always cite sources when referencing personal documents or specific information.
        """


# Function to be used by the router agent
async def run_personal_assistant(query: str) -> Dict[str, Any]:
    """
    Run the personal assistant on a query.
    This function is used as a tool by the router agent.

    Args:
        query: The user's query

    Returns:
        Response from the personal assistant
    """
    logger.info(f"Personal assistant processing query: {query}")

    # Initialize the assistant
    assistant = PersonalAssistant()

    # Process the query
    response = await assistant.process_query(query)

    # Convert TypedDict to regular dict for tool compatibility
    return dict(response)
