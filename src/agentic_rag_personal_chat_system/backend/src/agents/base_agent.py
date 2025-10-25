"""
Base agent class for all specialized assistants.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from loguru import logger

# Import our custom LLM and retriever
from src.agentic_rag_personal_chat_system.backend.src.llm.huggingface_llm import get_llm
from src.agentic_rag_personal_chat_system.backend.src.retrieval.retriever import Retriever
from src.agentic_rag_personal_chat_system.backend.src.config.agent_config import AgentConfig


class AgentResponse(TypedDict, total=False):
    """Standard response format for all agents."""

    response: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    error: Optional[str]


class BaseAgent(ABC):
    """Base agent class with common functionality for all specialized agents."""

    def __init__(
        self,
        collection_name: str,
        model_name: str = AgentConfig().model_name,
        temperature: float = AgentConfig().temperature,
        top_k: int = AgentConfig().retrieval_top_k,
    ):
        """
        Initialize the base agent.

        Args:
            collection_name: Name of the vector DB collection to query
            model_name: Name of the LLM model to use
            temperature: Temperature setting for the LLM (0.0 = deterministic)
            top_k: Number of relevant documents to retrieve
        """
        self.collection_name = collection_name
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k

        # Initialize LLM using our custom implementation
        self.llm = get_llm(model_name=self.model_name, temperature=self.temperature)

        # Initialize memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize retriever for vector DB access
        self.retriever = Retriever(collection_name=self.collection_name, top_k=self.top_k)

        # Initialize system prompt
        self.system_prompt = self._get_system_prompt()

        logger.info(f"Initialized {self.__class__.__name__} with collection {collection_name}")

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    async def retrieve_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector database.

        Args:
            query: The user query to find relevant documents for

        Returns:
            List of relevant documents with metadata
        """
        logger.info(f"Retrieving context for query: {query[:50]}...")

        try:
            # Use our retriever to get relevant documents
            documents = await self.retriever.retrieve(query)

            if not documents:
                logger.warning(f"No relevant documents found for query: {query[:50]}...")
                # Return empty list if no documents found
                return []

            logger.info(f"Retrieved {len(documents)} relevant documents")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            # Return empty list on error
            return []

    def _format_context_for_prompt(
        self,
        contexts: List[Dict[str, Any]],
    ) -> str:
        """
        Format retrieved contexts into a string for the prompt.

        Args:
            contexts: List of context documents

        Returns:
            Formatted context string
        """
        formatted_contexts = []

        for i, context in enumerate(contexts):
            formatted_contexts.append(
                f"Document {i + 1} [source: {context.get('source', 'unknown')}]:\n"
                f"{context.get('text', '')}\n"
            )

        return "\n\n".join(formatted_contexts)

    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Create the prompt template for the agent.

        Returns:
            ChatPromptTemplate: The prompt template for the agent
        """
        # Basic RAG prompt template
        template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", "Relevant context information:\n{context}"),
                ("human", "{query}"),
            ]
        )

        return template

    async def _generate_response(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            query: User query
            context: Retrieved context formatted as string

        Returns:
            Generated response
        """
        prompt = self._create_prompt()
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Generate response
        result = await chain.ainvoke(
            {"query": query, "context": context, "chat_history": self.memory.chat_memory.messages}
        )

        # Update memory with the interaction
        self.memory.save_context({"input": query}, {"output": result["text"]})

        return result["text"]

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Process a user query and return a response.

        Args:
            query: The user query
            context: Optional additional context

        Returns:
            AgentResponse with response text and sources
        """
        try:
            logger.info(f"Processing query with {self.__class__.__name__}")

            # Retrieve relevant documents
            relevant_docs = await self.retrieve_relevant_context(query)

            # Format contexts for the prompt
            formatted_context = self._format_context_for_prompt(relevant_docs)

            # Generate response
            response_text = await self._generate_response(query, formatted_context)

            return {
                "response": response_text,
                "sources": relevant_docs,
                "metadata": {
                    "agent_type": self.__class__.__name__,
                    "model": self.model_name,
                    "collection": self.collection_name,
                },
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "sources": [],
                "metadata": {},
                "error": str(e),
            }
