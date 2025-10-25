"""
LangGraph implementation for orchestrating the multi-agent system.
"""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from src.agentic_rag_personal_chat_system.backend.src.agents.personal_assistant import (
    run_personal_assistant,
)
from src.agentic_rag_personal_chat_system.backend.src.agents.technical_assistant import (
    run_technical_assistant,
)


# Define our state schema as TypedDict for LangGraph compatibility
class AgentStateRequired(TypedDict):
    """Required fields for the agent workflow."""

    query: str
    context: Dict[str, Any]
    mode: str


class AgentState(AgentStateRequired, total=False):
    """State for the agent workflow with optional fields."""

    agent_type: str
    response: Optional[str]
    sources: List[Dict[str, Any]]
    error: Optional[str]


def get_llm(model_name: str, temperature: float) -> Any:
    from src.agentic_rag_personal_chat_system.backend.src.config.llm_config import (
        HuggingFaceInferenceLLMConfig,
    )
    from src.agentic_rag_personal_chat_system.backend.src.llm.huggingface_llm import (
        HuggingFaceInferenceLLM,
    )

    return HuggingFaceInferenceLLM(
        config=HuggingFaceInferenceLLMConfig(
            model_name=model_name,
            temperature=temperature,
        )
    )


async def classify_query(query: str) -> str:
    """Simple query classification."""
    llm = get_llm(model_name="deepseek-ai/DeepSeek-V3.1-Terminus", temperature=0.0)

    prompt = f"""
    Classify this query as 'technical' or 'personal':

    Query: {query}

    Technical: programming, coding, software, AI, data science, development
    Personal: general knowledge, daily life, documents, leases, casual conversation

    Answer with just one word: technical or personal
    """

    try:
        response = await llm.ainvoke(prompt)
        classification = response.lower().strip()
        return "technical" if "technical" in classification else "personal"
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return "personal"  # Default fallback


async def route_and_process(state: AgentState) -> AgentState:
    """Single function to route and process query."""
    query = state["query"]

    try:
        # Simple classification
        agent_type = await classify_query(query)
        logger.info(f"Classified query as: {agent_type}")

        # Direct agent call
        if agent_type == "technical":
            response = await run_technical_assistant(query)
        else:
            response = await run_personal_assistant(query)

        # Update state
        state = state.copy()
        state.update({
            "agent_type": agent_type,
            "response": response.get("response", "No response generated"),
            "sources": response.get("sources", [])
        })

    except Exception as e:
        logger.error(f"Error in route_and_process: {e}")
        state = state.copy()
        state.update({
            "agent_type": "error",
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "sources": [],
            "error": str(e)
        })

    return state


def create_system_graph() -> CompiledStateGraph:
    """Create the simplified LangGraph for the agent workflow."""
    # Create a graph with the AgentState schema
    workflow = StateGraph(AgentState)

    # Single node handles routing and processing
    workflow.add_node("process", route_and_process)

    # Simple linear flow
    workflow.add_edge(START, "process")
    workflow.add_edge("process", END)

    # Compile with memory
    memory = MemorySaver()
    workflow = workflow.compile(checkpointer=memory)

    return workflow


# Create a singleton graph instance
_graph_instance = None


def get_graph_instance():
    """Get the singleton graph instance."""
    global _graph_instance
    if _graph_instance is None:
        logger.info("Creating new graph instance")
        _graph_instance = create_system_graph()
        # Compile the graph into a runnable
        # _graph_instance = graph.compile()
    return _graph_instance


# For testing purposes
if __name__ == "__main__":
    import asyncio

    async def test_graph():
        graph = get_graph_instance()

        # Test with a technical query
        result = await graph.ainvoke(
            {
                "query": "How do I fix a bug in my Python code?",
                "context": {},
                "mode": "auto",
            }
        )
        print("Technical query result:", result)

        # Test with a personal query
        result = await graph.ainvoke(
            {
                "query": "What's the weather like today?",
                "context": {},
                "mode": "auto",
            }
        )
        print("Personal query result:", result)

    asyncio.run(test_graph())
