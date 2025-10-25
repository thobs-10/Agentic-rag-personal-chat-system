"""
LangGraph implementation for orchestrating the multi-agent system.
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain.agents import AgentType, Tool, initialize_agent
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
    metadata: Dict[str, Any]
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


async def route_query_with_react(state: AgentState) -> Tuple[str, AgentState]:
    """Route using ReAct agent."""
    query = state["query"]
    llm = get_llm(model_name="deepseek-ai/DeepSeek-V3.1-Terminus", temperature=0.0)
    tools = [
        Tool(
            name="Personal Assistant",
            func=run_personal_assistant,
            description="Handles personal queries like daily life, general knowledge, and casual conversation, lease information and personal documents information",
        ),
        Tool(
            name="Technical Assistant",
            func=run_technical_assistant,
            description="Handles technical queries about coding, programming, data science, deeplearning, and AI",
        ),
    ]

    # Create the routing agent
    system_agent = initialize_agent(
        tools, llm=llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )
    system_agent_query = f"""
    Decide whether the following query should be handled by the 'Personal Assistant' or 'Technical Assistant' based on its content:
    
    USER QUERY: {query}
    
    Respond with either 'Personal Assistant' or 'Technical Assistant'.
    
    If you cannot route the query or decide which agent should be answering,
    respond with an appropriate nice message to inform the user that you are
    unable to route their query and they should rephrase it or ask something else.
    """
    response = system_agent.run(system_agent_query)
    logger.info(f"Router agent response: {response}")

    if "Personal Assistant" in response:
        return "personal", state
    elif "Technical Assistant" in response:
        return "technical", state
    else:
        raise ValueError("Could not determine the appropriate agent")


async def process_with_personal_assistant(state: AgentState) -> AgentState:
    """Process the query with the personal assistant."""
    # This is a placeholder that will be implemented later
    # In a real implementation, this would use the retrieval system and LLM
    state = state.copy()
    state["agent_type"] = "personal"
    state["response"] = f"Personal assistant response to: {state['query']}"
    state["sources"] = [
        {"text": "Example source", "source": "personal_collection", "relevance": 0.95}
    ]
    response = run_personal_assistant(state["query"])
    return state


async def process_with_technical_assistant(state: AgentState) -> AgentState:
    """Process the query with the technical assistant."""
    # This is a placeholder that will be implemented later
    # In a real implementation, this would use the retrieval system and LLM
    state = state.copy()
    state["agent_type"] = "technical"
    state["response"] = f"Technical assistant response to: {state['query']}"
    state["sources"] = [
        {"text": "Example technical source", "source": "technical_collection", "relevance": 0.98}
    ]
    return state


def create_system_graph() -> CompiledStateGraph:
    """Create the LangGraph for the agent workflow."""
    # Create a graph with the AgentState schema
    workflow = StateGraph(AgentState)

    # Add the nodes to the graph
    workflow.add_node("router", route_query_with_react)
    workflow.add_node("personal_assistant", process_with_personal_assistant)
    workflow.add_node("technical_assistant", process_with_technical_assistant)

    # Add conditional edges from the router to the assistants
    workflow.add_conditional_edges(
        "router",
        lambda x: x,  # Pass through the router's output as the key
        {
            "personal": "personal_assistant",
            "technical": "technical_assistant",
        },
    )

    # Add edges from the assistants to the end
    workflow.add_edge(START, "router")
    workflow.add_edge("personal_assistant", END)
    workflow.add_edge("technical_assistant", END)

    # Set the entry point
    workflow.set_entry_point("router")
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
