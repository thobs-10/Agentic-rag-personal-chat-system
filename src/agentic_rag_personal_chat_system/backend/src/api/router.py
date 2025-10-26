"""
API router for handling chat queries.
"""

from loguru import logger
from fastapi import APIRouter, FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Callable

from agentic_rag_personal_chat_system.backend.src.api.models import (
    QueryRequest,
    QueryResponse,
    ErrorResponse,
)
from agentic_rag_personal_chat_system.backend.src.config.backend_config import config
from agentic_rag_personal_chat_system.backend.src.graph import get_graph_instance, AgentState

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG Personal Chat System",
    description="API for the Agentic RAG Personal Chat System",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router
api_router = APIRouter()


# Dependency to get the LangGraph instance (to be implemented)
# async def get_lang_graph() -> callable():
#     """Dependency for getting the LangGraph instance."""
#     # This will be implemented when we create the LangGraph
#     # For now, it's a placeholder
#     try:
#         graph = get_graph_instance()
#     except Exception as e:
#         logger.error(f"Error getting graph instance: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")
#     return graph


@api_router.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def process_query(
    request: QueryRequest,
):
    """
    Process a user query and return a response.

    Args:
        request: The query request containing the user's question
        graph: The LangGraph instance for processing the query

    Returns:
        QueryResponse: The response to the user's query

    Raises:
        HTTPException: If there's an error processing the query
    """
    try:
        logger.info(f"Received query: {request.query}")
        graph = get_graph_instance()
        # Prepare inputs for the graph
        # Prepare input state
        input_state: AgentState = {
            "query": request.query,
            "context": request.context or {},
            "mode": request.mode,
        }
        # inputs = {
        #     "query": request.query,
        #     "context": request.context or {},
        #     "mode": request.mode,
        # }

        # Process the query through the graph
        # This will be implemented later, but this is how we'll interface with it
        result = await graph.ainvoke(input_state)

        logger.info(f"Query processed by {result.get('agent_type', 'unknown')} agent")

        # Return the response
        return QueryResponse(
            response=result.get("response", "No response generated"),
            sources=result.get("sources", []),
            agent_type=result.get("agent_type", "unknown"),
            metadata=result.get("metadata", {}),
        )

    except ValueError as e:
        logger.error(f"Value error processing query: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Register the router
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agentic_rag_personal_chat_system.backend.src.api.router:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
    )
