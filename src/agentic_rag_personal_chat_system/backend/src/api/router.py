"""
API router for handling chat queries.
"""

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from agentic_rag_personal_chat_system.backend.src.api.models import (
    ErrorResponse,
    QueryRequest,
    QueryResponse,
)
from agentic_rag_personal_chat_system.backend.src.config.backend_config import APIConfig
from agentic_rag_personal_chat_system.backend.src.graph import AgentState, get_graph_instance

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG Personal Chat System",
    description="API for the Agentic RAG Personal Chat System",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router
api_router = APIRouter()


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
        input_state: AgentState = {
            "query": request.query,
            "context": request.context or {},
            "mode": request.mode,
        }

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
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agentic_rag_personal_chat_system.backend.src.api.router:app",
        host=APIConfig.host,
        port=APIConfig.port,
        reload=APIConfig.reload,
    )
