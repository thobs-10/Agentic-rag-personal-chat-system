"""
API router for handling chat queries.
"""

import json

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from agentic_rag_personal_chat_system.backend.src.agents.personal_assistant import (
    PersonalAssistant,
)
from agentic_rag_personal_chat_system.backend.src.agents.technical_assistant import (
    TechnicalAssistant,
)
from agentic_rag_personal_chat_system.backend.src.api.models import (
    ErrorResponse,
    QueryRequest,
    QueryResponse,
)
from agentic_rag_personal_chat_system.backend.src.config.backend_config import APIConfig
from agentic_rag_personal_chat_system.backend.src.graph import (
    AgentState,
    classify_query,
    get_graph_instance,
)

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


@api_router.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    Stream a response token-by-token using Server-Sent Events.

    Events are newline-delimited JSON with a 'type' field:
      - agent_type: which assistant was selected
      - token:      a single text chunk from the LLM
      - sources:    retrieved documents (sent after streaming is complete)
      - error:      friendly error message
    Terminated by: data: [DONE]
    """

    async def generate():
        try:
            mode = request.mode
            agent_type = await classify_query(request.query) if mode == "auto" else mode
            yield f"data: {json.dumps({'type': 'agent_type', 'content': agent_type})}\n\n"

            agent = TechnicalAssistant() if agent_type == "technical" else PersonalAssistant()
            async for event in agent.astream_query(request.query):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': 'An error occurred. Please try again.'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
