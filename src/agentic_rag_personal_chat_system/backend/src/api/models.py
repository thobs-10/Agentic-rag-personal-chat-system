"""
Models for API requests and responses.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for user queries."""

    query: str = Field(..., description="The user's query text")
    context: Optional[Dict[str, str]] = Field(
        default=None, description="Optional contextual information"
    )
    mode: str = Field(
        default="auto",
        description="Query mode: 'personal', 'technical', or 'auto' for automatic routing",
    )


class ResponseItem(BaseModel):
    """A single piece of information in the response."""

    text: str = Field(..., description="The text content")
    source: Optional[str] = Field(None, description="Source document reference")
    relevance: Optional[float] = Field(None, description="Relevance score")


class QueryResponse(BaseModel):
    """Response model for query answers."""

    response: str = Field(..., description="The generated response to the query")
    sources: List[ResponseItem] = Field(
        default_factory=list, description="Sources used to generate the response"
    )
    agent_type: str = Field(
        ..., description="The agent that handled the query (personal/technical)"
    )
    metadata: Dict = Field(
        default_factory=dict, description="Additional metadata about the response"
    )


class ErrorResponse(BaseModel):
    """Model for error responses."""

    error: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(default=None, description="Additional error details")
