"""API request and response models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for memory queries."""
    
    query: str = Field(..., description="Natural language query")
    user_id: Optional[str] = Field(None, description="User identifier")
    limit: int = Field(10, description="Maximum number of results")
    include_metadata: bool = Field(True, description="Include metadata in response")


class QueryResponse(BaseModel):
    """Response model for memory queries."""
    
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    total_results: int = Field(..., description="Total number of results")
    processing_time: float = Field(..., description="Query processing time in seconds")
    suggestions: List[str] = Field(default_factory=list, description="Suggested follow-up queries")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Current timestamp")
