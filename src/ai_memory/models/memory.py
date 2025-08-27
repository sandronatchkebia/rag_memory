"""Memory and insight data models."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories that can be generated."""
    
    CONVERSATION = "conversation"
    INSIGHT = "insight"
    PATTERN = "pattern"
    REMINDER = "reminder"
    ANNIVERSARY = "anniversary"
    UNFINISHED = "unfinished"


class Memory(BaseModel):
    """Represents a memory or insight extracted from conversations."""
    
    id: str = Field(..., description="Unique identifier for the memory")
    type: MemoryType = Field(..., description="Type of memory")
    title: str = Field(..., description="Title or summary of the memory")
    content: str = Field(..., description="Detailed content of the memory")
    source_conversations: List[str] = Field(..., description="IDs of source conversations")
    participants: List[str] = Field(..., description="IDs of relevant participants")
    created_at: datetime = Field(default_factory=datetime.now, description="When the memory was created")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
