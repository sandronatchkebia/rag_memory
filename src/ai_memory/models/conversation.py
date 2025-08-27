"""Conversation data models."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Participant(BaseModel):
    """Represents a participant in a conversation."""
    
    id: str = Field(..., description="Unique identifier for the participant")
    name: str = Field(..., description="Display name of the participant")
    email: Optional[str] = Field(None, description="Email address if available")
    platform_id: Optional[str] = Field(None, description="Platform-specific ID")
    is_self: bool = Field(False, description="Whether this participant is the user")


class Message(BaseModel):
    """Represents a single message in a conversation."""
    
    id: str = Field(..., description="Unique identifier for the message")
    content: str = Field(..., description="Text content of the message")
    sender_id: str = Field(..., description="ID of the message sender")
    timestamp: datetime = Field(..., description="When the message was sent")
    platform: str = Field(..., description="Platform where the message originated")
    message_type: str = Field("text", description="Type of message (text, image, etc.)")
    language: Optional[str] = Field(None, description="Detected language of the message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Platform-specific metadata")


class Conversation(BaseModel):
    """Represents a conversation thread."""
    
    id: str = Field(..., description="Unique identifier for the conversation")
    title: Optional[str] = Field(None, description="Conversation title or subject")
    platform: str = Field(..., description="Platform where the conversation occurred")
    participants: List[Participant] = Field(..., description="List of conversation participants")
    messages: List[Message] = Field(..., description="Messages in the conversation")
    start_date: datetime = Field(..., description="When the conversation started")
    last_activity: datetime = Field(..., description="Last activity in the conversation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
