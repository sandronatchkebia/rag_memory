"""Clustering models for AI Memory system."""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime, date
from pydantic import BaseModel, Field
from enum import Enum


class ClusterType(str, Enum):
    """Types of clusters."""
    CONTENT = "content"           # Topic-based clustering
    TEMPORAL = "temporal"         # Time-based clustering
    RELATIONSHIP = "relationship" # Sender/participant clustering
    THREAD = "thread"            # Conversation thread clustering


class ContentCluster(BaseModel):
    """Content-based topic cluster."""
    id: str
    cluster_type: ClusterType = ClusterType.CONTENT
    topic: str
    keywords: List[str]
    message_ids: List[str]
    centroid_embedding: Optional[List[float]] = None
    cluster_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TemporalCluster(BaseModel):
    """Time-based cluster."""
    id: str
    cluster_type: ClusterType = ClusterType.TEMPORAL
    time_period: str  # "day", "week", "month", "year", "custom"
    start_date: datetime
    end_date: datetime
    message_ids: List[str]
    activity_level: float = Field(ge=0.0, le=1.0)  # How active this period was
    dominant_topics: List[str] = []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RelationshipCluster(BaseModel):
    """Relationship-based cluster."""
    id: str
    cluster_type: ClusterType = ClusterType.RELATIONSHIP
    participant_id: str
    participant_name: Optional[str] = None
    platforms: List[str]
    message_count: int
    first_interaction: datetime
    last_interaction: datetime
    conversation_ids: List[str]
    message_ids: List[str]
    dominant_topics: List[str] = []
    relationship_strength: float = Field(ge=0.0, le=1.0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ThreadCluster(BaseModel):
    """Conversation thread cluster."""
    id: str
    cluster_type: ClusterType = ClusterType.THREAD
    conversation_id: str
    platform: str
    participants: List[str]
    message_count: int
    start_date: datetime
    end_date: datetime
    dominant_topics: List[str] = []
    thread_length: int
    activity_pattern: str  # "continuous", "sporadic", "burst"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ClusteringResult(BaseModel):
    """Result of clustering operation."""
    cluster_type: ClusterType
    clusters: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QueryIntent(BaseModel):
    """Parsed query intent for clustering."""
    temporal_filters: List[Dict[str, Any]] = []
    content_filters: List[str] = []
    relationship_filters: List[str] = []
    platform_filters: List[str] = []
    query_type: str  # "first_occurrence", "most_frequent", "pattern_analysis", etc.
    confidence: float = Field(ge=0.0, le=1.0)
