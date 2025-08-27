"""Clustering manager for AI Memory system."""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from loguru import logger

from ..models.clustering import (
    ClusterType, ContentCluster, TemporalCluster, 
    RelationshipCluster, ThreadCluster, ClusteringResult
)
from ..models.conversation import Message, Conversation
from .embedding_manager import EmbeddingManager


class ClusteringManager:
    """Manages clustering operations for messages and conversations."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self._initialized = False
        
    async def initialize(self):
        """Initialize the clustering manager."""
        try:
            # Import HDBSCAN here to avoid import issues
            import hdbscan
            self._initialized = True
            logger.info("ClusteringManager initialized with HDBSCAN")
        except ImportError as e:
            logger.error(f"Failed to import HDBSCAN: {e}")
            raise
    
    async def cluster_content(self, messages: List[Message], 
                            min_cluster_size: int = 3,
                            min_samples: int = 2) -> ClusteringResult:
        """Cluster messages by content similarity using HDBSCAN."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Generate embeddings for all messages
            texts = [msg.content for msg in messages if msg.content.strip()]
            if len(texts) < min_cluster_size:
                return ClusteringResult(
                    cluster_type=ClusterType.CONTENT,
                    clusters=[],
                    metadata={"reason": "insufficient_messages"},
                    processing_time=time.time() - start_time,
                    created_at=datetime.now()
                )
            
            embeddings = await self.embedding_manager.generate_embeddings(texts)
            
            # Perform HDBSCAN clustering
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Create content clusters
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                    
                # Get messages in this cluster
                cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
                cluster_messages = [messages[i] for i in cluster_indices]
                cluster_embeddings = [embeddings[i] for i in cluster_indices]
                
                # Calculate centroid
                centroid = np.mean(cluster_embeddings, axis=0).tolist()
                
                # Extract keywords (simple approach - could be enhanced with NLP)
                all_text = " ".join([msg.content for msg in cluster_messages])
                keywords = self._extract_keywords(all_text)
                
                # Determine topic
                topic = self._determine_topic(cluster_messages, keywords)
                
                cluster = ContentCluster(
                    id=f"content_{label}_{int(time.time())}",
                    topic=topic,
                    keywords=keywords,
                    message_ids=[msg.id for msg in cluster_messages],
                    centroid_embedding=centroid,
                    cluster_score=clusterer.probabilities_[cluster_indices].mean() if hasattr(clusterer, 'probabilities_') else 0.8,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                clusters.append(cluster.dict())
            
            processing_time = time.time() - start_time
            
            return ClusteringResult(
                cluster_type=ClusterType.CONTENT,
                clusters=clusters,
                metadata={
                    "total_messages": len(messages),
                    "clustered_messages": len([c for c in cluster_labels if c != -1]),
                    "noise_points": len([c for c in cluster_labels if c == -1]),
                    "algorithm": "HDBSCAN"
                },
                processing_time=processing_time,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in content clustering: {e}")
            raise
    
    async def cluster_temporal(self, messages: List[Message], 
                              time_period: str = "month") -> ClusteringResult:
        """Cluster messages by temporal patterns."""
        start_time = time.time()
        
        try:
            # Group messages by time period
            temporal_groups = {}
            
            for msg in messages:
                if time_period == "day":
                    key = msg.timestamp.date()
                elif time_period == "week":
                    key = msg.timestamp.date() - timedelta(days=msg.timestamp.weekday())
                elif time_period == "month":
                    key = msg.timestamp.replace(day=1).date()
                elif time_period == "year":
                    key = msg.timestamp.replace(month=1, day=1).date()
                else:
                    key = msg.timestamp.date()
                
                if key not in temporal_groups:
                    temporal_groups[key] = []
                temporal_groups[key].append(msg)
            
            # Create temporal clusters
            clusters = []
            for period_start, period_messages in temporal_groups.items():
                if len(period_messages) == 0:
                    continue
                
                # Calculate activity level based on message count
                max_messages = max(len(msgs) for msgs in temporal_groups.values())
                activity_level = len(period_messages) / max_messages if max_messages > 0 else 0
                
                # Determine end date
                if time_period == "day":
                    end_date = period_start
                elif time_period == "week":
                    end_date = period_start + timedelta(days=6)
                elif time_period == "month":
                    if period_start.month == 12:
                        end_date = period_start.replace(year=period_start.year + 1, month=1) - timedelta(days=1)
                    else:
                        end_date = period_start.replace(month=period_start.month + 1) - timedelta(days=1)
                else:  # year
                    end_date = period_start.replace(year=period_start.year + 1) - timedelta(days=1)
                
                # Extract dominant topics (simple approach)
                all_text = " ".join([msg.content for msg in period_messages])
                dominant_topics = self._extract_keywords(all_text)[:5]
                
                cluster = TemporalCluster(
                    id=f"temporal_{period_start}_{int(time.time())}",
                    time_period=time_period,
                    start_date=datetime.combine(period_start, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.max.time()),
                    message_ids=[msg.id for msg in period_messages],
                    activity_level=activity_level,
                    dominant_topics=dominant_topics
                )
                clusters.append(cluster.dict())
            
            processing_time = time.time() - start_time
            
            return ClusteringResult(
                cluster_type=ClusterType.TEMPORAL,
                clusters=clusters,
                metadata={
                    "total_messages": len(messages),
                    "time_period": time_period,
                    "periods_analyzed": len(temporal_groups)
                },
                processing_time=processing_time,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in temporal clustering: {e}")
            raise
    
    async def cluster_relationships(self, conversations: List[Conversation]) -> ClusteringResult:
        """Cluster by relationships and interaction patterns."""
        start_time = time.time()
        
        try:
            # Analyze participant interactions
            participant_data = {}
            
            for conv in conversations:
                for msg in conv.messages:
                    sender = msg.sender_id
                    
                    if sender not in participant_data:
                        participant_data[sender] = {
                            'message_count': 0,
                            'platforms': set(),
                            'first_interaction': msg.timestamp,
                            'last_interaction': msg.timestamp,
                            'conversation_ids': set(),
                            'message_ids': [],
                            'content': []
                        }
                    
                    participant_data[sender]['message_count'] += 1
                    participant_data[sender]['platforms'].add(msg.platform)
                    participant_data[sender]['first_interaction'] = min(
                        participant_data[sender]['first_interaction'], msg.timestamp
                    )
                    participant_data[sender]['last_interaction'] = max(
                        participant_data[sender]['last_interaction'], msg.timestamp
                    )
                    participant_data[sender]['conversation_ids'].add(conv.id)
                    participant_data[sender]['message_ids'].append(msg.id)
                    participant_data[sender]['content'].append(msg.content)
            
            # Create relationship clusters
            clusters = []
            max_messages = max(data['message_count'] for data in participant_data.values()) if participant_data else 1
            
            for participant_id, data in participant_data.items():
                # Calculate relationship strength
                relationship_strength = data['message_count'] / max_messages
                
                # Extract dominant topics
                all_content = " ".join(data['content'])
                dominant_topics = self._extract_keywords(all_content)[:5]
                
                cluster = RelationshipCluster(
                    id=f"relationship_{participant_id}_{int(time.time())}",
                    participant_id=participant_id,
                    participant_name=participant_id,  # Could be enhanced with name resolution
                    platforms=list(data['platforms']),
                    message_count=data['message_count'],
                    first_interaction=data['first_interaction'],
                    last_interaction=data['last_interaction'],
                    conversation_ids=list(data['conversation_ids']),
                    message_ids=data['message_ids'],
                    dominant_topics=dominant_topics,
                    relationship_strength=relationship_strength
                )
                clusters.append(cluster.dict())
            
            processing_time = time.time() - start_time
            
            return ClusteringResult(
                cluster_type=ClusterType.RELATIONSHIP,
                clusters=clusters,
                metadata={
                    "total_conversations": len(conversations),
                    "total_participants": len(participant_data),
                    "total_messages": sum(data['message_count'] for data in participant_data.values())
                },
                processing_time=processing_time,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in relationship clustering: {e}")
            raise
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # Simple keyword extraction - could be enhanced with NLP libraries
        import re
        
        # Remove common words and punctuation
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(words)
        
        # Return most common words
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def _determine_topic(self, messages: List[Message], keywords: List[str]) -> str:
        """Determine the main topic of a cluster."""
        if not keywords:
            return "general"
        
        # Simple topic determination based on keywords
        topic_keywords = {
            'work': ['work', 'job', 'office', 'meeting', 'project', 'deadline'],
            'school': ['class', 'homework', 'assignment', 'exam', 'study', 'university'],
            'travel': ['travel', 'trip', 'vacation', 'hotel', 'flight', 'destination'],
            'family': ['family', 'mom', 'dad', 'sister', 'brother', 'home'],
            'friends': ['friend', 'hangout', 'party', 'dinner', 'coffee'],
            'technology': ['computer', 'phone', 'app', 'software', 'tech', 'coding']
        }
        
        # Find the best matching topic
        best_topic = "general"
        best_score = 0
        
        for topic, topic_words in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in topic_words)
            if score > best_score:
                best_score = score
                best_topic = topic
        
        return best_topic
