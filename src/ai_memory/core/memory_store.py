"""Memory store for managing conversation data."""

import json
import uuid
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
import logging

from ..models.conversation import Conversation, Message
from ..rag.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class MemoryStore:
    """Manages storage and retrieval of conversation memories using ChromaDB."""
    
    def __init__(self, db_path: str = "./data/chroma_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._embedding_manager = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the memory store and ChromaDB."""
        try:
            logger.info("Initializing ChromaDB memory store...")
            
            # Initialize ChromaDB client
            self._client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding manager
            self._embedding_manager = EmbeddingManager()
            await self._embedding_manager.initialize()
            
            # Get or create collection
            try:
                self._collection = self._client.get_collection(name="conversation_memories")
                # Check if existing collection has the right dimension
                collection_metadata = self._collection.metadata
                existing_dim = collection_metadata.get("embedding_dimension", 0)
                current_dim = self._embedding_manager.embedding_dimension
                
                if existing_dim != current_dim:
                    logger.warning(f"Existing collection has dimension {existing_dim}, but current model has {current_dim}")
                    logger.info("Deleting existing collection to recreate with correct dimensions...")
                    self._client.delete_collection(name="conversation_memories")
                    self._collection = self._client.create_collection(
                        name="conversation_memories",
                        metadata={
                            "description": "AI Memory conversation embeddings and metadata",
                            "embedding_dimension": current_dim
                        }
                    )
                    logger.info("Collection recreated with correct dimensions")
                else:
                    logger.info(f"Using existing collection with dimension {existing_dim}")
                    
            except Exception:
                # Collection doesn't exist, create new one
                self._collection = self._client.create_collection(
                    name="conversation_memories",
                    metadata={
                        "description": "AI Memory conversation embeddings and metadata",
                        "embedding_dimension": self._embedding_manager.embedding_dimension
                    }
                )
                logger.info(f"Created new collection with dimension {self._embedding_manager.embedding_dimension}")
            
            self._initialized = True
            logger.info("Memory store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory store: {e}")
            raise
    
    async def add_conversation(self, conversation: Conversation) -> str:
        """Add a conversation to the store."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check if conversation is too large and needs chunking
            if len(conversation.messages) > 1000:  # Threshold for mega-conversations
                logger.info(f"Conversation {conversation.id} has {len(conversation.messages)} messages, chunking...")
                return await self._add_chunked_conversation(conversation)
            
            # Filter messages and prepare for embedding
            valid_messages = []
            message_texts = []
            
            for msg in conversation.messages:
                if msg.content and msg.content.strip():
                    valid_messages.append(msg)
                    message_texts.append(msg.content)
            
            if not message_texts:
                logger.warning(f"Conversation {conversation.id} has no valid messages")
                return conversation.id
            
            # Generate embeddings
            embeddings = await self._embedding_manager.generate_embeddings(message_texts)
            
            # Ensure we have the same number of embeddings as valid messages
            if len(embeddings) != len(valid_messages):
                logger.error(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(valid_messages)} messages")
                # Truncate or pad embeddings to match
                if len(embeddings) > len(valid_messages):
                    embeddings = embeddings[:len(valid_messages)]
                else:
                    # Pad with zero vectors
                    dim = self._embedding_manager.embedding_dimension
                    while len(embeddings) < len(valid_messages):
                        embeddings.append([0.0] * dim)
            
            # Prepare documents for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, (message, embedding) in enumerate(zip(valid_messages, embeddings)):
                # Create unique ID for this message
                message_id = f"{conversation.id}:msg_{i}"
                
                # Prepare metadata
                metadata = {
                    "conversation_id": conversation.id,
                    "platform": conversation.platform,
                    "message_id": message.id,
                    "sender_id": message.sender_id,
                    "timestamp": message.timestamp.isoformat(),
                    "language": message.language or "unknown",
                    "participants": json.dumps([p.name for p in conversation.participants]),
                    "participant_emails": json.dumps([p.email for p in conversation.participants if p.email]),
                    "is_self": any(p.is_self for p in conversation.participants if p.id == message.sender_id),
                    "message_type": message.message_type,
                    "thread_index": message.metadata.get("thread_index", 0),
                    "platform_metadata": json.dumps(message.metadata)
                }
                
                # Add conversation-level metadata
                if conversation.title:
                    metadata["conversation_title"] = conversation.title
                
                documents.append(message.content)
                metadatas.append(metadata)
                ids.append(message_id)
            
            # Final safety check - ensure all arrays have the same length
            array_lengths = [len(documents), len(metadatas), len(ids), len(embeddings)]
            if len(set(array_lengths)) != 1:
                logger.error(f"Array length mismatch: documents={len(documents)}, metadatas={len(metadatas)}, ids={len(ids)}, embeddings={len(embeddings)}")
                # Use the shortest length
                min_length = min(array_lengths)
                documents = documents[:min_length]
                metadatas = metadatas[:min_length]
                ids = ids[:min_length]
                embeddings = embeddings[:min_length]
            
            # Add to ChromaDB
            if documents:
                self._collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                
                logger.info(f"Added conversation {conversation.id} with {len(documents)} messages")
            
            return conversation.id
            
        except Exception as e:
            logger.error(f"Error adding conversation {conversation.id}: {e}")
            raise
    
    async def _add_chunked_conversation(self, conversation: Conversation) -> str:
        """Add a large conversation by processing it in chunks."""
        try:
            # Sort messages by timestamp to maintain chronological order
            sorted_messages = sorted(conversation.messages, key=lambda x: x.timestamp)
            
            # Process in chunks of 1000 messages
            chunk_size = 1000
            total_processed = 0
            
            for i in range(0, len(sorted_messages), chunk_size):
                chunk_messages = sorted_messages[i:i + chunk_size]
                
                # Create a sub-conversation for this chunk
                chunk_id = f"{conversation.id}:chunk_{i//chunk_size}"
                chunk_conversation = Conversation(
                    id=chunk_id,
                    title=f"{conversation.title or 'Conversation'} (Part {i//chunk_size + 1})" if conversation.title else f"Conversation (Part {i//chunk_size + 1})",
                    platform=conversation.platform,
                    participants=conversation.participants,
                    messages=chunk_messages,
                    start_date=chunk_messages[0].timestamp if chunk_messages else conversation.start_date,
                    last_activity=chunk_messages[-1].timestamp if chunk_messages else conversation.last_activity,
                    metadata={
                        **conversation.metadata,
                        "original_conversation_id": conversation.id,
                        "chunk_index": i // chunk_size,
                        "total_chunks": (len(sorted_messages) + chunk_size - 1) // chunk_size,
                        "chunk_start_message": i,
                        "chunk_end_message": min(i + chunk_size, len(sorted_messages))
                    }
                )
                
                # Process this chunk normally
                await self.add_conversation(chunk_conversation)
                total_processed += len(chunk_messages)
                
                logger.info(f"Processed chunk {i//chunk_size + 1} of conversation {conversation.id}: {len(chunk_messages)} messages")
            
            logger.info(f"Successfully chunked and processed conversation {conversation.id}: {total_processed} total messages in {(len(sorted_messages) + chunk_size - 1) // chunk_size} chunks")
            return conversation.id
            
        except Exception as e:
            logger.error(f"Error chunking conversation {conversation.id}: {e}")
            raise
    
    async def search_conversations(self, query: str, limit: int = 10, 
                                  filters: Optional[Dict[str, Any]] = None,
                                  context_window: int = 5) -> List[Dict[str, Any]]:
        """Search conversations by query and return results with context.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            filters: Optional filters for platform, language, participant, etc.
            context_window: Number of messages before/after to include for context
            
        Returns:
            List of dictionaries containing:
            - target_message: The most relevant message
            - context_before: Messages before the target
            - context_after: Messages after the target
            - conversation_metadata: Platform, participants, etc.
            - relevance_score: Similarity score
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await self._embedding_manager.generate_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ["platform", "language", "sender_id"]:
                        where_clause[key] = value
                    elif key == "participant_name":
                        where_clause["participants"] = {"$contains": value}
                    elif key == "date_range":
                        # Handle date filtering
                        start_date, end_date = value
                        where_clause["timestamp"] = {
                            "$gte": start_date.isoformat(),
                            "$lte": end_date.isoformat()
                        }
            
            # Search in ChromaDB - get more results to account for context grouping
            search_limit = limit * 3  # Get more results to group by conversation
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=search_limit,
                where=where_clause if where_clause else None,
                include=["metadatas", "distances", "documents"]
            )
            
            # Process results and group by conversation
            conversation_groups = {}
            
            if results["ids"] and results["ids"][0] and results["documents"] and results["documents"][0]:
                for i, (doc_id, metadata, distance, content) in enumerate(zip(
                    results["ids"][0], 
                    results["metadatas"][0], 
                    results["distances"][0],
                    results["documents"][0]
                )):
                    conversation_id = metadata.get("conversation_id", "unknown")
                    message_index = int(doc_id.split(":msg_")[-1]) if ":msg_" in doc_id else 0
                    
                    if conversation_id not in conversation_groups:
                        conversation_groups[conversation_id] = []
                    
                    # Convert distance to similarity score
                    similarity = 1.0 - distance
                    
                    # Create message object
                    message = Message(
                        id=metadata["message_id"],
                        content=content,
                        sender_id=metadata["sender_id"],
                        timestamp=metadata["timestamp"],
                        platform=metadata["platform"],
                        message_type=metadata.get("message_type", "text"),
                        language=metadata.get("language"),
                        metadata=json.loads(metadata.get("platform_metadata", "{}"))
                    )
                    
                    conversation_groups[conversation_id].append({
                        "message": message,
                        "similarity": similarity,
                        "index": message_index,
                        "metadata": metadata
                    })
            
            # For each conversation, find the best message and build context
            context_results = []
            
            for conversation_id, messages in conversation_groups.items():
                # Sort by similarity score (best first)
                messages.sort(key=lambda x: x["similarity"], reverse=True)
                best_message_data = messages[0]
                
                # Get conversation metadata
                conv_metadata = best_message_data["metadata"]
                
                # Build conversation segment with context
                conversation_segment = await self._build_conversation_segment(
                    conversation_id, 
                    best_message_data["index"], 
                    context_window,
                    conv_metadata
                )
                
                if conversation_segment:
                    context_results.append({
                        "target_message": best_message_data["message"],
                        "context_before": conversation_segment.get("context_before", []),
                        "context_after": conversation_segment.get("context_after", []),
                        "conversation_metadata": {
                            "id": conversation_id,
                            "platform": conv_metadata.get("platform"),
                            "participants": json.loads(conv_metadata.get("participants", "[]")),
                            "title": conv_metadata.get("conversation_title"),
                            "timestamp": conv_metadata.get("timestamp")
                        },
                        "relevance_score": best_message_data["similarity"],
                        "total_messages_in_conversation": len(messages)
                    })
            
            # Sort by relevance score and return top results
            context_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return context_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []
    
    async def _build_conversation_segment(self, conversation_id: str, target_index: int, 
                                        context_window: int, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a conversation segment with context around a target message."""
        try:
            # Get all messages from this conversation
            results = self._collection.query(
                query_texts=["conversation"],
                n_results=10000,  # Get all messages
                where={"conversation_id": conversation_id},
                include=["metadatas", "documents"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                return None
            
            # Sort messages by index
            messages_with_indices = []
            for i, (doc_id, msg_metadata, content) in enumerate(zip(
                results["ids"][0],
                results["metadatas"][0],
                results["documents"][0]
            )):
                msg_index = int(doc_id.split(":msg_")[-1]) if ":msg_" in doc_id else 0
                messages_with_indices.append({
                    "index": msg_index,
                    "content": content,
                    "metadata": msg_metadata
                })
            
            messages_with_indices.sort(key=lambda x: x["index"])
            
            # Find target message and build context
            target_message = None
            context_before = []
            context_after = []
            
            for msg_data in messages_with_indices:
                if msg_data["index"] == target_index:
                    target_message = msg_data
                    break
            
            if not target_message:
                return None
            
            # Get context before
            start_idx = max(0, target_index - context_window)
            for i in range(start_idx, target_index):
                for msg_data in messages_with_indices:
                    if msg_data["index"] == i:
                        context_before.append(msg_data)
                        break
            
            # Get context after
            end_idx = min(len(messages_with_indices), target_index + context_window + 1)
            for i in range(target_index + 1, end_idx):
                for msg_data in messages_with_indices:
                    if msg_data["index"] == i:
                        context_after.append(msg_data)
                        break
            
            return {
                "context_before": context_before,
                "context_after": context_after
            }
            
        except Exception as e:
            logger.error(f"Error building conversation segment for {conversation_id}: {e}")
            return None
    
    async def search_by_participant(self, participant_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversations by participant."""
        return await self.search_conversations(
            query=f"conversations with {participant_name}",
            limit=limit,
            filters={"participant_name": participant_name}
        )
    
    async def search_by_time_range(self, start_date: str, end_date: str, 
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversations within a time range."""
        from datetime import datetime
        
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            return await self.search_conversations(
                query="conversations in this time period",
                limit=limit,
                filters={"date_range": (start_dt, end_dt)}
            )
        except Exception as e:
            logger.error(f"Error parsing date range: {e}")
            return []
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by ID."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Search for messages in this conversation
            results = self._collection.query(
                query_texts=["conversation"],
                n_results=1000,  # Get all messages
                where={"conversation_id": conversation_id},
                include=["metadatas", "documents"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                return None
            
            # Reconstruct conversation
            messages = []
            participants = set()
            
            for i, (doc_id, metadata, content) in enumerate(zip(
                results["ids"][0],
                results["metadatas"][0],
                results["documents"][0]
            )):
                # Create message
                message = Message(
                    id=metadata["message_id"],
                    content=content,
                    sender_id=metadata["sender_id"],
                    timestamp=metadata["timestamp"],
                    platform=metadata["platform"],
                    message_type=metadata["message_type"],
                    language=metadata["language"],
                    metadata=json.loads(metadata["platform_metadata"])
                )
                messages.append(message)
                
                # Collect participants
                participants.add(metadata["sender_id"])
            
            # Sort messages by timestamp
            messages.sort(key=lambda x: x.timestamp)
            
            # Create conversation object
            # Note: This is a simplified reconstruction - you might want to store full conversation data separately
            conversation = Conversation(
                id=conversation_id,
                platform=messages[0].platform if messages else "unknown",
                participants=[],  # Would need to reconstruct from stored data
                messages=messages,
                start_date=messages[0].timestamp if messages else None,
                last_activity=messages[-1].timestamp if messages else None
            )
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            return None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored conversations."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get collection info
            collection_info = self._collection.get()
            
            # Count by platform
            platform_counts = {}
            language_counts = {}
            
            if collection_info["metadatas"]:
                for metadata in collection_info["metadatas"]:
                    platform = metadata.get("platform", "unknown")
                    language = metadata.get("language", "unknown")
                    
                    platform_counts[platform] = platform_counts.get(platform, 0) + 1
                    language_counts[language] = language_counts.get(language, 0) + 1
            
            return {
                "total_messages": len(collection_info["ids"]),
                "platform_distribution": platform_counts,
                "language_distribution": language_counts,
                "embedding_dimension": self._embedding_manager.embedding_dimension
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    async def clear_all(self):
        """Clear all stored data."""
        if not self._initialized:
            await self.initialize()
        
        try:
            self._client.delete_collection("conversation_memories")
            self._collection = None
            self._initialized = False
            logger.info("Memory store cleared")
        except Exception as e:
            logger.error(f"Error clearing memory store: {e}")
            raise
    
    @property
    def is_ready(self) -> bool:
        """Check if the memory store is ready to use."""
        return self._initialized and self._collection is not None
