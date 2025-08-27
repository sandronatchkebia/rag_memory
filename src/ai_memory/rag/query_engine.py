"""Processes natural language queries and generates responses using LLM."""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
import openai
from loguru import logger

from ..models.conversation import Conversation, Message
from ..models.memory import Memory, MemoryType
from ..models.clustering import QueryIntent, ClusterType

if TYPE_CHECKING:
    from ..core.memory_store import MemoryStore
    from .clustering_manager import ClusteringManager
    from .query_parser import QueryParser


class QueryEngine:
    """Processes queries and generates intelligent responses using LLM."""
    
    def __init__(self, memory_store: 'MemoryStore', 
                 clustering_manager: Optional['ClusteringManager'] = None,
                 query_parser: Optional['QueryParser'] = None):
        self.memory_store = memory_store
        self.clustering_manager = clustering_manager
        self.query_parser = query_parser
        self._initialized = False
        self._openai_client = None
        self._conversation_history = []
        self._max_context_length = 8000  # Tokens for context
        self._max_results = 10  # Max search results to consider
    
    async def initialize(self):
        """Initialize the query engine with OpenAI client."""
        try:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self._openai_client = openai.OpenAI(api_key=api_key)
            self._initialized = True
            logger.info("QueryEngine initialized with OpenAI")
            
        except Exception as e:
            logger.error(f"Failed to initialize QueryEngine: {e}")
            raise
    
    async def query(self, question: str, limit: int = 5, 
                   include_metadata: bool = True, 
                   conversation_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process a natural language question and return an intelligent answer.
        
        Args:
            question: Natural language question
            limit: Maximum number of search results to consider
            include_metadata: Whether to include source metadata
            conversation_context: Previous conversation context for follow-up questions
        
        Returns:
            Dict containing answer, sources, and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Step 1: Parse query intent if parser is available
            query_intent = None
            if self.query_parser:
                query_intent = self.query_parser.parse_query(question)
                logger.info(f"Query intent: {query_intent.query_type} (confidence: {query_intent.confidence:.2f})")
            
            # Step 2: Handle clustering-specific queries
            if query_intent and query_intent.query_type in ["first_occurrence", "most_frequent", "temporal_analysis"]:
                return await self._handle_clustering_query(question, query_intent, limit, include_metadata)
            
            # Step 3: Standard query processing
            enhanced_question = await self._enhance_query_with_context(question, conversation_context)
            
            # Step 4: Search for relevant context
            search_results = await self._search_relevant_context(enhanced_question, limit)
            
            if not search_results:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "confidence": "low",
                    "query": question,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Step 5: Prepare context for LLM
            context_text = await self._prepare_context_for_llm(search_results, include_metadata)
            
            # Step 6: Generate answer using LLM
            answer = await self._generate_llm_answer(question, context_text)
            
            # Step 7: Prepare sources and metadata
            sources = await self._prepare_sources(search_results, include_metadata)
            
            # Step 8: Update conversation history
            self._conversation_history.append({
                "question": question,
                "answer": answer,
                "sources_count": len(sources),
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 10 conversations
            if len(self._conversation_history) > 10:
                self._conversation_history = self._conversation_history[-10:]
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": self._assess_confidence(search_results),
                "query": question,
                "timestamp": datetime.now().isoformat(),
                "context_length": len(context_text),
                "results_considered": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "confidence": "error",
                "query": question,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _enhance_query_with_context(self, question: str, 
                                        conversation_context: Optional[List[Dict]]) -> str:
        """Enhance the query using conversation context for better search."""
        if not conversation_context:
            return question
        
        # Extract key entities and topics from recent conversation
        context_summary = " ".join([ctx.get("question", "") + " " + ctx.get("answer", "") 
                                  for ctx in conversation_context[-3:]])
        
        # Create an enhanced query that includes context
        enhanced = f"{question} (Context: {context_summary[:200]}...)"
        return enhanced
    
    async def _search_relevant_context(self, query: str, limit: int) -> List[Tuple[Message, float]]:
        """Search for relevant context using the memory store."""
        try:
            # First try semantic search
            results = await self.memory_store.search_conversations(query, limit=limit)
            
            # If we don't get enough results, try with different strategies
            if len(results) < 3:
                # Try breaking down the query into keywords
                keywords = query.split()
                for keyword in keywords[:3]:  # Try first 3 keywords
                    if len(keyword) > 3:  # Only meaningful keywords
                        keyword_results = await self.memory_store.search_conversations(keyword, limit=2)
                        results.extend(keyword_results)
                        if len(results) >= limit:
                            break
            
            # Remove duplicates and sort by similarity
            unique_results = {}
            for message, similarity in results:
                if message.id not in unique_results:
                    unique_results[message.id] = (message, similarity)
            
            # Sort by similarity and return top results
            sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)
            return sorted_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching for context: {e}")
            return []
    
    async def _prepare_context_for_llm(self, search_results: List[Tuple[Message, float]], 
                                     include_metadata: bool) -> str:
        """Prepare context text for the LLM with relevant information."""
        context_parts = []
        
        for i, (message, similarity) in enumerate(search_results):
            # Format each message with relevant metadata
            if include_metadata:
                context_part = f"[Source {i+1}] Platform: {message.platform}, Sender: {message.sender_id}, Time: {message.timestamp}, Similarity: {similarity:.3f}\nContent: {message.content}\n"
            else:
                context_part = f"[Source {i+1}] {message.content}\n"
            
            context_parts.append(context_part)
        
        # Join all context parts
        full_context = "\n".join(context_parts)
        
        # Truncate if too long (approximate token count)
        if len(full_context) > self._max_context_length:
            full_context = full_context[:self._max_context_length] + "\n[Context truncated...]"
        
        return full_context
    
    async def _generate_llm_answer(self, question: str, context: str) -> str:
        """Generate an answer using OpenAI GPT-4."""
        try:
            system_prompt = """You are an AI Memory Assistant that helps users find and understand information from their personal conversations, emails, and messages across various platforms (Gmail, Facebook Messenger, WhatsApp, Instagram, etc.).

Your role is to:
1. Analyze the provided context from the user's messages
2. Answer their question based on the available information
3. Provide clear, helpful responses that reference specific sources when possible
4. Be conversational but professional
5. If the context doesn't contain enough information, say so clearly
6. Respect privacy and handle personal information appropriately

Always cite your sources using [Source X] format when referencing specific messages."""

            user_prompt = f"""Question: {question}

Context from your messages:
{context}

Please provide a helpful answer based on the context above. If you reference specific information, cite the source using [Source X] format."""

            response = self._openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM answer: {e}")
            return f"I couldn't generate an answer due to a technical issue: {str(e)}"
    
    async def _prepare_sources(self, search_results: List[Tuple[Message, float]], 
                              include_metadata: bool) -> List[Dict[str, Any]]:
        """Prepare source information for the response."""
        sources = []
        
        for i, (message, similarity) in enumerate(search_results):
            source = {
                "id": message.id,
                "platform": message.platform,
                "sender": message.sender_id,
                "timestamp": message.timestamp.isoformat() if hasattr(message.timestamp, 'isoformat') else str(message.timestamp),
                "similarity": similarity,
                "content_preview": message.content[:100] + "..." if len(message.content) > 100 else message.content
            }
            
            if include_metadata:
                source.update({
                    "language": message.language,
                    "message_type": message.message_type,
                    "full_content": message.content
                })
            
            sources.append(source)
        
        return sources
    
    def _assess_confidence(self, search_results: List[Tuple[Message, float]]) -> str:
        """Assess confidence level based on search results."""
        if not search_results:
            return "none"
        
        # Calculate average similarity
        avg_similarity = sum(similarity for _, similarity in search_results) / len(search_results)
        
        if avg_similarity > 0.8:
            return "high"
        elif avg_similarity > 0.6:
            return "medium"
        elif avg_similarity > 0.4:
            return "low"
        else:
            return "very_low"
    
    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history for context."""
        return self._conversation_history.copy()
    
    async def clear_conversation_history(self):
        """Clear the conversation history."""
        self._conversation_history = []
    
    async def suggest_queries(self, user_id: str = None) -> List[str]:
        """Suggest interesting queries based on available data."""
        try:
            # Get some recent messages to suggest queries
            recent_results = await self.memory_store.search_conversations("", limit=20)
            
            suggestions = [
                "What did I talk about with friends recently?",
                "Show me my recent job-related conversations",
                "What languages do I communicate in?",
                "Find conversations about travel or vacations",
                "What are my most active communication platforms?",
                "Show me messages from last week",
                "Find conversations about birthdays or celebrations"
            ]
            
            # Add platform-specific suggestions
            stats = await self.memory_store.get_statistics()
            if stats.get('platform_distribution'):
                for platform in stats['platform_distribution'].keys():
                    suggestions.append(f"What did I discuss on {platform} recently?")
            
            return suggestions[:10]  # Return top 10 suggestions
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return ["What did I talk about recently?", "Show me my recent conversations"]
    
    async def _handle_clustering_query(self, question: str, query_intent: QueryIntent, 
                                     limit: int, include_metadata: bool) -> Dict[str, Any]:
        """Handle clustering-specific queries like temporal analysis and first occurrences."""
        try:
            if not self.clustering_manager:
                return {
                    "answer": "I don't have clustering capabilities enabled to answer this type of question.",
                    "sources": [],
                    "confidence": "none",
                    "query": question,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get all conversations for clustering analysis
            # This is a simplified approach - in production, you'd want to cache clustering results
            all_conversations = await self._get_all_conversations()
            
            if query_intent.query_type == "first_occurrence":
                return await self._handle_first_occurrence_query(question, query_intent, all_conversations)
            elif query_intent.query_type == "most_frequent":
                return await self._handle_most_frequent_query(question, query_intent, all_conversations)
            elif query_intent.query_type == "temporal_analysis":
                return await self._handle_temporal_analysis_query(question, query_intent, all_conversations)
            else:
                return await self._handle_general_clustering_query(question, query_intent, all_conversations)
                
        except Exception as e:
            logger.error(f"Error handling clustering query: {e}")
            return {
                "answer": f"I encountered an error while analyzing your question: {str(e)}",
                "sources": [],
                "confidence": "error",
                "query": question,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _handle_first_occurrence_query(self, question: str, query_intent: QueryIntent, 
                                           conversations: List[Conversation]) -> Dict[str, Any]:
        """Handle 'first time' queries."""
        # Extract topic from content filters
        topics = query_intent.content_filters
        if not topics:
            return {
                "answer": "I couldn't determine what topic you're asking about. Please be more specific.",
                "sources": [],
                "confidence": "low",
                "query": question,
                "timestamp": datetime.now().isoformat()
            }
        
        topic = topics[0]
        
        # Search for messages containing the topic
        search_results = await self.memory_store.search_conversations(topic, limit=50)
        
        if not search_results:
            return {
                "answer": f"I couldn't find any conversations about '{topic}' in your messages.",
                "sources": [],
                "confidence": "none",
                "query": question,
                "timestamp": datetime.now().isoformat()
            }
        
        # Find the earliest message
        earliest_message = min(search_results, key=lambda x: x[0].timestamp)
        message, similarity = earliest_message
        
        # Format the answer
        answer = f"The first time you talked about '{topic}' was on {message.timestamp.strftime('%B %d, %Y')} "
        answer += f"on {message.platform}. "
        
        if hasattr(message, 'sender_id') and message.sender_id:
            if message.sender_id in ['aleksandre_natchkebia', 'aleksandre']:
                answer += f"You mentioned it in a message."
            else:
                answer += f"{message.sender_id} mentioned it in a message."
        
        answer += f" The message content was: \"{message.content[:100]}{'...' if len(message.content) > 100 else ''}\""
        
        return {
            "answer": answer,
            "sources": [{
                "id": message.id,
                "platform": message.platform,
                "sender": message.sender_id,
                "timestamp": message.timestamp.isoformat(),
                "similarity": similarity,
                "content_preview": message.content[:100]
            }],
            "confidence": "high" if similarity > 0.7 else "medium",
            "query": question,
            "timestamp": datetime.now().isoformat(),
            "query_type": "first_occurrence"
        }
    
    async def _handle_most_frequent_query(self, question: str, query_intent: QueryIntent, 
                                        conversations: List[Conversation]) -> Dict[str, Any]:
        """Handle 'most frequent' queries."""
        # Apply temporal filters if present
        temporal_filters = query_intent.temporal_filters
        
        # Get relationship clusters
        relationship_result = await self.clustering_manager.cluster_relationships(conversations)
        
        if not relationship_result.clusters:
            return {
                "answer": "I couldn't analyze your communication patterns.",
                "sources": [],
                "confidence": "low",
                "query": question,
                "timestamp": datetime.now().isoformat()
            }
        
        # Filter by temporal constraints if specified
        filtered_clusters = relationship_result.clusters
        if temporal_filters:
            filtered_clusters = []
            for cluster in relationship_result.clusters:
                # Check if cluster has interactions in the specified time period
                for filter_info in temporal_filters:
                    if filter_info['type'] == 'year':
                        year = filter_info['value']
                        cluster_start = datetime.fromisoformat(cluster['first_interaction'])
                        cluster_end = datetime.fromisoformat(cluster['last_interaction'])
                        
                        if cluster_start.year <= year <= cluster_end.year:
                            filtered_clusters.append(cluster)
                            break
        
        # Sort by message count
        sorted_clusters = sorted(filtered_clusters, key=lambda x: x['message_count'], reverse=True)
        
        if not sorted_clusters:
            return {
                "answer": "I couldn't find any communication patterns for the specified time period.",
                "sources": [],
                "confidence": "low",
                "query": question,
                "timestamp": datetime.now().isoformat()
            }
        
        # Get top contact
        top_contact = sorted_clusters[0]
        
        # Format answer
        time_period = ""
        if temporal_filters:
            for filter_info in temporal_filters:
                if filter_info['type'] == 'year':
                    time_period = f" in {filter_info['value']}"
                    break
        
        answer = f"Based on your message history{time_period}, you talked to "
        answer += f"**{top_contact['participant_id']}** the most, with "
        answer += f"{top_contact['message_count']} messages across "
        answer += f"{', '.join(top_contact['platforms'])}. "
        
        if top_contact['dominant_topics']:
            answer += f"Your main topics included: {', '.join(top_contact['dominant_topics'][:3])}."
        
        return {
            "answer": answer,
            "sources": [{
                "id": f"relationship_{top_contact['participant_id']}",
                "platform": ", ".join(top_contact['platforms']),
                "sender": top_contact['participant_id'],
                "message_count": top_contact['message_count'],
                "relationship_strength": top_contact['relationship_strength']
            }],
            "confidence": "high",
            "query": question,
            "timestamp": datetime.now().isoformat(),
            "query_type": "most_frequent"
        }
    
    async def _handle_temporal_analysis_query(self, question: str, query_intent: QueryIntent, 
                                            conversations: List[Conversation]) -> Dict[str, Any]:
        """Handle temporal analysis queries."""
        # Get all messages
        all_messages = []
        for conv in conversations:
            all_messages.extend(conv.messages)
        
        # Apply temporal clustering
        temporal_result = await self.clustering_manager.cluster_temporal(all_messages, "month")
        
        # Filter by temporal constraints
        temporal_filters = query_intent.temporal_filters
        platform_filters = query_intent.platform_filters
        
        relevant_clusters = []
        for cluster in temporal_result.clusters:
            # Check temporal filters
            if temporal_filters:
                cluster_start = datetime.fromisoformat(cluster['start_date'])
                cluster_end = datetime.fromisoformat(cluster['end_date'])
                
                for filter_info in temporal_filters:
                    if filter_info['type'] == 'year':
                        year = filter_info['value']
                        if cluster_start.year <= year <= cluster_end.year:
                            relevant_clusters.append(cluster)
                            break
            else:
                relevant_clusters.append(cluster)
        
        if not relevant_clusters:
            return {
                "answer": "I couldn't find any communication patterns for the specified time period.",
                "sources": [],
                "confidence": "low",
                "query": question,
                "timestamp": datetime.now().isoformat()
            }
        
        # Sort by activity level
        sorted_clusters = sorted(relevant_clusters, key=lambda x: x['activity_level'], reverse=True)
        
        # Format answer
        answer = "Based on your communication patterns, here are the key insights:\n\n"
        
        for i, cluster in enumerate(sorted_clusters[:3]):
            start_date = datetime.fromisoformat(cluster['start_date']).strftime('%B %Y')
            end_date = datetime.fromisoformat(cluster['end_date']).strftime('%B %Y')
            
            answer += f"**{start_date}**: {len(cluster['message_ids'])} messages "
            answer += f"(Activity level: {cluster['activity_level']:.1%})\n"
            
            if cluster['dominant_topics']:
                answer += f"   Main topics: {', '.join(cluster['dominant_topics'][:3])}\n"
            answer += "\n"
        
        return {
            "answer": answer,
            "sources": [{
                "id": f"temporal_{i}",
                "period": f"{cluster['start_date'][:7]} to {cluster['end_date'][:7]}",
                "message_count": len(cluster['message_ids']),
                "activity_level": cluster['activity_level']
            } for i, cluster in enumerate(sorted_clusters[:3])],
            "confidence": "high",
            "query": question,
            "timestamp": datetime.now().isoformat(),
            "query_type": "temporal_analysis"
        }
    
    async def _handle_general_clustering_query(self, question: str, query_intent: QueryIntent, 
                                             conversations: List[Conversation]) -> Dict[str, Any]:
        """Handle general clustering queries."""
        return {
            "answer": "I understand you're asking about patterns in your communication, but I need more specific information to provide a helpful answer.",
            "sources": [],
            "confidence": "low",
            "query": question,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_all_conversations(self) -> List[Conversation]:
        """Get all conversations from the memory store."""
        # This is a simplified approach - in production, you'd want to implement
        # a method to retrieve all conversations efficiently
        # For now, return empty list as placeholder
        return []
