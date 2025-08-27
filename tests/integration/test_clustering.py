#!/usr/bin/env python3
"""Test clustering functionality with the AI Memory system."""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dotenv import load_dotenv
from ai_memory.core.data_loader import DataLoader
from ai_memory.core.memory_store import MemoryStore
from ai_memory.rag.embedding_manager import EmbeddingManager
from ai_memory.rag.clustering_manager import ClusteringManager
from ai_memory.rag.query_parser import QueryParser

# Load environment variables
load_dotenv()


async def test_clustering():
    """Test the clustering functionality."""
    print("🧠 Testing Clustering System...")
    
    # Initialize components
    data_loader = DataLoader()
    memory_store = MemoryStore()
    embedding_manager = EmbeddingManager()
    clustering_manager = ClusteringManager(embedding_manager)
    query_parser = QueryParser()
    
    try:
        # Initialize components
        await memory_store.initialize()
        await embedding_manager.initialize()
        await clustering_manager.initialize()
        print("✅ Components initialized")
        
        # Load and process data
        print("📖 Loading data...")
        all_conversations = await data_loader.load_all_platforms()
        
        if not all_conversations:
            print("❌ No conversations loaded")
            return
        
        # Get the first few conversations for testing
        test_conversations = []
        for platform, conversations in all_conversations.items():
            test_conversations.extend(conversations[:2])  # First 2 from each platform
            if len(test_conversations) >= 6:  # Limit for testing
                break
        
        print(f"📝 Testing with {len(test_conversations)} conversations")
        
        # Test 1: Content Clustering
        print("\n🔍 Test 1: Content Clustering")
        print("=" * 40)
        
        all_messages = []
        for conv in test_conversations:
            all_messages.extend(conv.messages)
        
        print(f"   Messages to cluster: {len(all_messages)}")
        
        if len(all_messages) >= 3:
            content_result = await clustering_manager.cluster_content(all_messages)
            print(f"   ✅ Content clustering completed in {content_result.processing_time:.2f}s")
            print(f"   📊 Found {len(content_result.clusters)} content clusters")
            
            for i, cluster in enumerate(content_result.clusters[:3]):
                print(f"      Cluster {i+1}: {cluster['topic']} ({len(cluster['message_ids'])} messages)")
                print(f"         Keywords: {', '.join(cluster['keywords'][:5])}")
        else:
            print("   ⚠️  Not enough messages for content clustering (need at least 3)")
        
        # Test 2: Temporal Clustering
        print("\n⏰ Test 2: Temporal Clustering")
        print("=" * 40)
        
        temporal_result = await clustering_manager.cluster_temporal(all_messages, "month")
        print(f"   ✅ Temporal clustering completed in {temporal_result.processing_time:.2f}s")
        print(f"   📊 Found {len(temporal_result.clusters)} temporal clusters")
        
        for i, cluster in enumerate(temporal_result.clusters[:3]):
            start_date = cluster['start_date'][:10] if isinstance(cluster['start_date'], str) else str(cluster['start_date'])[:10]
            end_date = cluster['end_date'][:10] if isinstance(cluster['end_date'], str) else str(cluster['end_date'])[:10]
            print(f"      Period {i+1}: {start_date} to {end_date} ({len(cluster['message_ids'])} messages)")
            print(f"         Activity: {cluster['activity_level']:.2f}")
            print(f"         Topics: {', '.join(cluster['dominant_topics'][:3])}")
        
        # Test 3: Relationship Clustering
        print("\n👥 Test 3: Relationship Clustering")
        print("=" * 40)
        
        relationship_result = await clustering_manager.cluster_relationships(test_conversations)
        print(f"   ✅ Relationship clustering completed in {relationship_result.processing_time:.2f}s")
        print(f"   📊 Found {len(relationship_result.clusters)} relationship clusters")
        
        # Sort by message count
        sorted_relationships = sorted(relationship_result.clusters, key=lambda x: x['message_count'], reverse=True)
        
        for i, cluster in enumerate(sorted_relationships[:3]):
            print(f"      Contact {i+1}: {cluster['participant_id']} ({cluster['message_count']} messages)")
            print(f"         Platforms: {', '.join(cluster['platforms'])}")
            print(f"         Strength: {cluster['relationship_strength']:.2f}")
            print(f"         Topics: {', '.join(cluster['dominant_topics'][:3])}")
        
        # Test 4: Query Parsing
        print("\n🔍 Test 4: Query Parsing")
        print("=" * 40)
        
        test_queries = [
            "When was the first time I talked about Berkeley?",
            "Who did I talk to most in 2018?",
            "What did I discuss on WhatsApp last month?",
            "How did my communication with John evolve?",
            "What were the main topics in 2020?"
        ]
        
        for query in test_queries:
            intent = query_parser.parse_query(query)
            print(f"   Query: {query}")
            print(f"      Type: {intent.query_type}")
            print(f"      Confidence: {intent.confidence:.2f}")
            print(f"      Temporal: {len(intent.temporal_filters)} filters")
            print(f"      Content: {intent.content_filters}")
            print(f"      Relationships: {intent.relationship_filters}")
            print(f"      Platforms: {intent.platform_filters}")
            print()
        
        # Test 5: Query Suggestions
        print("\n💡 Test 5: Query Suggestions")
        print("=" * 40)
        
        suggestions = query_parser.get_query_suggestions()
        print(f"   Generated {len(suggestions)} query suggestions:")
        for i, suggestion in enumerate(suggestions[:5], 1):
            print(f"      {i}. {suggestion}")
        
        print("\n🎉 Clustering system test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await memory_store.clear_all()
        print("🧹 Database cleared")


if __name__ == "__main__":
    asyncio.run(test_clustering())
