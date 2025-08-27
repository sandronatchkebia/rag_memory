#!/usr/bin/env python3
"""Test the clustering-aware QueryEngine with meaningful queries."""

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
from ai_memory.rag.query_engine import QueryEngine

# Load environment variables
load_dotenv()


async def test_clustering_queries():
    """Test the clustering-aware QueryEngine with meaningful queries."""
    print("🧠 Testing Clustering-Aware QueryEngine...")
    
    # Initialize components
    data_loader = DataLoader()
    memory_store = MemoryStore()
    embedding_manager = EmbeddingManager()
    clustering_manager = ClusteringManager(embedding_manager)
    query_parser = QueryParser()
    query_engine = QueryEngine(memory_store, clustering_manager, query_parser)
    
    try:
        # Initialize components
        await memory_store.initialize()
        await embedding_manager.initialize()
        await clustering_manager.initialize()
        await query_engine.initialize()
        print("✅ Components initialized")
        
        # Load and process data
        print("📖 Loading data...")
        all_conversations = await data_loader.load_all_platforms()
        
        if not all_conversations:
            print("❌ No conversations loaded")
            return
        
        # Get conversations for testing
        test_conversations = []
        for platform, conversations in all_conversations.items():
            test_conversations.extend(conversations[:3])  # First 3 from each platform
            if len(test_conversations) >= 9:  # Limit for testing
                break
        
        print(f"📝 Testing with {len(test_conversations)} conversations")
        
        # Add conversations to memory store
        print("💾 Adding conversations to memory store...")
        for conv in test_conversations:
            await memory_store.add_conversation(conv)
        print("✅ Conversations added to memory store")
        
        # Test meaningful queries
        test_queries = [
            "When was the first time I talked about Berkeley?",
            "Who did I talk to most in 2020?",
            "What were my most active conversation topics last month?",
            "How did my communication with friends evolve?",
            "What did I discuss most frequently on WhatsApp?",
            "Who was my most frequent contact in 2018?",
            "When did I start talking about machine learning?",
            "What were the main topics of my conversations in 2021?"
        ]
        
        print("\n🔍 Testing Clustering-Aware Queries:")
        print("=" * 60)
        
        for i, question in enumerate(test_queries, 1):
            print(f"\n{i}. ❓ Question: {question}")
            print("-" * 50)
            
            try:
                # Parse query intent first
                intent = query_parser.parse_query(question)
                print(f"   📊 Query Type: {intent.query_type}")
                print(f"   🎯 Confidence: {intent.confidence:.2f}")
                print(f"   ⏰ Temporal Filters: {len(intent.temporal_filters)}")
                print(f"   📝 Content Filters: {intent.content_filters}")
                print(f"   👥 Relationship Filters: {intent.relationship_filters}")
                print(f"   📱 Platform Filters: {intent.platform_filters}")
                
                # Get answer from QueryEngine
                response = await query_engine.query(question)
                print(f"\n   🤖 Answer: {response['answer']}")
                print(f"   📊 Confidence: {response['confidence']}")
                print(f"   🔗 Sources: {len(response['sources'])} found")
                
                # Show source details
                if response['sources']:
                    print(f"   📋 Sources:")
                    for j, source in enumerate(response['sources'][:2], 1):
                        if 'content_preview' in source:
                            print(f"      {j}. {source['platform']} - {source['sender']} ({source['timestamp'][:10]})")
                            print(f"         Content: {source['content_preview']}")
                        elif 'message_count' in source:
                            print(f"      {j}. {source['sender']} - {source['message_count']} messages")
                        elif 'period' in source:
                            print(f"      {j}. {source['period']} - {source['message_count']} messages")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            print()
        
        print("🎉 Clustering-aware QueryEngine test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await memory_store.clear_all()
        print("🧹 Database cleared")


if __name__ == "__main__":
    asyncio.run(test_clustering_queries())
