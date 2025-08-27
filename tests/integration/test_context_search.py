#!/usr/bin/env python3
"""Test the context-aware search functionality."""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from ai_memory.core.memory_store import MemoryStore

# Load environment variables
load_dotenv(override=True)

async def test_context_search():
    """Test the context-aware search with context preservation."""
    print("🔍 Testing Context-Aware Search...")
    
    # Debug: Check environment variables
    print(f"🔧 Environment check:")
    print(f"   OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"   EMBEDDING_PROVIDER: {os.getenv('EMBEDDING_PROVIDER', 'NOT SET')}")
    print(f"   EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL', 'NOT SET')}")
    
    # Initialize memory store
    memory_store = MemoryStore()
    
    try:
        await memory_store.initialize()
        print("✅ Memory store initialized")
        
        # Clear existing collection to ensure correct dimensions
        print("🧹 Clearing existing collection to ensure correct embedding dimensions...")
        await memory_store.clear_all()
        await memory_store.initialize()
        print("✅ Collection recreated with correct dimensions")
        
        # Debug: Check embedding manager
        print(f"🔧 Embedding manager check:")
        print(f"   Provider: {memory_store._embedding_manager.provider}")
        print(f"   Model: {memory_store._embedding_manager.model_name}")
        print(f"   Initialized: {memory_store._embedding_manager._initialized}")
        
        # Test search with context
        query = "Berkeley"
        print(f"\n🔎 Searching for: '{query}'")
        print("=" * 50)
        
        results = await memory_store.search_conversations(
            query=query,
            limit=3,
            context_window=3  # Get 3 messages before/after
        )
        
        print(f"📊 Found {len(results)} conversation segments")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Conversation Segment:")
            print(f"   📱 Platform: {result['conversation_metadata']['platform']}")
            print(f"   👥 Participants: {', '.join(result['conversation_metadata']['participants'])}")
            print(f"   📊 Relevance Score: {result['relevance_score']:.3f}")
            print(f"   📝 Total Messages: {result['total_messages_in_conversation']}")
            
            # Show context before
            if result['context_before']:
                print(f"   📜 Context Before ({len(result['context_before'])} messages):")
                for ctx_msg in result['context_before'][-2:]:  # Show last 2
                    print(f"      {ctx_msg['metadata']['sender_id']}: {ctx_msg['content'][:100]}...")
            
            # Show target message
            target = result['target_message']
            print(f"   🎯 Target Message:")
            print(f"      {target.sender_id}: {target.content[:200]}...")
            
            # Show context after
            if result['context_after']:
                print(f"   📜 Context After ({len(result['context_after'])} messages):")
                for ctx_msg in result['context_after'][:2]:  # Show first 2
                    print(f"      {ctx_msg['metadata']['sender_id']}: {ctx_msg['content'][:100]}...")
            
            print()
        
        print("✅ Context-aware search test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_context_search())
