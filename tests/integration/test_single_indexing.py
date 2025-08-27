#!/usr/bin/env python3
"""Test indexing a single conversation with the fixed language detection."""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from ai_memory.core.data_loader import DataLoader
from ai_memory.core.memory_store import MemoryStore

# Load environment variables
load_dotenv()


async def test_single_indexing():
    """Test indexing a single conversation."""
    print("🔍 Testing single conversation indexing...")
    
    # Initialize components
    data_loader = DataLoader()
    memory_store = MemoryStore()
    
    try:
        # Initialize memory store
        await memory_store.initialize()
        print("✅ Memory store initialized")
        
        # Load and process data
        print("📖 Loading data...")
        all_conversations = await data_loader.load_all_platforms()
        
        if not all_conversations:
            print("❌ No conversations loaded")
            return
        
        # Get the first conversation from the first platform
        first_platform = list(all_conversations.keys())[0]
        first_conversation = all_conversations[first_platform][0]
        
        print(f"📝 Processing conversation: {first_conversation.id}")
        print(f"   Platform: {first_platform}")
        print(f"   Messages: {len(first_conversation.messages)}")
        
        # Show a sample message
        if first_conversation.messages:
            sample_msg = first_conversation.messages[0]
            print(f"   Sample message content: {repr(sample_msg.content)}")
            print(f"   Sample message language: {sample_msg.language}")
        
        # Add to memory store
        print("💾 Adding to memory store...")
        await memory_store.add_conversation(first_conversation)
        print("✅ Conversation added to memory store")
        
        # Test search
        print("🔍 Testing search...")
        search_results = await memory_store.search_conversations("Happy New Year", limit=5)
        print(f"   Search results: {len(search_results)} found")
        
        for i, (message, similarity) in enumerate(search_results[:3]):
            print(f"   Result {i+1}: {repr(message.content[:100])} (similarity: {similarity:.3f})")
        
        # Get statistics
        stats = await memory_store.get_statistics()
        print(f"📊 Database statistics: {stats}")
        
        print("🎉 Single indexing test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await memory_store.clear_all()
        print("🧹 Database cleared")


if __name__ == "__main__":
    asyncio.run(test_single_indexing())
