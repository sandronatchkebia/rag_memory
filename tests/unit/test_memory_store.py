#!/usr/bin/env python3
"""Test script for memory store functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_memory.core.memory_store import MemoryStore
from ai_memory.models.conversation import Conversation, Message, Participant
from datetime import datetime, timezone


async def test_memory_store():
    """Test memory store functionality."""
    print("🧪 Testing AI Memory store...")
    
    # Create memory store
    memory_store = MemoryStore("./data/chroma_db_test")
    
    try:
        # Initialize
        print("🔧 Initializing memory store...")
        await memory_store.initialize()
        print("✅ Memory store initialized")
        
        # Create a test conversation
        print("\n📝 Creating test conversation...")
        
        # Test participants
        participants = [
            Participant(
                id="test_user@example.com",
                name="test_user",
                email="test_user@example.com",
                is_self=True
            ),
            Participant(
                id="friend@example.com", 
                name="friend",
                email="friend@example.com",
                is_self=False
            )
        ]
        
        # Test messages
        messages = [
            Message(
                id="test_msg_1",
                content="Hello! How are you doing today?",
                sender_id="test_user@example.com",
                timestamp=datetime.now(timezone.utc),
                platform="test",
                message_type="text",
                language="en"
            ),
            Message(
                id="test_msg_2",
                content="I'm doing great! Thanks for asking.",
                sender_id="friend@example.com",
                timestamp=datetime.now(timezone.utc),
                platform="test",
                message_type="text",
                language="en"
            )
        ]
        
        # Create conversation
        conversation = Conversation(
            id="test:conv_123",
            platform="test",
            participants=participants,
            messages=messages,
            start_date=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc)
        )
        
        # Add to store
        print("💾 Adding conversation to store...")
        conv_id = await memory_store.add_conversation(conversation)
        print(f"✅ Conversation added with ID: {conv_id}")
        
        # Test search
        print("\n🔍 Testing search functionality...")
        search_results = await memory_store.search_conversations("hello how are you", limit=5)
        print(f"✅ Search returned {len(search_results)} results")
        
        for message, similarity in search_results:
            print(f"  - {message.content[:50]}... (similarity: {similarity:.3f})")
        
        # Test statistics
        print("\n📊 Getting statistics...")
        stats = await memory_store.get_statistics()
        print(f"✅ Statistics: {stats}")
        
        # Clean up
        print("\n🧹 Cleaning up test data...")
        await memory_store.clear_all()
        print("✅ Test data cleared")
        
    except Exception as e:
        print(f"❌ Error testing memory store: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Memory store test complete!")


if __name__ == "__main__":
    asyncio.run(test_memory_store())

