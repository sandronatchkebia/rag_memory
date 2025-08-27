#!/usr/bin/env python3
"""Test the LLM-powered QueryEngine with clean data."""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from ai_memory.core.data_loader import DataLoader
from ai_memory.core.memory_store import MemoryStore
from ai_memory.rag.query_engine import QueryEngine

# Load environment variables
load_dotenv()


async def test_llm_query():
    """Test the LLM-powered QueryEngine."""
    print("🧠 Testing LLM-powered QueryEngine...")
    
    # Initialize components
    data_loader = DataLoader()
    memory_store = MemoryStore()
    query_engine = QueryEngine(memory_store)
    
    try:
        # Initialize components
        await memory_store.initialize()
        await query_engine.initialize()
        print("✅ Components initialized")
        
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
        
        # Show sample messages
        for i, msg in enumerate(first_conversation.messages[:3]):
            print(f"   Message {i+1}: {repr(msg.content[:100])} (lang: {msg.language})")
        
        # Add to memory store
        print("💾 Adding to memory store...")
        await memory_store.add_conversation(first_conversation)
        print("✅ Conversation added to memory store")
        
        # Test LLM queries
        test_questions = [
            "What did someone say about New Year?",
            "What messages are in this conversation?",
            "What language are the messages in?",
            "Who sent the messages?",
        ]
        
        print("\n🔍 Testing LLM queries:")
        print("=" * 60)
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            try:
                response = await query_engine.query(question)
                print(f"🤖 Answer: {response['answer']}")
                
                # Handle confidence formatting properly
                confidence = response['confidence']
                if isinstance(confidence, (int, float)):
                    print(f"📊 Confidence: {confidence:.3f}")
                else:
                    print(f"📊 Confidence: {confidence}")
                    
                print(f"🔗 Sources: {len(response['sources'])} found")
                
                # Show first source
                if response['sources']:
                    source = response['sources'][0]
                    # Use the correct key for content
                    content_key = 'full_content' if 'full_content' in source else 'content_preview'
                    print(f"   Source: {repr(source[content_key][:100])}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n🎉 LLM query test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await memory_store.clear_all()
        print("🧹 Database cleared")


if __name__ == "__main__":
    asyncio.run(test_llm_query())
