#!/usr/bin/env python3
"""Test retrieval and query capabilities of the AI Memory system."""

import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ai_memory.core.memory_store import MemoryStore
from src.ai_memory.rag.query_engine import QueryEngine

async def test_retrieval():
    """Test various retrieval and query capabilities."""
    print("🔍 Testing AI Memory retrieval system...")
    
    try:
        # Initialize memory store
        print("📚 Initializing memory store...")
        memory_store = MemoryStore()
        await memory_store.initialize()
        
        # Get current statistics
        stats = await memory_store.get_statistics()
        print(f"📊 Current database: {stats['total_messages']} messages across {len(stats['platform_distribution'])} platforms")
        
        # Test 1: Simple keyword search
        print("\n🔎 Test 1: Simple keyword search for 'გამარჯობა' (Georgian hello)")
        results = await memory_store.search_conversations("გამარჯობა", limit=3)
        print(f"   Found {len(results)} results")
        for i, (message, similarity) in enumerate(results[:2]):
            print(f"   {i+1}. {message.platform} - {message.sender_id} (similarity: {similarity:.3f})")
            print(f"      Content: {message.content[:100]}...")
        
        # Test 2: English search
        print("\n🔎 Test 2: English search for 'resume'")
        results = await memory_store.search_conversations("resume", limit=3)
        print(f"   Found {len(results)} results")
        for i, (message, similarity) in enumerate(results[:2]):
            print(f"   {i+1}. {message.platform} - {message.sender_id} (similarity: {similarity:.3f})")
            print(f"      Content: {message.content[:100]}...")
        
        # Test 3: Platform-specific search
        print("\n🔎 Test 3: Platform-specific search in Gmail")
        results = await memory_store.search_conversations("job application", limit=3, filters={"platform": "gmail"})
        print(f"   Found {len(results)} Gmail results")
        for i, (message, similarity) in enumerate(results[:2]):
            print(f"   {i+1}. {message.sender_id} (similarity: {similarity:.3f})")
            print(f"      Content: {message.content[:100]}...")
        
        # Test 4: Date-based search (recent)
        print("\n🔎 Test 4: Recent conversations")
        results = await memory_store.search_conversations("", limit=5)  # Empty query to get recent
        print(f"   Found {len(results)} recent results")
        for i, (message, similarity) in enumerate(results[:2]):
            print(f"   {i+1}. {message.platform} - {message.timestamp} (similarity: {similarity:.3f})")
            print(f"      Content: {message.content[:100]}...")
        
        # Test 5: Query Engine (if available)
        print("\n🤖 Test 5: Testing Query Engine")
        try:
            query_engine = QueryEngine(memory_store)
            await query_engine.initialize()
            
            # Test a natural language query
            query = "What job applications did I send recently?"
            print(f"   Query: {query}")
            
            response = await query_engine.query(query, limit=3)
            print(f"   Response: {response}")
            
        except Exception as e:
            print(f"   Query Engine not fully implemented yet: {e}")
        
        print("\n🎉 Retrieval testing completed!")
        
    except Exception as e:
        print(f"❌ Error during retrieval test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_retrieval())
