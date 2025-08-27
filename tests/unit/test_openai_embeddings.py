#!/usr/bin/env python3
"""Test script for OpenAI embeddings."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from ai_memory.rag.embedding_manager import EmbeddingManager


async def test_openai_embeddings():
    """Test OpenAI embedding functionality."""
    print("🧪 Testing OpenAI embeddings...")
    
    # Check if API key is set
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Please set it in your .env file or environment.")
        print("   Example: export OPENAI_API_KEY=sk-your-key-here")
        return
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        # Create embedding manager
        embedding_manager = EmbeddingManager(
            model_name="text-embedding-ada-002",
            provider="openai"
        )
        
        # Initialize
        print("🔧 Initializing OpenAI embedding manager...")
        await embedding_manager.initialize()
        print(f"✅ Initialized: {embedding_manager.provider_info}")
        
        # Test single embedding
        print("\n📝 Testing single embedding...")
        text = "Hello! How are you doing today?"
        embedding = await embedding_manager.generate_embedding(text)
        print(f"✅ Generated embedding: {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Test batch embeddings
        print("\n📚 Testing batch embeddings...")
        texts = [
            "Hello! How are you doing today?",
            "I'm doing great! Thanks for asking.",
            "This is a test message in Georgian: გამარჯობა! როგორ ხარ?",
            "Another message to test batch processing."
        ]
        
        embeddings = await embedding_manager.generate_embeddings(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Test similarity
        print("\n🔍 Testing similarity calculation...")
        similarity = embedding_manager.similarity(embeddings[0], embeddings[1])
        print(f"   Similarity between messages 1 & 2: {similarity:.3f}")
        
        similarity = embedding_manager.similarity(embeddings[0], embeddings[2])
        print(f"   Similarity between messages 1 & 3: {similarity:.3f}")
        
        # Test batch similarity
        print("\n📊 Testing batch similarity...")
        similarities = embedding_manager.batch_similarity(embeddings[0], embeddings[1:])
        print(f"   Batch similarities: {[f'{s:.3f}' for s in similarities]}")
        
        print("\n🎉 OpenAI embedding test complete!")
        
    except Exception as e:
        print(f"❌ Error testing OpenAI embeddings: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_openai_embeddings())
