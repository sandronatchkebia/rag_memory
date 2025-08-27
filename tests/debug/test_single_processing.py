#!/usr/bin/env python3
"""Test single conversation processing with debug logging."""

import asyncio
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from src.ai_memory.core.data_loader import DataLoader

async def test_single_processing():
    """Test processing a single conversation with debug logging."""
    print("🔍 Testing single conversation processing...")
    
    try:
        # Initialize data loader
        data_loader = DataLoader()
        
        # Get the first few lines from messenger JSONL
        messenger_file = Path("data/raw/messenger_parsed.jsonl")
        if not messenger_file.exists():
            print("❌ Messenger file not found")
            return
        
        # Read first few lines
        conversations = []
        with open(messenger_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Just process first 10 lines
                    break
                
                try:
                    data = json.loads(line.strip())
                    conversations.append(data)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error on line {i}: {e}")
                    continue
        
        print(f"📖 Loaded {len(conversations)} raw conversations")
        
        # Process just the first conversation
        if conversations:
            first_conv = conversations[0]
            print(f"\n🔍 Processing conversation: {first_conv.get('conversation_id', 'unknown')}")
            
            # Check the raw content
            body_text = first_conv.get('body_text', '')
            print(f"   Raw body_text: {repr(body_text)}")
            
            # Test the normalization function directly
            print(f"\n🧪 Testing text normalization:")
            normalized = data_loader._normalize_text_content(body_text)
            print(f"   Normalized: {repr(normalized)}")
            
            # Check if corruption happened
            if normalized != body_text:
                print(f"   ⚠️  Content was modified during normalization!")
        
        print("\n🎉 Single processing test completed!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_processing())
