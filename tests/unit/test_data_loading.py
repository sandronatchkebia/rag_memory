#!/usr/bin/env python3
"""Test script for data loading functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_memory.core.data_loader import DataLoader


async def test_data_loading():
    """Test data loading with a small sample."""
    print("🧪 Testing AI Memory data loading...")
    
    # Create a test data loader
    loader = DataLoader("./data/raw", "./data/processed")
    
    # Test with just one platform first
    print("\n📧 Testing Gmail data loading...")
    
    try:
        # Process just Gmail data
        gmail_file = Path("./data/raw/gmail_parsed.jsonl")
        if gmail_file.exists():
            conversations = await loader._load_platform_data(gmail_file, "gmail")
            print(f"✅ Successfully loaded {len(conversations)} Gmail conversations")
            
            if conversations:
                # Show sample conversation
                sample_conv = conversations[0]  # conversations is already a list
                print(f"\n📝 Sample conversation:")
                print(f"  ID: {sample_conv.id}")
                print(f"  Platform: {sample_conv.platform}")
                print(f"  Participants: {len(sample_conv.participants)}")
                print(f"  Messages: {len(sample_conv.messages)}")
                print(f"  Date range: {sample_conv.start_date} to {sample_conv.last_activity}")
                
                if sample_conv.messages:
                    sample_msg = sample_conv.messages[0]
                    print(f"\n💬 Sample message:")
                    print(f"  Content: {sample_msg.content[:100]}...")
                    print(f"  Language: {sample_msg.language}")
                    print(f"  Sender: {sample_msg.sender_id}")
                    print(f"  Timestamp: {sample_msg.timestamp}")
                    
                    # Show participants
                    print(f"\n👥 Participants:")
                    for participant in sample_conv.participants:
                        print(f"  - {participant.name} ({participant.email or participant.id}) [Self: {participant.is_self}]")
        else:
            print("❌ Gmail file not found")
            
    except Exception as e:
        print(f"❌ Error testing Gmail loading: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Test complete!")


if __name__ == "__main__":
    asyncio.run(test_data_loading())
