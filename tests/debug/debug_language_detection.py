#!/usr/bin/env python3
"""Debug language detection on actual data."""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_memory.core.data_loader import DataLoader
from ai_memory.utils.language_detection import detect_language, classify_text


async def debug_language_detection():
    """Debug language detection on actual data."""
    print("🔍 Debugging language detection on actual data...")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    try:
        # Load and process data
        print("📖 Loading data...")
        all_conversations = await data_loader.load_all_platforms()
        
        if not all_conversations:
            print("❌ No conversations loaded")
            return
        
        # Sample messages from different platforms
        sample_messages = []
        
        for platform, conversations in all_conversations.items():
            if conversations:
                # Get first few messages from this platform
                for conv in conversations[:2]:  # First 2 conversations
                    for msg in conv.messages[:3]:  # First 3 messages
                        if msg.content and len(msg.content.strip()) > 5:
                            sample_messages.append((platform, msg.content, msg.language))
                            if len(sample_messages) >= 20:  # Limit samples
                                break
                    if len(sample_messages) >= 20:
                        break
                if len(sample_messages) >= 20:
                    break
        
        print(f"📝 Analyzing {len(sample_messages)} sample messages:")
        print("=" * 80)
        
        for i, (platform, content, detected_lang) in enumerate(sample_messages):
            # Test our language detection
            our_lang = detect_language(content)
            classification = classify_text(content)
            
            # Show content preview
            content_preview = content[:100] + "..." if len(content) > 100 else content
            
            print(f"{i+1:2d}. [{platform}] {detected_lang} -> {our_lang} ({classification})")
            print(f"     Content: {repr(content_preview)}")
            print()
            
            # If there's a mismatch, show more details
            if detected_lang != our_lang and detected_lang != "unknown":
                print(f"     ⚠️  Language mismatch! Original: {detected_lang}, Our detection: {our_lang}")
                print()
        
        print("=" * 80)
        print("🎯 Summary:")
        
        # Count language distributions
        original_langs = {}
        our_langs = {}
        
        for _, _, detected_lang in sample_messages:
            original_langs[detected_lang] = original_langs.get(detected_lang, 0) + 1
        
        for _, content, _ in sample_messages:
            our_lang = detect_language(content)
            our_langs[our_lang] = our_langs.get(our_lang, 0) + 1
        
        print("Original language detection:")
        for lang, count in sorted(original_langs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {count}")
        
        print("\nOur language detection:")
        for lang, count in sorted(our_langs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_language_detection())
