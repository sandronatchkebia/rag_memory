#!/usr/bin/env python3
"""Debug script to test text processing pipeline."""

import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ai_memory.utils.language_detection import detect_language, normalize_text, is_georgian, is_romanized_georgian, normalize_georgian_roman

def debug_text_processing():
    """Debug the text processing pipeline."""
    print("🔍 Debugging Text Processing Pipeline")
    print("=" * 50)
    
    # Test cases from your data
    test_cases = [
        "Happy New Year <3",
        "djinni.co", 
        "You're now friends with Sean.",
        "გამარჯობა",
        "gamarjoba",
        "Hello world",
        "როგორ ხარ?",
        "rogor khar?"
    ]
    
    print("\n📝 Testing text processing:")
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Original: '{text}'")
        
        # Test language detection
        lang = detect_language(text)
        print(f"   Language detected: {lang}")
        
        # Test Georgian checks
        is_geo = is_georgian(text)
        is_roman = is_romanized_georgian(text)
        print(f"   Is Georgian script: {is_geo}")
        print(f"   Is romanized Georgian: {is_roman}")
        
        # Test normalization
        normalized = normalize_text(text)
        print(f"   Normalized: '{normalized}'")
        
        # Test Georgian roman normalization specifically
        if is_roman:
            geo_converted = normalize_georgian_roman(text)
            print(f"   Georgian conversion: '{geo_converted}'")
    
    # Test with actual data from your files
    print(f"\n🔍 Testing with actual data from processed files:")
    
    processed_dir = Path("data/processed/messenger")
    if processed_dir.exists():
        # Find a small file
        small_files = [f for f in processed_dir.glob("*.json") if f.stat().st_size < 100000]
        if small_files:
            test_file = small_files[0]
            print(f"\n📖 Testing file: {test_file.name}")
            
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Look at first few messages
            for i, msg in enumerate(data['messages'][:3]):
                print(f"\n   Message {i+1}:")
                print(f"     ID: {msg['id']}")
                print(f"     Content: '{msg['content']}'")
                print(f"     Language: {msg.get('language', 'unknown')}")
                print(f"     Length: {len(msg['content'])}")
                
                # Check if content looks corrupted
                if len(msg['content']) > 0:
                    first_char = msg['content'][0]
                    last_char = msg['content'][-1]
                    print(f"     First char: '{first_char}' (ord: {ord(first_char)})")
                    print(f"     Last char: '{last_char}' (ord: {ord(last_char)})")
                    
                    # Check for encoding issues
                    if any(ord(c) > 127 for c in msg['content']):
                        print(f"     ⚠️  Contains high-ASCII characters (potential encoding issue)")
                    else:
                        print(f"     ✅ Only standard ASCII characters")

if __name__ == "__main__":
    debug_text_processing()
