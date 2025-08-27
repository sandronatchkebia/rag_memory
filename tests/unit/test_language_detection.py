#!/usr/bin/env python3
"""Test the improved language detection logic."""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_memory.utils.language_detection import (
    detect_language, classify_text, is_georgian, 
    is_romanized_georgian, normalize_georgian_roman
)


def test_language_detection():
    """Test various text samples for language detection."""
    
    test_cases = [
        # English texts
        ("Hello world!", "en"),
        ("Happy New Year <3", "en"),
        ("How are you doing today?", "en"),
        ("The quick brown fox jumps over the lazy dog.", "en"),
        
        # Georgian texts (Unicode)
        ("გამარჯობა", "ka"),
        ("როგორ ხარ?", "ka"),
        ("მე კარგად ვარ", "ka"),
        
        # Romanized Georgian
        ("gamarjoba", "ka"),
        ("rogor khar?", "ka"),
        ("me kargad var", "ka"),
        ("sakartvelo", "ka"),
        
        # Mixed cases
        ("Hello gamarjoba", "en"),  # Should default to English
        ("gamarjoba hello", "ka"),  # Should detect as Georgian
        ("ok", "en"),  # Short English
        ("ra", "ka"),  # Short Georgian
        
        # Edge cases
        ("", "unknown"),
        ("a", "unknown"),
        ("hi", "en"),
        ("123", "unknown"),
    ]
    
    print("Testing language detection:")
    print("=" * 50)
    
    for text, expected in test_cases:
        detected = detect_language(text)
        classification = classify_text(text) if len(text.strip()) >= 2 else "unknown"
        
        status = "✅" if detected == expected else "❌"
        print(f"{status} Text: {repr(text)}")
        print(f"   Expected: {expected}, Detected: {detected}, Classification: {classification}")
        print()
    
    print("\nTesting specific functions:")
    print("=" * 50)
    
    # Test is_georgian
    print("is_georgian tests:")
    print(f"  'გამარჯობა': {is_georgian('გამარჯობა')}")
    print(f"  'Hello': {is_georgian('Hello')}")
    
    # Test is_romanized_georgian
    print("\nis_romanized_georgian tests:")
    print(f"  'gamarjoba': {is_romanized_georgian('gamarjoba')}")
    print(f"  'Hello world': {is_romanized_georgian('Hello world')}")
    print(f"  'Happy New Year': {is_romanized_georgian('Happy New Year')}")
    
    # Test normalization
    print("\nnormalize_georgian_roman tests:")
    print(f"  'gamarjoba' -> {normalize_georgian_roman('gamarjoba')}")
    print(f"  'rogor khar' -> {normalize_georgian_roman('rogor khar')}")


if __name__ == "__main__":
    test_language_detection()
