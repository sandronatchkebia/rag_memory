#!/usr/bin/env python3
"""Interactive test of the AI Memory system."""

import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ai_memory.core.memory_store import MemoryStore

async def interactive_test():
    """Interactive test of the AI Memory system."""
    print("🧠 Welcome to AI Memory Interactive Test!")
    print("=" * 50)
    
    try:
        # Initialize memory store
        print("📚 Initializing memory store...")
        memory_store = MemoryStore()
        await memory_store.initialize()
        
        # Show current status
        stats = await memory_store.get_statistics()
        print(f"📊 Database: {stats['total_messages']} messages across {len(stats['platform_distribution'])} platforms")
        print(f"   Platforms: {', '.join(stats['platform_distribution'].keys())}")
        print(f"   Languages: {len(stats['language_distribution'])} detected")
        print(f"   Embeddings: {stats['embedding_dimension']} dimensions")
        
        print("\n🔍 Ready to search! Try these example queries or type your own:")
        print("   Examples: 'გამარჯობა', 'resume', 'job application', 'vacation', 'birthday'")
        print("   Type 'quit' to exit")
        
        while True:
            try:
                # Get user query
                query = input("\n🔍 Enter your search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    print("❌ Please enter a search query")
                    continue
                
                # Ask for platform filter
                platform_filter = input("📱 Filter by platform (gmail/messenger/whatsapp/instagram/berkeley_mail) or press Enter for all: ").strip()
                
                # Prepare filters
                filters = {}
                if platform_filter and platform_filter.lower() in ['gmail', 'messenger', 'whatsapp', 'instagram', 'berkeley_mail']:
                    filters['platform'] = platform_filter.lower()
                    print(f"🔍 Searching for '{query}' in {platform_filter}...")
                else:
                    print(f"🔍 Searching for '{query}' across all platforms...")
                
                # Perform search
                results = await memory_store.search_conversations(query, limit=5, filters=filters)
                
                if not results:
                    print("❌ No results found. Try a different query or check your spelling.")
                    continue
                
                print(f"\n✅ Found {len(results)} results:")
                print("-" * 60)
                
                # Display results
                for i, (message, similarity) in enumerate(results, 1):
                    print(f"\n{i}. 📱 {message.platform.upper()}")
                    print(f"   👤 {message.sender_id}")
                    print(f"   🕒 {message.timestamp}")
                    print(f"   🌍 {message.language or 'unknown'}")
                    print(f"   📊 Similarity: {similarity:.3f}")
                    print(f"   💬 Content: {message.content[:150]}{'...' if len(message.content) > 150 else ''}")
                    print("-" * 40)
                
                # Ask if user wants to see more details
                if len(results) > 0:
                    detail_choice = input(f"\n🔍 View full content of result 1-{len(results)} or press Enter to continue: ").strip()
                    
                    if detail_choice.isdigit():
                        choice = int(detail_choice)
                        if 1 <= choice <= len(results):
                            message, similarity = results[choice - 1]
                            print(f"\n📄 Full message content:")
                            print("=" * 60)
                            print(f"Platform: {message.platform}")
                            print(f"Sender: {message.sender_id}")
                            print(f"Time: {message.timestamp}")
                            print(f"Language: {message.language or 'unknown'}")
                            print(f"Type: {message.message_type}")
                            print(f"Similarity: {similarity:.3f}")
                            print("-" * 60)
                            print(message.content)
                            print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during search: {e}")
                continue
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(interactive_test())
