"""Command-line interface for AI Memory."""

import click
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .core.memory_store import MemoryStore
from .core.conversation_processor import ConversationProcessor
from .core.data_loader import DataLoader
from .utils.file_utils import load_json_directory, ensure_directory


@click.group()
@click.version_option(version="0.1.0")
def main():
    """AI Memory - Personal digital memory tool with advanced RAG capabilities."""
    pass


@main.command()
@click.option("--data-dir", default="./data/raw", help="Directory containing JSON exports")
@click.option("--output-dir", default="./data/processed", help="Output directory for processed data")
def process_data(data_dir: str, output_dir: str):
    """Process raw JSON exports into normalized format."""
    click.echo(f"Processing data from {data_dir}...")
    
    try:
        # Ensure directories exist
        ensure_directory(output_dir)
        
        # Load raw data
        raw_data = load_json_directory(data_dir)
        click.echo(f"Loaded {len(raw_data)} JSON files")
        
        # TODO: Implement actual processing
        click.echo("Data processing not yet implemented")
        
    except Exception as e:
        click.echo(f"Error processing data: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option("--raw-dir", default="./data/raw", help="Directory containing JSONL exports")
@click.option("--processed-dir", default="./data/processed", help="Output directory for processed data")
@click.option("--platform", help="Process only specific platform (gmail, messenger, whatsapp, instagram)")
def load_data(raw_dir: str, processed_dir: str, platform: str = None):
    """Load and normalize conversation data from JSONL files."""
    click.echo("Loading and normalizing conversation data...")
    
    async def _load_data():
        loader = DataLoader(raw_dir, processed_dir)
        
        if platform:
            click.echo(f"Processing only {platform} data...")
            # TODO: Implement single platform loading
            click.echo("Single platform loading not yet implemented")
        else:
            click.echo("Processing all platforms...")
            results = await loader.load_all_platforms()
            
            total_conversations = sum(len(convs) for convs in results.values())
            total_messages = sum(
                sum(len(conv.messages) for conv in convs) 
                for convs in results.values()
            )
            
            click.echo(f"\n✅ Data loading complete!")
            click.echo(f"📊 Summary:")
            for platform_name, conversations in results.items():
                platform_messages = sum(len(conv.messages) for conv in conversations)
                click.echo(f"  {platform_name}: {len(conversations)} conversations, {platform_messages} messages")
            
            click.echo(f"\nTotal: {total_conversations} conversations, {total_messages} messages")
            click.echo(f"Processed data saved to: {processed_dir}")
    
    asyncio.run(_load_data())


@main.command()
@click.option("--processed-dir", default="./data/processed", help="Directory containing processed conversations")
@click.option("--db-path", default="./data/chroma_db", help="ChromaDB database path")
@click.option("--platform", help="Index only specific platform")
@click.option("--batch-size", default=100, help="Number of conversations to process in each batch")
@click.option("--sample-size", default=0, help="Process only N conversations per platform (for testing)")
def index_data(processed_dir: str, db_path: str, platform: str = None, batch_size: int = 100, sample_size: int = 0):
    """Index processed conversations into the vector database."""
    if sample_size > 0:
        click.echo(f"🧪 TESTING MODE: Indexing only {sample_size} conversations per platform")
    else:
        click.echo("Indexing conversations into vector database...")
    
    async def _index_data():
        # Initialize memory store
        memory_store = MemoryStore(db_path)
        await memory_store.initialize()
        
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            click.echo(f"❌ Processed data directory not found: {processed_dir}")
            return
        
        # Get platforms to process
        if platform:
            platforms = [platform]
        else:
            platforms = [d.name for d in processed_path.iterdir() if d.is_dir()]
        
        total_indexed = 0
        skipped = {
            "files_missing_required_fields": [],
            "messages_missing_required_fields": [],
        }
        
        for platform_name in platforms:
            platform_dir = processed_path / platform_name
            if not platform_dir.exists():
                continue
            
            click.echo(f"\n📱 Indexing {platform_name} conversations...")
            
            # Get all conversation files, skip summaries or non-conversation files
            conversation_files = [
                p for p in platform_dir.glob("*.json")
                if p.name not in {"summary.json", "stats.json", "metadata.json"}
            ]
            if not conversation_files:
                click.echo(f"  No conversations found for {platform_name}")
                continue
            
            # Apply sampling if requested
            if sample_size > 0:
                import random
                random.shuffle(conversation_files)
                conversation_files = conversation_files[:sample_size]
                click.echo(f"  Testing with {len(conversation_files)} conversations (random sample)")
            
            # Process in batches
            for i in range(0, len(conversation_files), batch_size):
                batch_files = conversation_files[i:i + batch_size]
                click.echo(f"  Processing batch {i//batch_size + 1}/{(len(conversation_files) + batch_size - 1)//batch_size}")
                
                for conv_file in batch_files:
                    try:
                        # Load conversation
                        with open(conv_file, 'r', encoding='utf-8') as f:
                            conv_data = json.load(f)
                        
                        # Convert to Conversation object
                        from .models.conversation import Conversation, Message, Participant
                        
                        # Reconstruct participants
                        participants = []
                        for p_data in conv_data.get("participants", []):
                            participant = Participant(
                                id=p_data["id"],
                                name=p_data["name"],
                                email=p_data.get("email"),
                                platform_id=p_data.get("platform_id"),
                                is_self=p_data.get("is_self", False)
                            )
                            participants.append(participant)
                        
                        # Reconstruct messages
                        messages = []
                        for msg_data in conv_data.get("messages", []):
                            # Validate required fields
                            if not all(k in msg_data for k in ("id", "content", "sender_id", "timestamp", "platform")):
                                skipped["messages_missing_required_fields"].append({
                                    "file": conv_file.name,
                                    "reason": "missing_fields",
                                    "keys_present": list(msg_data.keys())
                                })
                                continue
                            message = Message(
                                id=msg_data["id"],
                                content=msg_data["content"],
                                sender_id=msg_data["sender_id"],
                                timestamp=msg_data["timestamp"],
                                platform=msg_data["platform"],
                                message_type=msg_data.get("message_type", "text"),
                                language=msg_data.get("language"),
                                metadata=msg_data.get("metadata", {})
                            )
                            messages.append(message)
                        
                        # Create conversation object
                        if not all(k in conv_data for k in ("id", "platform", "start_date", "last_activity")):
                            skipped["files_missing_required_fields"].append({
                                "file": conv_file.name,
                                "reason": "missing_fields",
                                "keys_present": list(conv_data.keys())
                            })
                            raise ValueError("Missing required conversation fields")
                        conversation = Conversation(
                            id=conv_data["id"],
                            title=conv_data.get("title"),
                            platform=conv_data["platform"],
                            participants=participants,
                            messages=messages,
                            start_date=conv_data["start_date"],
                            last_activity=conv_data["last_activity"],
                            metadata=conv_data.get("metadata", {})
                        )
                        
                        # Add to memory store
                        await memory_store.add_conversation(conversation)
                        total_indexed += 1
                        
                    except Exception as e:
                        click.echo(f"  ❌ Error processing {conv_file.name}: {e}")
                        continue
                
                click.echo(f"  ✅ Processed {len(batch_files)} conversations")
        
        # Show final statistics
        if sample_size > 0:
            click.echo(f"\n🧪 Testing complete!")
            click.echo(f"📊 Total conversations indexed: {total_indexed}")
            click.echo(f"💡 This was a test with {sample_size} conversations per platform")
            click.echo(f"💡 Run without --sample-size to index the full dataset")
        else:
            click.echo(f"\n🎉 Indexing complete!")
            click.echo(f"📊 Total conversations indexed: {total_indexed}")
        
        # Get database statistics
        stats = await memory_store.get_statistics()
        if stats:
            click.echo(f"\n📈 Database statistics:")
            click.echo(f"  Total messages: {stats.get('total_messages', 0)}")
            click.echo(f"  Platform distribution: {stats.get('platform_distribution', {})}")
            click.echo(f"  Language distribution: {stats.get('language_distribution', {})}")
            click.echo(f"  Embedding dimension: {stats.get('embedding_dimension', 0)}")

        # Write skip report
        try:
            reports_dir = Path("data/index_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            import datetime
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            report_path = reports_dir / f"index_report_{ts}.json"
            with open(report_path, 'w', encoding='utf-8') as rf:
                json.dump({
                    "processed_dir": processed_dir,
                    "platform": platform,
                    "batch_size": batch_size,
                    "sample_size": sample_size,
                    "total_indexed": total_indexed,
                    "skipped": skipped
                }, rf, ensure_ascii=False, indent=2)
            click.echo(f"📝 Skip report saved to: {report_path}")
        except Exception as e:
            click.echo(f"⚠️ Failed to write skip report: {e}")
    
    asyncio.run(_index_data())


@main.command()
@click.option("--host", default="localhost", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def serve(host: str, port: int):
    """Start the AI Memory API server."""
    click.echo(f"Starting AI Memory server on {host}:{port}...")
    
    try:
        import uvicorn
        uvicorn.run(
            "ai_memory.api.routes:router",
            host=host,
            port=port,
            reload=True
        )
    except ImportError:
        click.echo("uvicorn not installed. Install with: uv pip install uvicorn[standard]", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument("query")
@click.option("--limit", default=10, help="Maximum number of results")
def query(query: str, limit: int):
    """Query memories using natural language."""
    click.echo(f"Querying: {query}")
    
    # TODO: Implement actual querying
    click.echo("Query functionality not yet implemented")


if __name__ == "__main__":
    main()
