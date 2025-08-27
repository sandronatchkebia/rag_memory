# AI Memory

A personal digital memory tool that uses advanced RAG (Retrieval-Augmented Generation) techniques to help you query, analyze, and rediscover meaningful moments from your digital communications.

## Features

- **Multi-source ingestion**: Emails, Facebook messages, WhatsApp chats, Instagram DMs
- **Natural language querying**: Ask questions in plain English about your conversations
- **Context preservation**: Maintains who said what, when, and where
- **Insight generation**: Discover patterns, relationship evolution, and topic importance
- **Proactive resurfacing**: Rediscover forgotten moments, anniversaries, and unfinished plans
- **Multilingual support**: English and Georgian (both keyboard layouts)
- **Privacy-first**: Local processing with user control over data

## Project Structure

```
ai_memory/
├── src/ai_memory/          # Source code
│   ├── core/               # Core functionality
│   ├── models/             # Data models
│   ├── rag/                # RAG engine
│   ├── utils/              # Utility functions
│   └── api/                # API endpoints
├── data/                   # Data storage
│   ├── raw/                # Raw JSON exports
│   └── processed/          # Processed data
├── tests/                  # Test files
├── docs/                   # Documentation
└── pyproject.toml          # Project configuration
```

## Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd ai_memory
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

3. **Environment setup**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

### Data Layout
- Place raw JSONL exports under `data/raw/` (by platform if available)
- Processed, normalized conversations live under `data/processed/<platform>/`

### Environment
Create a `.env` file (never commit this) with at least:
```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_API_KEY=sk-...
CHROMA_DB_PATH=./data/chroma_db_main
```

### Indexing
- Full indexing (recommended DB path for production queries):
```bash
python -m ai_memory.cli index-data --processed-dir data/processed --db-path data/chroma_db_main --batch-size 200
```

- Quick test indexing (isolated test DB):
```bash
python -m ai_memory.cli index-data --processed-dir data/processed --db-path data/chroma_db_test --sample-size 2 --batch-size 50
```

Notes:
- Very large conversations are chunked automatically.
- Embedding batches bisect on failures and skip only bad inputs; skips are recorded in `data/index_reports/`.
- Collection dimensions are validated and recreated if mismatched.

### Context-Aware Search
The search API returns the target message plus surrounding context.

Run the integration test (uses whichever DB path you configured in `.env` and MemoryStore init):
```bash
python tests/integration/test_context_search.py
```

### Query Examples
- "What did I discuss with Sarah about travel in 2022?"
- "Show me conversations about AI from last month"
- "What topics were most important to me in 2023?"
- "Find unanswered questions from my conversations"

## Development

- **Code formatting**: `uv run black src/ tests/`
- **Import sorting**: `uv run isort src/ tests/`
- **Type checking**: `uv run mypy src/`
- **Testing**: `uv run pytest`

## Architecture

The system uses a modular architecture with:
- **Data Models**: Pydantic models for type safety
- **Vector Database**: ChromaDB for semantic search
- **Embeddings**: OpenAI `text-embedding-ada-002` (1536-dim) by default
- **RAG Engine**: Advanced retrieval with context awareness
- **API Layer**: FastAPI for web interface

## Privacy & Security

- All data processing happens locally, except OpenAI embeddings/LLM calls if enabled
- No data is sent to external services without explicit consent
- User maintains full control over their data
- Configurable privacy settings and data retention policies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
