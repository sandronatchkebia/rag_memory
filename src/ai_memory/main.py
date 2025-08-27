"""Main application entry point for AI Memory."""

import asyncio
import uvicorn
from fastapi import FastAPI
from .api.routes import router
from .core.memory_store import MemoryStore
from .core.conversation_processor import ConversationProcessor
from .rag.embedding_manager import EmbeddingManager
from .rag.retriever import MemoryRetriever
from .rag.query_engine import QueryEngine


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Memory",
        description="Personal digital memory tool with advanced RAG capabilities",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize components on startup."""
        # TODO: Initialize all components
        pass
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        # TODO: Cleanup resources
        pass
    
    return app


async def main():
    """Main application entry point."""
    app = create_app()
    
    # Run the application
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
