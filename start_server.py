#!/usr/bin/env python3
"""Simple script to start the AI Memory API server."""

import uvicorn
from src.ai_memory.main import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
