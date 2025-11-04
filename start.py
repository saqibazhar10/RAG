#!/usr/bin/env python3
"""
Startup script for RAG Document Processing API
"""

import uvicorn
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    print(f"Starting RAG API on {host}:{port}")
    print("Make sure PostgreSQL is running and configured in .env file")
    print("API will be available at: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

if __name__ == "__main__":
    main()
