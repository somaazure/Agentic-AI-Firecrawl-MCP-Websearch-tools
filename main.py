#!/usr/bin/env python3
"""
Autonomous AI Agent System
Main application entry point
"""

import os
import sys
import asyncio
import json
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.orchestrator import OrchestratorAgent
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Autonomous AI Agent System",
    description="AI Agent System with SSE support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup"""
    global orchestrator
    logger.info("Initializing Autonomous AI Agent System...")
    orchestrator = OrchestratorAgent()
    logger.info("System initialized successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Autonomous AI Agent System",
        "version": "1.0.0",
        "endpoints": {
            "sse": "/sse",
            "scrape": "/scrape",
            "query": "/query",
            "status": "/status"
        }
    }

@app.get("/sse")
async def sse_endpoint():
    """Server-Sent Events endpoint"""
    async def event_stream():
        session_id = str(uuid.uuid4())
        
        # Send initial connection event
        yield f"event: endpoint\ndata: /messages?sessionId={session_id}\n\n"
        
        # Keep connection alive
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            yield f"event: heartbeat\ndata: {json.dumps({'sessionId': session_id, 'timestamp': asyncio.get_event_loop().time()})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.post("/scrape")
async def scrape_url(request: Request):
    """Scrape a URL and store content"""
    try:
        data = await request.json()
        url = data.get("url")
        deep_scrape = data.get("deep_scrape", False)
        
        if not url:
            return {"success": False, "error": "URL is required"}
        
        if not orchestrator:
            return {"success": False, "error": "System not initialized"}
        
        result = await orchestrator.process_url_scraping(url, deep_scrape)
        return result
        
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/query")
async def query_endpoint(request: Request):
    """Process a query"""
    try:
        data = await request.json()
        query = data.get("query")
        
        if not query:
            return {"success": False, "error": "Query is required"}
        
        if not orchestrator:
            return {"success": False, "error": "System not initialized"}
        
        result = await orchestrator.process_query(query)
        return result
        
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/status")
async def system_status():
    """Get system status"""
    try:
        status = {
            "system": "running",
            "timestamp": asyncio.get_event_loop().time(),
            "components": {}
        }
        
        # Check MCP Tools
        try:
            from services.mcp_client import MCPClient
            mcp_client = MCPClient()
            status["components"]["mcp"] = "connected"
        except Exception:
            status["components"]["mcp"] = "failed"
        
        # Check Pinecone
        try:
            from services.pinecone_service import PineconeService
            pinecone_service = PineconeService()
            status["components"]["pinecone"] = "connected"
            
            # Check if there's content in the database
            try:
                # Query with a dummy vector to see if there's any content
                dummy_vector = [0.0] * 1536
                results = pinecone_service.query_similar(
                    np.array(dummy_vector), 
                    top_k=1, 
                    min_score=0.0
                )
                status["components"]["pinecone_content"] = f"{len(results)} vectors found"
            except Exception as e:
                status["components"]["pinecone_content"] = f"error: {str(e)}"
                
        except Exception:
            status["components"]["pinecone"] = "failed"
        
        # Check Tavily
        try:
            from services.tavily_service import TavilyService
            tavily_service = TavilyService()
            status["components"]["tavily"] = "connected"
        except Exception:
            status["components"]["tavily"] = "failed"
        
        # Check Embedding Service
        try:
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            status["components"]["embedding"] = "loaded"
        except Exception:
            status["components"]["embedding"] = "failed"
        
        return status
        
    except Exception as e:
        logger.error(f"Error checking system status: {str(e)}")
        return {"system": "error", "error": str(e)}

def main():
    """Main application entry point"""
    logger.info("Starting Autonomous AI Agent System...")
    
    # Check required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY", 
        "TAVILY_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not getattr(settings, var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please check your .env file and ensure all required variables are set.")
        sys.exit(1)
    
    try:
        # Launch the FastAPI server
        logger.info(f"Launching FastAPI server on {settings.GRADIO_HOST}:{settings.GRADIO_PORT}")
        
        uvicorn.run(
            app,
            host=settings.GRADIO_HOST,
            port=settings.GRADIO_PORT,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()