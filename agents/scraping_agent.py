import asyncio
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from services.mcp_client import MCPClient
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService
from utils.text_processing import TextProcessor

class ScrapingAgent(BaseAgent):
    def __init__(self):
        super().__init__("ScrapingAgent")
        self.mcp_client = MCPClient()
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
        self.text_processor = TextProcessor()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process URL scraping and vector storage"""
        url = input_data.get("url", "")
        deep_scrape = input_data.get("deep_scrape", False)
        
        if not url:
            return {"error": "No URL provided", "success": False}
        
        self.log_action("Starting scraping", f"URL: {url}")
        
        try:
            # Scrape content
            if deep_scrape:
                scraped_data = self.mcp_client.deep_scrape_url(url)
            else:
                scraped_data = self.mcp_client.scrape_url(url)
            
            if "error" in scraped_data:
                return {"error": scraped_data["error"], "success": False}
            
            content = scraped_data.get("content", "")
            if not content:
                return {"error": "No content extracted", "success": False}
            
            # Process and chunk text
            chunks = self.text_processor.chunk_text(content, url)
            self.log_action("Text chunked", f"Created {len(chunks)} chunks")
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)
            
            if not embeddings:
                return {"error": "Failed to generate embeddings", "success": False}
            
            # Store in Pinecone
            success = self.pinecone_service.upsert_vectors(chunks, embeddings)
            
            if success:
                self.log_action("Successfully stored", f"{len(chunks)} chunks in vector DB")
                return {
                    "success": True,
                    "chunks_stored": len(chunks),
                    "url": url,
                    "content_preview": content[:200] + "..."
                }
            else:
                return {"error": "Failed to store in vector database", "success": False}
                
        except Exception as e:
            self.logger.error(f"Error in scraping process: {str(e)}")
            return {"error": str(e), "success": False} 