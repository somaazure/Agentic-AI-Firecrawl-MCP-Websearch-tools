import asyncio
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from agents.scraping_agent import ScrapingAgent
from agents.search_agent import SearchAgent
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService
from config.settings import settings

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("OrchestratorAgent")
        self.scraping_agent = ScrapingAgent()
        self.search_agent = SearchAgent()
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
    
    async def process_url_scraping(self, url: str, deep_scrape: bool = False) -> Dict[str, Any]:
        """Process URL scraping request"""
        input_data = {
            "url": url,
            "deep_scrape": deep_scrape
        }
        return await self.scraping_agent.process(input_data)
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query - check vector DB first, then web search if needed"""
        self.log_action("Processing query", query)
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.embed_text(query)
            
            if query_embedding.size == 0:
                return await self._fallback_to_search(query)
            
            # Search in vector database
            similar_results = self.pinecone_service.query_similar(
                query_embedding, 
                top_k=5, 
                min_score=settings.SIMILARITY_THRESHOLD
            )
            
            if similar_results:
                # Found relevant content in vector DB
                self.log_action("Found relevant content", f"{len(similar_results)} matches")
                
                # Combine results into a coherent answer
                answer = self._format_vector_results(similar_results, query)
                
                return {
                    "success": True,
                    "answer": answer,
                    "source": "vector_database",
                    "num_sources": len(similar_results),
                    "sources": [result["source_url"] for result in similar_results]
                }
            else:
                # No relevant content found, fallback to web search
                self.log_action("No relevant content found", "Falling back to web search")
                return await self._fallback_to_search(query)
                
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return await self._fallback_to_search(query)
    
    async def _fallback_to_search(self, query: str) -> Dict[str, Any]:
        """Fallback to web search when vector DB has no relevant content"""
        search_input = {
            "query": query,
            "search_type": "qna"
        }
        return await self.search_agent.process(search_input)
    
    def _format_vector_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format vector search results into a coherent answer"""
        if not results:
            return "No relevant information found."
        
        # Combine the most relevant chunks
        combined_text = ""
        sources = set()
        
        for result in results[:3]:  # Use top 3 results
            combined_text += f"{result['text']}\n\n"
            sources.add(result['source_url'])
        
        # Create a formatted response
        answer = f"Based on the scraped content, here's what I found:\n\n{combined_text.strip()}"
        
        if sources:
            source_list = "\n".join([f"- {source}" for source in sources])
            answer += f"\nSources:\n{source_list}"
        
        return answer