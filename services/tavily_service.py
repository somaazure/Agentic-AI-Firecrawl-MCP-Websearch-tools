from tavily import TavilyClient
from typing import List, Dict, Any
from utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class TavilyService:
    def __init__(self):
        try:
            self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            logger.info("Initialized Tavily search client")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily: {str(e)}")
            raise
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Tavily API"""
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=None,
                exclude_domains=None
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching with Tavily: {str(e)}")
            return []
    
    def search_and_summarize(self, query: str) -> str:
        """Search and get a summarized answer"""
        try:
            response = self.client.qna_search(query=query)
            # Handle both string and dict responses
            if isinstance(response, str):
                return response
            elif isinstance(response, dict):
                return response.get("answer", "No answer found")
            else:
                return "No answer found"
            
        except Exception as e:
            logger.error(f"Error in Tavily QnA search: {str(e)}")
            # Fallback to regular search
            try:
                results = self.search(query, max_results=3)
                if results:
                    # Combine top results
                    combined_content = " ".join([result.get("content", "") for result in results[:2]])
                    return f"Based on search results: {combined_content[:500]}..."
                else:
                    return "No relevant information found"
            except Exception as search_error:
                logger.error(f"Fallback search also failed: {str(search_error)}")
                return "Search failed" 