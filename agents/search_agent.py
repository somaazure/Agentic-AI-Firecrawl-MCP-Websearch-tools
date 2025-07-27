from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from services.tavily_service import TavilyService

class SearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("SearchAgent")
        self.tavily_service = TavilyService()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process web search using Tavily"""
        query = input_data.get("query", "")
        search_type = input_data.get("search_type", "search")  # 'search' or 'qna'
        
        if not query:
            return {"error": "No query provided", "success": False}
        
        self.log_action("Starting web search", f"Query: {query}")
        
        try:
            if search_type == "qna":
                # Get direct answer
                answer = self.tavily_service.search_and_summarize(query)
                return {
                    "success": True,
                    "answer": answer,
                    "source": "web_search",
                    "query": query
                }
            else:
                # Get search results
                results = self.tavily_service.search(query, max_results=5)
                
                # Format results for response
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "title": result["title"],
                        "url": result["url"],
                        "content": result["content"][:300] + "...",
                        "score": result["score"]
                    })
                
                return {
                    "success": True,
                    "results": formatted_results,
                    "source": "web_search",
                    "query": query
                }
                
        except Exception as e:
            self.logger.error(f"Error in search process: {str(e)}")
            return {"error": str(e), "success": False} 