import asyncio
import json
import requests
from typing import Dict, Any, List, Optional
from utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class MCPClient:
    def __init__(self, server_url: str = None):
        self.server_url = server_url or settings.MCP_SERVER_URL
        self.session = requests.Session()
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape URL using MCP firecrawl tools"""
        try:
            # Use firecrawl_scrape for basic scraping
            response = self._call_mcp_tool("firecrawl_scrape", {"url": url})
            if response.get("error"):
                # If MCP returns an error, use fallback
                logger.warning(f"MCP returned error, using fallback: {response.get('error')}")
                return self._fallback_scrape(url)
            return response
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            # Fallback to basic HTTP request
            return self._fallback_scrape(url)
    
    def _fallback_scrape(self, url: str) -> Dict[str, Any]:
        """Fallback scraping method when MCP is not available"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return {
                "content": text[:5000],  # Limit content length
                "url": url,
                "title": soup.title.string if soup.title else "No title"
            }
            
        except Exception as e:
            logger.error(f"Fallback scraping failed for {url}: {str(e)}")
            return {"error": f"Scraping failed: {str(e)}", "content": ""}
    
    def deep_scrape_url(self, url: str) -> Dict[str, Any]:
        """Deep scrape URL using MCP firecrawl tools"""
        try:
            # Use firecrawl_deep_research for comprehensive scraping
            response = self._call_mcp_tool("firecrawl_deep_research", {"url": url})
            return response
        except Exception as e:
            logger.error(f"Error deep scraping URL {url}: {str(e)}")
            return {"error": str(e), "content": ""}
    
    def search_and_scrape(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search and scrape multiple URLs"""
        try:
            # First search for relevant URLs
            search_response = self._call_mcp_tool("firecrawl_search", {
                "query": query,
                "max_results": max_results
            })
            
            results = []
            if "urls" in search_response:
                for url in search_response["urls"]:
                    scraped_data = self.scrape_url(url)
                    results.append(scraped_data)
            
            return results
        except Exception as e:
            logger.error(f"Error in search and scrape: {str(e)}")
            return []
    
    def _call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool via HTTP API"""
        try:
            payload = {
                "tool": tool_name,
                "parameters": params
            }
            
            response = self.session.post(
                f"{self.server_url}/mcp/call",
                json=payload,
                timeout=30  # Reduced timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP API call failed: {str(e)}")
            # Return a more graceful error response
            return {"error": f"MCP service unavailable: {str(e)}", "content": ""} 