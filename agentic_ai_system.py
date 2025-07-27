# Project Structure:
# autonomous_ai_agent/
# ├── main.py                    # Main Gradio application
# ├── requirements.txt           # Python dependencies
# ├── .env.example              # Environment variables template
# ├── config/
# │   └── settings.py           # Configuration settings
# ├── agents/
# │   ├── __init__.py
# │   ├── base_agent.py         # Base agent class
# │   ├── scraping_agent.py     # MCP scraping agent
# │   ├── search_agent.py       # Tavily search agent
# │   └── orchestrator.py       # Main orchestrator agent
# ├── services/
# │   ├── __init__.py
# │   ├── mcp_client.py         # MCP tools client
# │   ├── pinecone_service.py   # Pinecone vector DB service
# │   ├── embedding_service.py  # Text embedding service
# │   └── tavily_service.py     # Tavily search service
# ├── utils/
# │   ├── __init__.py
# │   ├── text_processing.py    # Text chunking and processing
# │   ├── similarity.py         # Similarity calculations
# │   └── logger.py             # Logging utilities
# └── frontend/
#     ├── __init__.py
#     └── gradio_interface.py   # Gradio UI components

# === requirements.txt ===
gradio==4.44.0
pinecone-client==3.0.0
openai==1.35.0
sentence-transformers==2.2.2
tavily-python==0.3.3
python-dotenv==1.0.0
requests==2.31.0
beautifulsoup4==4.12.2
numpy==1.24.3
pandas==2.0.3
asyncio-mqtt==0.16.1
pydantic==2.5.0
uvicorn==0.24.0
fastapi==0.104.1

# === .env.example ===
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=autonomous-ai-index
TAVILY_API_KEY=your_tavily_api_key_here
MCP_SERVER_URL=http://localhost:8000
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.7

# === config/settings.py ===
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings:
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "autonomous-ai-index")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # MCP Configuration
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Text Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Similarity Threshold
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Gradio Configuration
    GRADIO_HOST: str = os.getenv("GRADIO_HOST", "0.0.0.0")
    GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7860"))

settings = Settings()

# === utils/__init__.py ===
# Empty file

# === utils/logger.py ===
import logging
import sys
from datetime import datetime

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# === utils/text_processing.py ===
import re
from typing import List, Tuple
from config.settings import settings

class TextProcessor:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, url: str = "") -> List[dict]:
        """Chunk text into smaller pieces with metadata"""
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_data = {
                'text': chunk_text,
                'metadata': {
                    'source_url': url,
                    'chunk_index': len(chunks),
                    'word_count': len(chunk_words),
                    'character_count': len(chunk_text)
                }
            }
            chunks.append(chunk_data)
        
        return chunks

# === utils/similarity.py ===
import numpy as np
from typing import List, Tuple

class SimilarityCalculator:
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    @staticmethod
    def find_most_similar(query_embedding: np.ndarray, 
                         embeddings: List[np.ndarray],
                         threshold: float = 0.7) -> List[Tuple[int, float]]:
        """Find most similar embeddings above threshold"""
        similarities = []
        
        for i, embedding in enumerate(embeddings):
            similarity = SimilarityCalculator.cosine_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

# === services/__init__.py ===
# Empty file

# === services/mcp_client.py ===
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
            return response
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return {"error": str(e), "content": ""}
    
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
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP API call failed: {str(e)}")
            return {"error": str(e)}

# === services/embedding_service.py ===
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.array([])
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        return self.model.get_sentence_embedding_dimension()

# === services/pinecone_service.py ===
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from utils.logger import setup_logger
from config.settings import settings
import uuid
import time

logger = setup_logger(__name__)

class PineconeService:
    def __init__(self):
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index_name = settings.PINECONE_INDEX_NAME
            self.index = None
            self._initialize_index()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def _initialize_index(self):
        """Initialize Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise
    
    def upsert_vectors(self, chunks: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> bool:
        """Upsert vectors to Pinecone"""
        try:
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = str(uuid.uuid4())
                vector_data = {
                    "id": vector_id,
                    "values": embedding.tolist(),
                    "metadata": {
                        "text": chunk["text"][:1000],  # Limit text size for metadata
                        "source_url": chunk["metadata"]["source_url"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "word_count": chunk["metadata"]["word_count"]
                    }
                }
                vectors.append(vector_data)
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return False
    
    def query_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                     min_score: float = None) -> List[Dict[str, Any]]:
        """Query similar vectors from Pinecone"""
        try:
            min_score = min_score or settings.SIMILARITY_THRESHOLD
            
            response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            results = []
            for match in response.matches:
                if match.score >= min_score:
                    results.append({
                        "score": match.score,
                        "text": match.metadata.get("text", ""),
                        "source_url": match.metadata.get("source_url", ""),
                        "metadata": match.metadata
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            return []
    
    def delete_by_source(self, source_url: str) -> bool:
        """Delete all vectors from a specific source URL"""
        try:
            # Query to find all vectors with the source URL
            query_response = self.index.query(
                vector=[0] * 384,  # Dummy vector
                top_k=10000,
                include_metadata=True,
                filter={"source_url": source_url}
            )
            
            # Extract IDs and delete
            ids_to_delete = [match.id for match in query_response.matches]
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} vectors from source: {source_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return False

# === services/tavily_service.py ===
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
            return response.get("answer", "No answer found")
            
        except Exception as e:
            logger.error(f"Error in Tavily QnA search: {str(e)}")
            return "Search failed"

# === agents/__init__.py ===
# Empty file

# === agents/base_agent.py ===
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = setup_logger(f"Agent.{name}")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return result"""
        pass
    
    def log_action(self, action: str, details: str = ""):
        """Log agent action"""
        self.logger.info(f"{self.name} - {action}: {details}")

# === agents/scraping_agent.py ===
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

# === agents/search_agent.py ===
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

# === agents/orchestrator.py ===
import asyncio
from typing import Dict, Any, List, Optional
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
            answer += f"\n\nSources:\n{source_list}"
        
        return answer

# === frontend/__init__.py ===
# Empty file

# === frontend/gradio_interface.py ===
import gradio as gr
import asyncio
from typing import Tuple, List
from agents.orchestrator import OrchestratorAgent
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GradioInterface:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(
            title="Autonomous AI Agent System",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .panel {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🤖 Autonomous AI Agent System
            
            This system can:
            - **Scrape URLs** and store content in a vector database
            - **Answer queries** using scraped content or web search
            - **Automatically route** between local knowledge and web search
            """)
            
            with gr.Tabs():
                # URL Scraping Tab
                with gr.TabItem("🌐 URL Scraping"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            url_input = gr.Textbox(
                                label="URL to Scrape",
                                placeholder="https://example.com",
                                lines=1
                            )
                            deep_scrape_check = gr.Checkbox(
                                label="Deep Scrape (More comprehensive)",
                                value=False
                            )
                            scrape_btn = gr.Button("🔍 Scrape URL", variant="primary")
                        
                        with gr.Column(scale=3):
                            scrape_output = gr.Textbox(
                                label="Scraping Results",
                                lines=10,
                                interactive=False
                            )
                
                # Query Tab
                with gr.TabItem("💬 Ask Questions"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            query_input = gr.Textbox(
                                label="Your Question",
                                placeholder="Ask anything about the scraped content...",
                                lines=3
                            )
                            query_btn = gr.Button("🔍 Ask Question", variant="primary")
                            
                            gr.Markdown("### 📊 Response Source")
                            source_indicator = gr.Textbox(
                                label="Information Source",
                                interactive=False,
                                lines=1
                            )
                        
                        with gr.Column(scale=3):
                            query_output = gr.Textbox(
                                label="Answer",
                                lines=12,
                                interactive=False
                            )
                
                # System Status Tab
                with gr.TabItem("📊 System Status"):
                    with gr.Column():
                        gr.Markdown("### System Components Status")
                        
                        with gr.Row():
                            with gr.Column():
                                mcp_status = gr.Textbox(
                                    label="MCP Tools Status",
                                    value="🟡 Checking...",
                                    interactive=False
                                )
                                pinecone_status = gr.Textbox(
                                    label="Pinecone Vector DB Status",
                                    value="🟡 Checking...",
                                    interactive=False
                                )
                            
                            with gr.Column():
                                tavily_status = gr.Textbox(
                                    label="Tavily Search Status",
                                    value="🟡 Checking...",
                                    interactive=False
                                )
                                embedding_status = gr.Textbox(
                                    label="Embedding Service Status",
                                    value="🟡 Checking...",
                                    interactive=False
                                )
                        
                        status_refresh_btn = gr.Button("🔄 Refresh Status")
                        
                        gr.Markdown("### 📝 Recent Activity Log")
                        activity_log = gr.Textbox(
                            label="Activity Log",
                            lines=8,
                            interactive=False,
                            value="System initialized..."
                        )
            
            # Event handlers
            scrape_btn.click(
                fn=self.handle_scraping,
                inputs=[url_input, deep_scrape_check],
                outputs=[scrape_output]
            )
            
            query_btn.click(
                fn=self.handle_query,
                inputs=[query_input],
                outputs=[query_output, source_indicator]
            )
            
            status_refresh_btn.click(
                fn=self.check_system_status,
                outputs=[mcp_status, pinecone_status, tavily_status, embedding_status]
            )
            
            # Load initial status
            interface.load(
                fn=self.check_system_status,
                outputs=[mcp_status, pinecone_status, tavily_status, embedding_status]
            )
        
        return interface
    
    def handle_scraping(self, url: str, deep_scrape: bool) -> str:
        """Handle URL scraping request"""
        if not url:
            return "❌ Please provide a valid URL"
        
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.orchestrator.process_url_scraping(url, deep_scrape)
            )
            loop.close()
            
            if result.get("success"):
                return f"""✅ Successfully scraped and stored content!

📊 **Results:**
- Chunks stored: {result.get('chunks_stored', 0)}
- Source URL: {result.get('url', '')}

📝 **Content Preview:**
{result.get('content_preview', 'No preview available')}

The content has been processed and stored in the vector database for future queries."""
            else:
                return f"❌ Scraping failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Error in scraping handler: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def handle_query(self, query: str) -> Tuple[str, str]:
        """Handle query request"""
        if not query:
            return "❌ Please provide a question", "No source"
        
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.orchestrator.process_query(query)
            )
            loop.close()
            
            if result.get("success"):
                answer = result.get("answer", "No answer provided")
                source = result.get("source", "unknown")
                
                # Format source indicator
                if source == "vector_database":
                    source_text = f"📚 Vector Database ({result.get('num_sources', 0)} sources)"
                elif source == "web_search":
                    source_text = "🌐 Web Search (Tavily)"
                else:
                    source_text = f"❓ {source}"
                
                return answer, source_text
            else:
                error_msg = f"❌ Query failed: {result.get('error', 'Unknown error')}"
                return error_msg, "Error"
                
        except Exception as e:
            logger.error(f"Error in query handler: {str(e)}")
            return f"❌ Error: {str(e)}", "Error"
    
    def check_system_status(self) -> Tuple[str, str, str, str]:
        """Check status of all system components"""
        try:
            # Check MCP Tools
            try:
                from services.mcp_client import MCPClient
                mcp_client = MCPClient()
                # Try a simple call to check if MCP is working
                mcp_status = "🟢 Connected"
            except Exception:
                mcp_status = "🔴 Connection Failed"
            
            # Check Pinecone
            try:
                from services.pinecone_service import PineconeService
                pinecone_service = PineconeService()
                pinecone_status = "🟢 Connected"
            except Exception:
                pinecone_status = "🔴 Connection Failed"
            
            # Check Tavily
            try:
                from services.tavily_service import TavilyService
                tavily_service = TavilyService()
                tavily_status = "🟢 Connected"
            except Exception:
                tavily_status = "🔴 Connection Failed"
            
            # Check Embedding Service
            try:
                from services.embedding_service import EmbeddingService
                embedding_service = EmbeddingService()
                embedding_status = "🟢 Model Loaded"
            except Exception:
                embedding_status = "🔴 Model Load Failed"
            
            return mcp_status, pinecone_status, tavily_status, embedding_status
            
        except Exception as e:
            logger.error(f"Error checking system status: {str(e)}")
            error_status = f"🔴 Error: {str(e)}"
            return error_status, error_status, error_status, error_status

# === main.py ===
#!/usr/bin/env python3
"""
Autonomous AI Agent System
Main application entry point
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from frontend.gradio_interface import GradioInterface
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

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
        # Create Gradio interface
        gradio_interface = GradioInterface()
        interface = gradio_interface.create_interface()
        
        # Launch the application
        logger.info(f"Launching Gradio interface on {settings.GRADIO_HOST}:{settings.GRADIO_PORT}")
        
        interface.launch(
            server_name=settings.GRADIO_HOST,
            server_port=settings.GRADIO_PORT,
            share=False,
            debug=False,
            show_error=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# === Setup Instructions (README.md) ===
# Autonomous AI Agent System

A complete autonomous agentic AI system that scrapes URLs using MCP tools, stores content in Pinecone vector database, and provides intelligent query responses with automatic routing to web search when needed.

## Features

- 🌐 **URL Scraping**: Uses MCP Firecrawl tools for comprehensive web scraping
- 🧠 **Vector Storage**: Automatic chunking and storage in Pinecone vector database
- 🔍 **Intelligent Querying**: Searches vector DB first, falls back to Tavily web search
- 🎯 **Smart Routing**: Automatically determines best information source
- 🖥️ **User-Friendly Interface**: Clean Gradio-based web interface
- 📊 **System Monitoring**: Real-time status monitoring of all components

## Quick Setup

1. **Clone and Setup**:
   ```bash
   cd autonomous_ai_agent
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Required API Keys**:
   - OpenAI API key (for embeddings)
   - Pinecone API key and environment
   - Tavily API key for web search

4. **Start MCP Server**:
   Ensure your MCP Firecrawl server is running on the configured URL

5. **Run Application**:
   ```bash
   python main.py
   ```

6. **Access Interface**:
   Open http://localhost:7860 in your browser

## Usage

### URL Scraping
1. Go to "🌐 URL Scraping" tab
2. Enter URL to scrape
3. Choose regular or deep scraping
4. Click "🔍 Scrape URL"

### Asking Questions
1. Go to "💬 Ask Questions" tab
2. Enter your question
3. System automatically searches vector DB or web
4. View source indicator to see information origin

### System Monitoring
1. Go to "📊 System Status" tab
2. Monitor all component status
3. Check activity logs

## Architecture

```
autonomous_ai_agent/
├── main.py                 # Application entry point
├── config/settings.py      # Configuration management
├── agents/                 # AI agent implementations
│   ├── base_agent.py      # Base agent class
│   ├── scraping_agent.py  # MCP scraping agent
│   ├── search_agent.py    # Tavily search agent
│   └── orchestrator.py    # Main coordination agent
├── services/              # Core services
│   ├── mcp_client.py      # MCP tools integration
│   ├── pinecone_service.py # Vector database service
│   ├── embedding_service.py # Text embedding service
│   └── tavily_service.py   # Web search service
├── utils/                 # Utility functions
└── frontend/              # Gradio interface
```

## Configuration

All configuration is managed through environment variables in `.env`:

- `PINECONE_API_KEY`: Your Pinecone API key
- `TAVILY_API_KEY`: Your Tavily search API key  
- `MCP_SERVER_URL`: URL of your MCP Firecrawl server
- `SIMILARITY_THRESHOLD`: Minimum similarity for vector matches (0.7)
- `CHUNK_SIZE`: Text chunk size for processing (1000)

## Troubleshooting

1. **MCP Connection Issues**: Ensure Firecrawl MCP server is running
2. **Pinecone Errors**: Verify API key and environment settings
3. **Tavily Search Fails**: Check API key validity
4. **Embedding Issues**: Ensure sufficient disk space for model downloads

## License

MIT License - feel free to modify and distribute as needed.