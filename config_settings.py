import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
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