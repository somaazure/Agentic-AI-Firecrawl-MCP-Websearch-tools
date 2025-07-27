import numpy as np
from typing import List, Union
from utils.logger import setup_logger
from config.settings import settings
import openai

logger = setup_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        try:
            # Set API key globally
            openai.api_key = settings.OPENAI_API_KEY
            logger.info(f"Initialized OpenAI embedding service with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding service: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model_name
            )
            embedding = response['data'][0]['embedding']
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.array([])
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        try:
            response = openai.Embedding.create(
                input=texts,
                model=self.model_name
            )
            embeddings = [np.array(item['embedding']) for item in response['data']]
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        # OpenAI text-embedding-ada-002 produces 1536-dimensional vectors
        return 1536 