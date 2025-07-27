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
                    dimension=1536,  # Dimension for text-embedding-ada-002
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
                vector=[0] * 1536,  # Dummy vector for text-embedding-ada-002
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