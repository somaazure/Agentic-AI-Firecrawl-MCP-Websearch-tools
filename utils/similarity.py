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