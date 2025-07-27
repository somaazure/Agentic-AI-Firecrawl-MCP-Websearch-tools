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