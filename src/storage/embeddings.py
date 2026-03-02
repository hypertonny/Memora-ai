"""
Embedding generation using sentence-transformers.
"""

import asyncio
from typing import Optional
import logging

from sentence_transformers import SentenceTransformer

from ..config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for text using sentence-transformers.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name or settings.embedding_model
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.dimension
        
        try:
            # Run in thread pool (sentence-transformers is synchronous)
            embedding = await asyncio.get_running_loop().run_in_executor(
                None, self._embed_sync, text
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * self.dimension
    
    def _embed_sync(self, text: str) -> list[float]:
        """Synchronous embedding generation."""
        # Truncate text if too long (most models have 512 token limit)
        max_chars = 5000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            embeddings = await asyncio.get_running_loop().run_in_executor(
                None, self._embed_batch_sync, texts
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [[0.0] * self.dimension for _ in texts]
    
    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous batch embedding."""
        # Truncate and clean texts
        max_chars = 5000
        cleaned = [
            t[:max_chars] if len(t) > max_chars else t
            for t in texts
        ]
        
        embeddings = self.model.encode(cleaned, convert_to_numpy=True)
        return embeddings.tolist()
    
    async def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        from numpy import dot
        from numpy.linalg import norm
        
        emb1 = await self.embed(text1)
        emb2 = await self.embed(text2)
        
        # Cosine similarity
        return float(dot(emb1, emb2) / (norm(emb1) * norm(emb2)))
