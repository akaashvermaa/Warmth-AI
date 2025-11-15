"""
Embedding Manager for vector search functionality.
Handles generation, storage, and retrieval of semantic embeddings.
"""

import logging
import numpy as np
from functools import lru_cache
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Try to import sentence-transformers; gracefully fail if not installed
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Vector search features will be disabled.")


class EmbeddingManager:
    """
    Manages semantic embeddings for memory items.
    Generates, caches, and computes similarity between text embeddings.
    """
    
    # Default embedding model (fast, 22MB, 384-dim)
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: HuggingFace model identifier for sentence-transformers
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = self.EMBEDDING_DIM
        
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers library not available. Install with: pip install sentence-transformers")
            return
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if embedding model is available."""
        return self.model is not None and _SENTENCE_TRANSFORMERS_AVAILABLE
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array of shape (embedding_dim,)
        """
        if not self.is_available():
            logger.warning("Embedding model not available. Returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text for embedding: {type(text)}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            # Truncate long texts to avoid memory issues
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
                logger.debug(f"Text truncated to {max_length} chars for embedding")
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings as numpy arrays
        """
        if not self.is_available():
            logger.warning("Embedding model not available. Returning zero vectors.")
            return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
        
        if not texts:
            return []
        
        try:
            # Filter and truncate texts
            processed_texts = []
            for text in texts:
                if text and isinstance(text, str):
                    max_length = 512
                    processed_texts.append(text[:max_length] if len(text) > max_length else text)
                else:
                    processed_texts.append("")
            
            embeddings = self.model.encode(processed_texts, convert_to_numpy=True, batch_size=32)
            return [emb.astype(np.float32) for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
    
    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """
        Convert numpy embedding array to bytes for database storage.
        
        Args:
            embedding: Numpy array of floats
            
        Returns:
            Bytes representation
        """
        if embedding is None or len(embedding) == 0:
            return b''
        return embedding.astype(np.float32).tobytes()
    
    @staticmethod
    def bytes_to_embedding(data: bytes, dim: int = EMBEDDING_DIM) -> np.ndarray:
        """
        Convert bytes from database back to numpy array.
        
        Args:
            data: Bytes from database
            dim: Expected embedding dimension
            
        Returns:
            Numpy array
        """
        if not data:
            return np.zeros(dim, dtype=np.float32)
        try:
            return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error converting bytes to embedding: {e}")
            return np.zeros(dim, dtype=np.float32)
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1 (typically 0 to 1)
        """
        if embedding1 is None or embedding2 is None or len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0.0
    
    @staticmethod
    def batch_cosine_similarity(query_embedding: np.ndarray, embeddings: List[np.ndarray]) -> List[float]:
        """
        Compute cosine similarity between one query embedding and multiple embeddings.
        Vectorized for efficiency.
        
        Args:
            query_embedding: Query vector
            embeddings: List of embedding vectors
            
        Returns:
            List of similarity scores
        """
        if query_embedding is None or not embeddings:
            return []
        
        try:
            # Stack embeddings into a matrix
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            
            # Normalize query
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return [0.0] * len(embeddings)
            
            # Normalize each embedding
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            
            # Compute cosine similarities (vectorized)
            similarities = np.dot(embedding_matrix, query_embedding) / (norms.flatten() * query_norm)
            
            return [float(sim) for sim in similarities]
        except Exception as e:
            logger.error(f"Error computing batch cosine similarity: {e}")
            return [0.0] * len(embeddings)
