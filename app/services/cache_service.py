"""
Cache Manager for optimization.
Provides multi-layer caching: Redis (primary) with in-memory fallback.
Caches embeddings, mood contexts, and search results.
"""

import logging
import json
import time
from typing import Optional, Any, List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try to import redis; gracefully fallback if not installed
try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    logger.warning("redis not installed. Using in-memory cache only. Install: pip install redis")


class CacheManager:
    """
    Multi-layer cache: Redis (primary) + in-memory fallback.
    Optimized for embedding caching and search results.
    """
    
    # Cache TTL (time-to-live) in seconds
    TTL_EMBEDDING = 24 * 60 * 60      # 24 hours for embeddings (immutable)
    TTL_MOOD_CONTEXT = 5 * 60          # 5 minutes for mood context (changes frequently)
    TTL_SEARCH_RESULT = 10 * 60        # 10 minutes for search results (user-specific)
    TTL_MEMORY_IMPORTANCE = 60 * 60    # 1 hour for importance scores
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, redis_password: Optional[str] = None):
        """
        Initialize cache manager.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (optional)
        """
        self.redis_client = None
        self.in_memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if not _REDIS_AVAILABLE:
            logger.warning("Redis not available. Using in-memory cache only.")
            return
        
        try:
            logger.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True,  # Automatically decode to strings
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✓ Redis connection established")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory cache.")
            self.redis_client = None
    
    def is_redis_available(self) -> bool:
        """Check if Redis is available and connected."""
        if self.redis_client is None:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.debug(f"Redis unavailable: {e}")
            return False
    
    def _get_cache_key(self, prefix: str, key: str) -> str:
        """Generate cache key with prefix."""
        return f"warmth:{prefix}:{key}"
    
    def get(self, prefix: str, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Args:
            prefix: Cache namespace (e.g., 'embedding', 'mood_context')
            key: Unique key within namespace
            
        Returns:
            Cached value or None if not found
        """
        cache_key = self._get_cache_key(prefix, key)
        
        # Try Redis first
        if self.is_redis_available():
            try:
                value = self.redis_client.get(cache_key)
                if value:
                    self.cache_hits += 1
                    logger.debug(f"Redis HIT: {cache_key}")
                    # Deserialize JSON
                    return json.loads(value)
                else:
                    self.cache_misses += 1
                    logger.debug(f"Redis MISS: {cache_key}")
                    return None
            except Exception as e:
                logger.debug(f"Redis get error: {e}. Falling back to in-memory.")
        
        # Fallback to in-memory
        if cache_key in self.in_memory_cache:
            entry = self.in_memory_cache[cache_key]
            # Check expiration
            if entry['expires_at'] > time.time():
                self.cache_hits += 1
                logger.debug(f"In-memory HIT: {cache_key}")
                return entry['value']
            else:
                # Expired, remove and miss
                del self.in_memory_cache[cache_key]
                self.cache_misses += 1
                logger.debug(f"In-memory EXPIRED: {cache_key}")
                return None
        
        self.cache_misses += 1
        logger.debug(f"Cache MISS: {cache_key}")
        return None
    
    def set(self, prefix: str, key: str, value: Any, ttl: int = TTL_SEARCH_RESULT) -> bool:
        """
        Store value in cache.
        
        Args:
            prefix: Cache namespace
            key: Unique key
            value: Value to cache (JSON-serializable)
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        cache_key = self._get_cache_key(prefix, key)
        
        try:
            json_value = json.dumps(value)
        except Exception as e:
            logger.warning(f"Failed to serialize cache value: {e}")
            return False
        
        # Try Redis first
        if self.is_redis_available():
            try:
                self.redis_client.setex(cache_key, ttl, json_value)
                logger.debug(f"Redis SET: {cache_key} (TTL: {ttl}s)")
                return True
            except Exception as e:
                logger.debug(f"Redis set error: {e}. Falling back to in-memory.")
        
        # Fallback to in-memory
        try:
            self.in_memory_cache[cache_key] = {
                'value': value,
                'expires_at': time.time() + ttl
            }
            logger.debug(f"In-memory SET: {cache_key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Failed to set cache: {e}")
            return False
    
    def delete(self, prefix: str, key: str) -> bool:
        """Delete cache entry."""
        cache_key = self._get_cache_key(prefix, key)
        
        if self.is_redis_available():
            try:
                self.redis_client.delete(cache_key)
                logger.debug(f"Redis DELETE: {cache_key}")
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")
        
        if cache_key in self.in_memory_cache:
            del self.in_memory_cache[cache_key]
            logger.debug(f"In-memory DELETE: {cache_key}")
        
        return True
    
    def clear_prefix(self, prefix: str) -> int:
        """
        Clear all cache entries with given prefix.
        
        Args:
            prefix: Prefix pattern to clear
            
        Returns:
            Number of entries deleted
        """
        pattern = self._get_cache_key(prefix, "*")
        deleted = 0
        
        if self.is_redis_available():
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted = self.redis_client.delete(*keys)
                    logger.info(f"Redis CLEAR: {prefix} ({deleted} entries)")
            except Exception as e:
                logger.debug(f"Redis clear error: {e}")
        
        # In-memory
        prefix_key = f"warmth:{prefix}:"
        keys_to_delete = [k for k in self.in_memory_cache.keys() if k.startswith(prefix_key)]
        for key in keys_to_delete:
            del self.in_memory_cache[key]
            deleted += 1
        
        if keys_to_delete:
            logger.info(f"In-memory CLEAR: {prefix} ({len(keys_to_delete)} entries)")
        
        return deleted
    
    def clear_all(self) -> bool:
        """Clear entire cache."""
        if self.is_redis_available():
            try:
                self.redis_client.flushdb()
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.warning(f"Redis flush error: {e}")
        
        self.in_memory_cache.clear()
        logger.info("In-memory cache cleared")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        in_memory_size = len(self.in_memory_cache)
        redis_size = 0
        
        if self.is_redis_available():
            try:
                redis_info = self.redis_client.info('memory')
                redis_size = redis_info.get('used_memory', 0)
            except Exception as e:
                logger.debug(f"Failed to get Redis stats: {e}")
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate': round(hit_rate * 100, 2),
            'in_memory_entries': in_memory_size,
            'redis_available': self.is_redis_available(),
            'redis_memory_bytes': redis_size
        }


class EmbeddingCache:
    """
    Specialized cache for embeddings.
    Stores binary embeddings efficiently.
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize embedding cache.
        
        Args:
            cache_manager: CacheManager instance
        """
        self.cache = cache_manager
    
    def get_embedding(self, memory_id: int, user_id: str) -> Optional[List[float]]:
        """
        Get cached embedding for a memory.
        
        Args:
            memory_id: Memory ID
            user_id: User ID (for namespace isolation)
            
        Returns:
            Embedding as list of floats or None
        """
        key = f"user_{user_id}_mem_{memory_id}"
        return self.cache.get("embedding", key)
    
    def set_embedding(self, memory_id: int, user_id: str, embedding: List[float]) -> bool:
        """
        Cache an embedding.
        
        Args:
            memory_id: Memory ID
            user_id: User ID
            embedding: Embedding vector
            
        Returns:
            True if successful
        """
        key = f"user_{user_id}_mem_{memory_id}"
        return self.cache.set(
            "embedding",
            key,
            embedding,
            ttl=CacheManager.TTL_EMBEDDING
        )
    
    def clear_user_embeddings(self, user_id: str) -> int:
        """Clear all embeddings for a user."""
        # This is approximate - clears all embeddings
        # In production, might want per-user namespace
        return self.cache.clear_prefix("embedding")


class MoodContextCache:
    """
    Specialized cache for mood analysis results.
    """
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize mood context cache."""
        self.cache = cache_manager
    
    def get_mood_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached mood context for user.
        
        Args:
            user_id: User ID
            
        Returns:
            Mood context dict or None
        """
        return self.cache.get("mood_context", user_id)
    
    def set_mood_context(self, user_id: str, context: Dict[str, Any]) -> bool:
        """
        Cache mood context.
        
        Args:
            user_id: User ID
            context: Mood context analysis result
            
        Returns:
            True if successful
        """
        return self.cache.set(
            "mood_context",
            user_id,
            context,
            ttl=CacheManager.TTL_MOOD_CONTEXT
        )
    
    def invalidate_mood_context(self, user_id: str) -> bool:
        """Invalidate mood context cache when new mood logged."""
        return self.cache.delete("mood_context", user_id)


class SearchResultCache:
    """
    Specialized cache for semantic search results.
    """
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize search result cache."""
        self.cache = cache_manager
    
    def _make_key(self, user_id: str, query: str, top_k: int) -> str:
        """Generate cache key for search."""
        return f"user_{user_id}_q_{hash(query)}_k_{top_k}"
    
    def get_search_results(self, user_id: str, query: str, top_k: int) -> Optional[List[Dict]]:
        """
        Get cached search results.
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Result count
            
        Returns:
            List of memory dicts or None
        """
        key = self._make_key(user_id, query, top_k)
        return self.cache.get("search_result", key)
    
    def set_search_results(self, user_id: str, query: str, top_k: int, 
                          results: List[Dict]) -> bool:
        """
        Cache search results.
        
        Args:
            user_id: User ID
            query: Search query
            top_k: Result count
            results: Memory results
            
        Returns:
            True if successful
        """
        key = self._make_key(user_id, query, top_k)
        return self.cache.set(
            "search_result",
            key,
            results,
            ttl=CacheManager.TTL_SEARCH_RESULT
        )
    
    def invalidate_user_results(self, user_id: str) -> int:
        """Invalidate all search results for user when memory changes."""
        # In production, might want more granular invalidation
        return self.cache.clear_prefix("search_result")
