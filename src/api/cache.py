import redis
import json
import logging
from typing import Any, Optional
import os

logger = logging.getLogger(__name__)

class CacheService:
    """Redis caching service for search results and embeddings"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        
        logger.info(f"Connecting to Redis at {self.host}:{self.port}")
        
        self.redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=db,
            decode_responses=True
        )
        
        # Test connection
        self.redis.ping()
        logger.info("✅ Connected to Redis")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL (seconds)"""
        try:
            self.redis.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )
            logger.debug(f"Cached: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete key from cache"""
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        try:
            info = self.redis.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "total_keys": self.redis.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info)
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}
    
    def _calculate_hit_rate(self, info: dict) -> float:
        """Calculate cache hit rate"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return round((hits / total) * 100, 2)

# Global instance
_cache_service = None

def get_cache_service() -> CacheService:
    """Get or create cache service singleton"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
