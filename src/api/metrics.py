from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'API request latency',
    ['method', 'endpoint']
)

EMBEDDING_LATENCY = Histogram(
    'embedding_generation_seconds',
    'Embedding generation latency',
    ['batch_size']
)

SEARCH_LATENCY = Histogram(
    'search_latency_seconds',
    'Search operation latency',
    ['top_k']
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits'
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses'
)

ACTIVE_DOCUMENTS = Gauge(
    'active_documents',
    'Number of documents in system'
)

ACTIVE_CHUNKS = Gauge(
    'active_chunks',
    'Number of chunks in system'
)

def track_request_metrics(func):
    """Decorator to track request metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        status = "success"
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = "error"
            raise e
        finally:
            duration = time.time() - start_time
            
            # Get endpoint name
            endpoint = func.__name__
            
            # Track metrics
            REQUEST_COUNT.labels(
                method="POST",
                endpoint=endpoint,
                status=status
            ).inc()
            
            REQUEST_LATENCY.labels(
                method="POST",
                endpoint=endpoint
            ).observe(duration)
    
    return wrapper

def track_embedding_time(batch_size: int):
    """Decorator to track embedding generation time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            EMBEDDING_LATENCY.labels(
                batch_size=str(batch_size)
            ).observe(duration)
            
            logger.info(f"Embedding batch ({batch_size}): {duration:.3f}s")
            
            return result
        return wrapper
    return decorator

def track_search_time(top_k: int):
    """Decorator to track search time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            SEARCH_LATENCY.labels(
                top_k=str(top_k)
            ).observe(duration)
            
            logger.info(f"Search (k={top_k}): {duration:.3f}s")
            
            return result
        return wrapper
    return decorator

class MetricsCollector:
    """Collect and report metrics"""
    
    @staticmethod
    def get_metrics_summary():
        """Get summary of all metrics"""
        return {
            "requests": {
                "total": REQUEST_COUNT._value.get(),
            },
            "cache": {
                "hits": CACHE_HITS._value.get(),
                "misses": CACHE_MISSES._value.get(),
                "hit_rate": MetricsCollector._calculate_cache_hit_rate()
            },
            "storage": {
                "documents": ACTIVE_DOCUMENTS._value.get(),
                "chunks": ACTIVE_CHUNKS._value.get()
            }
        }
    
    @staticmethod
    def _calculate_cache_hit_rate():
        """Calculate cache hit rate percentage"""
        hits = CACHE_HITS._value.get()
        misses = CACHE_MISSES._value.get()
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return round((hits / total) * 100, 2)
    
    @staticmethod
    def export_prometheus():
        """Export metrics in Prometheus format"""
        return generate_latest()
