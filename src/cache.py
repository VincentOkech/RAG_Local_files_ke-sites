from functools import lru_cache
import diskcache
import time
from typing import Any, Optional

class Cache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache = diskcache.Cache(cache_dir)
        self.ttl = 3600  # 1 hour default TTL

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        self.cache.set(key, value, expire=ttl or self.ttl)

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self.cache.delete(key)

    @lru_cache(maxsize=100)
    def memoize(self, func):
        """Decorator for function-level caching."""
        def wrapper(*args, **kwargs):
            # Create a cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            result = self.get(key)
            
            if result is None:
                result = func(*args, **kwargs)
                self.set(key, result)
            
            return result
        return wrapper 