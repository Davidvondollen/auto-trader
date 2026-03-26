"""
Caching and Performance Optimization Utilities
Intelligent caching for market data and computed results.
"""

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps
from loguru import logger
import pickle
from pathlib import Path
import hashlib
import json


class DataCache:
    """
    Intelligent cache for market data with TTL and invalidation.
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        default_ttl_minutes: int = 30,
        max_memory_mb: int = 500
    ):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory for persistent cache
            default_ttl_minutes: Default time-to-live for cached items
            default_ttl_minutes: Default TTL in minutes
            max_memory_mb: Maximum memory for in-memory cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

        # In-memory cache
        self.memory_cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, datetime] = {}
        self.cache_sizes: Dict[str, int] = {}

        logger.info(f"Initialized DataCache: dir={cache_dir}, ttl={default_ttl_minutes}m")

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first
        if key in self.memory_cache:
            cache_entry = self.memory_cache[key]
            ttl = cache_entry.get('ttl', self.default_ttl)
            cached_at = cache_entry.get('cached_at', datetime.min)

            # Check if expired
            if datetime.now() - cached_at < ttl:
                self.access_times[key] = datetime.now()
                return cache_entry['data']
            else:
                # Expired, remove from memory
                self._remove_from_memory(key)

        # Check disk cache
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_entry = pickle.load(f)

                ttl = cache_entry.get('ttl', self.default_ttl)
                cached_at = cache_entry.get('cached_at', datetime.min)

                if datetime.now() - cached_at < ttl:
                    # Load into memory for faster access
                    self._add_to_memory(key, cache_entry)
                    return cache_entry['data']
                else:
                    # Expired, delete file
                    cache_file.unlink()

            except Exception as e:
                logger.error(f"Failed to load cache for {key}: {e}")

        return None

    def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[timedelta] = None,
        persist: bool = True
    ) -> None:
        """
        Set item in cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live (uses default if None)
            persist: Whether to persist to disk
        """
        if ttl is None:
            ttl = self.default_ttl

        cache_entry = {
            'data': data,
            'cached_at': datetime.now(),
            'ttl': ttl
        }

        # Add to memory
        self._add_to_memory(key, cache_entry)

        # Persist to disk if requested
        if persist:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_entry, f)
            except Exception as e:
                logger.error(f"Failed to persist cache for {key}: {e}")

    def invalidate(self, key: str) -> None:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key to invalidate
        """
        # Remove from memory
        self._remove_from_memory(key)

        # Remove from disk
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        if cache_file.exists():
            cache_file.unlink()

    def clear_all(self) -> None:
        """Clear all cache (memory and disk)."""
        self.memory_cache.clear()
        self.access_times.clear()
        self.cache_sizes.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

        logger.info("Cache cleared")

    def _add_to_memory(self, key: str, cache_entry: Dict) -> None:
        """Add item to memory cache, managing size limits."""
        # Estimate size
        try:
            size = len(pickle.dumps(cache_entry['data']))
        except:
            size = 1024  # Default estimate

        # Check if we need to evict
        current_size = sum(self.cache_sizes.values())
        while current_size + size > self.max_memory_bytes and self.memory_cache:
            # Evict least recently used
            lru_key = min(self.access_times, key=self.access_times.get)
            self._remove_from_memory(lru_key)
            current_size = sum(self.cache_sizes.values())

        # Add to cache
        self.memory_cache[key] = cache_entry
        self.access_times[key] = datetime.now()
        self.cache_sizes[key] = size

    def _remove_from_memory(self, key: str) -> None:
        """Remove item from memory cache."""
        self.memory_cache.pop(key, None)
        self.access_times.pop(key, None)
        self.cache_sizes.pop(key, None)

    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'memory_items': len(self.memory_cache),
            'memory_size_mb': sum(self.cache_sizes.values()) / (1024 * 1024),
            'disk_items': len(list(self.cache_dir.glob("*.pkl")))
        }


def cached(
    cache: DataCache,
    ttl_minutes: Optional[int] = None,
    key_prefix: str = ""
) -> Callable:
    """
    Decorator for caching function results.

    Args:
        cache: DataCache instance
        ttl_minutes: TTL in minutes (uses cache default if None)
        key_prefix: Prefix for cache key

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix or func.__name__]

            # Add args
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(type(arg).__name__)

            # Add kwargs
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}={v}")

            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result

            # Compute result
            logger.debug(f"Cache miss: {cache_key}")
            result = func(*args, **kwargs)

            # Cache result
            ttl = timedelta(minutes=ttl_minutes) if ttl_minutes else None
            cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


class ResultMemoizer:
    """
    Memoize expensive computations with automatic invalidation.
    """

    def __init__(self, maxsize: int = 128):
        """
        Initialize memoizer.

        Args:
            maxsize: Maximum number of cached results
        """
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def memoize(self, func: Callable) -> Callable:
        """
        Memoize function results.

        Args:
            func: Function to memoize

        Returns:
            Memoized function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create key from function name and arguments
            key = self._make_key(func.__name__, args, kwargs)

            # Check cache
            if key in self.cache:
                result, _ = self.cache[key]
                self.hits += 1
                return result

            # Compute result
            self.misses += 1
            result = func(*args, **kwargs)

            # Add to cache
            self.cache[key] = (result, datetime.now())

            # Evict if over size
            if len(self.cache) > self.maxsize:
                # Remove oldest
                oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            return result

        return wrapper

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function arguments."""
        key_parts = [func_name]

        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(str(hash(tuple(arg))))
            elif isinstance(arg, dict):
                key_parts.append(str(hash(tuple(sorted(arg.items())))))
            else:
                key_parts.append(type(arg).__name__)

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        return ":".join(key_parts)

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class BatchProcessor:
    """
    Batch process operations for better performance.
    """

    @staticmethod
    def batch_dataframe_operations(
        dfs: list,
        operation: Callable,
        batch_size: int = 100
    ) -> list:
        """
        Process DataFrames in batches.

        Args:
            dfs: List of DataFrames
            operation: Operation to apply to each DataFrame
            batch_size: Batch size

        Returns:
            List of results
        """
        results = []

        for i in range(0, len(dfs), batch_size):
            batch = dfs[i:i + batch_size]

            # Process batch
            batch_results = [operation(df) for df in batch]
            results.extend(batch_results)

        return results

    @staticmethod
    def vectorized_calculation(
        df: pd.DataFrame,
        calculations: Dict[str, Callable]
    ) -> pd.DataFrame:
        """
        Apply multiple calculations in vectorized manner.

        Args:
            df: Input DataFrame
            calculations: Dict mapping column names to calculation functions

        Returns:
            DataFrame with calculated columns
        """
        result = df.copy()

        # Apply all calculations at once (vectorized)
        for col_name, calc_func in calculations.items():
            try:
                result[col_name] = calc_func(result)
            except Exception as e:
                logger.error(f"Calculation failed for {col_name}: {e}")

        return result


if __name__ == "__main__":
    # Test caching
    logger.info("Testing Cache System")

    # Create cache
    cache = DataCache(cache_dir="./test_cache", default_ttl_minutes=5)

    # Test basic caching
    cache.set("test_key", {"data": [1, 2, 3]})
    result = cache.get("test_key")
    print(f"Cache get: {result}")

    # Test decorator
    @cached(cache, ttl_minutes=1, key_prefix="expensive_calc")
    def expensive_calculation(x, y):
        logger.info(f"Computing {x} + {y}")
        return x + y

    # First call - cache miss
    result1 = expensive_calculation(5, 3)
    print(f"Result 1: {result1}")

    # Second call - cache hit
    result2 = expensive_calculation(5, 3)
    print(f"Result 2: {result2}")

    # Different args - cache miss
    result3 = expensive_calculation(10, 20)
    print(f"Result 3: {result3}")

    # Test memoizer
    memoizer = ResultMemoizer(maxsize=10)

    @memoizer.memoize
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    print(f"\nFibonacci(10) = {fibonacci(10)}")
    print(f"Memoizer stats: {memoizer.get_stats()}")

    # Cache stats
    print(f"\nCache stats: {cache.get_stats()}")
