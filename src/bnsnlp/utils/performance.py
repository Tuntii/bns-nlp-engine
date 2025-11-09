"""Performance utilities for batch processing, streaming, multiprocessing, GPU, pooling, and caching."""

import asyncio
import hashlib
import multiprocessing
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, AsyncIterator, Callable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    """Efficient batch processing with configurable batch size."""

    def __init__(self, batch_size: int = 32):
        """Initialize batch processor.

        Args:
            batch_size: Number of items to process in each batch
        """
        self.batch_size = batch_size

    async def process_batches(
        self, items: List[T], process_fn: Callable[[List[T]], Any]
    ) -> List[R]:
        """Process items in batches.

        Args:
            items: List of items to process
            process_fn: Async function to process each batch

        Returns:
            List of results from all batches
        """
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batch_results = await process_fn(batch)
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        return results

    def create_batches(self, items: List[T]) -> List[List[T]]:
        """Split items into batches.

        Args:
            items: List of items to split

        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i : i + self.batch_size])
        return batches


class StreamProcessor:
    """Async streaming utilities for processing data streams."""

    @staticmethod
    async def process_stream(
        input_stream: AsyncIterator[T], process_fn: Callable[[T], Any]
    ) -> AsyncIterator[R]:
        """Process streaming data without loading all into memory.

        Args:
            input_stream: Async iterator of input items
            process_fn: Async function to process each item

        Yields:
            Processed results
        """
        async for item in input_stream:
            result = await process_fn(item)
            yield result

    @staticmethod
    async def batch_stream(
        input_stream: AsyncIterator[T], batch_size: int = 32
    ) -> AsyncIterator[List[T]]:
        """Batch items from a stream.

        Args:
            input_stream: Async iterator of input items
            batch_size: Number of items per batch

        Yields:
            Batches of items
        """
        batch = []
        async for item in input_stream:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining items
        if batch:
            yield batch

    @staticmethod
    async def stream_from_list(items: List[T]) -> AsyncIterator[T]:
        """Convert a list to an async stream.

        Args:
            items: List of items

        Yields:
            Items from the list
        """
        for item in items:
            yield item


class MultiprocessingExecutor:
    """Execute CPU-bound tasks across multiple processes."""

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize multiprocessing executor.

        Args:
            max_workers: Maximum number of worker processes (defaults to CPU count)
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

    async def execute(self, func: Callable[[T], R], items: List[T]) -> List[R]:
        """Execute function across multiple processes.

        Args:
            func: Function to execute (must be picklable)
            items: List of items to process

        Returns:
            List of results
        """
        loop = asyncio.get_event_loop()
        # Use executor.map instead of lambda to avoid pickling issues
        results = await loop.run_in_executor(None, lambda: list(self.executor.map(func, items)))
        return results

    async def map(self, func: Callable[[T], R], items: List[T]) -> List[R]:
        """Map function over items using multiprocessing.

        Args:
            func: Function to map (must be picklable)
            items: List of items to process

        Returns:
            List of results
        """
        loop = asyncio.get_event_loop()
        # Use executor.map directly
        results = await loop.run_in_executor(None, lambda: list(self.executor.map(func, items)))
        return results

    def shutdown(self, wait: bool = True):
        """Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


class GPUAccelerator:
    """Manage GPU resources for model inference."""

    def __init__(self):
        """Initialize GPU accelerator and detect GPU availability."""
        self._gpu_available = None
        self._device = None

    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available.

        Returns:
            True if GPU is available, False otherwise
        """
        if self._gpu_available is None:
            try:
                import torch

                self._gpu_available = torch.cuda.is_available()
            except ImportError:
                self._gpu_available = False
        return self._gpu_available

    @property
    def device(self) -> str:
        """Get the device to use for computation.

        Returns:
            'cuda' if GPU is available, 'cpu' otherwise
        """
        if self._device is None:
            self._device = "cuda" if self.gpu_available else "cpu"
        return self._device

    def get_device_name(self) -> str:
        """Get the name of the GPU device.

        Returns:
            GPU device name or 'CPU'
        """
        if self.gpu_available:
            try:
                import torch

                return torch.cuda.get_device_name(0)
            except Exception:
                return "Unknown GPU"
        return "CPU"

    def get_device_count(self) -> int:
        """Get the number of available GPU devices.

        Returns:
            Number of GPU devices
        """
        if self.gpu_available:
            try:
                import torch

                return torch.cuda.device_count()
            except Exception:
                return 0
        return 0

    def to_device(self, tensor):
        """Move tensor to the appropriate device.

        Args:
            tensor: Tensor to move

        Returns:
            Tensor on the appropriate device
        """
        return tensor.to(self.device)

    def clear_cache(self):
        """Clear GPU cache if available."""
        if self.gpu_available:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass


class ConnectionPool:
    """Manage connection pool for external services."""

    def __init__(self, max_connections: int = 10):
        """Initialize connection pool.

        Args:
            max_connections: Maximum number of connections in the pool
        """
        self.max_connections = max_connections
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._initialized = False
        self._create_connection_fn: Optional[Callable] = None

    def set_connection_factory(self, create_fn: Callable):
        """Set the function to create new connections.

        Args:
            create_fn: Async function that creates a new connection
        """
        self._create_connection_fn = create_fn

    async def initialize(self):
        """Initialize connection pool with connections."""
        if self._initialized:
            return

        if self._create_connection_fn is None:
            raise ValueError("Connection factory not set. Call set_connection_factory first.")

        for _ in range(self.max_connections):
            conn = await self._create_connection_fn()
            await self.pool.put(conn)

        self._initialized = True

    async def acquire(self):
        """Acquire connection from pool.

        Returns:
            Connection from the pool
        """
        if not self._initialized:
            await self.initialize()
        return await self.pool.get()

    async def release(self, conn):
        """Release connection back to pool.

        Args:
            conn: Connection to release
        """
        await self.pool.put(conn)

    async def close_all(self):
        """Close all connections in the pool."""
        while not self.pool.empty():
            conn = await self.pool.get()
            if hasattr(conn, "close"):
                await conn.close()
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_all()


class CacheManager:
    """Manage caching for frequently accessed data with LRU eviction."""

    def __init__(self, max_size: int = 1000):
        """Initialize cache manager.

        Args:
            max_size: Maximum number of items in cache
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()

    def cache_key(self, text: str) -> str:
        """Generate cache key from text.

        Args:
            text: Text to generate key for

        Returns:
            MD5 hash of the text
        """
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)

        self.cache[key] = value

    async def get_or_compute(self, key: str, compute_fn: Callable) -> Any:
        """Get from cache or compute if not found.

        Args:
            key: Cache key
            compute_fn: Async function to compute value if not cached

        Returns:
            Cached or computed value
        """
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value

        # Compute value
        result = await compute_fn()
        self.set(key, result)
        return result

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()

    def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of items in cache
        """
        return len(self.cache)

    def remove(self, key: str) -> bool:
        """Remove item from cache.

        Args:
            key: Cache key

        Returns:
            True if item was removed, False if not found
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
