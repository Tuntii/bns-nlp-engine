"""Tests for performance utilities."""

import asyncio
from typing import List

import pytest

from bnsnlp.utils.performance import (
    BatchProcessor,
    CacheManager,
    ConnectionPool,
    GPUAccelerator,
    MultiprocessingExecutor,
    StreamProcessor,
)


# Module-level functions for multiprocessing tests (must be picklable)
def _square(x: int) -> int:
    """Square a number."""
    return x * x


def _double(x: int) -> int:
    """Double a number."""
    return x * 2


class TestBatchProcessor:
    """Tests for BatchProcessor."""

    @pytest.mark.asyncio
    async def test_process_batches(self):
        """Test batch processing."""
        processor = BatchProcessor(batch_size=3)

        async def process_fn(batch: List[int]) -> List[int]:
            return [x * 2 for x in batch]

        items = [1, 2, 3, 4, 5, 6, 7]
        results = await processor.process_batches(items, process_fn)

        assert results == [2, 4, 6, 8, 10, 12, 14]

    @pytest.mark.asyncio
    async def test_process_batches_single_result(self):
        """Test batch processing with single result per batch."""
        processor = BatchProcessor(batch_size=2)

        async def process_fn(batch: List[int]) -> int:
            return sum(batch)

        items = [1, 2, 3, 4]
        results = await processor.process_batches(items, process_fn)

        assert results == [3, 7]

    def test_create_batches(self):
        """Test batch creation."""
        processor = BatchProcessor(batch_size=3)
        items = [1, 2, 3, 4, 5, 6, 7]
        batches = processor.create_batches(items)

        assert len(batches) == 3
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5, 6]
        assert batches[2] == [7]


class TestStreamProcessor:
    """Tests for StreamProcessor."""

    @pytest.mark.asyncio
    async def test_process_stream(self):
        """Test stream processing."""

        async def input_stream():
            for i in range(5):
                yield i

        async def process_fn(x: int) -> int:
            return x * 2

        results = []
        async for result in StreamProcessor.process_stream(input_stream(), process_fn):
            results.append(result)

        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_batch_stream(self):
        """Test batching from stream."""

        async def input_stream():
            for i in range(7):
                yield i

        batches = []
        async for batch in StreamProcessor.batch_stream(input_stream(), batch_size=3):
            batches.append(batch)

        assert len(batches) == 3
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6]

    @pytest.mark.asyncio
    async def test_stream_from_list(self):
        """Test converting list to stream."""
        items = [1, 2, 3, 4, 5]
        results = []

        async for item in StreamProcessor.stream_from_list(items):
            results.append(item)

        assert results == items


class TestMultiprocessingExecutor:
    """Tests for MultiprocessingExecutor."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test multiprocessing execution."""
        executor = MultiprocessingExecutor(max_workers=2)
        items = [1, 2, 3, 4, 5]
        results = await executor.execute(_square, items)

        assert results == [1, 4, 9, 16, 25]
        executor.shutdown()

    @pytest.mark.asyncio
    async def test_map(self):
        """Test multiprocessing map."""
        executor = MultiprocessingExecutor(max_workers=2)
        items = [1, 2, 3, 4, 5]
        results = await executor.map(_double, items)

        assert results == [2, 4, 6, 8, 10]
        executor.shutdown()

    def test_context_manager(self):
        """Test executor as context manager."""
        with MultiprocessingExecutor(max_workers=2) as executor:
            assert executor is not None


class TestGPUAccelerator:
    """Tests for GPUAccelerator."""

    def test_gpu_detection(self):
        """Test GPU availability detection."""
        accelerator = GPUAccelerator()

        # Should not raise an error
        gpu_available = accelerator.gpu_available
        assert isinstance(gpu_available, bool)

    def test_device_property(self):
        """Test device property."""
        accelerator = GPUAccelerator()
        device = accelerator.device

        assert device in ["cuda", "cpu"]

    def test_device_name(self):
        """Test getting device name."""
        accelerator = GPUAccelerator()
        device_name = accelerator.get_device_name()

        assert isinstance(device_name, str)
        assert len(device_name) > 0

    def test_device_count(self):
        """Test getting device count."""
        accelerator = GPUAccelerator()
        device_count = accelerator.get_device_count()

        assert isinstance(device_count, int)
        assert device_count >= 0

    def test_clear_cache(self):
        """Test clearing GPU cache."""
        accelerator = GPUAccelerator()

        # Should not raise an error
        accelerator.clear_cache()


class TestConnectionPool:
    """Tests for ConnectionPool."""

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test acquiring and releasing connections."""
        pool = ConnectionPool(max_connections=2)

        # Mock connection factory
        connection_counter = [0]

        async def create_connection():
            connection_counter[0] += 1
            return f"connection_{connection_counter[0]}"

        pool.set_connection_factory(create_connection)

        # Acquire connection
        conn1 = await pool.acquire()
        assert conn1 in ["connection_1", "connection_2"]

        # Release connection
        await pool.release(conn1)

        # Acquire again (should get a connection from pool)
        conn2 = await pool.acquire()
        assert conn2 in ["connection_1", "connection_2"]

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_multiple_connections(self):
        """Test multiple connections."""
        pool = ConnectionPool(max_connections=3)

        connection_counter = [0]

        async def create_connection():
            connection_counter[0] += 1
            return f"connection_{connection_counter[0]}"

        pool.set_connection_factory(create_connection)

        # Acquire multiple connections
        conn1 = await pool.acquire()
        conn2 = await pool.acquire()
        conn3 = await pool.acquire()

        assert conn1 == "connection_1"
        assert conn2 == "connection_2"
        assert conn3 == "connection_3"

        # Release all
        await pool.release(conn1)
        await pool.release(conn2)
        await pool.release(conn3)

        await pool.close_all()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test pool as context manager."""

        async def create_connection():
            return "connection"

        pool = ConnectionPool(max_connections=2)
        pool.set_connection_factory(create_connection)

        async with pool:
            conn = await pool.acquire()
            assert conn == "connection"

    @pytest.mark.asyncio
    async def test_no_factory_error(self):
        """Test error when factory not set."""
        pool = ConnectionPool(max_connections=2)

        with pytest.raises(ValueError, match="Connection factory not set"):
            await pool.acquire()


class TestCacheManager:
    """Tests for CacheManager."""

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = CacheManager()
        key1 = cache.cache_key("test")
        key2 = cache.cache_key("test")
        key3 = cache.cache_key("different")

        assert key1 == key2
        assert key1 != key3

    def test_get_set(self):
        """Test getting and setting cache values."""
        cache = CacheManager()

        # Set value
        cache.set("key1", "value1")

        # Get value
        value = cache.get("key1")
        assert value == "value1"

        # Get non-existent key
        value = cache.get("key2")
        assert value is None

    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = CacheManager(max_size=3)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Add one more (should evict key1)
        cache.set("key4", "value4")

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_access_order(self):
        """Test LRU access order."""
        cache = CacheManager(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 (moves it to end)
        cache.get("key1")

        # Add key4 (should evict key2, not key1)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_get_or_compute(self):
        """Test get or compute pattern."""
        cache = CacheManager()
        compute_count = [0]

        async def compute_fn():
            compute_count[0] += 1
            return "computed_value"

        # First call should compute
        value1 = await cache.get_or_compute("key1", compute_fn)
        assert value1 == "computed_value"
        assert compute_count[0] == 1

        # Second call should use cache
        value2 = await cache.get_or_compute("key1", compute_fn)
        assert value2 == "computed_value"
        assert compute_count[0] == 1  # Not incremented

    def test_clear(self):
        """Test clearing cache."""
        cache = CacheManager()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.size() == 2

        cache.clear()

        assert cache.size() == 0
        assert cache.get("key1") is None

    def test_size(self):
        """Test cache size."""
        cache = CacheManager()

        assert cache.size() == 0

        cache.set("key1", "value1")
        assert cache.size() == 1

        cache.set("key2", "value2")
        assert cache.size() == 2

    def test_remove(self):
        """Test removing items."""
        cache = CacheManager()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Remove existing key
        removed = cache.remove("key1")
        assert removed is True
        assert cache.get("key1") is None
        assert cache.size() == 1

        # Remove non-existent key
        removed = cache.remove("key3")
        assert removed is False
