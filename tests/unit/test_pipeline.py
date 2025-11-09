"""
Unit tests for Pipeline orchestrator.
"""

import pytest
from typing import Any, Dict, AsyncIterator

from bnsnlp.core import Pipeline, PipelineStep, Config, PluginRegistry, ProcessingError


# Mock plugin classes for testing
class MockPreprocessor:
    """Mock preprocessor plugin."""
    
    name = "mock_preprocessor"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin."""
        pass
    
    async def process(self, text: str) -> str:
        """Mock preprocessing - converts to lowercase."""
        return text.lower()


class MockEmbedder:
    """Mock embedder plugin."""
    
    name = "mock_embedder"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin."""
        pass
    
    async def process(self, text: str) -> list:
        """Mock embedding - returns list of character codes."""
        return [ord(c) for c in text]


class MockFailingPlugin:
    """Mock plugin that always fails."""
    
    name = "mock_failing"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin."""
        pass
    
    async def process(self, data: Any) -> Any:
        """Always raises an error."""
        raise ValueError("Mock plugin failure")


class MockSyncPlugin:
    """Mock plugin with synchronous process method."""
    
    name = "mock_sync"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin."""
        pass
    
    def process(self, text: str) -> str:
        """Synchronous process method."""
        return text.upper()


@pytest.fixture
def mock_registry():
    """Create a mock plugin registry with test plugins."""
    registry = PluginRegistry()
    registry.register('preprocess', 'mock', MockPreprocessor)
    registry.register('embed', 'mock', MockEmbedder)
    registry.register('classify', 'failing', MockFailingPlugin)
    registry.register('classify', 'sync', MockSyncPlugin)
    return registry


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def pipeline(config, mock_registry):
    """Create a pipeline with mock registry."""
    return Pipeline(config, mock_registry)


class TestPipelineStepManagement:
    """Tests for pipeline step management."""
    
    def test_add_step(self, pipeline):
        """Test adding a step to the pipeline."""
        result = pipeline.add_step('preprocess', 'mock')
        
        assert result is pipeline  # Should return self for chaining
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].module == 'preprocess'
        assert pipeline.steps[0].plugin == 'mock'
    
    def test_add_step_with_config(self, pipeline):
        """Test adding a step with custom configuration."""
        custom_config = {'batch_size': 64, 'custom_param': 'value'}
        pipeline.add_step('embed', 'mock', config=custom_config)
        
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].config == custom_config
    
    def test_add_multiple_steps(self, pipeline):
        """Test adding multiple steps."""
        pipeline.add_step('preprocess', 'mock')
        pipeline.add_step('embed', 'mock')
        
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].module == 'preprocess'
        assert pipeline.steps[1].module == 'embed'
    
    def test_method_chaining(self, pipeline):
        """Test method chaining for adding steps."""
        result = (pipeline
                  .add_step('preprocess', 'mock')
                  .add_step('embed', 'mock'))
        
        assert result is pipeline
        assert len(pipeline.steps) == 2
    
    def test_clear_steps(self, pipeline):
        """Test clearing all pipeline steps."""
        pipeline.add_step('preprocess', 'mock')
        pipeline.add_step('embed', 'mock')
        
        pipeline.clear_steps()
        
        assert len(pipeline.steps) == 0
    
    def test_get_steps(self, pipeline):
        """Test getting a copy of pipeline steps."""
        pipeline.add_step('preprocess', 'mock')
        
        steps = pipeline.get_steps()
        
        assert len(steps) == 1
        assert steps[0].module == 'preprocess'
        
        # Verify it's a copy
        steps.clear()
        assert len(pipeline.steps) == 1


class TestSingleItemProcessing:
    """Tests for single item processing."""
    
    @pytest.mark.asyncio
    async def test_process_single_step(self, pipeline):
        """Test processing with a single step."""
        pipeline.add_step('preprocess', 'mock')
        
        result = await pipeline.process("HELLO WORLD")
        
        assert result == "hello world"
    
    @pytest.mark.asyncio
    async def test_process_multiple_steps(self, pipeline):
        """Test processing with multiple steps."""
        pipeline.add_step('preprocess', 'mock')  # lowercase
        pipeline.add_step('embed', 'mock')  # convert to char codes
        
        result = await pipeline.process("ABC")
        
        # Should be lowercase then converted to char codes
        assert result == [ord('a'), ord('b'), ord('c')]
    
    @pytest.mark.asyncio
    async def test_process_with_sync_plugin(self, pipeline):
        """Test processing with synchronous plugin."""
        pipeline.add_step('classify', 'sync')
        
        result = await pipeline.process("hello")
        
        assert result == "HELLO"
    
    @pytest.mark.asyncio
    async def test_process_empty_pipeline(self, pipeline):
        """Test processing with no steps raises error."""
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.process("test")
        
        assert "no steps configured" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_process_error_handling(self, pipeline):
        """Test error handling during processing."""
        pipeline.add_step('classify', 'failing')
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.process("test")
        
        error = exc_info.value
        assert "step 1" in str(error).lower()
        assert error.context['module'] == 'classify'
        assert error.context['plugin'] == 'failing'
    
    @pytest.mark.asyncio
    async def test_process_passes_data_between_steps(self, pipeline):
        """Test that data is correctly passed between steps."""
        pipeline.add_step('preprocess', 'mock')
        pipeline.add_step('classify', 'sync')
        
        result = await pipeline.process("hello")
        
        # First step: lowercase -> "hello"
        # Second step: uppercase -> "HELLO"
        assert result == "HELLO"


class TestBatchProcessing:
    """Tests for batch processing."""
    
    @pytest.mark.asyncio
    async def test_process_batch_empty_list(self, pipeline):
        """Test processing empty batch."""
        pipeline.add_step('preprocess', 'mock')
        
        result = await pipeline.process_batch([])
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_process_batch_single_item(self, pipeline):
        """Test processing batch with single item."""
        pipeline.add_step('preprocess', 'mock')
        
        result = await pipeline.process_batch(["HELLO"])
        
        assert result == ["hello"]
    
    @pytest.mark.asyncio
    async def test_process_batch_multiple_items(self, pipeline):
        """Test processing batch with multiple items."""
        pipeline.add_step('preprocess', 'mock')
        
        inputs = ["HELLO", "WORLD", "TEST"]
        result = await pipeline.process_batch(inputs)
        
        assert result == ["hello", "world", "test"]
    
    @pytest.mark.asyncio
    async def test_process_batch_with_custom_batch_size(self, pipeline):
        """Test processing with custom batch size."""
        pipeline.add_step('preprocess', 'mock')
        
        inputs = ["A", "B", "C", "D", "E"]
        result = await pipeline.process_batch(inputs, batch_size=2)
        
        assert len(result) == 5
        assert result == ["a", "b", "c", "d", "e"]
    
    @pytest.mark.asyncio
    async def test_process_batch_preserves_order(self, pipeline):
        """Test that batch processing preserves input order."""
        pipeline.add_step('embed', 'mock')
        
        inputs = ["A", "B", "C"]
        result = await pipeline.process_batch(inputs)
        
        assert result[0] == [ord('A')]
        assert result[1] == [ord('B')]
        assert result[2] == [ord('C')]
    
    @pytest.mark.asyncio
    async def test_process_batch_empty_pipeline(self, pipeline):
        """Test batch processing with no steps raises error."""
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.process_batch(["test"])
        
        assert "no steps configured" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_process_batch_error_handling(self, pipeline):
        """Test error handling in batch processing."""
        pipeline.add_step('classify', 'failing')
        
        with pytest.raises(ProcessingError) as exc_info:
            await pipeline.process_batch(["test1", "test2"])
        
        error = exc_info.value
        assert "batch" in str(error).lower()


class TestStreamProcessing:
    """Tests for streaming data processing."""
    
    async def async_generator(self, items):
        """Helper to create async generator."""
        for item in items:
            yield item
    
    @pytest.mark.asyncio
    async def test_process_stream_single_item(self, pipeline):
        """Test streaming with single item."""
        pipeline.add_step('preprocess', 'mock')
        
        stream = self.async_generator(["HELLO"])
        results = []
        
        async for result in pipeline.process_stream(stream):
            results.append(result)
        
        assert results == ["hello"]
    
    @pytest.mark.asyncio
    async def test_process_stream_multiple_items(self, pipeline):
        """Test streaming with multiple items."""
        pipeline.add_step('preprocess', 'mock')
        
        stream = self.async_generator(["HELLO", "WORLD", "TEST"])
        results = []
        
        async for result in pipeline.process_stream(stream):
            results.append(result)
        
        assert results == ["hello", "world", "test"]
    
    @pytest.mark.asyncio
    async def test_process_stream_empty(self, pipeline):
        """Test streaming with empty iterator."""
        pipeline.add_step('preprocess', 'mock')
        
        stream = self.async_generator([])
        results = []
        
        async for result in pipeline.process_stream(stream):
            results.append(result)
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_process_stream_empty_pipeline(self, pipeline):
        """Test streaming with no steps raises error."""
        stream = self.async_generator(["test"])
        
        with pytest.raises(ProcessingError) as exc_info:
            async for _ in pipeline.process_stream(stream):
                pass
        
        assert "no steps configured" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_process_stream_error_handling(self, pipeline):
        """Test error handling in stream processing."""
        pipeline.add_step('classify', 'failing')
        
        stream = self.async_generator(["test1", "test2"])
        
        with pytest.raises(ProcessingError) as exc_info:
            async for _ in pipeline.process_stream(stream):
                pass
        
        error = exc_info.value
        assert "stream item" in str(error).lower()
    
    @pytest.mark.asyncio
    async def test_process_stream_yields_immediately(self, pipeline):
        """Test that stream processing yields results immediately."""
        pipeline.add_step('preprocess', 'mock')
        
        stream = self.async_generator(["A", "B", "C"])
        results = []
        
        async for result in pipeline.process_stream(stream):
            results.append(result)
            # Verify we get results one at a time
            if len(results) == 1:
                assert results == ["a"]
            elif len(results) == 2:
                assert results == ["a", "b"]
        
        assert results == ["a", "b", "c"]


class TestConfigurationMerging:
    """Tests for configuration merging."""
    
    @pytest.mark.asyncio
    async def test_merge_config_uses_step_config(self, pipeline):
        """Test that step config is used when provided."""
        # Add step with custom config
        pipeline.add_step('preprocess', 'mock', config={'custom': 'value'})
        
        step = pipeline.steps[0]
        merged = pipeline._merge_config(step)
        
        assert 'custom' in merged
        assert merged['custom'] == 'value'
    
    @pytest.mark.asyncio
    async def test_merge_config_includes_global_config(self, config, mock_registry):
        """Test that global config is included."""
        # Modify global config
        config.preprocess.batch_size = 64
        
        pipeline = Pipeline(config, mock_registry)
        pipeline.add_step('preprocess', 'mock')
        
        step = pipeline.steps[0]
        merged = pipeline._merge_config(step)
        
        assert merged['batch_size'] == 64
    
    @pytest.mark.asyncio
    async def test_merge_config_step_overrides_global(self, config, mock_registry):
        """Test that step config overrides global config."""
        config.preprocess.batch_size = 64
        
        pipeline = Pipeline(config, mock_registry)
        pipeline.add_step('preprocess', 'mock', config={'batch_size': 128})
        
        step = pipeline.steps[0]
        merged = pipeline._merge_config(step)
        
        assert merged['batch_size'] == 128
