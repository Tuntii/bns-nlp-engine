"""
Pipeline orchestrator for bns-nlp-engine.

This module provides the Pipeline class that orchestrates NLP processing
by coordinating multiple processing steps (preprocess, embed, search, classify).
"""

from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field

from bnsnlp.core.config import Config
from bnsnlp.core.exceptions import ProcessingError
from bnsnlp.core.registry import PluginRegistry


class PipelineStep(BaseModel):
    """Represents a single step in the processing pipeline.

    Attributes:
        module: The module category ('preprocess', 'embed', 'search', 'classify')
        plugin: The specific plugin name to use
        config: Configuration dictionary for this step
    """

    module: str = Field(..., description="Module category (preprocess, embed, search, classify)")
    plugin: str = Field(..., description="Plugin name to use for this step")
    config: Dict[str, Any] = Field(default_factory=dict, description="Step-specific configuration")


class Pipeline:
    """Orchestrates NLP processing pipeline.

    The Pipeline class manages the execution flow and coordinates multiple
    processing modules. It supports single item processing, batch processing,
    and streaming data processing.

    Example:
        >>> pipeline = Pipeline(config, registry)
        >>> pipeline.add_step('preprocess', 'turkish')
        >>> pipeline.add_step('embed', 'openai')
        >>> result = await pipeline.process("Merhaba dünya!")
    """

    def __init__(self, config: Config, registry: Optional[PluginRegistry] = None):
        """Initialize the pipeline.

        Args:
            config: Global configuration object
            registry: Plugin registry instance (creates new if None)
        """
        self.config = config
        self.registry = registry or PluginRegistry()
        self.steps: List[PipelineStep] = []

    def add_step(
        self, module: str, plugin: str, config: Optional[Dict[str, Any]] = None
    ) -> "Pipeline":
        """Add a processing step to the pipeline.

        Steps are executed in the order they are added. Each step receives
        the output of the previous step as input.

        Args:
            module: Module category ('preprocess', 'embed', 'search', 'classify')
            plugin: Specific plugin name to use
            config: Optional step-specific configuration

        Returns:
            Self for method chaining

        Raises:
            ProcessingError: If module or plugin is invalid
        """
        step = PipelineStep(module=module, plugin=plugin, config=config or {})
        self.steps.append(step)
        return self

    def clear_steps(self) -> None:
        """Clear all pipeline steps."""
        self.steps.clear()

    def get_steps(self) -> List[PipelineStep]:
        """Get a copy of all pipeline steps.

        Returns:
            List of pipeline steps
        """
        return self.steps.copy()

    async def process(self, input_data: Any) -> Any:
        """Process data through the pipeline.

        Executes all pipeline steps sequentially, passing the output of each
        step as input to the next step. Handles errors gracefully and provides
        context for debugging.

        Args:
            input_data: Input data to process (type depends on first step)

        Returns:
            Processed result (type depends on last step)

        Raises:
            ProcessingError: If any step fails during processing
        """
        if not self.steps:
            raise ProcessingError(
                "Pipeline has no steps configured",
                context={"input_type": type(input_data).__name__},
            )

        result = input_data

        for step_index, step in enumerate(self.steps):
            try:
                # Get plugin class from registry
                plugin_class = self.registry.get(step.module, step.plugin)

                # Instantiate plugin with merged configuration
                merged_config = self._merge_config(step)
                plugin = plugin_class(config=merged_config)

                # Process data through plugin
                result = await self._execute_step(plugin, result)

            except Exception as e:
                # Wrap and re-raise with context
                raise ProcessingError(
                    f"Error in pipeline step {step_index + 1} ({step.module}.{step.plugin}): {str(e)}",
                    context={
                        "step_index": step_index,
                        "module": step.module,
                        "plugin": step.plugin,
                        "input_type": type(input_data).__name__,
                        "error_type": type(e).__name__,
                        "original_error": str(e),
                    },
                ) from e

        return result

    def _merge_config(self, step: PipelineStep) -> Dict[str, Any]:
        """Merge global config with step-specific config.

        Step-specific configuration takes precedence over global configuration.

        Args:
            step: Pipeline step with configuration

        Returns:
            Merged configuration dictionary
        """
        # Get module-specific config from global config
        module_config = {}
        if step.module == "preprocess":
            module_config = self.config.preprocess.model_dump()
        elif step.module == "embed":
            module_config = self.config.embed.model_dump()
        elif step.module == "search":
            module_config = self.config.search.model_dump()

        # Merge with step config (step config takes precedence)
        merged = {**module_config, **step.config}
        return merged

    async def _execute_step(self, plugin: Any, data: Any) -> Any:
        """Execute a single plugin step.

        Handles both sync and async plugin methods.

        Args:
            plugin: Plugin instance
            data: Input data

        Returns:
            Processed data
        """
        # Check if plugin has process method
        if not hasattr(plugin, "process"):
            raise ProcessingError(
                f"Plugin {type(plugin).__name__} does not have a 'process' method",
                context={"plugin_type": type(plugin).__name__},
            )

        # Execute process method (async or sync)
        process_method = getattr(plugin, "process")

        # Check if it's a coroutine function
        import inspect

        if inspect.iscoroutinefunction(process_method):
            return await process_method(data)
        else:
            return process_method(data)

    async def process_batch(self, inputs: List[Any], batch_size: Optional[int] = None) -> List[Any]:
        """Process multiple items efficiently in batches.

        Processes items in configurable batch sizes to optimize performance
        and resource usage. Each batch is processed through all pipeline steps
        before moving to the next batch.

        Args:
            inputs: List of input items to process
            batch_size: Optional batch size (uses config default if None)

        Returns:
            List of processed results in the same order as inputs

        Raises:
            ProcessingError: If batch processing fails
        """
        if not inputs:
            return []

        if not self.steps:
            raise ProcessingError(
                "Pipeline has no steps configured", context={"input_count": len(inputs)}
            )

        # Determine batch size
        effective_batch_size = batch_size or self._get_default_batch_size()

        results = []

        # Process in batches
        for batch_start in range(0, len(inputs), effective_batch_size):
            batch_end = min(batch_start + effective_batch_size, len(inputs))
            batch = inputs[batch_start:batch_end]

            try:
                # Process each item in the batch
                batch_results = []
                for item in batch:
                    result = await self.process(item)
                    batch_results.append(result)

                results.extend(batch_results)

            except Exception as e:
                raise ProcessingError(
                    f"Error processing batch {batch_start}-{batch_end}: {str(e)}",
                    context={
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "batch_size": len(batch),
                        "total_items": len(inputs),
                        "error_type": type(e).__name__,
                    },
                ) from e

        return results

    def _get_default_batch_size(self) -> int:
        """Get default batch size from configuration.

        Returns:
            Default batch size (32 if not configured)
        """
        # Check if any step has batch_size configured
        for step in self.steps:
            if "batch_size" in step.config:
                return step.config["batch_size"]

        # Check module configs
        if hasattr(self.config.preprocess, "batch_size"):
            return self.config.preprocess.batch_size
        if hasattr(self.config.embed, "batch_size"):
            return self.config.embed.batch_size

        # Default batch size
        return 32

    async def process_stream(self, inputs: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Process streaming data without loading all into memory.

        Processes items as they arrive from an async iterator, yielding
        results immediately. This is memory-efficient for large datasets.

        Args:
            inputs: Async iterator of input items

        Yields:
            Processed results as they become available

        Raises:
            ProcessingError: If stream processing fails
        """
        if not self.steps:
            raise ProcessingError(
                "Pipeline has no steps configured", context={"processing_mode": "stream"}
            )

        item_count = 0

        try:
            async for item in inputs:
                item_count += 1
                try:
                    result = await self.process(item)
                    yield result
                except Exception as e:
                    raise ProcessingError(
                        f"Error processing stream item {item_count}: {str(e)}",
                        context={
                            "item_index": item_count,
                            "error_type": type(e).__name__,
                            "original_error": str(e),
                        },
                    ) from e
        except Exception as e:
            if not isinstance(e, ProcessingError):
                raise ProcessingError(
                    f"Error in stream processing: {str(e)}",
                    context={"items_processed": item_count, "error_type": type(e).__name__},
                ) from e
            raise
