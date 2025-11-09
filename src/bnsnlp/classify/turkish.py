"""
Turkish-specific classifier implementation.

This module provides intent classification and named entity recognition
for Turkish text using HuggingFace transformers.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Union

from bnsnlp.classify.base import BaseClassifier, ClassifyResult, Entity
from bnsnlp.core.exceptions import ProcessingError


class TurkishClassifier(BaseClassifier):
    """Turkish intent classifier and entity recognizer.

    This classifier uses HuggingFace transformers for intent classification
    and named entity recognition on Turkish text. It supports GPU acceleration
    and async processing via thread pools.

    Configuration options:
        - intent_model: HuggingFace model for intent classification
        - entity_model: HuggingFace model for NER
        - use_gpu: Whether to use GPU if available (default: True)
        - batch_size: Batch size for processing (default: 8)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Turkish classifier.

        Args:
            config: Configuration dictionary with model settings
        """
        super().__init__(config)

        self.intent_model_name = config.get(
            "intent_model", "savasy/bert-base-turkish-sentiment-cased"
        )
        self.entity_model_name = config.get("entity_model", "savasy/bert-turkish-ner-cased")
        self.use_gpu = config.get("use_gpu", True)
        self.batch_size = config.get("batch_size", 8)

        # Lazy loading - models will be loaded on first use
        self._intent_pipeline = None
        self._entity_pipeline = None
        self._device = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _initialize_models(self):
        """Initialize the models (lazy loading)."""
        if self._intent_pipeline is not None:
            return

        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise ProcessingError(
                "transformers and torch are required for TurkishClassifier. "
                "Install with: pip install bns-nlp-engine[huggingface]",
                context={"error": str(e)},
            )

        # Determine device
        if self.use_gpu and torch.cuda.is_available():
            self._device = 0  # Use first GPU
        else:
            self._device = -1  # Use CPU

        # Load intent classification pipeline
        try:
            self._intent_pipeline = pipeline(
                "text-classification", model=self.intent_model_name, device=self._device
            )
        except Exception as e:
            raise ProcessingError(
                f"Failed to load intent model: {self.intent_model_name}",
                context={"error": str(e), "model": self.intent_model_name},
            )

        # Load entity recognition pipeline
        try:
            self._entity_pipeline = pipeline(
                "ner",
                model=self.entity_model_name,
                device=self._device,
                aggregation_strategy="simple",
            )
        except Exception as e:
            raise ProcessingError(
                f"Failed to load entity model: {self.entity_model_name}",
                context={"error": str(e), "model": self.entity_model_name},
            )

    async def classify(
        self, text: Union[str, List[str]]
    ) -> Union[ClassifyResult, List[ClassifyResult]]:
        """Classify text and extract entities.

        Args:
            text: Input text or list of texts to classify

        Returns:
            ClassifyResult for single text, or list of ClassifyResult for batch

        Raises:
            ProcessingError: If classification fails
        """
        # Initialize models if needed
        self._initialize_models()

        # Handle single text vs batch
        if isinstance(text, str):
            return await self._classify_single(text)
        else:
            return await self._classify_batch(text)

    async def _classify_single(self, text: str) -> ClassifyResult:
        """Classify a single text.

        Args:
            text: Input text to classify

        Returns:
            ClassifyResult with intent and entities
        """
        try:
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            # Get intent
            intent_result = await loop.run_in_executor(
                self._executor, lambda: self._intent_pipeline(text)[0]
            )

            # Get entities
            entity_results = await loop.run_in_executor(
                self._executor, lambda: self._entity_pipeline(text)
            )

            # Convert to Entity objects
            entities = [
                Entity(
                    text=e["word"],
                    type=e["entity_group"],
                    start=e["start"],
                    end=e["end"],
                    confidence=e["score"],
                )
                for e in entity_results
            ]

            return ClassifyResult(
                intent=intent_result["label"],
                intent_confidence=intent_result["score"],
                entities=entities,
                metadata={
                    "intent_model": self.intent_model_name,
                    "entity_model": self.entity_model_name,
                    "device": "gpu" if self._device >= 0 else "cpu",
                },
            )

        except Exception as e:
            raise ProcessingError(
                f"Classification failed for text: {text[:50]}...",
                context={"error": str(e), "text_length": len(text)},
            )

    async def _classify_batch(self, texts: List[str]) -> List[ClassifyResult]:
        """Classify a batch of texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassifyResult objects
        """
        try:
            # Run inference in thread pool
            loop = asyncio.get_event_loop()

            # Get intents for all texts
            intent_results = await loop.run_in_executor(
                self._executor, lambda: self._intent_pipeline(texts)
            )

            # Get entities for all texts
            entity_results = await loop.run_in_executor(
                self._executor, lambda: [self._entity_pipeline(text) for text in texts]
            )

            # Build results
            results = []
            for i, text in enumerate(texts):
                entities = [
                    Entity(
                        text=e["word"],
                        type=e["entity_group"],
                        start=e["start"],
                        end=e["end"],
                        confidence=e["score"],
                    )
                    for e in entity_results[i]
                ]

                results.append(
                    ClassifyResult(
                        intent=intent_results[i]["label"],
                        intent_confidence=intent_results[i]["score"],
                        entities=entities,
                        metadata={
                            "intent_model": self.intent_model_name,
                            "entity_model": self.entity_model_name,
                            "device": "gpu" if self._device >= 0 else "cpu",
                        },
                    )
                )

            return results

        except Exception as e:
            raise ProcessingError(
                f"Batch classification failed for {len(texts)} texts",
                context={"error": str(e), "batch_size": len(texts)},
            )

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
