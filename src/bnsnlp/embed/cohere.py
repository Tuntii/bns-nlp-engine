"""
Cohere embedding adapter.
"""

import asyncio
from typing import Any, Dict, List, Union

from bnsnlp.core.exceptions import AdapterError
from bnsnlp.embed.base import BaseEmbedder, EmbedResult


class CohereEmbedder(BaseEmbedder):
    """Cohere embedding adapter with batch support and error handling.

    This adapter uses the Cohere API to generate embeddings for text.
    It supports batch processing and implements retry logic for transient failures.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Cohere embedder.

        Args:
            config: Configuration dictionary containing:
                - api_key: Cohere API key (required)
                - model: Model name (default: "embed-multilingual-v3.0")
                - batch_size: Batch size for processing (default: 96)
                - max_retries: Maximum number of retries (default: 3)
                - retry_delay: Initial retry delay in seconds (default: 1.0)
                - input_type: Input type for embeddings (default: "search_document")

        Raises:
            AdapterError: If Cohere package is not installed or API key is missing
        """
        super().__init__(config)

        # Import Cohere here to make it an optional dependency
        try:
            from cohere import AsyncClient
        except ImportError as e:
            raise AdapterError(
                "Cohere package is not installed. Install it with: pip install cohere",
                context={"package": "cohere"},
            ) from e

        # Get configuration
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise AdapterError(
                "Cohere API key is required. Set it in config or BNSNLP_EMBED__API_KEY environment variable.",
                context={"config_key": "api_key"},
            )

        self.model = config.get("model", "embed-multilingual-v3.0")
        self.batch_size = config.get("batch_size", 96)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.input_type = config.get("input_type", "search_document")

        # Initialize async client
        self.client = AsyncClient(api_key=self.api_key)

    async def embed(self, texts: Union[str, List[str]]) -> EmbedResult:
        """Generate embeddings for text(s).

        Args:
            texts: Single text string or list of text strings to embed

        Returns:
            EmbedResult containing embeddings and metadata

        Raises:
            AdapterError: If embedding generation fails after retries
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise AdapterError(
                "No texts provided for embedding",
                context={"texts_count": 0},
            )

        # Process in batches
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return EmbedResult(
            embeddings=all_embeddings,
            model=self.model,
            dimensions=len(all_embeddings[0]) if all_embeddings else 0,
            metadata={
                "batch_size": self.batch_size,
                "total_texts": len(texts),
                "provider": "cohere",
                "input_type": self.input_type,
            },
        )

    async def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """Embed a batch of texts with retry logic.

        Args:
            batch: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            AdapterError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return await self._embed_batch(batch)
            except Exception as e:
                last_error = e

                # Check if we should retry
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                    continue

        # All retries failed
        raise AdapterError(
            f"Failed to generate embeddings after {self.max_retries} attempts: {str(last_error)}",
            context={
                "batch_size": len(batch),
                "max_retries": self.max_retries,
                "error": str(last_error),
            },
        ) from last_error

    async def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """Embed a batch of texts using Cohere API.

        Args:
            batch: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If API call fails
        """
        try:
            response = await self.client.embed(
                texts=batch, model=self.model, input_type=self.input_type
            )

            # Extract embeddings
            embeddings = response.embeddings

            return embeddings

        except Exception as e:
            # Re-raise with more context
            raise AdapterError(
                f"Cohere API error: {str(e)}",
                context={
                    "model": self.model,
                    "batch_size": len(batch),
                    "error_type": type(e).__name__,
                },
            ) from e
