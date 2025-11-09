"""
HuggingFace embedding adapter with GPU support.
"""

import asyncio
from typing import Any, Dict, List, Union

from bnsnlp.core.exceptions import AdapterError
from bnsnlp.embed.base import BaseEmbedder, EmbedResult


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace local model adapter with GPU acceleration support.

    This adapter uses sentence-transformers models to generate embeddings locally.
    It supports GPU acceleration when available and implements async processing
    using thread pools to avoid blocking the event loop.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize HuggingFace embedder.

        Args:
            config: Configuration dictionary containing:
                - model: Model name (default: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                - use_gpu: Whether to use GPU if available (default: True)
                - batch_size: Batch size for processing (default: 32)
                - normalize_embeddings: Whether to normalize embeddings (default: True)
                - device: Specific device to use (optional, overrides use_gpu)

        Raises:
            AdapterError: If required packages are not installed
        """
        super().__init__(config)

        # Import dependencies here to make them optional
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise AdapterError(
                "sentence-transformers package is not installed. "
                "Install it with: pip install sentence-transformers",
                context={"package": "sentence-transformers"},
            ) from e

        try:
            import torch
        except ImportError as e:
            raise AdapterError(
                "torch package is not installed. Install it with: pip install torch",
                context={"package": "torch"},
            ) from e

        # Get configuration
        self.model_name = config.get(
            "model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.use_gpu = config.get("use_gpu", True)
        self.batch_size = config.get("batch_size", 32)
        self.normalize_embeddings = config.get("normalize_embeddings", True)

        # Determine device
        if "device" in config:
            self.device = config["device"]
        else:
            self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"

        # Store torch reference for GPU checks
        self._torch = torch

        # Initialize model
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            raise AdapterError(
                f"Failed to load HuggingFace model '{self.model_name}': {str(e)}",
                context={
                    "model": self.model_name,
                    "device": self.device,
                    "error_type": type(e).__name__,
                },
            ) from e

        # Get embedding dimensions
        self.dimensions = self.model.get_sentence_embedding_dimension()

    async def embed(self, texts: Union[str, List[str]]) -> EmbedResult:
        """Generate embeddings for text(s).

        Args:
            texts: Single text string or list of text strings to embed

        Returns:
            EmbedResult containing embeddings and metadata

        Raises:
            AdapterError: If embedding generation fails
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise AdapterError(
                "No texts provided for embedding",
                context={"texts_count": 0},
            )

        # Run encoding in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()

        try:
            embeddings_array = await loop.run_in_executor(None, self._encode_texts, texts)

            # Convert to list of lists
            embeddings = embeddings_array.tolist()

            return EmbedResult(
                embeddings=embeddings,
                model=self.model_name,
                dimensions=self.dimensions,
                metadata={
                    "batch_size": self.batch_size,
                    "total_texts": len(texts),
                    "provider": "huggingface",
                    "device": self.device,
                    "gpu_available": self._torch.cuda.is_available(),
                    "normalize_embeddings": self.normalize_embeddings,
                },
            )

        except Exception as e:
            raise AdapterError(
                f"Failed to generate embeddings: {str(e)}",
                context={
                    "model": self.model_name,
                    "texts_count": len(texts),
                    "device": self.device,
                    "error_type": type(e).__name__,
                },
            ) from e

    def _encode_texts(self, texts: List[str]):
        """Encode texts using the model (runs in thread pool).

        Args:
            texts: List of texts to encode

        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
