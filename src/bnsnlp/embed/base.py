"""
Base embedder interface and models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class EmbedResult(BaseModel):
    """Result of text embedding operation.

    Attributes:
        embeddings: List of embedding vectors (each vector is a list of floats)
        model: Name of the model used for embedding
        dimensions: Dimensionality of the embedding vectors
        metadata: Additional metadata about the embedding operation
    """

    embeddings: List[List[float]]
    model: str
    dimensions: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "model": "text-embedding-3-small",
                "dimensions": 3,
                "metadata": {"batch_size": 2},
            }
        }


class BaseEmbedder(ABC):
    """Base interface for text embedders.

    All embedder implementations must inherit from this class
    and implement the embed() method.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize embedder with configuration.

        Args:
            config: Configuration dictionary for the embedder
        """
        self.config = config

    @abstractmethod
    async def embed(self, texts: Union[str, List[str]]) -> EmbedResult:
        """Generate embeddings for text(s).

        Args:
            texts: Single text string or list of text strings to embed

        Returns:
            EmbedResult containing embeddings and metadata

        Raises:
            AdapterError: If embedding generation fails
        """
        ...
