"""
Base search interface and models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Single search result.

    Attributes:
        id: Unique identifier of the document
        score: Similarity score (higher is more similar)
        text: Text content of the document
        metadata: Additional metadata associated with the document
    """

    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "id": "doc_123",
                "score": 0.95,
                "text": "Merhaba dünya",
                "metadata": {"source": "example.txt", "timestamp": "2024-01-15"},
            }
        }


class SearchResponse(BaseModel):
    """Response from a search operation.

    Attributes:
        results: List of search results ordered by relevance
        query_time_ms: Time taken to execute the query in milliseconds
        metadata: Additional metadata about the search operation
    """

    results: List[SearchResult]
    query_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "doc_123",
                        "score": 0.95,
                        "text": "Merhaba dünya",
                        "metadata": {},
                    }
                ],
                "query_time_ms": 15.5,
                "metadata": {"total_documents": 1000},
            }
        }


class BaseSearch(ABC):
    """Base interface for search backends.

    All search implementations must inherit from this class
    and implement the index() and search() methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize search backend with configuration.

        Args:
            config: Configuration dictionary for the search backend
        """
        self.config = config

    @abstractmethod
    async def index(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Index documents with their embeddings.

        Args:
            texts: List of text documents to index
            embeddings: List of embedding vectors corresponding to texts
            ids: List of unique identifiers for the documents
            metadata: Optional list of metadata dictionaries for each document

        Raises:
            AdapterError: If indexing operation fails
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Search for similar documents.

        Args:
            query_embedding: Embedding vector of the search query
            top_k: Maximum number of results to return
            filters: Optional filters to apply to the search

        Returns:
            SearchResponse containing search results and metadata

        Raises:
            AdapterError: If search operation fails
        """
        ...
