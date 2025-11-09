"""
Qdrant vector database adapter for semantic search.
"""

import time
from typing import Any, Dict, List, Optional

from bnsnlp.core.exceptions import AdapterError
from bnsnlp.search.base import BaseSearch, SearchResponse, SearchResult

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, Filter, PointStruct, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


class QdrantSearch(BaseSearch):
    """Qdrant vector database adapter.

    This adapter provides integration with Qdrant for semantic search
    operations including document indexing and similarity search.

    Attributes:
        url: Qdrant server URL
        collection: Name of the collection to use
        client: Async Qdrant client instance
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Qdrant search adapter.

        Args:
            config: Configuration dictionary with keys:
                - url: Qdrant server URL (default: 'http://localhost:6333')
                - collection: Collection name (default: 'bnsnlp')
                - api_key: Optional API key for authentication
                - timeout: Request timeout in seconds (default: 30)

        Raises:
            AdapterError: If Qdrant client is not available
        """
        super().__init__(config)

        if not QDRANT_AVAILABLE:
            raise AdapterError(
                "Qdrant client not available. Install with: pip install qdrant-client",
                context={"adapter": "qdrant"},
            )

        self.url = config.get("url", "http://localhost:6333")
        self.collection = config.get("collection", "bnsnlp")
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30)

        # Initialize async client
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    async def _ensure_collection(self, vector_size: int) -> None:
        """Ensure collection exists, create if it doesn't.

        Args:
            vector_size: Dimensionality of the vectors

        Raises:
            AdapterError: If collection creation fails
        """
        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection not in collection_names:
                await self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as e:
            raise AdapterError(
                f"Failed to ensure collection exists: {str(e)}",
                context={"collection": self.collection, "vector_size": vector_size},
            )

    async def index(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Index documents with their embeddings in Qdrant.

        Args:
            texts: List of text documents to index
            embeddings: List of embedding vectors corresponding to texts
            ids: List of unique identifiers for the documents
            metadata: Optional list of metadata dictionaries for each document

        Raises:
            AdapterError: If indexing operation fails
        """
        if not texts or not embeddings or not ids:
            raise AdapterError(
                "texts, embeddings, and ids must be non-empty lists",
                context={
                    "texts_len": len(texts),
                    "embeddings_len": len(embeddings),
                    "ids_len": len(ids),
                },
            )

        if not (len(texts) == len(embeddings) == len(ids)):
            raise AdapterError(
                "texts, embeddings, and ids must have the same length",
                context={
                    "texts_len": len(texts),
                    "embeddings_len": len(embeddings),
                    "ids_len": len(ids),
                },
            )

        # Ensure collection exists
        vector_size = len(embeddings[0])
        await self._ensure_collection(vector_size)

        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(texts)
        elif len(metadata) != len(texts):
            raise AdapterError(
                "metadata must have the same length as texts",
                context={"metadata_len": len(metadata), "texts_len": len(texts)},
            )

        # Create points
        points = []
        for id_, text, embedding, meta in zip(ids, texts, embeddings, metadata):
            payload = {"text": text, **meta}
            points.append(
                PointStruct(
                    id=id_,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Index with retry logic
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                await self.client.upsert(
                    collection_name=self.collection,
                    points=points,
                )
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise AdapterError(
                        f"Failed to index documents after {max_retries} attempts: {str(e)}",
                        context={
                            "collection": self.collection,
                            "num_documents": len(texts),
                            "attempt": attempt + 1,
                        },
                    )
                # Wait before retry with exponential backoff
                await self._sleep(retry_delay * (2**attempt))

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Search for similar documents in Qdrant.

        Args:
            query_embedding: Embedding vector of the search query
            top_k: Maximum number of results to return
            filters: Optional filters to apply to the search (Qdrant filter format)

        Returns:
            SearchResponse containing search results and metadata

        Raises:
            AdapterError: If search operation fails
        """
        if not query_embedding:
            raise AdapterError(
                "query_embedding must be a non-empty list",
                context={"query_embedding_len": len(query_embedding)},
            )

        if top_k <= 0:
            raise AdapterError(
                "top_k must be a positive integer",
                context={"top_k": top_k},
            )

        start_time = time.time()

        # Convert filters to Qdrant format if provided
        qdrant_filter = None
        if filters:
            try:
                qdrant_filter = Filter(**filters)
            except Exception as e:
                raise AdapterError(
                    f"Invalid filter format: {str(e)}",
                    context={"filters": filters},
                )

        # Search with retry logic
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                results = await self.client.search(
                    collection_name=self.collection,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter=qdrant_filter,
                )

                query_time_ms = (time.time() - start_time) * 1000

                search_results = [
                    SearchResult(
                        id=str(result.id),
                        score=result.score,
                        text=result.payload.get("text", ""),
                        metadata={k: v for k, v in result.payload.items() if k != "text"},
                    )
                    for result in results
                ]

                return SearchResponse(
                    results=search_results,
                    query_time_ms=query_time_ms,
                    metadata={
                        "collection": self.collection,
                        "top_k": top_k,
                        "num_results": len(search_results),
                    },
                )

            except Exception as e:
                if attempt == max_retries - 1:
                    raise AdapterError(
                        f"Failed to search after {max_retries} attempts: {str(e)}",
                        context={
                            "collection": self.collection,
                            "top_k": top_k,
                            "attempt": attempt + 1,
                        },
                    )
                # Wait before retry with exponential backoff
                await self._sleep(retry_delay * (2**attempt))

        # This should never be reached due to the raise in the loop
        raise AdapterError("Unexpected error in search operation")

    async def _sleep(self, seconds: float) -> None:
        """Sleep for the specified number of seconds.

        Args:
            seconds: Number of seconds to sleep
        """
        import asyncio

        await asyncio.sleep(seconds)
