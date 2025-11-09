"""
Pinecone vector database adapter for semantic search.
"""

import time
from typing import Any, Dict, List, Optional

from bnsnlp.core.exceptions import AdapterError
from bnsnlp.search.base import BaseSearch, SearchResponse, SearchResult

try:
    from pinecone import Pinecone, ServerlessSpec

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False


class PineconeSearch(BaseSearch):
    """Pinecone vector database adapter.

    This adapter provides integration with Pinecone for semantic search
    operations including document indexing and similarity search.

    Attributes:
        api_key: Pinecone API key
        environment: Pinecone environment
        index_name: Name of the index to use
        client: Pinecone client instance
        index: Pinecone index instance
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize Pinecone search adapter.

        Args:
            config: Configuration dictionary with keys:
                - api_key: Pinecone API key (required)
                - index_name: Index name (default: 'bnsnlp')
                - environment: Pinecone environment (default: 'us-east-1-aws')
                - metric: Distance metric (default: 'cosine')
                - dimension: Vector dimension (required for index creation)

        Raises:
            AdapterError: If Pinecone client is not available or API key is missing
        """
        super().__init__(config)

        if not PINECONE_AVAILABLE:
            raise AdapterError(
                "Pinecone client not available. Install with: pip install pinecone-client",
                context={"adapter": "pinecone"},
            )

        self.api_key = config.get("api_key")
        if not self.api_key:
            raise AdapterError(
                "Pinecone API key is required",
                context={"adapter": "pinecone"},
            )

        self.index_name = config.get("index_name", "bnsnlp")
        self.environment = config.get("environment", "us-east-1-aws")
        self.metric = config.get("metric", "cosine")
        self.dimension = config.get("dimension")

        # Initialize Pinecone client
        self.client = Pinecone(api_key=self.api_key)
        self._index = None

    def _get_index(self):
        """Get or create Pinecone index.

        Returns:
            Pinecone index instance

        Raises:
            AdapterError: If index retrieval fails
        """
        if self._index is not None:
            return self._index

        try:
            # Check if index exists
            existing_indexes = self.client.list_indexes()
            index_names = [idx.name for idx in existing_indexes]

            if self.index_name not in index_names:
                if self.dimension is None:
                    raise AdapterError(
                        "dimension must be specified to create a new Pinecone index",
                        context={"index_name": self.index_name},
                    )

                # Create index with serverless spec
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud="aws", region=self.environment),
                )

            self._index = self.client.Index(self.index_name)
            return self._index

        except Exception as e:
            raise AdapterError(
                f"Failed to get or create Pinecone index: {str(e)}",
                context={"index_name": self.index_name},
            )

    async def index(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Index documents with their embeddings in Pinecone.

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

        # Set dimension if not already set
        if self.dimension is None:
            self.dimension = len(embeddings[0])

        # Get index
        index = self._get_index()

        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(texts)
        elif len(metadata) != len(texts):
            raise AdapterError(
                "metadata must have the same length as texts",
                context={"metadata_len": len(metadata), "texts_len": len(texts)},
            )

        # Prepare vectors for upsert
        vectors = []
        for id_, text, embedding, meta in zip(ids, texts, embeddings, metadata):
            vector_metadata = {"text": text, **meta}
            vectors.append((id_, embedding, vector_metadata))

        # Upsert with retry logic
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Pinecone upsert in batches of 100
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i : i + batch_size]
                    # Pinecone upsert is synchronous
                    index.upsert(vectors=batch)
                return

            except Exception as e:
                if attempt == max_retries - 1:
                    raise AdapterError(
                        f"Failed to index documents after {max_retries} attempts: {str(e)}",
                        context={
                            "index_name": self.index_name,
                            "num_documents": len(texts),
                            "attempt": attempt + 1,
                        },
                    )
                # Wait before retry with exponential backoff
                import time as time_module

                time_module.sleep(retry_delay * (2**attempt))

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Search for similar documents in Pinecone.

        Args:
            query_embedding: Embedding vector of the search query
            top_k: Maximum number of results to return
            filters: Optional filters to apply to the search (Pinecone filter format)

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

        # Get index
        index = self._get_index()

        start_time = time.time()

        # Search with retry logic
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Query Pinecone (synchronous)
                query_response = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    filter=filters,
                    include_metadata=True,
                )

                query_time_ms = (time.time() - start_time) * 1000

                # Convert to SearchResult objects
                search_results = []
                for match in query_response.matches:
                    metadata_dict = match.metadata or {}
                    text = metadata_dict.pop("text", "")

                    search_results.append(
                        SearchResult(
                            id=match.id,
                            score=match.score,
                            text=text,
                            metadata=metadata_dict,
                        )
                    )

                return SearchResponse(
                    results=search_results,
                    query_time_ms=query_time_ms,
                    metadata={
                        "index_name": self.index_name,
                        "top_k": top_k,
                        "num_results": len(search_results),
                    },
                )

            except Exception as e:
                if attempt == max_retries - 1:
                    raise AdapterError(
                        f"Failed to search after {max_retries} attempts: {str(e)}",
                        context={
                            "index_name": self.index_name,
                            "top_k": top_k,
                            "attempt": attempt + 1,
                        },
                    )
                # Wait before retry with exponential backoff
                import time as time_module

                time_module.sleep(retry_delay * (2**attempt))

        # This should never be reached due to the raise in the loop
        raise AdapterError("Unexpected error in search operation")
