"""
FAISS local vector index adapter for semantic search.
"""

import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bnsnlp.core.exceptions import AdapterError
from bnsnlp.search.base import BaseSearch, SearchResponse, SearchResult

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSSearch(BaseSearch):
    """FAISS local vector index adapter.

    This adapter provides integration with FAISS for local semantic search
    operations including document indexing, similarity search, and index persistence.

    Attributes:
        index_path: Path to save/load the FAISS index
        dimension: Dimensionality of the vectors
        metric: Distance metric ('cosine' or 'l2')
        index: FAISS index instance
        documents: Dictionary mapping document IDs to their data
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize FAISS search adapter.

        Args:
            config: Configuration dictionary with keys:
                - index_path: Path to save/load index (default: 'faiss.index')
                - dimension: Vector dimension (required for new index)
                - metric: Distance metric 'cosine' or 'l2' (default: 'cosine')
                - use_gpu: Whether to use GPU if available (default: False)

        Raises:
            AdapterError: If FAISS is not available
        """
        super().__init__(config)

        if not FAISS_AVAILABLE:
            raise AdapterError(
                "FAISS not available. Install with: pip install faiss-cpu or faiss-gpu",
                context={"adapter": "faiss"},
            )

        self.index_path = config.get("index_path", "faiss.index")
        self.dimension = config.get("dimension")
        self.metric = config.get("metric", "cosine")
        self.use_gpu = config.get("use_gpu", False)

        # Initialize index and document storage
        self._index = None
        self.documents: Dict[str, Dict[str, Any]] = {}

        # Load existing index if available
        if os.path.exists(self.index_path):
            self._load_index()

    def _create_index(self, dimension: int) -> None:
        """Create a new FAISS index.

        Args:
            dimension: Dimensionality of the vectors

        Raises:
            AdapterError: If index creation fails
        """
        try:
            self.dimension = dimension

            # Create index based on metric
            if self.metric == "cosine":
                # For cosine similarity, normalize vectors and use L2
                self._index = faiss.IndexFlatIP(dimension)
            elif self.metric == "l2":
                self._index = faiss.IndexFlatL2(dimension)
            else:
                raise AdapterError(
                    f"Unsupported metric: {self.metric}. Use 'cosine' or 'l2'",
                    context={"metric": self.metric},
                )

            # Move to GPU if requested and available
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)

        except Exception as e:
            raise AdapterError(
                f"Failed to create FAISS index: {str(e)}",
                context={"dimension": dimension, "metric": self.metric},
            )

    def _save_index(self) -> None:
        """Save FAISS index and documents to disk.

        Raises:
            AdapterError: If saving fails
        """
        try:
            # Create directory if it doesn't exist
            index_dir = Path(self.index_path).parent
            if index_dir and not index_dir.exists():
                index_dir.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            if self.use_gpu and faiss.get_num_gpus() > 0:
                # Move to CPU before saving
                cpu_index = faiss.index_gpu_to_cpu(self._index)
                faiss.write_index(cpu_index, self.index_path)
            else:
                faiss.write_index(self._index, self.index_path)

            # Save documents metadata
            docs_path = f"{self.index_path}.docs"
            with open(docs_path, "wb") as f:
                pickle.dump(
                    {
                        "documents": self.documents,
                        "dimension": self.dimension,
                        "metric": self.metric,
                    },
                    f,
                )

        except Exception as e:
            raise AdapterError(
                f"Failed to save FAISS index: {str(e)}",
                context={"index_path": self.index_path},
            )

    def _load_index(self) -> None:
        """Load FAISS index and documents from disk.

        Raises:
            AdapterError: If loading fails
        """
        try:
            # Load FAISS index
            self._index = faiss.read_index(self.index_path)

            # Move to GPU if requested and available
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)

            # Load documents metadata
            docs_path = f"{self.index_path}.docs"
            if os.path.exists(docs_path):
                with open(docs_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", {})
                    self.dimension = data.get("dimension")
                    self.metric = data.get("metric", "cosine")

        except Exception as e:
            raise AdapterError(
                f"Failed to load FAISS index: {str(e)}",
                context={"index_path": self.index_path},
            )

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity.

        Args:
            vectors: Array of vectors to normalize

        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    async def index(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Index documents with their embeddings in FAISS.

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

        # Create index if it doesn't exist
        if self._index is None:
            vector_dim = len(embeddings[0])
            self._create_index(vector_dim)

        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(texts)
        elif len(metadata) != len(texts):
            raise AdapterError(
                "metadata must have the same length as texts",
                context={"metadata_len": len(metadata), "texts_len": len(texts)},
            )

        try:
            # Convert embeddings to numpy array
            vectors = np.array(embeddings, dtype=np.float32)

            # Normalize vectors for cosine similarity
            if self.metric == "cosine":
                vectors = self._normalize_vectors(vectors)

            # Get current index size to generate internal IDs
            start_idx = self._index.ntotal

            # Add vectors to FAISS index
            self._index.add(vectors)

            # Store document metadata with mapping from internal ID to document ID
            for i, (doc_id, text, meta) in enumerate(zip(ids, texts, metadata)):
                internal_id = start_idx + i
                self.documents[str(internal_id)] = {
                    "id": doc_id,
                    "text": text,
                    "metadata": meta,
                }

            # Save index to disk
            self._save_index()

        except Exception as e:
            raise AdapterError(
                f"Failed to index documents: {str(e)}",
                context={"num_documents": len(texts)},
            )

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Search for similar documents in FAISS.

        Args:
            query_embedding: Embedding vector of the search query
            top_k: Maximum number of results to return
            filters: Optional filters to apply to the search (metadata filters)

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

        if self._index is None or self._index.ntotal == 0:
            # Return empty results if index is empty
            return SearchResponse(
                results=[],
                query_time_ms=0.0,
                metadata={"num_results": 0, "total_documents": 0},
            )

        start_time = time.time()

        try:
            # Convert query to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)

            # Normalize for cosine similarity
            if self.metric == "cosine":
                query_vector = self._normalize_vectors(query_vector)

            # Search in FAISS
            # Request more results if we need to filter
            search_k = top_k * 10 if filters else top_k
            search_k = min(search_k, self._index.ntotal)

            distances, indices = self._index.search(query_vector, search_k)

            # Convert results
            search_results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue

                doc_data = self.documents.get(str(idx))
                if doc_data is None:
                    continue

                # Apply filters if provided
                if filters:
                    match = True
                    for key, value in filters.items():
                        if doc_data["metadata"].get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                # Convert distance to similarity score
                # For cosine: score = distance (already similarity with IP)
                # For L2: score = 1 / (1 + distance)
                if self.metric == "cosine":
                    score = float(distance)
                else:
                    score = 1.0 / (1.0 + float(distance))

                search_results.append(
                    SearchResult(
                        id=doc_data["id"],
                        score=score,
                        text=doc_data["text"],
                        metadata=doc_data["metadata"],
                    )
                )

                # Stop if we have enough results
                if len(search_results) >= top_k:
                    break

            query_time_ms = (time.time() - start_time) * 1000

            return SearchResponse(
                results=search_results,
                query_time_ms=query_time_ms,
                metadata={
                    "index_path": self.index_path,
                    "top_k": top_k,
                    "num_results": len(search_results),
                    "total_documents": self._index.ntotal,
                },
            )

        except Exception as e:
            raise AdapterError(
                f"Failed to search: {str(e)}",
                context={"top_k": top_k},
            )
