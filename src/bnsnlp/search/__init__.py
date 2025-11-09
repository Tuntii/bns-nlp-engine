"""
Search module for semantic search.

This module provides adapters for vector databases including
Qdrant, Pinecone, and FAISS for semantic search operations.
"""

from bnsnlp.search.base import BaseSearch, SearchResponse, SearchResult

__all__ = [
    "BaseSearch",
    "SearchResult",
    "SearchResponse",
]

# Conditionally import adapters if dependencies are available
try:
    from bnsnlp.search.qdrant import QdrantSearch

    __all__.append("QdrantSearch")
except ImportError:
    pass

try:
    from bnsnlp.search.pinecone import PineconeSearch

    __all__.append("PineconeSearch")
except ImportError:
    pass

try:
    from bnsnlp.search.faiss import FAISSSearch

    __all__.append("FAISSSearch")
except ImportError:
    pass
