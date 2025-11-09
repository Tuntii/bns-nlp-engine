"""
Embedding module for text vectorization.

This module provides adapters for various embedding providers including
OpenAI, Cohere, and HuggingFace models.
"""

from bnsnlp.embed.base import BaseEmbedder, EmbedResult

__all__ = ["BaseEmbedder", "EmbedResult"]

# Optional imports for adapters
try:
    from bnsnlp.embed.openai import OpenAIEmbedder

    __all__.append("OpenAIEmbedder")
except ImportError:
    pass

try:
    from bnsnlp.embed.cohere import CohereEmbedder

    __all__.append("CohereEmbedder")
except ImportError:
    pass

try:
    from bnsnlp.embed.huggingface import HuggingFaceEmbedder

    __all__.append("HuggingFaceEmbedder")
except ImportError:
    pass
