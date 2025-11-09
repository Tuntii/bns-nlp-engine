"""
Type definitions and protocols for bns-nlp-engine.

This module defines common type aliases, protocols, and interfaces
used throughout the library for type safety and consistency.
"""

from typing import Any, Dict, List, Protocol, Union, runtime_checkable


# Type aliases for common data structures
ConfigDict = Dict[str, Any]
"""Configuration dictionary type."""

MetadataDict = Dict[str, Any]
"""Metadata dictionary type."""

EmbeddingVector = List[float]
"""Type for embedding vectors."""

TextInput = Union[str, List[str]]
"""Type for text input - can be single string or list of strings."""


@runtime_checkable
class PluginInterface(Protocol):
    """
    Base protocol for all plugins.
    
    All plugins must implement this interface to be registered
    and used within the bns-nlp-engine system.
    
    Attributes:
        name: Unique name of the plugin
        version: Version string of the plugin (e.g., "1.0.0")
    """
    
    name: str
    version: str
    
    def initialize(self, config: ConfigDict) -> None:
        """
        Initialize the plugin with configuration.
        
        This method is called when the plugin is loaded and should
        perform any necessary setup such as loading models, establishing
        connections, or validating configuration.
        
        Args:
            config: Configuration dictionary for the plugin
            
        Raises:
            ConfigurationError: If configuration is invalid
            PluginError: If initialization fails
        """
        ...


@runtime_checkable
class PreprocessorInterface(Protocol):
    """
    Protocol for text preprocessing plugins.
    
    Preprocessors handle text normalization, tokenization,
    and other text cleaning operations.
    """
    
    name: str
    version: str
    
    async def process(self, text: TextInput) -> Any:
        """
        Process text or batch of texts.
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            PreprocessResult or list of PreprocessResult objects
            
        Raises:
            ProcessingError: If preprocessing fails
        """
        ...


@runtime_checkable
class EmbedderInterface(Protocol):
    """
    Protocol for text embedding plugins.
    
    Embedders convert text into dense vector representations
    using various embedding models or services.
    """
    
    name: str
    version: str
    
    async def embed(self, texts: TextInput) -> Any:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            EmbedResult containing embeddings and metadata
            
        Raises:
            AdapterError: If embedding generation fails
        """
        ...


@runtime_checkable
class SearchInterface(Protocol):
    """
    Protocol for semantic search plugins.
    
    Search plugins provide vector similarity search capabilities
    using various vector database backends.
    """
    
    name: str
    version: str
    
    async def index(
        self,
        texts: List[str],
        embeddings: List[EmbeddingVector],
        ids: List[str],
        metadata: List[MetadataDict]
    ) -> None:
        """
        Index documents with their embeddings.
        
        Args:
            texts: List of text documents
            embeddings: List of embedding vectors
            ids: List of document IDs
            metadata: List of metadata dictionaries
            
        Raises:
            AdapterError: If indexing fails
        """
        ...
    
    async def search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        filters: MetadataDict
    ) -> Any:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            SearchResponse containing results
            
        Raises:
            AdapterError: If search fails
        """
        ...


@runtime_checkable
class ClassifierInterface(Protocol):
    """
    Protocol for text classification plugins.
    
    Classifiers perform intent classification and entity extraction
    on text inputs.
    """
    
    name: str
    version: str
    
    async def classify(self, text: str) -> Any:
        """
        Classify intent and extract entities from text.
        
        Args:
            text: Input text to classify
            
        Returns:
            ClassifyResult containing intent and entities
            
        Raises:
            ProcessingError: If classification fails
        """
        ...


__all__ = [
    "ConfigDict",
    "MetadataDict",
    "EmbeddingVector",
    "TextInput",
    "PluginInterface",
    "PreprocessorInterface",
    "EmbedderInterface",
    "SearchInterface",
    "ClassifierInterface",
]
