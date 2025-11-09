"""
bns-nlp-engine: Turkish NLP Engine - Modular and extensible NLP library

This package provides Turkish-focused natural language processing capabilities
including preprocessing, embeddings, semantic search, and classification.
"""

from bnsnlp.__version__ import __version__

# Classification
from bnsnlp.classify import BaseClassifier, ClassifyResult, Entity, TurkishClassifier

# Core components
from bnsnlp.core import (
    AdapterError,
    BNSNLPError,
    ClassifierInterface,
    Config,
    ConfigDict,
    ConfigurationError,
    EmbedConfig,
    EmbedderInterface,
    EmbeddingVector,
    LoggingConfig,
    MetadataDict,
    Pipeline,
    PipelineStep,
    PluginError,
    PluginInterface,
    PluginRegistry,
    PreprocessConfig,
    PreprocessorInterface,
    ProcessingError,
    SearchConfig,
    SearchInterface,
    TelemetryConfig,
    TextInput,
    ValidationError,
)

# Embedding
from bnsnlp.embed import BaseEmbedder, EmbedResult

# Preprocessing
from bnsnlp.preprocess import (
    BasePreprocessor,
    DummyLemmatizer,
    PreprocessResult,
    StopWords,
    Tokenizer,
    TurkishLemmatizer,
    TurkishNormalizer,
    TurkishPreprocessor,
    load_turkish_stopwords,
)

# Search
from bnsnlp.search import BaseSearch, SearchResponse, SearchResult

# Utilities
from bnsnlp.utils import (
    BatchProcessor,
    CacheManager,
    ConnectionPool,
    CorrelationLoggerAdapter,
    GPUAccelerator,
    JSONFormatter,
    MultiprocessingExecutor,
    SecureConfig,
    StreamProcessor,
    Telemetry,
    clear_correlation_id,
    generate_correlation_id,
    get_correlation_id,
    get_logger,
    get_telemetry,
    initialize_telemetry,
    set_correlation_id,
    set_telemetry,
    setup_logging,
    track_event,
)

# Optional adapters - only export if dependencies are available
__all__ = [
    # Version
    "__version__",
    # Core - Exceptions
    "BNSNLPError",
    "ConfigurationError",
    "PluginError",
    "ProcessingError",
    "AdapterError",
    "ValidationError",
    # Core - Types
    "ConfigDict",
    "MetadataDict",
    "EmbeddingVector",
    "TextInput",
    "PluginInterface",
    "PreprocessorInterface",
    "EmbedderInterface",
    "SearchInterface",
    "ClassifierInterface",
    # Core - Configuration
    "Config",
    "LoggingConfig",
    "TelemetryConfig",
    "PreprocessConfig",
    "EmbedConfig",
    "SearchConfig",
    # Core - Registry & Pipeline
    "PluginRegistry",
    "Pipeline",
    "PipelineStep",
    # Preprocessing
    "BasePreprocessor",
    "PreprocessResult",
    "TurkishNormalizer",
    "Tokenizer",
    "StopWords",
    "load_turkish_stopwords",
    "TurkishLemmatizer",
    "DummyLemmatizer",
    "TurkishPreprocessor",
    # Embedding
    "BaseEmbedder",
    "EmbedResult",
    # Search
    "BaseSearch",
    "SearchResult",
    "SearchResponse",
    # Classification
    "BaseClassifier",
    "ClassifyResult",
    "Entity",
    "TurkishClassifier",
    # Utilities - Logging
    "JSONFormatter",
    "setup_logging",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "generate_correlation_id",
    "clear_correlation_id",
    "CorrelationLoggerAdapter",
    # Utilities - Performance
    "BatchProcessor",
    "StreamProcessor",
    "MultiprocessingExecutor",
    "GPUAccelerator",
    "ConnectionPool",
    "CacheManager",
    # Utilities - Security
    "SecureConfig",
    # Utilities - Telemetry
    "Telemetry",
    "get_telemetry",
    "set_telemetry",
    "initialize_telemetry",
    "track_event",
]

# Conditionally add optional adapters
try:
    from bnsnlp.embed import OpenAIEmbedder

    __all__.append("OpenAIEmbedder")
except ImportError:
    pass

try:
    from bnsnlp.embed import CohereEmbedder

    __all__.append("CohereEmbedder")
except ImportError:
    pass

try:
    from bnsnlp.embed import HuggingFaceEmbedder

    __all__.append("HuggingFaceEmbedder")
except ImportError:
    pass

try:
    from bnsnlp.search import QdrantSearch

    __all__.append("QdrantSearch")
except ImportError:
    pass

try:
    from bnsnlp.search import PineconeSearch

    __all__.append("PineconeSearch")
except ImportError:
    pass

try:
    from bnsnlp.search import FAISSSearch

    __all__.append("FAISSSearch")
except ImportError:
    pass
