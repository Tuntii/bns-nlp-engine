# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and core infrastructure
- Configuration system with YAML and environment variable support
- Plugin registry system with entry_points discovery
- Structured logging with JSON formatter and correlation IDs
- Pipeline orchestrator for processing workflows

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- Secure API key management (environment variables only)
- Opt-in telemetry (disabled by default)

## [1.0.0] - TBD

### Added
- **Core Features**
  - Modular, plugin-based architecture
  - Type-safe API with Pydantic models and type hints
  - Async/await support for I/O-bound operations
  - Comprehensive error handling with custom exception hierarchy

- **Preprocessing Module**
  - Turkish text normalization (ı, ğ, ü, ş, ö, ç)
  - Tokenization and punctuation removal
  - Turkish stop words removal
  - Turkish lemmatization
  - Batch processing support

- **Embedding Module**
  - OpenAI embedding adapter (text-embedding-3-small, text-embedding-3-large)
  - Cohere embedding adapter
  - HuggingFace local model adapter with GPU support
  - Batch embedding operations
  - Async embedding requests

- **Search Module**
  - Qdrant vector database adapter
  - Pinecone vector database adapter
  - FAISS local index adapter
  - Similarity search with configurable top-k
  - Metadata filtering support
  - Connection retry logic

- **Classification Module**
  - Intent classification for Turkish text
  - Named entity recognition (NER)
  - Custom model support
  - Batch classification
  - GPU acceleration

- **Performance Utilities**
  - Batch processor with configurable batch sizes
  - Streaming utilities for large datasets
  - Multiprocessing executor for CPU-bound tasks
  - GPU accelerator utility
  - Connection pooling
  - LRU caching manager

- **CLI Interface**
  - `bnsnlp preprocess` - Text preprocessing
  - `bnsnlp embed` - Generate embeddings
  - `bnsnlp search` - Semantic search
  - `bnsnlp classify` - Intent and entity extraction
  - JSON output format
  - Verbose mode for debugging

- **FastAPI Service**
  - REST API endpoints for all operations
  - `/preprocess` - Text preprocessing endpoint
  - `/embed` - Embedding generation endpoint
  - `/search` - Semantic search endpoint
  - `/classify` - Classification endpoint
  - `/health` - Health check endpoint
  - CORS middleware support
  - OpenAPI documentation

- **Security & Privacy**
  - Secure API key management (environment variables only)
  - Opt-in telemetry with anonymous data collection
  - No user content storage or transmission
  - Sensitive data masking in logs

- **Development Tools**
  - Pre-commit hooks (black, isort, ruff, mypy)
  - Tox configuration for multi-environment testing
  - GitHub Actions CI/CD workflows
  - Comprehensive test suite (unit, integration)
  - Code coverage reporting

- **Documentation**
  - Comprehensive README with quickstart guide
  - API documentation
  - CLI reference
  - Plugin development guide
  - Example notebooks
  - MIT License

### Technical Details
- Python 3.10+ support
- Pydantic v2 for data validation
- Type hints throughout codebase
- Async/await for I/O operations
- Plugin discovery via entry_points
- Configurable via YAML or environment variables

### Installation
```bash
# Basic installation
pip install bns-nlp-engine

# With all features
pip install bns-nlp-engine[all]

# Specific features
pip install bns-nlp-engine[openai,qdrant]
```

### Breaking Changes
- N/A (initial release)

### Migration Guide
- N/A (initial release)

---

## Release Notes Format

Each release will include:
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

## Version History

- **1.0.0** - Initial release with core NLP features
- Future versions will be documented here

## Links

- [PyPI Package](https://pypi.org/project/bns-nlp-engine/)
- [GitHub Repository](https://github.com/yourusername/bns-nlp-engine)
- [Documentation](https://yourusername.github.io/bns-nlp-engine/)
- [Issue Tracker](https://github.com/yourusername/bns-nlp-engine/issues)
