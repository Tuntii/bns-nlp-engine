# Requirements Document

## Introduction

bns-nlp-engine, Türkçe doğal dil işleme (NLP) için tasarlanmış açık kaynak, modüler ve genişletilebilir bir Python kütüphanesidir. Kütüphane, metin ön işleme, embedding, semantik arama ve sınıflandırma yeteneklerini plugin tabanlı bir mimari ile sunar. Hem programatik API hem de CLI ve opsiyonel FastAPI servisi olarak kullanılabilir.

## Glossary

- **NLP Engine**: bns-nlp-engine kütüphanesinin ana sistemi
- **Pipeline**: Metin işleme adımlarının sıralı zinciri
- **Adapter**: Üçüncü taraf servislere (OpenAI, Qdrant vb.) bağlantı sağlayan modül
- **Plugin**: Entry points mekanizması ile dinamik olarak yüklenebilen genişletme modülü
- **Registry**: Plugin'lerin kaydedildiği ve yönetildiği merkezi sistem
- **CLI**: Command-line interface (komut satırı arayüzü)
- **Preprocess Module**: Metin normalizasyon ve temizleme modülü
- **Embed Module**: Metin vektörleştirme modülü
- **Search Module**: Semantik arama modülü
- **Classify Module**: Intent ve entity extraction modülü
- **Configuration System**: YAML/env dosyaları ile yapılandırma yönetim sistemi
- **Telemetry**: Kullanım istatistikleri toplama mekanizması

## Requirements

### Requirement 1

**User Story:** Geliştirici olarak, kütüphaneyi pip ile kurabilmek istiyorum, böylece hızlıca projeme entegre edebilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL be installable via pip with command "pip install bns-nlp-engine"
2. THE NLP Engine SHALL support Python versions 3.10 and above
3. THE NLP Engine SHALL include all core dependencies in the package manifest
4. THE NLP Engine SHALL provide optional dependency groups for adapters (e.g., "pip install bns-nlp-engine[openai,qdrant]")
5. WHEN installed, THE NLP Engine SHALL be importable as "import bnsnlp"

### Requirement 2

**User Story:** Geliştirici olarak, Türkçe metinleri ön işlemden geçirebilmek istiyorum, böylece temiz ve normalize edilmiş veri elde edebilirim

#### Acceptance Criteria

1. THE Preprocess Module SHALL normalize Turkish characters (ı, ğ, ü, ş, ö, ç) correctly
2. THE Preprocess Module SHALL provide lowercase conversion functionality
3. THE Preprocess Module SHALL remove punctuation marks when configured
4. THE Preprocess Module SHALL remove Turkish stop words when configured
5. THE Preprocess Module SHALL perform lemmatization for Turkish language
6. THE Preprocess Module SHALL process text in batches when batch size is specified
7. THE Preprocess Module SHALL support async processing for I/O-bound operations

### Requirement 3

**User Story:** Geliştirici olarak, farklı embedding servislerini kullanabilmek istiyorum, böylece projemin ihtiyaçlarına göre en uygun çözümü seçebilirim

#### Acceptance Criteria

1. THE Embed Module SHALL provide adapters for OpenAI embedding API
2. THE Embed Module SHALL provide adapters for Cohere embedding API
3. THE Embed Module SHALL provide adapters for local HuggingFace models
4. THE Embed Module SHALL support batch embedding operations
5. THE Embed Module SHALL support async embedding requests
6. WHEN GPU is available, THE Embed Module SHALL utilize GPU acceleration for local models
7. THE Embed Module SHALL return embeddings as numpy arrays or torch tensors based on configuration

### Requirement 4

**User Story:** Geliştirici olarak, farklı vektör veritabanlarında semantik arama yapabilmek istiyorum, böylece altyapıma uygun çözümü kullanabilirim

#### Acceptance Criteria

1. THE Search Module SHALL provide adapter for Qdrant vector database
2. THE Search Module SHALL provide adapter for Pinecone vector database
3. THE Search Module SHALL provide adapter for FAISS local index
4. THE Search Module SHALL support similarity search with configurable top-k results
5. THE Search Module SHALL support filtering based on metadata
6. THE Search Module SHALL support async search operations
7. THE Search Module SHALL handle connection errors with retry logic

### Requirement 5

**User Story:** Geliştirici olarak, metinlerden intent ve entity çıkarabilmek istiyorum, böylece kullanıcı niyetlerini ve önemli bilgileri tespit edebilirim

#### Acceptance Criteria

1. THE Classify Module SHALL extract intent from Turkish text
2. THE Classify Module SHALL extract named entities from Turkish text
3. THE Classify Module SHALL support custom intent classification models
4. THE Classify Module SHALL support custom entity recognition models
5. THE Classify Module SHALL return structured results with confidence scores
6. THE Classify Module SHALL process classifications in batches

### Requirement 6

**User Story:** Geliştirici olarak, kendi adapter'larımı ve modüllerimi ekleyebilmek istiyorum, böylece kütüphaneyi özel ihtiyaçlarıma göre genişletebilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL implement a plugin registry system
2. THE NLP Engine SHALL discover plugins via Python entry_points mechanism
3. THE NLP Engine SHALL load third-party plugins without modifying core code
4. THE NLP Engine SHALL validate plugin interfaces at registration time
5. THE NLP Engine SHALL provide base classes for custom adapters
6. THE NLP Engine SHALL document plugin development in API documentation

### Requirement 7

**User Story:** Geliştirici olarak, CLI üzerinden NLP işlemlerini çalıştırabilmek istiyorum, böylece kod yazmadan hızlı testler yapabilirim

#### Acceptance Criteria

1. THE CLI SHALL provide "bnsnlp preprocess" command for text preprocessing
2. THE CLI SHALL provide "bnsnlp embed" command for text embedding
3. THE CLI SHALL provide "bnsnlp search" command for semantic search
4. THE CLI SHALL provide "bnsnlp classify" command for classification
5. THE CLI SHALL accept input from stdin or file paths
6. THE CLI SHALL output results in JSON format by default
7. THE CLI SHALL support verbose mode for debugging

### Requirement 8

**User Story:** Geliştirici olarak, kütüphaneyi YAML veya environment variables ile yapılandırabilmek istiyorum, böylece farklı ortamlarda kolayca ayarlayabilirim

#### Acceptance Criteria

1. THE Configuration System SHALL load settings from YAML files
2. THE Configuration System SHALL load settings from environment variables
3. WHEN both exist, THE Configuration System SHALL prioritize environment variables over YAML
4. THE Configuration System SHALL validate configuration schema using pydantic
5. THE Configuration System SHALL provide default configuration values
6. THE Configuration System SHALL support nested configuration structures
7. THE Configuration System SHALL raise clear errors for invalid configurations

### Requirement 9

**User Story:** Geliştirici olarak, tip güvenliği ve IDE desteği istiyorum, böylece hataları geliştirme aşamasında yakalayabilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL include type hints for all public APIs
2. THE NLP Engine SHALL use pydantic models for data validation
3. THE NLP Engine SHALL pass mypy strict type checking
4. THE NLP Engine SHALL provide py.typed marker for type information
5. THE NLP Engine SHALL document parameter types in docstrings

### Requirement 10

**User Story:** Geliştirici olarak, yapılandırılabilir logging istiyorum, böylece üretim ortamında sorunları takip edebilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL provide structured logging in JSON format
2. THE NLP Engine SHALL support configurable log levels (DEBUG, INFO, WARNING, ERROR)
3. THE NLP Engine SHALL log to stdout by default
4. THE NLP Engine SHALL support custom log handlers via configuration
5. THE NLP Engine SHALL include correlation IDs in async operations
6. THE NLP Engine SHALL not log sensitive data (API keys, user content)

### Requirement 11

**User Story:** Geliştirici olarak, tutarlı hata yönetimi istiyorum, böylece hataları kolayca yakalayıp işleyebilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL define custom exception hierarchy
2. THE NLP Engine SHALL raise specific exceptions for different error types
3. THE NLP Engine SHALL include error codes in exception messages
4. THE NLP Engine SHALL provide error context in exception attributes
5. THE NLP Engine SHALL document all exceptions in API documentation

### Requirement 12

**User Story:** Geliştirici olarak, kapsamlı testler istiyorum, böylece kütüphanenin güvenilir olduğundan emin olabilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL achieve minimum 90% code coverage
2. THE NLP Engine SHALL include unit tests for all modules
3. THE NLP Engine SHALL include integration tests for adapters
4. THE NLP Engine SHALL use pytest as test framework
5. THE NLP Engine SHALL mock external API calls in tests
6. THE NLP Engine SHALL run tests in CI/CD pipeline

### Requirement 13

**User Story:** Geliştirici olarak, kod kalitesi araçları istiyorum, böylece tutarlı ve temiz kod yazabilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL use black for code formatting
2. THE NLP Engine SHALL use isort for import sorting
3. THE NLP Engine SHALL use ruff for linting
4. THE NLP Engine SHALL use mypy for type checking
5. THE NLP Engine SHALL configure pre-commit hooks for all quality tools
6. THE NLP Engine SHALL enforce quality checks in CI/CD pipeline

### Requirement 14

**User Story:** Geliştirici olarak, CI/CD pipeline istiyorum, böylece her değişiklik otomatik test edilip yayınlanabilsin

#### Acceptance Criteria

1. THE NLP Engine SHALL use GitHub Actions for CI/CD
2. THE NLP Engine SHALL run tests on multiple Python versions (3.10, 3.11, 3.12)
3. THE NLP Engine SHALL build wheel packages in CI
4. THE NLP Engine SHALL publish to PyPI on tagged releases
5. THE NLP Engine SHALL run tox for multi-environment testing
6. THE NLP Engine SHALL fail CI on quality check failures

### Requirement 15

**User Story:** Geliştirici olarak, kapsamlı dokümantasyon istiyorum, böylece kütüphaneyi kolayca öğrenip kullanabilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL include comprehensive README with quickstart guide
2. THE NLP Engine SHALL use MkDocs for API documentation
3. THE NLP Engine SHALL include Jupyter notebooks in examples/ directory
4. THE NLP Engine SHALL include FastAPI service example
5. THE NLP Engine SHALL document all public APIs with docstrings
6. THE NLP Engine SHALL include architecture diagrams in documentation
7. THE NLP Engine SHALL publish documentation to GitHub Pages

### Requirement 16

**User Story:** Geliştirici olarak, performans optimizasyonları istiyorum, böylece büyük veri setlerini verimli işleyebilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL support batch processing for all operations
2. THE NLP Engine SHALL support streaming for large datasets
3. THE NLP Engine SHALL provide multiprocessing option for CPU-bound tasks
4. THE NLP Engine SHALL provide multithreading option for I/O-bound tasks
5. WHEN GPU is available, THE NLP Engine SHALL utilize GPU acceleration
6. THE NLP Engine SHALL implement connection pooling for external services
7. THE NLP Engine SHALL cache frequently used resources

### Requirement 17

**User Story:** Kullanıcı olarak, gizliliğimin korunmasını istiyorum, böylece verilerim izinsiz toplanmasın

#### Acceptance Criteria

1. THE Telemetry SHALL be disabled by default
2. WHEN enabled, THE Telemetry SHALL only collect anonymous usage statistics
3. THE Telemetry SHALL not collect user content or API keys
4. THE Telemetry SHALL provide opt-in mechanism via configuration
5. THE NLP Engine SHALL document telemetry behavior in README

### Requirement 18

**User Story:** Geliştirici olarak, kütüphaneyi FastAPI servisi olarak çalıştırabilmek istiyorum, böylece REST API üzerinden kullanabilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL provide optional FastAPI service implementation
2. THE NLP Engine SHALL expose REST endpoints for all core operations
3. THE NLP Engine SHALL include OpenAPI documentation
4. THE NLP Engine SHALL support async request handling
5. THE NLP Engine SHALL implement rate limiting
6. THE NLP Engine SHALL include health check endpoint

### Requirement 19

**User Story:** Proje yöneticisi olarak, semantic versioning istiyorum, böylece breaking change'leri kolayca takip edebilirim

#### Acceptance Criteria

1. THE NLP Engine SHALL follow SEMVER (Semantic Versioning) specification
2. THE NLP Engine SHALL document version changes in CHANGELOG
3. THE NLP Engine SHALL tag releases in git repository
4. THE NLP Engine SHALL include version number in package metadata
5. THE NLP Engine SHALL expose version via __version__ attribute

### Requirement 20

**User Story:** Açık kaynak topluluğu olarak, MIT lisansı istiyoruz, böylece kütüphaneyi özgürce kullanabilelim

#### Acceptance Criteria

1. THE NLP Engine SHALL be licensed under MIT License
2. THE NLP Engine SHALL include LICENSE file in repository root
3. THE NLP Engine SHALL include license header in source files
4. THE NLP Engine SHALL document license in README
5. THE NLP Engine SHALL include third-party license attributions
