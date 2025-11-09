# API Referansı

bns-nlp-engine'in tüm modüllerinin detaylı API dokümantasyonu.

## Modül Yapısı

```
bnsnlp/
├── core/          # Temel bileşenler
├── preprocess/    # Metin ön işleme
├── embed/         # Embedding oluşturma
├── search/        # Semantik arama
├── classify/      # Sınıflandırma
└── utils/         # Yardımcı araçlar
```

## Core Modülleri

Core modüller, kütüphanenin temel işlevselliğini sağlar:

- **[Pipeline](core/pipeline.md)**: İşlem adımlarını orkestra eder
- **[Registry](core/registry.md)**: Plugin'leri yönetir
- **[Config](core/config.md)**: Yapılandırma yönetimi
- **[Exceptions](core/exceptions.md)**: Özel exception'lar
- **[Types](core/types.md)**: Ortak tip tanımları

## Preprocess Modülü

Türkçe metin ön işleme bileşenleri:

- **[Base](preprocess/base.md)**: Temel preprocessor interface
- **[Turkish](preprocess/turkish.md)**: Türkçe preprocessor
- **[Normalizer](preprocess/normalizer.md)**: Metin normalizasyonu
- **[Tokenizer](preprocess/tokenizer.md)**: Tokenization
- **[Lemmatizer](preprocess/lemmatizer.md)**: Lemmatization
- **[Stopwords](preprocess/stopwords.md)**: Stop words yönetimi

## Embed Modülü

Embedding oluşturma adapter'ları:

- **[Base](embed/base.md)**: Temel embedder interface
- **[OpenAI](embed/openai.md)**: OpenAI API adapter
- **[Cohere](embed/cohere.md)**: Cohere API adapter
- **[HuggingFace](embed/huggingface.md)**: HuggingFace modelleri

## Search Modülü

Vector database adapter'ları:

- **[Base](search/base.md)**: Temel search interface
- **[Qdrant](search/qdrant.md)**: Qdrant adapter
- **[Pinecone](search/pinecone.md)**: Pinecone adapter
- **[FAISS](search/faiss.md)**: FAISS adapter

## Classify Modülü

Intent ve entity extraction:

- **[Base](classify/base.md)**: Temel classifier interface
- **[Turkish](classify/turkish.md)**: Türkçe classifier

## Utils Modülleri

Yardımcı araçlar:

- **[Logging](utils/logging.md)**: Structured logging
- **[Performance](utils/performance.md)**: Performans optimizasyonları
- **[Security](utils/security.md)**: Güvenlik araçları
- **[Telemetry](utils/telemetry.md)**: Telemetry yönetimi

## Kullanım Örnekleri

Her modül sayfasında detaylı kullanım örnekleri bulabilirsiniz. Ayrıca [Örnekler](../examples/index.md) bölümünde daha kapsamlı senaryolar mevcuttur.

## Type Hints

Tüm public API'ler type hint'ler ile donatılmıştır. IDE'nizde otomatik tamamlama ve tip kontrolü için:

```python
from bnsnlp.preprocess import TurkishPreprocessor
from bnsnlp.preprocess.base import PreprocessResult

# IDE otomatik tamamlama çalışır
preprocessor: TurkishPreprocessor = TurkishPreprocessor({})
result: PreprocessResult = await preprocessor.process("metin")
```

## Async/Await

Çoğu işlem async olarak tasarlanmıştır:

```python
import asyncio

async def main():
    result = await preprocessor.process("metin")
    embeddings = await embedder.embed(["metin1", "metin2"])
    search_results = await searcher.search(query_embedding)

asyncio.run(main())
```
