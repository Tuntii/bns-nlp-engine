# bns-nlp-engine

[![CI](https://github.com/yourusername/bns-nlp-engine/workflows/CI/badge.svg)](https://github.com/yourusername/bns-nlp-engine/actions)
[![PyPI version](https://badge.fury.io/py/bns-nlp-engine.svg)](https://badge.fury.io/py/bns-nlp-engine)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Türkçe doğal dil işleme (NLP) için modüler, genişletilebilir ve açık kaynak Python kütüphanesi.

## ✨ Özellikler

- 🇹🇷 **Türkçe Odaklı**: Türkçe diline özel normalizasyon, lemmatizasyon ve stop words desteği
- 🔌 **Plugin Tabanlı Mimari**: Core'u değiştirmeden yeni özellikler ekleyin
- ⚡ **Async/Await Desteği**: I/O-bound işlemler için yüksek performans
- 🎯 **Type-Safe API**: Pydantic ve type hints ile güçlü tip kontrolü
- 🚀 **Performans Optimizasyonları**: Batch processing, streaming, multiprocessing ve GPU desteği
- 🔒 **Gizlilik Odaklı**: Telemetry varsayılan olarak kapalı, veri toplama yok
- 📦 **Kolay Kurulum**: pip ile tek komutla kurulum
- 🛠️ **Çoklu Arayüz**: Python API, CLI ve opsiyonel FastAPI servisi

## 📋 İçindekiler

- [Kurulum](#-kurulum)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Modüller](#-modüller)
- [CLI Kullanımı](#-cli-kullanımı)
- [Yapılandırma](#-yapılandırma)
- [Plugin Geliştirme](#-plugin-geliştirme)
- [Örnekler](#-örnekler)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)

## 🚀 Kurulum

### Temel Kurulum

```bash
pip install bns-nlp-engine
```

### Tüm Özelliklerle Kurulum

```bash
pip install bns-nlp-engine[all]
```

### Belirli Özelliklerle Kurulum

```bash
# OpenAI embeddings ve Qdrant search
pip install bns-nlp-engine[openai,qdrant]

# HuggingFace modelleri ve FAISS
pip install bns-nlp-engine[huggingface,faiss]

# FastAPI servisi
pip install bns-nlp-engine[api]

# Geliştirme araçları
pip install bns-nlp-engine[dev]
```

### Kaynak Koddan Kurulum

```bash
git clone https://github.com/yourusername/bns-nlp-engine.git
cd bns-nlp-engine
pip install -e ".[dev,all]"
```

## 🎯 Hızlı Başlangıç

### Python API

```python
import asyncio
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

async def main():
    # Yapılandırma oluştur
    config = Config()
    
    # Plugin registry'yi başlat
    registry = PluginRegistry()
    registry.discover_plugins()
    
    # Pipeline oluştur
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish')
    pipeline.add_step('embed', 'openai')
    
    # Tek metin işle
    result = await pipeline.process("Merhaba dünya! Bu bir test metnidir.")
    print(result)
    
    # Batch işleme
    texts = [
        "İlk metin",
        "İkinci metin",
        "Üçüncü metin"
    ]
    results = await pipeline.process_batch(texts)
    for r in results:
        print(r)

asyncio.run(main())
```

### Metin Ön İşleme

```python
from bnsnlp.preprocess import TurkishPreprocessor

async def preprocess_example():
    config = {
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'lemmatize': True
    }
    
    preprocessor = TurkishPreprocessor(config)
    result = await preprocessor.process("Merhaba DÜNYA! Bu bir TEST metnidir.")
    
    print(f"Orijinal: {result.metadata['original_text']}")
    print(f"İşlenmiş: {result.text}")
    print(f"Tokenlar: {result.tokens}")

asyncio.run(preprocess_example())
```

### Embedding Oluşturma

```python
from bnsnlp.embed import OpenAIEmbedder, HuggingFaceEmbedder

async def embedding_example():
    # OpenAI ile
    openai_config = {
        'api_key': 'your-api-key',
        'model': 'text-embedding-3-small'
    }
    embedder = OpenAIEmbedder(openai_config)
    result = await embedder.embed("Merhaba dünya")
    print(f"Embedding boyutu: {result.dimensions}")
    
    # HuggingFace ile (yerel)
    hf_config = {
        'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'use_gpu': True
    }
    embedder = HuggingFaceEmbedder(hf_config)
    result = await embedder.embed(["Metin 1", "Metin 2"])
    print(f"Batch embedding: {len(result.embeddings)} vektör")

asyncio.run(embedding_example())
```

### Semantik Arama

```python
from bnsnlp.search import FAISSSearch, QdrantSearch

async def search_example():
    # FAISS ile yerel arama
    faiss_config = {'dimension': 768}
    search = FAISSSearch(faiss_config)
    
    # Dökümanları indeksle
    texts = ["Merhaba dünya", "Python programlama", "Makine öğrenmesi"]
    embeddings = [...]  # Embedding'leri al
    ids = ["1", "2", "3"]
    await search.index(texts, embeddings, ids)
    
    # Ara
    query_embedding = [...]  # Query embedding'i al
    results = await search.search(query_embedding, top_k=2)
    
    for result in results.results:
        print(f"ID: {result.id}, Score: {result.score}, Text: {result.text}")

asyncio.run(search_example())
```

### Intent ve Entity Extraction

```python
from bnsnlp.classify import TurkishClassifier

async def classify_example():
    config = {
        'intent_model': 'your-intent-model',
        'entity_model': 'your-entity-model',
        'use_gpu': True
    }
    
    classifier = TurkishClassifier(config)
    result = await classifier.classify("Yarın saat 14:00'te İstanbul'da toplantı var")
    
    print(f"Intent: {result.intent} (confidence: {result.intent_confidence})")
    for entity in result.entities:
        print(f"Entity: {entity.text} ({entity.type}) - {entity.confidence}")

asyncio.run(classify_example())
```

## 📦 Modüller

### Core

- **Pipeline**: İşlem adımlarını orkestra eder
- **Registry**: Plugin'leri yönetir ve keşfeder
- **Config**: YAML ve environment variable'lardan yapılandırma yükler
- **Exceptions**: Özel exception hiyerarşisi

### Preprocess

- **TurkishPreprocessor**: Türkçe metin ön işleme
  - Normalizasyon (ı, ğ, ü, ş, ö, ç)
  - Tokenization
  - Stop words kaldırma
  - Lemmatization

### Embed

- **OpenAIEmbedder**: OpenAI API adapter
- **CohereEmbedder**: Cohere API adapter
- **HuggingFaceEmbedder**: Yerel HuggingFace modelleri (GPU desteği)

### Search

- **QdrantSearch**: Qdrant vector database adapter
- **PineconeSearch**: Pinecone adapter
- **FAISSSearch**: Yerel FAISS index

### Classify

- **TurkishClassifier**: Intent classification ve entity extraction

## 🖥️ CLI Kullanımı

### Metin Ön İşleme

```bash
# Stdin'den
echo "Merhaba DÜNYA!" | bnsnlp preprocess

# Dosyadan
bnsnlp preprocess -i input.txt -o output.json

# Yapılandırma ile
bnsnlp preprocess -i input.txt -c config.yaml -v
```

### Embedding Oluşturma

```bash
# OpenAI ile
bnsnlp embed -i text.txt -p openai -o embeddings.json

# HuggingFace ile
bnsnlp embed -i text.txt -p huggingface -o embeddings.json
```

### Semantik Arama

```bash
# FAISS ile
bnsnlp search "arama sorgusu" -p faiss -k 10

# Qdrant ile
bnsnlp search "arama sorgusu" -p qdrant -k 5 -c config.yaml
```

### Sınıflandırma

```bash
# Intent ve entity extraction
bnsnlp classify -i text.txt -o results.json

# Verbose mode
bnsnlp classify -i text.txt -v
```

## ⚙️ Yapılandırma

### YAML Dosyası

```yaml
# config.yaml
logging:
  level: INFO
  format: json
  output: stdout

telemetry:
  enabled: false

preprocess:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true
  lemmatize: true
  batch_size: 32

embed:
  provider: openai
  model: text-embedding-3-small
  batch_size: 16
  use_gpu: true

search:
  provider: faiss
  top_k: 10
  similarity_threshold: 0.7
```

### Environment Variables

```bash
# API Keys
export BNSNLP_EMBED_API_KEY=sk-...
export BNSNLP_COHERE_API_KEY=...

# Service URLs
export BNSNLP_QDRANT_URL=http://localhost:6333
export BNSNLP_PINECONE_API_KEY=...

# Logging
export BNSNLP_LOG_LEVEL=DEBUG
```

### Python'dan Yapılandırma

```python
from bnsnlp.core.config import Config
from pathlib import Path

# YAML'dan yükle
config = Config.from_yaml(Path("config.yaml"))

# Environment variable'lardan yükle
config = Config.from_env()

# Manuel oluştur
config = Config(
    logging={'level': 'INFO'},
    embed={'provider': 'openai', 'model': 'text-embedding-3-small'}
)
```

## 🔌 Plugin Geliştirme

### Custom Preprocessor

```python
from bnsnlp.preprocess.base import BasePreprocessor, PreprocessResult
from typing import Union, List

class CustomPreprocessor(BasePreprocessor):
    name = "custom"
    version = "1.0.0"
    
    def __init__(self, config: dict):
        super().__init__(config)
        # Özel initialization
    
    async def process(self, text: Union[str, List[str]]) -> Union[PreprocessResult, List[PreprocessResult]]:
        # Özel işleme mantığı
        if isinstance(text, str):
            processed = self._process_single(text)
            return PreprocessResult(
                text=processed,
                tokens=processed.split(),
                metadata={'custom': True}
            )
        # Batch işleme...
```

### Plugin Kaydı

```python
# pyproject.toml
[project.entry-points."bnsnlp.preprocess"]
custom = "mypackage.preprocessor:CustomPreprocessor"
```

Daha fazla bilgi için [Plugin Development Guide](docs/plugins/creating.md) sayfasına bakın.

## 📚 Örnekler

### Jupyter Notebooks

- [Quickstart](examples/notebooks/quickstart.ipynb)
- [Preprocessing](examples/notebooks/preprocessing.ipynb)
- [Embeddings](examples/notebooks/embeddings.ipynb)
- [Search](examples/notebooks/search.ipynb)

### FastAPI Service

```python
# examples/fastapi_service/main.py
from fastapi import FastAPI
from bnsnlp.api.service import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# Servisi çalıştır
cd examples/fastapi_service
pip install -r requirements.txt
python main.py
```

API dokümantasyonu: http://localhost:8000/docs

## 🧪 Test

```bash
# Tüm testleri çalıştır
pytest

# Coverage ile
pytest --cov=bnsnlp --cov-report=html

# Belirli bir modül
pytest tests/unit/test_preprocess.py

# Tox ile multi-environment
tox
```

## 🛠️ Geliştirme

### Geliştirme Ortamı Kurulumu

```bash
# Repository'yi clone'la
git clone https://github.com/yourusername/bns-nlp-engine.git
cd bns-nlp-engine

# Geliştirme bağımlılıklarını yükle
pip install -e ".[dev,all]"

# Pre-commit hooks'ları kur
pre-commit install
```

### Kod Kalitesi

```bash
# Format
black src tests
isort src tests

# Lint
ruff check src tests

# Type check
mypy src/bnsnlp

# Tüm kontroller
pre-commit run --all-files
```

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları takip edin:

1. Repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

### Katkı Kuralları

- Tüm testler geçmeli
- Code coverage %90'ın üzerinde olmalı
- Type hints kullanılmalı
- Docstring'ler eklenm eli
- Pre-commit hooks geçmeli

## 📖 Dokümantasyon

Detaylı dokümantasyon için: [https://yourusername.github.io/bns-nlp-engine](https://yourusername.github.io/bns-nlp-engine)

## 🔐 Güvenlik ve Gizlilik

- **API Key'ler**: Sadece environment variable'lardan yüklenir, asla kodda saklanmaz
- **Telemetry**: Varsayılan olarak kapalı, opt-in gerektirir
- **Veri Toplama**: Kullanıcı içeriği veya kişisel veriler toplanmaz
- **Yerel İşleme**: FAISS ve HuggingFace ile tamamen offline çalışma imkanı

## 📝 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [OpenAI](https://openai.com/) - Embedding API
- [Cohere](https://cohere.ai/) - Embedding API
- [HuggingFace](https://huggingface.co/) - Transformers ve modeller
- [Qdrant](https://qdrant.tech/) - Vector database
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search
- Türkçe NLP topluluğu

## 📞 İletişim

- GitHub Issues: [https://github.com/yourusername/bns-nlp-engine/issues](https://github.com/yourusername/bns-nlp-engine/issues)
- Email: your.email@example.com

## 🗺️ Roadmap

- [ ] Ek dil desteği
- [ ] Daha fazla embedding provider
- [ ] Gelişmiş sınıflandırma modelleri
- [ ] Web UI dashboard
- [ ] Model fine-tuning araçları
- [ ] Apache Airflow entegrasyonu

---

**Not**: Bu kütüphane aktif geliştirme aşamasındadır. Production kullanımı için lütfen stable release'leri bekleyin.
