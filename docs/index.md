# bns-nlp-engine Dokümantasyonu

Türkçe doğal dil işleme (NLP) için modüler, genişletilebilir ve açık kaynak Python kütüphanesi.

## Hoş Geldiniz

bns-nlp-engine, Türkçe metinler üzerinde çeşitli NLP işlemleri gerçekleştirmenizi sağlayan güçlü bir kütüphanedir. Plugin tabanlı mimarisi sayesinde kolayca genişletilebilir ve farklı use case'lere adapte edilebilir.

## Temel Özellikler

### 🇹🇷 Türkçe Odaklı

- Türkçe karakterlerin doğru işlenmesi (ı, ğ, ü, ş, ö, ç)
- Türkçe stop words listesi
- Türkçe lemmatization desteği
- Türkçe intent ve entity extraction

### 🔌 Modüler Mimari

- **Preprocess**: Metin normalizasyon ve temizleme
- **Embed**: Çoklu provider desteği (OpenAI, Cohere, HuggingFace)
- **Search**: Vector database entegrasyonları (Qdrant, Pinecone, FAISS)
- **Classify**: Intent ve entity extraction

### ⚡ Yüksek Performans

- Async/await desteği
- Batch processing
- Streaming support
- GPU acceleration
- Multiprocessing/threading
- Connection pooling
- Intelligent caching

### 🎯 Type-Safe

- Pydantic modelleri ile veri validasyonu
- Comprehensive type hints
- Mypy strict mode uyumlu
- IDE autocomplete desteği

## Hızlı Başlangıç

```python
import asyncio
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

async def main():
    config = Config()
    registry = PluginRegistry()
    registry.discover_plugins()
    
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish')
    pipeline.add_step('embed', 'openai')
    
    result = await pipeline.process("Merhaba dünya!")
    print(result)

asyncio.run(main())
```

## Kurulum

```bash
# Temel kurulum
pip install bns-nlp-engine

# Tüm özelliklerle
pip install bns-nlp-engine[all]

# Belirli özellikler
pip install bns-nlp-engine[openai,qdrant]
```

## Dokümantasyon Yapısı

- **[Başlangıç](getting-started/index.md)**: Kurulum ve ilk adımlar
- **[Kullanım Kılavuzu](guide/index.md)**: Detaylı kullanım örnekleri
- **[API Referansı](api/index.md)**: Tüm modüllerin API dokümantasyonu
- **[Plugin Geliştirme](plugins/index.md)**: Kendi plugin'lerinizi oluşturun
- **[CLI Referansı](cli/index.md)**: Komut satırı kullanımı
- **[API Servisi](api-service/index.md)**: FastAPI servisi kurulumu
- **[Örnekler](examples/index.md)**: Pratik kullanım örnekleri

## Topluluk ve Destek


## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [Lisans](about/license.md) sayfasına bakın.
