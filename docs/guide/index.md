# Kullanım Kılavuzu

bns-nlp-engine'i kullanarak Türkçe NLP işlemlerini nasıl gerçekleştireceğinizi öğrenin.

## Kılavuz İçeriği

### Temel Modüller

- **[Metin Ön İşleme](preprocessing.md)**: Türkçe metinleri normalize edin, tokenize edin ve temizleyin
- **[Embedding Oluşturma](embeddings.md)**: Metinleri vektörlere dönüştürün
- **[Semantik Arama](search.md)**: Vector database'lerde semantik arama yapın
- **[Sınıflandırma](classification.md)**: Intent ve entity extraction

### İleri Seviye Konular

- **[Pipeline Kullanımı](pipeline.md)**: İşlem adımlarını zincirleyin
- **[Batch İşleme](batch-processing.md)**: Büyük veri setlerini verimli işleyin
- **[Streaming](streaming.md)**: Gerçek zamanlı veri işleme
- **[Hata Yönetimi](error-handling.md)**: Exception'ları yakalayın ve işleyin

## Hızlı Başlangıç

### Basit Örnek

```python
import asyncio
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

async def main():
    # Setup
    config = Config()
    registry = PluginRegistry()
    registry.discover_plugins()
    
    # Pipeline oluştur
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish')
    
    # İşle
    result = await pipeline.process("Merhaba DÜNYA! Bu bir test metnidir.")
    print(result)

asyncio.run(main())
```

### Tam Örnek

```python
import asyncio
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry
from bnsnlp.preprocess import TurkishPreprocessor
from bnsnlp.embed import OpenAIEmbedder
from bnsnlp.search import FAISSSearch

async def full_example():
    # 1. Metin ön işleme
    preprocessor = TurkishPreprocessor({
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'lemmatize': True
    })
    
    text = "Merhaba DÜNYA! Bu bir TEST metnidir."
    preprocess_result = await preprocessor.process(text)
    print(f"İşlenmiş: {preprocess_result.text}")
    
    # 2. Embedding oluştur
    embedder = OpenAIEmbedder({
        'api_key': 'your-api-key',
        'model': 'text-embedding-3-small'
    })
    
    embed_result = await embedder.embed(preprocess_result.text)
    print(f"Embedding boyutu: {embed_result.dimensions}")
    
    # 3. Arama yap
    search = FAISSSearch({'dimension': embed_result.dimensions})
    
    # Dökümanları indeksle
    docs = ["Merhaba dünya", "Python programlama", "Makine öğrenmesi"]
    doc_embeddings = await embedder.embed(docs)
    await search.index(docs, doc_embeddings.embeddings, ["1", "2", "3"])
    
    # Ara
    search_results = await search.search(embed_result.embeddings[0], top_k=2)
    for result in search_results.results:
        print(f"Sonuç: {result.text} (score: {result.score})")

asyncio.run(full_example())
```

## Yapılandırma

### YAML ile

```yaml
# config.yaml
preprocess:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true
  lemmatize: true

embed:
  provider: openai
  model: text-embedding-3-small
  batch_size: 16

search:
  provider: faiss
  top_k: 10
```

```python
from bnsnlp.core.config import Config
from pathlib import Path

config = Config.from_yaml(Path("config.yaml"))
```

### Environment Variables ile

```bash
export BNSNLP_EMBED_API_KEY=sk-...
export BNSNLP_LOG_LEVEL=INFO
```

```python
config = Config.from_env()
```

## Best Practices

### 1. Async/Await Kullanın

```python
# ✅ İyi
async def process_texts(texts):
    results = []
    for text in texts:
        result = await preprocessor.process(text)
        results.append(result)
    return results

# ❌ Kötü
def process_texts(texts):
    results = []
    for text in texts:
        result = asyncio.run(preprocessor.process(text))  # Her seferinde yeni event loop
        results.append(result)
    return results
```

### 2. Batch İşleme Kullanın

```python
# ✅ İyi - Batch işleme
texts = ["metin1", "metin2", "metin3"]
results = await preprocessor.process(texts)

# ❌ Kötü - Tek tek işleme
results = []
for text in texts:
    result = await preprocessor.process(text)
    results.append(result)
```

### 3. Hata Yönetimi

```python
from bnsnlp.core.exceptions import ProcessingError, AdapterError

try:
    result = await pipeline.process(text)
except ProcessingError as e:
    print(f"İşleme hatası: {e.message}")
    # Alternatif işlem
except AdapterError as e:
    print(f"Adapter hatası: {e.message}")
    # Yeniden dene veya fallback
```

### 4. Resource Yönetimi

```python
# Context manager kullanın (varsa)
async with embedder:
    results = await embedder.embed(texts)

# Veya manuel cleanup
try:
    results = await embedder.embed(texts)
finally:
    await embedder.close()
```

## Sonraki Adımlar

- [Metin Ön İşleme](preprocessing.md) - Detaylı preprocessing kılavuzu
- [Pipeline Kullanımı](pipeline.md) - Pipeline ile çalışma
- [Örnekler](../examples/index.md) - Daha fazla örnek
