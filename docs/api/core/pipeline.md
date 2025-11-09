# Pipeline

Pipeline, NLP işlem adımlarını sıralı bir şekilde yönetir ve orkestra eder.

## Sınıf Tanımı

::: bnsnlp.core.pipeline.Pipeline
    options:
      show_source: true
      heading_level: 3

## Kullanım

### Temel Kullanım

```python
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

# Registry ve config oluştur
registry = PluginRegistry()
registry.discover_plugins()
config = Config()

# Pipeline oluştur
pipeline = Pipeline(config, registry)

# Adımları ekle
pipeline.add_step('preprocess', 'turkish')
pipeline.add_step('embed', 'openai')

# Tek metin işle
result = await pipeline.process("Merhaba dünya!")
```

### Batch İşleme

```python
# Birden fazla metni aynı anda işle
texts = [
    "İlk metin",
    "İkinci metin",
    "Üçüncü metin"
]

results = await pipeline.process_batch(texts)
for result in results:
    print(result)
```

### Streaming İşleme

```python
async def text_generator():
    """Async generator ile metin üret"""
    for i in range(100):
        yield f"Metin {i}"

# Stream'i işle
async for result in pipeline.process_stream(text_generator()):
    print(result)
```

### Özel Yapılandırma ile Adım Ekleme

```python
# Her adım için özel config
pipeline.add_step('preprocess', 'turkish', config={
    'lowercase': True,
    'remove_punctuation': True,
    'lemmatize': False
})

pipeline.add_step('embed', 'openai', config={
    'model': 'text-embedding-3-large',
    'batch_size': 32
})
```

## Pipeline Adımları

Pipeline'a ekleyebileceğiniz adım türleri:

### Preprocess

```python
pipeline.add_step('preprocess', 'turkish', config={
    'lowercase': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatize': True,
    'batch_size': 32
})
```

### Embed

```python
# OpenAI
pipeline.add_step('embed', 'openai', config={
    'model': 'text-embedding-3-small',
    'batch_size': 16
})

# HuggingFace
pipeline.add_step('embed', 'huggingface', config={
    'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'use_gpu': True
})
```

### Search

```python
# FAISS
pipeline.add_step('search', 'faiss', config={
    'dimension': 768,
    'top_k': 10
})

# Qdrant
pipeline.add_step('search', 'qdrant', config={
    'url': 'http://localhost:6333',
    'collection': 'my_collection',
    'top_k': 10
})
```

### Classify

```python
pipeline.add_step('classify', 'turkish', config={
    'intent_model': 'path/to/intent/model',
    'entity_model': 'path/to/entity/model',
    'use_gpu': True
})
```

## Hata Yönetimi

Pipeline, her adımda oluşabilecek hataları yakalar ve uygun exception'ları fırlatır:

```python
from bnsnlp.core.exceptions import ProcessingError, AdapterError

try:
    result = await pipeline.process("metin")
except ProcessingError as e:
    print(f"İşleme hatası: {e.message}")
    print(f"Hata kodu: {e.code}")
    print(f"Context: {e.context}")
except AdapterError as e:
    print(f"Adapter hatası: {e.message}")
```

## Performans İpuçları

### Batch Size Optimizasyonu

```python
# Küçük metinler için daha büyük batch
pipeline.add_step('preprocess', 'turkish', config={'batch_size': 64})

# Büyük metinler için daha küçük batch
pipeline.add_step('preprocess', 'turkish', config={'batch_size': 16})
```

### Streaming Kullanımı

Büyük veri setleri için streaming kullanın:

```python
# Belleği verimli kullan
async def process_large_dataset(file_path):
    async def read_lines():
        with open(file_path) as f:
            for line in f:
                yield line.strip()
    
    async for result in pipeline.process_stream(read_lines()):
        # Sonuçları anında işle
        save_to_database(result)
```

## İlgili Bölümler

- [Registry](registry.md) - Plugin yönetimi
- [Config](config.md) - Yapılandırma
- [Kullanım Kılavuzu: Pipeline](../../guide/pipeline.md) - Detaylı örnekler
