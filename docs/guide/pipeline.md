# Pipeline Kullanımı

Pipeline, birden fazla NLP işlemini sıralı bir şekilde yönetmenizi sağlar.

## Temel Kullanım

### Pipeline Oluşturma

```python
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

# Registry ve config
registry = PluginRegistry()
registry.discover_plugins()
config = Config()

# Pipeline oluştur
pipeline = Pipeline(config, registry)
```

### Adım Ekleme

```python
# Preprocessing adımı
pipeline.add_step('preprocess', 'turkish', config={
    'lowercase': True,
    'remove_punctuation': True,
    'lemmatize': True
})

# Embedding adımı
pipeline.add_step('embed', 'openai', config={
    'model': 'text-embedding-3-small',
    'batch_size': 16
})

# Search adımı (opsiyonel)
pipeline.add_step('search', 'faiss', config={
    'dimension': 1536,
    'top_k': 10
})
```

### Pipeline Çalıştırma

```python
# Tek metin
result = await pipeline.process("Merhaba dünya!")

# Batch işleme
texts = ["Metin 1", "Metin 2", "Metin 3"]
results = await pipeline.process_batch(texts)

# Streaming
async for result in pipeline.process_stream(text_generator()):
    print(result)
```

## Pipeline Senaryoları

### 1. Metin Ön İşleme Pipeline

```python
pipeline = Pipeline(config, registry)

# Sadece preprocessing
pipeline.add_step('preprocess', 'turkish', config={
    'lowercase': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'lemmatize': True,
    'batch_size': 32
})

# Kullanım
texts = [
    "Merhaba DÜNYA!",
    "Bu bir TEST metnidir.",
    "Python PROGRAMLAMA dili."
]

results = await pipeline.process_batch(texts)
for result in results:
    print(f"Orijinal: {result.metadata.get('original_text')}")
    print(f"İşlenmiş: {result.text}")
    print(f"Tokenlar: {result.tokens}")
    print("---")
```

### 2. Embedding Pipeline

```python
pipeline = Pipeline(config, registry)

# Preprocessing + Embedding
pipeline.add_step('preprocess', 'turkish')
pipeline.add_step('embed', 'openai', config={
    'model': 'text-embedding-3-small'
})

# Kullanım
text = "Türkçe doğal dil işleme kütüphanesi"
result = await pipeline.process(text)

print(f"Embedding boyutu: {result.dimensions}")
print(f"Embedding: {result.embeddings[0][:5]}...")
```

### 3. Semantik Arama Pipeline

```python
pipeline = Pipeline(config, registry)

# Tam pipeline: Preprocess + Embed + Search
pipeline.add_step('preprocess', 'turkish')
pipeline.add_step('embed', 'openai')
pipeline.add_step('search', 'faiss', config={
    'dimension': 1536,
    'top_k': 5
})

# Önce dökümanları indeksle
documents = [
    "Python programlama dili",
    "Makine öğrenmesi algoritmaları",
    "Doğal dil işleme teknikleri",
    "Veri bilimi ve analitik"
]

# Her döküman için embedding oluştur ve indeksle
# (Bu kısım search adapter'ına bağlı olarak değişir)

# Arama yap
query = "NLP teknikleri"
results = await pipeline.process(query)

for result in results.results:
    print(f"Döküman: {result.text}")
    print(f"Benzerlik: {result.score}")
```

### 4. Sınıflandırma Pipeline

```python
pipeline = Pipeline(config, registry)

# Preprocess + Classify
pipeline.add_step('preprocess', 'turkish')
pipeline.add_step('classify', 'turkish', config={
    'intent_model': 'path/to/intent/model',
    'entity_model': 'path/to/entity/model'
})

# Kullanım
text = "Yarın saat 14:00'te İstanbul'da toplantı var"
result = await pipeline.process(text)

print(f"Intent: {result.intent} ({result.intent_confidence:.2f})")
for entity in result.entities:
    print(f"  {entity.type}: {entity.text} ({entity.confidence:.2f})")
```

## İleri Seviye Kullanım

### Dinamik Pipeline

```python
def create_pipeline(use_lemmatization=True, use_gpu=False):
    """Parametrelere göre pipeline oluştur"""
    pipeline = Pipeline(config, registry)
    
    # Preprocessing config
    preprocess_config = {
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'lemmatize': use_lemmatization
    }
    pipeline.add_step('preprocess', 'turkish', config=preprocess_config)
    
    # Embedding config
    if use_gpu:
        pipeline.add_step('embed', 'huggingface', config={
            'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'use_gpu': True
        })
    else:
        pipeline.add_step('embed', 'openai', config={
            'model': 'text-embedding-3-small'
        })
    
    return pipeline

# Kullanım
pipeline_with_lemma = create_pipeline(use_lemmatization=True, use_gpu=False)
pipeline_without_lemma = create_pipeline(use_lemmatization=False, use_gpu=True)
```

### Conditional Steps

```python
async def process_with_conditions(text, include_search=False):
    """Koşullu adımlarla işleme"""
    pipeline = Pipeline(config, registry)
    
    # Her zaman preprocessing
    pipeline.add_step('preprocess', 'turkish')
    pipeline.add_step('embed', 'openai')
    
    # Koşullu search
    if include_search:
        pipeline.add_step('search', 'faiss', config={'top_k': 10})
    
    return await pipeline.process(text)
```

### Pipeline Chaining

```python
async def multi_stage_processing(texts):
    """Çok aşamalı işleme"""
    # Stage 1: Preprocessing
    preprocess_pipeline = Pipeline(config, registry)
    preprocess_pipeline.add_step('preprocess', 'turkish')
    
    preprocessed = await preprocess_pipeline.process_batch(texts)
    
    # Stage 2: Embedding
    embed_pipeline = Pipeline(config, registry)
    embed_pipeline.add_step('embed', 'openai')
    
    embedded = await embed_pipeline.process_batch([p.text for p in preprocessed])
    
    # Stage 3: Classification
    classify_pipeline = Pipeline(config, registry)
    classify_pipeline.add_step('classify', 'turkish')
    
    classified = await classify_pipeline.process_batch([p.text for p in preprocessed])
    
    return {
        'preprocessed': preprocessed,
        'embedded': embedded,
        'classified': classified
    }
```

## Performans Optimizasyonu

### Batch Size Ayarlama

```python
# Küçük metinler için büyük batch
pipeline.add_step('preprocess', 'turkish', config={
    'batch_size': 64
})

# Büyük metinler için küçük batch
pipeline.add_step('preprocess', 'turkish', config={
    'batch_size': 16
})
```

### Streaming ile Bellek Yönetimi

```python
async def process_large_file(file_path):
    """Büyük dosyayı streaming ile işle"""
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish')
    pipeline.add_step('embed', 'openai')
    
    async def read_lines():
        with open(file_path) as f:
            for line in f:
                yield line.strip()
    
    # Stream'i işle
    results = []
    async for result in pipeline.process_stream(read_lines()):
        results.append(result)
        
        # Periyodik olarak kaydet
        if len(results) >= 100:
            save_to_database(results)
            results = []
    
    # Kalan sonuçları kaydet
    if results:
        save_to_database(results)
```

### Paralel Pipeline Execution

```python
async def parallel_pipelines(texts):
    """Birden fazla pipeline'ı paralel çalıştır"""
    # Pipeline 1: OpenAI embeddings
    pipeline1 = Pipeline(config, registry)
    pipeline1.add_step('preprocess', 'turkish')
    pipeline1.add_step('embed', 'openai')
    
    # Pipeline 2: HuggingFace embeddings
    pipeline2 = Pipeline(config, registry)
    pipeline2.add_step('preprocess', 'turkish')
    pipeline2.add_step('embed', 'huggingface')
    
    # Paralel çalıştır
    results1, results2 = await asyncio.gather(
        pipeline1.process_batch(texts),
        pipeline2.process_batch(texts)
    )
    
    return {
        'openai': results1,
        'huggingface': results2
    }
```

## Hata Yönetimi

### Try-Catch ile

```python
from bnsnlp.core.exceptions import ProcessingError, AdapterError

async def safe_pipeline_execution(text):
    """Güvenli pipeline çalıştırma"""
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish')
    pipeline.add_step('embed', 'openai')
    
    try:
        result = await pipeline.process(text)
        return result
    except ProcessingError as e:
        print(f"İşleme hatası: {e.message}")
        print(f"Context: {e.context}")
        # Fallback işlem
        return None
    except AdapterError as e:
        print(f"Adapter hatası: {e.message}")
        # Yeniden dene veya alternatif adapter kullan
        return None
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def resilient_pipeline_execution(text):
    """Retry ile pipeline çalıştırma"""
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish')
    pipeline.add_step('embed', 'openai')
    
    return await pipeline.process(text)
```

## Best Practices

### 1. Pipeline'ı Yeniden Kullanın

```python
# ✅ İyi - Pipeline'ı bir kez oluştur, birden fazla kullan
pipeline = Pipeline(config, registry)
pipeline.add_step('preprocess', 'turkish')
pipeline.add_step('embed', 'openai')

for text in texts:
    result = await pipeline.process(text)

# ❌ Kötü - Her seferinde yeni pipeline
for text in texts:
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish')
    result = await pipeline.process(text)
```

### 2. Batch İşleme Tercih Edin

```python
# ✅ İyi - Batch işleme
results = await pipeline.process_batch(texts)

# ❌ Kötü - Tek tek işleme
results = []
for text in texts:
    result = await pipeline.process(text)
    results.append(result)
```

### 3. Yapılandırmayı Dışarıda Tutun

```python
# ✅ İyi - Config dosyası kullan
config = Config.from_yaml(Path("config.yaml"))
pipeline = Pipeline(config, registry)

# ❌ Kötü - Hardcoded config
pipeline.add_step('embed', 'openai', config={
    'api_key': 'sk-hardcoded-key'  # Asla böyle yapmayın!
})
```

## İlgili Bölümler

- [API: Pipeline](../api/core/pipeline.md) - Pipeline API referansı
- [Batch İşleme](batch-processing.md) - Batch processing detayları
- [Streaming](streaming.md) - Streaming kullanımı
- [Hata Yönetimi](error-handling.md) - Exception handling
