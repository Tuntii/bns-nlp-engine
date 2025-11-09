# Embedding Oluşturma

Metinleri vektörlere dönüştürmek için embedding modülünü kullanın. Farklı provider'lar (OpenAI, Cohere, HuggingFace) arasından seçim yapabilirsiniz.

## Genel Bakış

Embedding modülü, metinleri sayısal vektörlere dönüştürür:

- **OpenAI**: text-embedding-3-small, text-embedding-3-large
- **Cohere**: embed-multilingual-v3.0
- **HuggingFace**: Yerel transformer modelleri (GPU desteği ile)

## Hızlı Başlangıç

### OpenAI Embeddings

```python
import asyncio
from bnsnlp.embed import OpenAIEmbedder

async def main():
    # Embedder oluştur
    embedder = OpenAIEmbedder({
        'api_key': 'your-api-key',
        'model': 'text-embedding-3-small',
        'batch_size': 16
    })
    
    # Tek metin
    text = "Merhaba dünya"
    result = await embedder.embed(text)
    
    print(f"Model: {result.model}")
    print(f"Boyut: {result.dimensions}")
    print(f"Embedding: {result.embeddings[0][:5]}...")

asyncio.run(main())
```

**Çıktı:**
```
Model: text-embedding-3-small
Boyut: 1536
Embedding: [0.123, -0.456, 0.789, -0.234, 0.567]...
```

### Cohere Embeddings

```python
from bnsnlp.embed import CohereEmbedder

embedder = CohereEmbedder({
    'api_key': 'your-api-key',
    'model': 'embed-multilingual-v3.0',
    'batch_size': 96
})

text = "Türkçe doğal dil işleme"
result = await embedder.embed(text)
print(f"Boyut: {result.dimensions}")
```

### HuggingFace Embeddings (Yerel)

```python
from bnsnlp.embed import HuggingFaceEmbedder

embedder = HuggingFaceEmbedder({
    'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'use_gpu': True
})

text = "Makine öğrenmesi"
result = await embedder.embed(text)
print(f"GPU kullanıldı: {embedder.device == 'cuda'}")
print(f"Boyut: {result.dimensions}")
```

## Provider Karşılaştırması

### OpenAI

**Avantajlar:**
- Yüksek kalite embeddings
- Büyük model seçenekleri
- Güvenilir API

**Dezavantajlar:**
- API key gerekli
- Maliyet (kullanım başına)
- İnternet bağlantısı gerekli

**Kullanım:**
```python
embedder = OpenAIEmbedder({
    'api_key': 'sk-...',
    'model': 'text-embedding-3-small',  # veya text-embedding-3-large
    'batch_size': 16
})
```

**Modeller:**
- `text-embedding-3-small`: 1536 boyut, hızlı, ekonomik
- `text-embedding-3-large`: 3072 boyut, yüksek kalite, pahalı

### Cohere

**Avantajlar:**
- Çok dilli destek
- Büyük batch size (96)
- Rekabetçi fiyatlandırma

**Dezavantajlar:**
- API key gerekli
- Maliyet
- İnternet bağlantısı gerekli

**Kullanım:**
```python
embedder = CohereEmbedder({
    'api_key': 'your-key',
    'model': 'embed-multilingual-v3.0',
    'batch_size': 96
})
```

### HuggingFace

**Avantajlar:**
- Tamamen yerel (offline)
- Ücretsiz
- GPU acceleration
- Model seçenekleri

**Dezavantajlar:**
- İlk indirme gerekli
- GPU için CUDA gerekli
- Daha yavaş (API'lere göre)

**Kullanım:**
```python
embedder = HuggingFaceEmbedder({
    'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'use_gpu': True
})
```

**Önerilen Modeller:**
- `paraphrase-multilingual-MiniLM-L12-v2`: 384 boyut, hızlı
- `paraphrase-multilingual-mpnet-base-v2`: 768 boyut, dengeli
- `LaBSE`: 768 boyut, çok dilli

## Kullanım Senaryoları

### 1. Tek Metin Embedding

```python
embedder = OpenAIEmbedder({'api_key': 'sk-...'})

text = "Türkçe doğal dil işleme kütüphanesi"
result = await embedder.embed(text)

print(f"Embedding boyutu: {result.dimensions}")
print(f"İlk 5 değer: {result.embeddings[0][:5]}")
```

### 2. Batch Embedding

Birden fazla metni verimli şekilde işleyin:

```python
embedder = OpenAIEmbedder({
    'api_key': 'sk-...',
    'batch_size': 16
})

texts = [
    "Python programlama dili",
    "Makine öğrenmesi algoritmaları",
    "Doğal dil işleme teknikleri",
    "Veri bilimi ve analitik"
]

result = await embedder.embed(texts)

print(f"İşlenen metin sayısı: {len(result.embeddings)}")
for i, text in enumerate(texts):
    print(f"{text}: {result.embeddings[i][:3]}...")
```

### 3. Büyük Veri Setleri

Büyük veri setlerini chunk'lar halinde işleyin:

```python
async def embed_large_dataset(texts, embedder, chunk_size=100):
    """Büyük veri setini chunk'lar halinde embed et"""
    all_embeddings = []
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        result = await embedder.embed(chunk)
        all_embeddings.extend(result.embeddings)
        
        print(f"İşlenen: {i + len(chunk)}/{len(texts)}")
    
    return all_embeddings

# Kullanım
texts = ["metin" + str(i) for i in range(1000)]
embeddings = await embed_large_dataset(texts, embedder)
```

### 4. Preprocessing ile Entegrasyon

Önce preprocessing, sonra embedding:

```python
from bnsnlp.preprocess import TurkishPreprocessor

# Preprocessor ve embedder
preprocessor = TurkishPreprocessor({
    'lowercase': True,
    'remove_punctuation': True,
    'lemmatize': True
})

embedder = OpenAIEmbedder({'api_key': 'sk-...'})

# İşlem
text = "Merhaba DÜNYA! Bu bir test metnidir."
preprocess_result = await preprocessor.process(text)
embed_result = await embedder.embed(preprocess_result.text)

print(f"Orijinal: {text}")
print(f"İşlenmiş: {preprocess_result.text}")
print(f"Embedding boyutu: {embed_result.dimensions}")
```

### 5. Pipeline ile Kullanım

```python
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

# Pipeline oluştur
registry = PluginRegistry()
registry.discover_plugins()
config = Config()

pipeline = Pipeline(config, registry)
pipeline.add_step('preprocess', 'turkish')
pipeline.add_step('embed', 'openai', config={
    'model': 'text-embedding-3-small'
})

# Kullanım
text = "Türkçe NLP kütüphanesi"
result = await pipeline.process(text)
print(f"Embedding: {result.embeddings[0][:5]}...")
```

## İleri Seviye Kullanım

### GPU Acceleration (HuggingFace)

```python
import torch

# GPU kontrolü
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("GPU bulunamadı, CPU kullanılacak")

# GPU ile embedder
embedder = HuggingFaceEmbedder({
    'model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'use_gpu': True
})

# Büyük batch ile GPU'dan faydalanın
texts = ["metin" + str(i) for i in range(100)]
result = await embedder.embed(texts)
```

### Custom Batch Processing

```python
async def custom_batch_embedding(texts, embedder, batch_size=32):
    """Özel batch işleme mantığı"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            result = await embedder.embed(batch)
            results.extend(result.embeddings)
        except Exception as e:
            print(f"Batch {i}-{i+len(batch)} hatası: {e}")
            # Tek tek dene
            for text in batch:
                try:
                    result = await embedder.embed(text)
                    results.extend(result.embeddings)
                except Exception as e2:
                    print(f"Metin atlandı: {e2}")
                    results.append(None)
    
    return results
```

### Embedding Cache

Aynı metinler için embedding'leri cache'leyin:

```python
import hashlib
import json
from pathlib import Path

class CachedEmbedder:
    """Cache ile embedder"""
    
    def __init__(self, embedder, cache_dir='./embedding_cache'):
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
    
    async def embed(self, text: str):
        """Cache'li embedding"""
        key = self._cache_key(text)
        cache_path = self._cache_path(key)
        
        # Cache'de var mı?
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
                return EmbedResult(**data)
        
        # Yoksa hesapla
        result = await self.embedder.embed(text)
        
        # Cache'e kaydet
        with open(cache_path, 'w') as f:
            json.dump(result.dict(), f)
        
        return result

# Kullanım
base_embedder = OpenAIEmbedder({'api_key': 'sk-...'})
cached_embedder = CachedEmbedder(base_embedder)

# İlk çağrı - API'ye gider
result1 = await cached_embedder.embed("Merhaba dünya")

# İkinci çağrı - cache'den gelir
result2 = await cached_embedder.embed("Merhaba dünya")
```

### Paralel Embedding

Farklı provider'ları paralel kullanın:

```python
async def parallel_embeddings(text):
    """Birden fazla provider'dan embedding al"""
    openai_embedder = OpenAIEmbedder({'api_key': 'sk-...'})
    cohere_embedder = CohereEmbedder({'api_key': 'your-key'})
    hf_embedder = HuggingFaceEmbedder({'use_gpu': True})
    
    # Paralel çalıştır
    results = await asyncio.gather(
        openai_embedder.embed(text),
        cohere_embedder.embed(text),
        hf_embedder.embed(text)
    )
    
    return {
        'openai': results[0],
        'cohere': results[1],
        'huggingface': results[2]
    }

# Kullanım
text = "Türkçe NLP"
all_embeddings = await parallel_embeddings(text)
print(f"OpenAI boyut: {all_embeddings['openai'].dimensions}")
print(f"Cohere boyut: {all_embeddings['cohere'].dimensions}")
print(f"HuggingFace boyut: {all_embeddings['huggingface'].dimensions}")
```

### Streaming Embedding

Büyük dosyaları streaming ile işleyin:

```python
async def stream_embeddings(file_path, embedder, batch_size=32):
    """Dosyayı streaming ile embed et"""
    batch = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                batch.append(text)
                
                if len(batch) >= batch_size:
                    result = await embedder.embed(batch)
                    for emb in result.embeddings:
                        yield emb
                    batch = []
        
        # Kalan batch
        if batch:
            result = await embedder.embed(batch)
            for emb in result.embeddings:
                yield emb

# Kullanım
embedder = OpenAIEmbedder({'api_key': 'sk-...'})
async for embedding in stream_embeddings('large_file.txt', embedder):
    # Her embedding'i işle
    save_to_database(embedding)
```

## Yapılandırma

### Environment Variables

```bash
# .env dosyası
BNSNLP_EMBED_API_KEY=sk-...
BNSNLP_COHERE_API_KEY=your-key
```

```python
import os
from bnsnlp.core.config import Config

# Environment'tan yükle
config = Config.from_env()

embedder = OpenAIEmbedder({
    'api_key': os.getenv('BNSNLP_EMBED_API_KEY'),
    'model': config.embed.model
})
```

### YAML Configuration

```yaml
# config.yaml
embed:
  provider: openai
  model: text-embedding-3-small
  batch_size: 16
  use_gpu: false
```

```python
from pathlib import Path

config = Config.from_yaml(Path('config.yaml'))

# Provider'a göre embedder oluştur
if config.embed.provider == 'openai':
    embedder = OpenAIEmbedder(config.embed.dict())
elif config.embed.provider == 'cohere':
    embedder = CohereEmbedder(config.embed.dict())
elif config.embed.provider == 'huggingface':
    embedder = HuggingFaceEmbedder(config.embed.dict())
```

## Performans İpuçları

### 1. Batch Size Optimizasyonu

```python
# OpenAI: 16-32 optimal
openai_embedder = OpenAIEmbedder({'batch_size': 16})

# Cohere: 96'ya kadar
cohere_embedder = CohereEmbedder({'batch_size': 96})

# HuggingFace: GPU'ya göre ayarlayın
hf_embedder = HuggingFaceEmbedder({'batch_size': 64})
```

### 2. GPU Kullanımı

```python
# GPU varsa kullan
embedder = HuggingFaceEmbedder({
    'model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'use_gpu': torch.cuda.is_available()
})
```

### 3. Connection Pooling

```python
# Aynı embedder'ı yeniden kullanın
embedder = OpenAIEmbedder({'api_key': 'sk-...'})

# Birden fazla işlem için
for batch in batches:
    result = await embedder.embed(batch)
    process_results(result)

# Yeni embedder oluşturmayın
```

### 4. Async Batch Processing

```python
async def efficient_batch_embedding(texts, embedder, max_concurrent=5):
    """Concurrent batch işleme"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def embed_with_semaphore(batch):
        async with semaphore:
            return await embedder.embed(batch)
    
    # Batch'lara böl
    batch_size = 32
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    # Concurrent işle
    results = await asyncio.gather(*[embed_with_semaphore(b) for b in batches])
    
    # Birleştir
    all_embeddings = []
    for result in results:
        all_embeddings.extend(result.embeddings)
    
    return all_embeddings
```

## Hata Yönetimi

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def resilient_embedding(text, embedder):
    """Retry ile embedding"""
    return await embedder.embed(text)

# Kullanım
try:
    result = await resilient_embedding("Merhaba dünya", embedder)
except Exception as e:
    print(f"3 denemeden sonra başarısız: {e}")
```

### Fallback Strategy

```python
async def embedding_with_fallback(text):
    """Fallback ile embedding"""
    # Önce OpenAI dene
    try:
        embedder = OpenAIEmbedder({'api_key': 'sk-...'})
        return await embedder.embed(text)
    except Exception as e:
        print(f"OpenAI hatası: {e}, HuggingFace'e geçiliyor")
        
        # Fallback: HuggingFace
        try:
            embedder = HuggingFaceEmbedder({'use_gpu': False})
            return await embedder.embed(text)
        except Exception as e2:
            print(f"HuggingFace hatası: {e2}")
            raise
```

### Exception Handling

```python
from bnsnlp.core.exceptions import AdapterError

async def safe_embedding(text, embedder):
    """Güvenli embedding"""
    try:
        result = await embedder.embed(text)
        return result
    except AdapterError as e:
        print(f"Adapter hatası: {e.message}")
        print(f"Context: {e.context}")
        return None
    except Exception as e:
        print(f"Beklenmeyen hata: {str(e)}")
        return None
```

## Best Practices

### 1. API Key Güvenliği

```python
# ✅ İyi - Environment variable
import os
api_key = os.getenv('BNSNLP_EMBED_API_KEY')
embedder = OpenAIEmbedder({'api_key': api_key})

# ❌ Kötü - Hardcoded
embedder = OpenAIEmbedder({'api_key': 'sk-hardcoded-key'})
```

### 2. Batch İşleme

```python
# ✅ İyi - Batch
texts = ["metin1", "metin2", "metin3"]
result = await embedder.embed(texts)

# ❌ Kötü - Tek tek
results = []
for text in texts:
    result = await embedder.embed(text)
    results.append(result)
```

### 3. Resource Management

```python
# ✅ İyi - Tek embedder
embedder = OpenAIEmbedder({'api_key': 'sk-...'})
for text in texts:
    result = await embedder.embed(text)

# ❌ Kötü - Her seferinde yeni
for text in texts:
    embedder = OpenAIEmbedder({'api_key': 'sk-...'})
    result = await embedder.embed(text)
```

### 4. Model Seçimi

```python
# Hız için: text-embedding-3-small
fast_embedder = OpenAIEmbedder({'model': 'text-embedding-3-small'})

# Kalite için: text-embedding-3-large
quality_embedder = OpenAIEmbedder({'model': 'text-embedding-3-large'})

# Offline için: HuggingFace
offline_embedder = HuggingFaceEmbedder({
    'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
})
```

## İlgili Bölümler

- [API: Embed](../api/embed/index.md) - Embedding API referansı
- [Semantik Arama](search.md) - Embedding'leri arama için kullanma
- [Pipeline Kullanımı](pipeline.md) - Pipeline ile entegrasyon
- [Örnekler](../examples/notebooks.md) - Jupyter notebook örnekleri
