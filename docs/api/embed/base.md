# Embed Base

Tüm embedder'ların implement etmesi gereken temel interface.

## Sınıf Tanımları

::: bnsnlp.embed.base.BaseEmbedder
    options:
      show_source: true
      heading_level: 3

::: bnsnlp.embed.base.EmbedResult
    options:
      show_source: true
      heading_level: 3

## Kullanım

### Mevcut Embedder Kullanımı

```python
from bnsnlp.embed import OpenAIEmbedder, HuggingFaceEmbedder

# OpenAI
openai_embedder = OpenAIEmbedder({
    'api_key': 'your-api-key',
    'model': 'text-embedding-3-small',
    'batch_size': 16
})

result = await openai_embedder.embed("Merhaba dünya")
print(f"Dimensions: {result.dimensions}")
print(f"Embedding: {result.embeddings[0][:5]}...")  # İlk 5 değer

# Batch
texts = ["Metin 1", "Metin 2", "Metin 3"]
result = await openai_embedder.embed(texts)
print(f"Generated {len(result.embeddings)} embeddings")
```

### Custom Embedder Oluşturma

```python
from bnsnlp.embed.base import BaseEmbedder, EmbedResult
from typing import Union, List, Dict, Any
import numpy as np

class CustomEmbedder(BaseEmbedder):
    """Özel embedder implementasyonu"""
    
    name = "custom"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model', 'default-model')
        self.dimension = config.get('dimension', 768)
        # Model yükleme
        self.model = self._load_model()
    
    def _load_model(self):
        """Model yükle"""
        # Özel model yükleme mantığı
        return None
    
    async def embed(
        self, 
        texts: Union[str, List[str]]
    ) -> EmbedResult:
        """Embedding oluştur"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Embedding oluşturma mantığı
        embeddings = await self._generate_embeddings(texts)
        
        return EmbedResult(
            embeddings=embeddings,
            model=self.model_name,
            dimensions=self.dimension,
            metadata={
                'batch_size': len(texts),
                'model_version': '1.0'
            }
        )
    
    async def _generate_embeddings(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        """Embedding'leri oluştur"""
        # CPU-bound işlem için thread pool kullan
        import asyncio
        loop = asyncio.get_event_loop()
        
        embeddings = await loop.run_in_executor(
            None,
            self._compute_embeddings,
            texts
        )
        
        return embeddings
    
    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Senkron embedding hesaplama"""
        # Gerçek embedding hesaplama
        embeddings = []
        for text in texts:
            # Örnek: random embedding (gerçekte model inference)
            embedding = np.random.rand(self.dimension).tolist()
            embeddings.append(embedding)
        return embeddings
```

### API-Based Embedder Örneği

```python
import aiohttp
from bnsnlp.core.exceptions import AdapterError

class APIEmbedder(BaseEmbedder):
    """API tabanlı embedder"""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_url = config['api_url']
        self.api_key = config['api_key']
        self.batch_size = config.get('batch_size', 16)
    
    async def embed(self, texts: Union[str, List[str]]) -> EmbedResult:
        if isinstance(texts, str):
            texts = [texts]
        
        # Batch'lere böl
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = await self._call_api(batch)
            all_embeddings.extend(embeddings)
        
        return EmbedResult(
            embeddings=all_embeddings,
            model='api-model',
            dimensions=len(all_embeddings[0])
        )
    
    async def _call_api(self, texts: List[str]) -> List[List[float]]:
        """API çağrısı yap"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.api_url,
                    json={'texts': texts},
                    headers={'Authorization': f'Bearer {self.api_key}'}
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data['embeddings']
            except aiohttp.ClientError as e:
                raise AdapterError(
                    f"API call failed: {str(e)}",
                    context={'batch_size': len(texts)}
                )
```

## EmbedResult

Embedding sonucu dönen veri modeli.

### Alanlar

- **embeddings** (List[List[float]]): Embedding vektörleri
- **model** (str): Kullanılan model adı
- **dimensions** (int): Embedding boyutu
- **metadata** (Dict[str, Any]): Ek bilgiler (opsiyonel)

### Örnek

```python
result = EmbedResult(
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    model='text-embedding-3-small',
    dimensions=3,
    metadata={
        'batch_size': 2,
        'processing_time_ms': 150
    }
)

# NumPy array'e çevir
import numpy as np
embeddings_array = np.array(result.embeddings)
```

## Best Practices

### 1. Batch Processing

Verimli batch işleme:

```python
async def embed(self, texts: Union[str, List[str]]) -> EmbedResult:
    if isinstance(texts, str):
        texts = [texts]
    
    all_embeddings = []
    
    # Batch'lere böl
    for i in range(0, len(texts), self.batch_size):
        batch = texts[i:i + self.batch_size]
        batch_embeddings = await self._process_batch(batch)
        all_embeddings.extend(batch_embeddings)
    
    return EmbedResult(
        embeddings=all_embeddings,
        model=self.model_name,
        dimensions=len(all_embeddings[0])
    )
```

### 2. GPU Kullanımı

GPU varsa kullan:

```python
import torch

class GPUEmbedder(BaseEmbedder):
    def __init__(self, config: Dict[str, Any]):
        self.use_gpu = config.get('use_gpu', True)
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        self.model = self._load_model().to(self.device)
    
    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            # Model inference on GPU
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().tolist()
```

### 3. Retry Logic

API çağrıları için retry:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientEmbedder(BaseEmbedder):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Retry ile API çağrısı"""
        # API call implementation
        pass
```

### 4. Caching

Sık kullanılan embedding'leri cache'le:

```python
from functools import lru_cache
import hashlib

class CachedEmbedder(BaseEmbedder):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cache = {}
    
    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    async def embed(self, texts: Union[str, List[str]]) -> EmbedResult:
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []
        
        # Cache'den al
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self.cache:
                embeddings.append(self.cache[key])
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
        
        # Yeni embedding'leri hesapla
        if texts_to_compute:
            new_embeddings = await self._generate_embeddings(texts_to_compute)
            for text, embedding in zip(texts_to_compute, new_embeddings):
                key = self._cache_key(text)
                self.cache[key] = embedding
                embeddings.append(embedding)
        
        return EmbedResult(
            embeddings=embeddings,
            model=self.model_name,
            dimensions=len(embeddings[0])
        )
```

## İlgili Bölümler

- [OpenAI Embedder](openai.md) - OpenAI implementasyonu
- [HuggingFace Embedder](huggingface.md) - HuggingFace implementasyonu
- [Plugin Geliştirme](../../plugins/embedder.md) - Detaylı plugin kılavuzu
- [Kullanım Kılavuzu: Embeddings](../../guide/embeddings.md) - Kullanım örnekleri
