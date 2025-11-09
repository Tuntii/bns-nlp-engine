# Search Base

Tüm search adapter'larının implement etmesi gereken temel interface.

## Sınıf Tanımları

::: bnsnlp.search.base.BaseSearch
    options:
      show_source: true
      heading_level: 3

::: bnsnlp.search.base.SearchResult
    options:
      show_source: true
      heading_level: 3

::: bnsnlp.search.base.SearchResponse
    options:
      show_source: true
      heading_level: 3

## Kullanım

### Mevcut Search Adapter Kullanımı

```python
from bnsnlp.search import FAISSSearch, QdrantSearch

# FAISS (yerel)
faiss_search = FAISSSearch({'dimension': 768})

# Dökümanları indeksle
texts = ["Merhaba dünya", "Python programlama", "Makine öğrenmesi"]
embeddings = [...]  # Embedding'leri al
ids = ["1", "2", "3"]
metadata = [
    {'category': 'greeting'},
    {'category': 'programming'},
    {'category': 'ml'}
]

await faiss_search.index(texts, embeddings, ids, metadata)

# Ara
query_embedding = [...]  # Query embedding'i al
response = await faiss_search.search(
    query_embedding,
    top_k=2,
    filters={'category': 'programming'}
)

for result in response.results:
    print(f"ID: {result.id}, Score: {result.score}")
    print(f"Text: {result.text}")
    print(f"Metadata: {result.metadata}")
```

### Custom Search Adapter Oluşturma

```python
from bnsnlp.search.base import BaseSearch, SearchResult, SearchResponse
from typing import List, Dict, Any, Optional
import time

class CustomSearch(BaseSearch):
    """Özel search adapter implementasyonu"""
    
    name = "custom"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dimension = config.get('dimension', 768)
        # Storage initialization
        self.documents = {}
        self.embeddings = {}
    
    async def index(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Dökümanları indeksle"""
        if metadata is None:
            metadata = [{}] * len(texts)
        
        for text, embedding, id_, meta in zip(texts, embeddings, ids, metadata):
            self.documents[id_] = {
                'text': text,
                'embedding': embedding,
                'metadata': meta
            }
            self.embeddings[id_] = embedding
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """Semantik arama yap"""
        start_time = time.time()
        
        # Similarity hesapla
        scores = []
        for id_, doc in self.documents.items():
            # Filter uygula
            if filters and not self._match_filters(doc['metadata'], filters):
                continue
            
            # Cosine similarity
            score = self._cosine_similarity(query_embedding, doc['embedding'])
            scores.append((id_, score))
        
        # Sırala ve top-k al
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]
        
        # SearchResult'ları oluştur
        results = []
        for id_, score in top_results:
            doc = self.documents[id_]
            results.append(SearchResult(
                id=id_,
                score=score,
                text=doc['text'],
                metadata=doc['metadata']
            ))
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=results,
            query_time_ms=query_time_ms,
            metadata={
                'total_documents': len(self.documents),
                'filtered_documents': len(scores)
            }
        )
    
    def _match_filters(
        self, 
        metadata: Dict[str, Any], 
        filters: Dict[str, Any]
    ) -> bool:
        """Metadata filter'ları kontrol et"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def _cosine_similarity(
        self, 
        vec1: List[float], 
        vec2: List[float]
    ) -> float:
        """Cosine similarity hesapla"""
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
```

### Vector Database Adapter Örneği

```python
from bnsnlp.core.exceptions import AdapterError

class VectorDBSearch(BaseSearch):
    """Vector database adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        self.url = config['url']
        self.collection = config.get('collection', 'default')
        self.client = self._init_client()
    
    def _init_client(self):
        """Client'ı başlat"""
        # Vector DB client initialization
        return None
    
    async def index(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Dökümanları indeksle"""
        try:
            # Batch upsert
            points = []
            for i, (text, embedding, id_) in enumerate(zip(texts, embeddings, ids)):
                meta = metadata[i] if metadata else {}
                meta['text'] = text
                
                points.append({
                    'id': id_,
                    'vector': embedding,
                    'payload': meta
                })
            
            # Vector DB'ye yaz
            await self.client.upsert(
                collection_name=self.collection,
                points=points
            )
        except Exception as e:
            raise AdapterError(
                f"Indexing failed: {str(e)}",
                context={'document_count': len(texts)}
            )
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """Semantik arama yap"""
        try:
            start_time = time.time()
            
            # Vector DB'de ara
            results = await self.client.search(
                collection_name=self.collection,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=filters
            )
            
            # SearchResult'lara çevir
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    id=str(result.id),
                    score=result.score,
                    text=result.payload.get('text', ''),
                    metadata=result.payload
                ))
            
            query_time_ms = (time.time() - start_time) * 1000
            
            return SearchResponse(
                results=search_results,
                query_time_ms=query_time_ms
            )
        except Exception as e:
            raise AdapterError(
                f"Search failed: {str(e)}",
                context={'top_k': top_k}
            )
```

## Data Models

### SearchResult

Tek bir arama sonucu.

```python
result = SearchResult(
    id="doc_123",
    score=0.95,
    text="Merhaba dünya",
    metadata={
        'category': 'greeting',
        'language': 'tr',
        'created_at': '2024-01-01'
    }
)
```

### SearchResponse

Arama sonuçlarının tamamı.

```python
response = SearchResponse(
    results=[result1, result2, result3],
    query_time_ms=45.2,
    metadata={
        'total_documents': 1000,
        'filtered_documents': 500
    }
)
```

## Best Practices

### 1. Batch Indexing

Büyük veri setleri için batch indexing:

```python
async def index_large_dataset(
    self,
    texts: List[str],
    embeddings: List[List[float]],
    ids: List[str],
    batch_size: int = 100
):
    """Büyük veri setini batch'lerle indeksle"""
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        await self.index(batch_texts, batch_embeddings, batch_ids)
```

### 2. Connection Pooling

Verimli bağlantı yönetimi:

```python
from bnsnlp.utils.performance import ConnectionPool

class PooledSearch(BaseSearch):
    def __init__(self, config: Dict[str, Any]):
        self.pool = ConnectionPool(max_connections=10)
    
    async def search(self, query_embedding, top_k=10, filters=None):
        conn = await self.pool.acquire()
        try:
            results = await conn.search(...)
            return results
        finally:
            await self.pool.release(conn)
```

### 3. Retry Logic

Geçici hatalar için retry:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientSearch(BaseSearch):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def search(self, query_embedding, top_k=10, filters=None):
        """Retry ile arama"""
        # Search implementation
        pass
```

### 4. Filtering Optimization

Verimli filtering:

```python
def _build_filter_query(self, filters: Dict[str, Any]) -> Any:
    """Filter'ları optimize edilmiş query'ye çevir"""
    if not filters:
        return None
    
    # Vector DB'ye özgü filter format
    query_filters = []
    for key, value in filters.items():
        if isinstance(value, list):
            # IN query
            query_filters.append({'key': key, 'match': {'any': value}})
        else:
            # Exact match
            query_filters.append({'key': key, 'match': {'value': value}})
    
    return {'must': query_filters}
```

### 5. Caching

Sık yapılan aramalar için cache:

```python
from bnsnlp.utils.performance import CacheManager

class CachedSearch(BaseSearch):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cache = CacheManager(max_size=1000)
    
    async def search(self, query_embedding, top_k=10, filters=None):
        # Cache key oluştur
        cache_key = self._create_cache_key(query_embedding, top_k, filters)
        
        # Cache'den al veya hesapla
        return await self.cache.get_or_compute(
            cache_key,
            lambda: self._do_search(query_embedding, top_k, filters)
        )
```

## İlgili Bölümler

- [FAISS Search](faiss.md) - FAISS implementasyonu
- [Qdrant Search](qdrant.md) - Qdrant implementasyonu
- [Plugin Geliştirme](../../plugins/search.md) - Detaylı plugin kılavuzu
- [Kullanım Kılavuzu: Search](../../guide/search.md) - Kullanım örnekleri
