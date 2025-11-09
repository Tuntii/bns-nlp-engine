# Semantik Arama

Vector database'lerde semantik arama yapmak için search modülünü kullanın. Qdrant, Pinecone ve FAISS desteği mevcuttur.

## Genel Bakış

Search modülü, embedding'leri kullanarak semantik arama yapar:

- **Qdrant**: Cloud veya self-hosted vector database
- **Pinecone**: Managed vector database servisi
- **FAISS**: Facebook'un yerel vector search kütüphanesi

## Hızlı Başlangıç

### FAISS (Yerel)

```python
import asyncio
from bnsnlp.embed import OpenAIEmbedder
from bnsnlp.search import FAISSSearch

async def main():
    # Embedder ve search
    embedder = OpenAIEmbedder({'api_key': 'sk-...'})
    search = FAISSSearch({'dimension': 1536})
    
    # Dökümanları indeksle
    documents = [
        "Python programlama dili",
        "Makine öğrenmesi algoritmaları",
        "Doğal dil işleme teknikleri",
        "Veri bilimi ve analitik"
    ]
    
    # Embedding'leri oluştur
    embed_result = await embedder.embed(documents)
    
    # İndeksle
    await search.index(
        texts=documents,
        embeddings=embed_result.embeddings,
        ids=["1", "2", "3", "4"]
    )
    
    # Arama yap
    query = "NLP teknikleri"
    query_embedding = await embedder.embed(query)
    
    results = await search.search(
        query_embedding=query_embedding.embeddings[0],
        top_k=2
    )
    
    for result in results.results:
        print(f"Döküman: {result.text}")
        print(f"Benzerlik: {result.score:.4f}")
        print("---")

asyncio.run(main())
```

**Çıktı:**
```
Döküman: Doğal dil işleme teknikleri
Benzerlik: 0.8923
---
Döküman: Makine öğrenmesi algoritmaları
Benzerlik: 0.7654
---
```

### Qdrant

```python
from bnsnlp.search import QdrantSearch

# Qdrant search
search = QdrantSearch({
    'url': 'http://localhost:6333',
    'collection': 'my_documents'
})

# İndeksle
await search.index(
    texts=documents,
    embeddings=embeddings,
    ids=["1", "2", "3", "4"],
    metadata=[
        {'category': 'programming'},
        {'category': 'ml'},
        {'category': 'nlp'},
        {'category': 'data'}
    ]
)

# Metadata filtresi ile ara
results = await search.search(
    query_embedding=query_embedding,
    top_k=5,
    filters={'category': 'nlp'}
)
```

### Pinecone

```python
from bnsnlp.search import PineconeSearch

# Pinecone search
search = PineconeSearch({
    'api_key': 'your-api-key',
    'environment': 'us-west1-gcp',
    'index_name': 'my-index'
})

# İndeksle
await search.index(
    texts=documents,
    embeddings=embeddings,
    ids=["1", "2", "3", "4"]
)

# Ara
results = await search.search(
    query_embedding=query_embedding,
    top_k=10
)
```

## Provider Karşılaştırması

### FAISS

**Avantajlar:**
- Tamamen yerel (offline)
- Ücretsiz
- Çok hızlı
- Kolay kurulum

**Dezavantajlar:**
- Metadata filtreleme sınırlı
- Ölçeklendirme zorluğu
- Manuel persistence

**Kullanım Senaryoları:**
- Prototipleme
- Küçük-orta ölçekli projeler
- Offline uygulamalar

### Qdrant

**Avantajlar:**
- Güçlü metadata filtreleme
- Self-hosted veya cloud
- Ölçeklenebilir
- Açık kaynak

**Dezavantajlar:**
- Kurulum gerekli (self-hosted)
- Maliyet (cloud)

**Kullanım Senaryoları:**
- Üretim uygulamaları
- Karmaşık filtreleme
- Büyük ölçekli projeler

### Pinecone

**Avantajlar:**
- Fully managed
- Kolay kurulum
- Otomatik ölçeklendirme
- Güvenilir

**Dezavantajlar:**
- Maliyet
- Vendor lock-in
- İnternet gerekli

**Kullanım Senaryoları:**
- Hızlı deployment
- Managed servis tercihi
- Ölçeklenebilir uygulamalar

## Kullanım Senaryoları

### 1. Basit Semantik Arama

```python
async def simple_search():
    """Basit semantik arama"""
    # Setup
    embedder = OpenAIEmbedder({'api_key': 'sk-...'})
    search = FAISSSearch({'dimension': 1536})
    
    # Dökümanlar
    documents = [
        "Python ile web geliştirme",
        "JavaScript framework'leri",
        "Makine öğrenmesi temelleri",
        "Derin öğrenme uygulamaları"
    ]
    
    # İndeksle
    embeddings = await embedder.embed(documents)
    await search.index(documents, embeddings.embeddings, 
                      [str(i) for i in range(len(documents))])
    
    # Ara
    query = "web programlama"
    query_emb = await embedder.embed(query)
    results = await search.search(query_emb.embeddings[0], top_k=2)
    
    return results
```

### 2. Metadata ile Filtreleme

```python
async def filtered_search():
    """Metadata filtresi ile arama"""
    search = QdrantSearch({
        'url': 'http://localhost:6333',
        'collection': 'articles'
    })
    
    # Metadata ile indeksle
    documents = [
        "Python tutorial",
        "JavaScript guide",
        "Machine learning basics",
        "Deep learning advanced"
    ]
    
    metadata = [
        {'language': 'python', 'level': 'beginner'},
        {'language': 'javascript', 'level': 'beginner'},
        {'language': 'python', 'level': 'intermediate'},
        {'language': 'python', 'level': 'advanced'}
    ]
    
    embeddings = await embedder.embed(documents)
    await search.index(documents, embeddings.embeddings,
                      [str(i) for i in range(len(documents))],
                      metadata=metadata)
    
    # Sadece Python dökümanlarında ara
    query_emb = await embedder.embed("programming tutorial")
    results = await search.search(
        query_emb.embeddings[0],
        top_k=5,
        filters={'language': 'python'}
    )
    
    return results
```

### 3. Batch İndeksleme

```python
async def batch_indexing(documents, batch_size=100):
    """Büyük veri setini batch'ler halinde indeksle"""
    embedder = OpenAIEmbedder({'api_key': 'sk-...', 'batch_size': 16})
    search = FAISSSearch({'dimension': 1536})
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        # Embedding'leri oluştur
        embeddings = await embedder.embed(batch)
        
        # İndeksle
        ids = [str(j) for j in range(i, i + len(batch))]
        await search.index(batch, embeddings.embeddings, ids)
        
        print(f"İndekslenen: {i + len(batch)}/{len(documents)}")
    
    return search
```

### 4. Incremental İndeksleme

```python
async def incremental_indexing(search, new_documents):
    """Mevcut index'e yeni dökümanlar ekle"""
    embedder = OpenAIEmbedder({'api_key': 'sk-...'})
    
    # Yeni dökümanlar için embedding
    embeddings = await embedder.embed(new_documents)
    
    # Mevcut index'e ekle
    new_ids = [f"new_{i}" for i in range(len(new_documents))]
    await search.index(new_documents, embeddings.embeddings, new_ids)
    
    print(f"{len(new_documents)} yeni döküman eklendi")
```

### 5. Pipeline ile Entegrasyon

```python
from bnsnlp import Pipeline, Config
from bnsnlp.core.registry import PluginRegistry

async def search_pipeline():
    """Tam pipeline: Preprocess + Embed + Search"""
    # Setup
    registry = PluginRegistry()
    registry.discover_plugins()
    config = Config()
    
    # Pipeline
    pipeline = Pipeline(config, registry)
    pipeline.add_step('preprocess', 'turkish')
    pipeline.add_step('embed', 'openai')
    
    # Dökümanları işle ve indeksle
    documents = [
        "Merhaba DÜNYA!",
        "Python PROGRAMLAMA",
        "Makine ÖĞRENMESI"
    ]
    
    search = FAISSSearch({'dimension': 1536})
    
    for i, doc in enumerate(documents):
        result = await pipeline.process(doc)
        await search.index([doc], [result.embeddings[0]], [str(i)])
    
    # Arama yap
    query = "programlama dili"
    query_result = await pipeline.process(query)
    results = await search.search(query_result.embeddings[0], top_k=2)
    
    return results
```

## İleri Seviye Kullanım

### Index Persistence (FAISS)

```python
import pickle
from pathlib import Path

class PersistentFAISSSearch:
    """Kalıcı FAISS index"""
    
    def __init__(self, config, index_path='faiss_index.pkl'):
        self.search = FAISSSearch(config)
        self.index_path = Path(index_path)
        self.documents = {}
        
        # Varsa yükle
        if self.index_path.exists():
            self.load()
    
    async def index(self, texts, embeddings, ids, metadata=None):
        """İndeksle ve kaydet"""
        await self.search.index(texts, embeddings, ids, metadata)
        
        # Dökümanları sakla
        for id_, text in zip(ids, texts):
            self.documents[id_] = text
        
        self.save()
    
    def save(self):
        """Index'i diske kaydet"""
        data = {
            'index': self.search.index,
            'documents': self.documents
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self):
        """Index'i diskten yükle"""
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
            self.search.index = data['index']
            self.documents = data['documents']
    
    async def search(self, query_embedding, top_k=10):
        """Ara"""
        return await self.search.search(query_embedding, top_k)

# Kullanım
search = PersistentFAISSSearch({'dimension': 1536})
await search.index(documents, embeddings, ids)

# Yeniden başlatıldığında otomatik yüklenir
search2 = PersistentFAISSSearch({'dimension': 1536})
results = await search2.search(query_embedding, top_k=5)
```

### Hybrid Search (Keyword + Semantic)

```python
from typing import List, Dict

class HybridSearch:
    """Keyword ve semantic search kombinasyonu"""
    
    def __init__(self, semantic_search, documents: List[str]):
        self.semantic_search = semantic_search
        self.documents = documents
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Basit keyword search"""
        query_lower = query.lower()
        results = []
        
        for i, doc in enumerate(self.documents):
            if query_lower in doc.lower():
                # Basit scoring: kelime sayısı
                score = doc.lower().count(query_lower)
                results.append({
                    'id': str(i),
                    'text': doc,
                    'score': score
                })
        
        # Score'a göre sırala
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    async def hybrid_search(self, query: str, query_embedding, 
                           top_k: int = 10, alpha: float = 0.5):
        """Hybrid search: alpha * semantic + (1-alpha) * keyword"""
        # Semantic search
        semantic_results = await self.semantic_search.search(
            query_embedding, top_k=top_k*2
        )
        
        # Keyword search
        keyword_results = self.keyword_search(query, top_k=top_k*2)
        
        # Skorları normalize et ve birleştir
        combined = {}
        
        # Semantic skorları ekle
        for result in semantic_results.results:
            combined[result.id] = {
                'text': result.text,
                'score': alpha * result.score
            }
        
        # Keyword skorları ekle
        max_keyword_score = max([r['score'] for r in keyword_results]) if keyword_results else 1
        for result in keyword_results:
            normalized_score = result['score'] / max_keyword_score
            if result['id'] in combined:
                combined[result['id']]['score'] += (1 - alpha) * normalized_score
            else:
                combined[result['id']] = {
                    'text': result['text'],
                    'score': (1 - alpha) * normalized_score
                }
        
        # Sırala ve döndür
        sorted_results = sorted(
            combined.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        return sorted_results[:top_k]

# Kullanım
hybrid = HybridSearch(faiss_search, documents)
results = await hybrid.hybrid_search(
    query="Python programlama",
    query_embedding=query_emb,
    top_k=5,
    alpha=0.7  # %70 semantic, %30 keyword
)
```

### Re-ranking

```python
from typing import List

class ReRanker:
    """Arama sonuçlarını yeniden sırala"""
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    async def rerank(self, query: str, results: List, top_k: int = 10):
        """Cross-encoder ile re-ranking"""
        # Query embedding
        query_emb = await self.embedder.embed(query)
        
        # Her sonuç için yeni skor hesapla
        reranked = []
        for result in results:
            # Döküman embedding
            doc_emb = await self.embedder.embed(result.text)
            
            # Cosine similarity
            similarity = self._cosine_similarity(
                query_emb.embeddings[0],
                doc_emb.embeddings[0]
            )
            
            reranked.append({
                'text': result.text,
                'original_score': result.score,
                'rerank_score': similarity,
                'combined_score': (result.score + similarity) / 2
            })
        
        # Yeni skora göre sırala
        reranked.sort(key=lambda x: x['combined_score'], reverse=True)
        return reranked[:top_k]
    
    def _cosine_similarity(self, vec1, vec2):
        """Cosine similarity hesapla"""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Kullanım
reranker = ReRanker(embedder)
initial_results = await search.search(query_embedding, top_k=20)
final_results = await reranker.rerank(query, initial_results.results, top_k=5)
```

### Multi-Index Search

```python
class MultiIndexSearch:
    """Birden fazla index'te ara"""
    
    def __init__(self, searches: Dict[str, any]):
        self.searches = searches
    
    async def search_all(self, query_embedding, top_k: int = 10):
        """Tüm index'lerde ara"""
        all_results = {}
        
        for name, search in self.searches.items():
            results = await search.search(query_embedding, top_k=top_k)
            all_results[name] = results
        
        return all_results
    
    async def search_best(self, query_embedding, top_k: int = 10):
        """En iyi sonuçları birleştir"""
        all_results = await self.search_all(query_embedding, top_k=top_k*2)
        
        # Tüm sonuçları birleştir
        combined = []
        for name, results in all_results.items():
            for result in results.results:
                combined.append({
                    'source': name,
                    'text': result.text,
                    'score': result.score
                })
        
        # Skora göre sırala
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:top_k]

# Kullanım
multi_search = MultiIndexSearch({
    'faiss': faiss_search,
    'qdrant': qdrant_search
})

results = await multi_search.search_best(query_embedding, top_k=5)
for result in results:
    print(f"[{result['source']}] {result['text']} ({result['score']:.4f})")
```

## Performans Optimizasyonu

### 1. Batch İndeksleme

```python
# ✅ İyi - Batch indeksleme
embeddings = await embedder.embed(documents)
await search.index(documents, embeddings.embeddings, ids)

# ❌ Kötü - Tek tek indeksleme
for doc, id_ in zip(documents, ids):
    emb = await embedder.embed(doc)
    await search.index([doc], [emb.embeddings[0]], [id_])
```

### 2. Index Caching

```python
# FAISS index'i cache'le
search = PersistentFAISSSearch({'dimension': 1536})

# İlk çalıştırma - index oluştur
if not search.index_path.exists():
    await search.index(documents, embeddings, ids)

# Sonraki çalıştırmalar - cache'den yükle
results = await search.search(query_embedding, top_k=10)
```

### 3. Top-K Optimizasyonu

```python
# Sadece ihtiyacınız kadar sonuç alın
results = await search.search(query_embedding, top_k=5)  # 5 yeterli

# Gereksiz yere çok sonuç almayın
# results = await search.search(query_embedding, top_k=1000)  # Yavaş
```

### 4. Similarity Threshold

```python
# Düşük skorlu sonuçları filtrele
results = await search.search(query_embedding, top_k=20)

filtered_results = [
    r for r in results.results 
    if r.score >= 0.7  # Minimum benzerlik
]
```

## Hata Yönetimi

### Connection Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def resilient_search(search, query_embedding, top_k=10):
    """Retry ile arama"""
    return await search.search(query_embedding, top_k)

# Kullanım
try:
    results = await resilient_search(qdrant_search, query_emb, top_k=5)
except Exception as e:
    print(f"3 denemeden sonra başarısız: {e}")
```

### Fallback Strategy

```python
async def search_with_fallback(query_embedding, top_k=10):
    """Fallback ile arama"""
    # Önce Qdrant dene
    try:
        results = await qdrant_search.search(query_embedding, top_k)
        return results
    except Exception as e:
        print(f"Qdrant hatası: {e}, FAISS'e geçiliyor")
        
        # Fallback: FAISS
        try:
            results = await faiss_search.search(query_embedding, top_k)
            return results
        except Exception as e2:
            print(f"FAISS hatası: {e2}")
            raise
```

## Best Practices

### 1. Index Boyutu Yönetimi

```python
# Büyük index'ler için batch işleme
async def manage_large_index(documents, max_batch=1000):
    """Büyük index'i yönet"""
    search = FAISSSearch({'dimension': 1536})
    
    for i in range(0, len(documents), max_batch):
        batch = documents[i:i + max_batch]
        embeddings = await embedder.embed(batch)
        ids = [str(j) for j in range(i, i + len(batch))]
        await search.index(batch, embeddings.embeddings, ids)
        
        # Periyodik kaydetme
        if i % 5000 == 0:
            save_checkpoint(search, i)
```

### 2. Metadata Kullanımı

```python
# ✅ İyi - Zengin metadata
metadata = {
    'category': 'technology',
    'date': '2024-01-15',
    'author': 'John Doe',
    'tags': ['python', 'ml', 'nlp']
}

# ❌ Kötü - Metadata yok
metadata = {}
```

### 3. Query Optimization

```python
# ✅ İyi - Preprocessing ile query
query = "Python PROGRAMLAMA dili"
preprocessed = await preprocessor.process(query)
query_emb = await embedder.embed(preprocessed.text)

# ❌ Kötü - Ham query
query_emb = await embedder.embed(query)
```

### 4. Result Validation

```python
# Sonuçları validate et
results = await search.search(query_embedding, top_k=10)

valid_results = [
    r for r in results.results
    if r.score >= 0.5 and len(r.text) > 10
]
```

## İlgili Bölümler

- [API: Search](../api/search/index.md) - Search API referansı
- [Embedding Oluşturma](embeddings.md) - Embedding'leri oluşturma
- [Pipeline Kullanımı](pipeline.md) - Pipeline ile entegrasyon
- [Örnekler](../examples/notebooks.md) - Jupyter notebook örnekleri
