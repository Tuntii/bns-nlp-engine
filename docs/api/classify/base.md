# Classify Base

Tüm classifier'ların implement etmesi gereken temel interface.

## Sınıf Tanımları

::: bnsnlp.classify.base.BaseClassifier
    options:
      show_source: true
      heading_level: 3

::: bnsnlp.classify.base.Entity
    options:
      show_source: true
      heading_level: 3

::: bnsnlp.classify.base.ClassifyResult
    options:
      show_source: true
      heading_level: 3

## Kullanım

### Mevcut Classifier Kullanımı

```python
from bnsnlp.classify import TurkishClassifier

config = {
    'intent_model': 'path/to/intent/model',
    'entity_model': 'path/to/entity/model',
    'use_gpu': True
}

classifier = TurkishClassifier(config)

# Tek metin
text = "Yarın saat 14:00'te İstanbul'da toplantı var"
result = await classifier.classify(text)

print(f"Intent: {result.intent} ({result.intent_confidence:.2f})")
for entity in result.entities:
    print(f"  {entity.type}: {entity.text} ({entity.confidence:.2f})")

# Batch işleme
texts = [
    "Bugün hava nasıl?",
    "Yarın İzmir'e gidiyorum",
    "Saat kaç oldu?"
]
results = await classifier.classify_batch(texts)
```

### Custom Classifier Oluşturma

```python
from bnsnlp.classify.base import BaseClassifier, Entity, ClassifyResult
from typing import Dict, Any, List

class CustomClassifier(BaseClassifier):
    """Özel classifier implementasyonu"""
    
    name = "custom"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intent_labels = config.get('intent_labels', [])
        self.entity_types = config.get('entity_types', [])
        # Model yükleme
        self.intent_model = self._load_intent_model()
        self.entity_model = self._load_entity_model()
    
    def _load_intent_model(self):
        """Intent model yükle"""
        # Model loading logic
        return None
    
    def _load_entity_model(self):
        """Entity model yükle"""
        # Model loading logic
        return None
    
    async def classify(self, text: str) -> ClassifyResult:
        """Metni sınıflandır"""
        # Intent classification
        intent, intent_conf = await self._classify_intent(text)
        
        # Entity extraction
        entities = await self._extract_entities(text)
        
        return ClassifyResult(
            intent=intent,
            intent_confidence=intent_conf,
            entities=entities,
            metadata={
                'text_length': len(text),
                'entity_count': len(entities)
            }
        )
    
    async def _classify_intent(self, text: str) -> tuple[str, float]:
        """Intent'i sınıflandır"""
        import asyncio
        loop = asyncio.get_event_loop()
        
        # CPU-bound işlem için thread pool
        result = await loop.run_in_executor(
            None,
            self._compute_intent,
            text
        )
        
        return result
    
    def _compute_intent(self, text: str) -> tuple[str, float]:
        """Senkron intent hesaplama"""
        # Model inference
        # Örnek: basit keyword matching
        text_lower = text.lower()
        
        if 'hava' in text_lower or 'sıcaklık' in text_lower:
            return ('weather', 0.95)
        elif 'toplantı' in text_lower or 'randevu' in text_lower:
            return ('meeting', 0.90)
        else:
            return ('general', 0.50)
    
    async def _extract_entities(self, text: str) -> List[Entity]:
        """Entity'leri çıkar"""
        import asyncio
        loop = asyncio.get_event_loop()
        
        entities = await loop.run_in_executor(
            None,
            self._compute_entities,
            text
        )
        
        return entities
    
    def _compute_entities(self, text: str) -> List[Entity]:
        """Senkron entity extraction"""
        entities = []
        
        # Basit pattern matching (gerçekte NER model kullanılır)
        import re
        
        # Tarih pattern
        date_pattern = r'\b(bugün|yarın|dün)\b'
        for match in re.finditer(date_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(),
                type='DATE',
                start=match.start(),
                end=match.end(),
                confidence=0.95
            ))
        
        # Saat pattern
        time_pattern = r'\b\d{1,2}:\d{2}\b'
        for match in re.finditer(time_pattern, text):
            entities.append(Entity(
                text=match.group(),
                type='TIME',
                start=match.start(),
                end=match.end(),
                confidence=0.90
            ))
        
        # Şehir pattern (basit örnek)
        cities = ['İstanbul', 'Ankara', 'İzmir', 'Bursa']
        for city in cities:
            if city in text:
                start = text.index(city)
                entities.append(Entity(
                    text=city,
                    type='LOCATION',
                    start=start,
                    end=start + len(city),
                    confidence=0.85
                ))
        
        return entities
    
    async def classify_batch(self, texts: List[str]) -> List[ClassifyResult]:
        """Batch sınıflandırma"""
        results = []
        for text in texts:
            result = await self.classify(text)
            results.append(result)
        return results
```

### Transformer-Based Classifier

```python
from transformers import pipeline
import torch

class TransformerClassifier(BaseClassifier):
    """Transformer tabanlı classifier"""
    
    def __init__(self, config: Dict[str, Any]):
        self.use_gpu = config.get('use_gpu', True)
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Pipeline'ları yükle
        self.intent_pipeline = pipeline(
            'text-classification',
            model=config['intent_model'],
            device=self.device
        )
        
        self.entity_pipeline = pipeline(
            'ner',
            model=config['entity_model'],
            device=self.device,
            aggregation_strategy='simple'
        )
    
    async def classify(self, text: str) -> ClassifyResult:
        """Transformer ile sınıflandır"""
        import asyncio
        loop = asyncio.get_event_loop()
        
        # Paralel olarak intent ve entity extraction
        intent_task = loop.run_in_executor(None, self._get_intent, text)
        entity_task = loop.run_in_executor(None, self._get_entities, text)
        
        intent_result, entity_results = await asyncio.gather(intent_task, entity_task)
        
        return ClassifyResult(
            intent=intent_result['label'],
            intent_confidence=intent_result['score'],
            entities=entity_results
        )
    
    def _get_intent(self, text: str) -> Dict[str, Any]:
        """Intent al"""
        results = self.intent_pipeline(text)
        return results[0]
    
    def _get_entities(self, text: str) -> List[Entity]:
        """Entity'leri al"""
        results = self.entity_pipeline(text)
        
        entities = []
        for result in results:
            entities.append(Entity(
                text=result['word'],
                type=result['entity_group'],
                start=result['start'],
                end=result['end'],
                confidence=result['score']
            ))
        
        return entities
```

## Data Models

### Entity

Tek bir entity.

```python
entity = Entity(
    text="İstanbul",
    type="LOCATION",
    start=25,
    end=33,
    confidence=0.95,
    metadata={'country': 'TR'}
)
```

### ClassifyResult

Sınıflandırma sonucu.

```python
result = ClassifyResult(
    intent="meeting",
    intent_confidence=0.92,
    entities=[
        Entity(text="yarın", type="DATE", start=0, end=5, confidence=0.95),
        Entity(text="14:00", type="TIME", start=11, end=16, confidence=0.90),
        Entity(text="İstanbul", type="LOCATION", start=20, end=28, confidence=0.95)
    ],
    metadata={
        'text_length': 45,
        'processing_time_ms': 120
    }
)
```

## Best Practices

### 1. GPU Acceleration

GPU kullanımı:

```python
import torch

class GPUClassifier(BaseClassifier):
    def __init__(self, config: Dict[str, Any]):
        self.use_gpu = config.get('use_gpu', True)
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Model'i GPU'ya taşı
        self.model = self._load_model().to(self.device)
    
    def _compute_intent(self, text: str) -> tuple[str, float]:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Process outputs
            return intent, confidence
```

### 2. Batch Processing

Verimli batch işleme:

```python
async def classify_batch(self, texts: List[str]) -> List[ClassifyResult]:
    """Batch sınıflandırma"""
    import asyncio
    
    # Paralel işleme
    tasks = [self.classify(text) for text in texts]
    results = await asyncio.gather(*tasks)
    
    return results
```

### 3. Confidence Thresholding

Düşük confidence'lı sonuçları filtrele:

```python
async def classify(self, text: str, min_confidence: float = 0.5) -> ClassifyResult:
    """Confidence threshold ile sınıflandır"""
    result = await self._classify_internal(text)
    
    # Intent confidence kontrolü
    if result.intent_confidence < min_confidence:
        result.intent = 'unknown'
    
    # Entity confidence kontrolü
    result.entities = [
        e for e in result.entities 
        if e.confidence >= min_confidence
    ]
    
    return result
```

### 4. Entity Post-Processing

Entity'leri normalize et:

```python
def _normalize_entities(self, entities: List[Entity]) -> List[Entity]:
    """Entity'leri normalize et"""
    normalized = []
    
    for entity in entities:
        # Duplicate'leri kaldır
        if not any(e.text == entity.text and e.type == entity.type for e in normalized):
            # Text'i normalize et
            entity.text = entity.text.strip()
            
            # Type'ı standardize et
            entity.type = entity.type.upper()
            
            normalized.append(entity)
    
    return normalized
```

### 5. Caching

Sık kullanılan sınıflandırmaları cache'le:

```python
from functools import lru_cache
import hashlib

class CachedClassifier(BaseClassifier):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cache = {}
    
    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    async def classify(self, text: str) -> ClassifyResult:
        key = self._cache_key(text)
        
        if key in self.cache:
            return self.cache[key]
        
        result = await self._classify_internal(text)
        self.cache[key] = result
        
        return result
```

## İlgili Bölümler

- [Turkish Classifier](turkish.md) - Türkçe implementasyonu
- [Plugin Geliştirme](../../plugins/classifier.md) - Detaylı plugin kılavuzu
- [Kullanım Kılavuzu: Classification](../../guide/classification.md) - Kullanım örnekleri
